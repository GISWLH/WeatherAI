"""Aurora's multi-variable, multi-level Perceiver encoder.

Surface (+static) variables and every pressure level are patch-embedded with
:class:`LevelPatchEmbed`; the ``C_A`` pressure levels of each patch are compressed into
``latent_levels - 1`` latent tokens with a Perceiver cross-attention (``level_agg``); the
surface token is kept as its own latent level. Position, patch-scale, lead-time and
absolute-time Fourier embeddings are added. Parameter names follow microsoft/aurora (MIT).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .fourier import (
    absolute_time_expansion,
    lead_time_expansion,
    lead_time_expansion_v3,
    levels_expansion,
    pos_scale_encoding,
)
from .patch_embed import LevelPatchEmbed
from .perceiver import MLP, PerceiverResampler


def init_weights(m: nn.Module) -> None:
    """Truncated-normal(0.02) for Linear / Conv, LayerNorm = (1, 0) (as upstream)."""
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.0)


class Perceiver3DEncoder(nn.Module):
    def __init__(
        self,
        surf_vars: Sequence[str],
        static_vars: Sequence[str],
        atmos_vars: Sequence[str],
        patch_size: int = 4,
        latent_levels: int = 4,
        embed_dim: int = 512,
        num_heads: int = 16,
        head_dim: int = 32,
        drop_rate: float = 0.0,
        depth: int = 1,
        mlp_ratio: float = 4.0,
        max_history_size: int = 2,
        perceiver_ln_eps: float = 1e-5,
        stabilise_level_agg: bool = False,
        use_updated_lead_time_embedding: bool = False,
    ):
        super().__init__()
        assert latent_levels > 1, "At least two latent levels are required."
        self.embed_dim, self.patch_size, self.latent_levels = embed_dim, patch_size, latent_levels
        self.use_updated_lead_time_embedding = use_updated_lead_time_embedding
        self.static_vars = tuple(static_vars)
        # static variables are embedded as extra *surface* channels
        surf_names = tuple(surf_vars) + self.static_vars

        # one latent level is reserved for the surface; the others aggregate the pressure levels
        self.atmos_latents = nn.Parameter(torch.randn(latent_levels - 1, embed_dim))
        self.surf_level_encoding = nn.Parameter(torch.randn(embed_dim))
        self.surf_mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout=drop_rate)
        self.surf_norm = nn.LayerNorm(embed_dim)

        self.pos_embed = nn.Linear(embed_dim, embed_dim)
        self.scale_embed = nn.Linear(embed_dim, embed_dim)
        self.lead_time_embed = nn.Linear(embed_dim, embed_dim)
        self.absolute_time_embed = nn.Linear(embed_dim, embed_dim)
        self.atmos_levels_embed = nn.Linear(embed_dim, embed_dim)

        self.surf_token_embeds = LevelPatchEmbed(surf_names, patch_size, embed_dim, max_history_size)
        self.atmos_token_embeds = LevelPatchEmbed(tuple(atmos_vars), patch_size, embed_dim, max_history_size)
        self.level_agg = PerceiverResampler(
            embed_dim, embed_dim, depth, head_dim, num_heads, mlp_ratio, drop_rate, True,
            perceiver_ln_eps, stabilise_level_agg,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.apply(init_weights)
        nn.init.trunc_normal_(self.atmos_latents, std=0.02)  # as the HF Perceiver
        nn.init.trunc_normal_(self.surf_level_encoding, std=0.02)

    def aggregate_levels(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C_A, L, D) -> (B, C, L, D)``: per patch, cross-attend latents to the levels."""
        B, _, L, _ = x.shape
        latents = self.atmos_latents.to(dtype=x.dtype)[None, :, None, :].expand(B, -1, L, -1)
        x = x.permute(0, 2, 1, 3).flatten(0, 1)  # (B*L, C_A, D)
        latents = latents.permute(0, 2, 1, 3).flatten(0, 1)  # (B*L, C, D)
        x = self.level_agg(latents, x)  # (B*L, C, D)
        return x.unflatten(0, (B, L)).permute(0, 2, 1, 3)  # (B, C, L, D)

    def forward(
        self,
        surf: torch.Tensor,
        static: torch.Tensor,
        atmos: torch.Tensor,
        lat: torch.Tensor,
        lon: torch.Tensor,
        atmos_levels: Sequence[float],
        times: Sequence,
        lead_times: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            surf: ``(B, T, V_S, H, W)``; static: ``(B, T, V_static, H, W)`` (already expanded);
            atmos: ``(B, T, V_A, C_A, H, W)``; lat ``(H,)``, lon ``(W,)`` in degrees;
            times: ``B`` datetimes; lead_times: ``(B,)`` hours.
        Returns:
            tokens ``(B, (latent_levels) * (H/P) * (W/P), D)`` ordered ``(C, H/P, W/P)``.
        """
        B, T, _, C, H, W = atmos.shape
        x_surf = torch.cat((surf, static), dim=2).permute(0, 2, 1, 3, 4)  # (B, V_S+V_st, T, H, W)
        x_surf = self.surf_token_embeds(x_surf, self.surf_token_embeds.var_names)  # (B, L, D)
        dtype = x_surf.dtype

        x_atmos = atmos.permute(0, 3, 2, 1, 4, 5).reshape(B * C, -1, T, H, W)  # ((B C), V_A, T, H, W)
        x_atmos = self.atmos_token_embeds(x_atmos, self.atmos_token_embeds.var_names)  # ((B C), L, D)
        x_atmos = x_atmos.reshape(B, C, *x_atmos.shape[1:])  # (B, C_A, L, D)

        x_surf = x_surf + self.surf_level_encoding[None, None, :].to(dtype=dtype)
        x_surf = x_surf + self.surf_norm(self.surf_mlp(x_surf))  # surface is not aggregated: MLP only

        lv = torch.tensor(list(atmos_levels), device=x_atmos.device)
        lv_embed = self.atmos_levels_embed(levels_expansion(lv, self.embed_dim).to(dtype=dtype))
        x_atmos = x_atmos + lv_embed[None, :, None, :]
        x_atmos = self.aggregate_levels(x_atmos)  # (B, C, L, D)

        x = torch.cat((x_surf.unsqueeze(1), x_atmos), dim=1)  # (B, C+1, L, D)

        pos, scale = pos_scale_encoding(self.embed_dim, lat.float(), lon.float(), self.patch_size)
        x = x + self.pos_embed(pos[None, None].to(device=x.device, dtype=dtype))
        x = x + self.scale_embed(scale[None, None].to(device=x.device, dtype=dtype))
        x = x.reshape(B, -1, self.embed_dim)

        expand = lead_time_expansion_v3 if self.use_updated_lead_time_embedding else lead_time_expansion
        x = x + self.lead_time_embed(expand(lead_times, self.embed_dim).to(dtype=dtype)).unsqueeze(1)
        abs_t = torch.tensor([t.timestamp() / 3600 for t in times], dtype=torch.float32, device=x.device)
        x = x + self.absolute_time_embed(absolute_time_expansion(abs_t, self.embed_dim).to(dtype=dtype)).unsqueeze(1)
        return self.pos_drop(x)
