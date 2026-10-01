"""Aurora's Perceiver decoder: de-aggregate latent levels into pressure levels, then per-variable
linear patch reconstruction. Parameter names follow microsoft/aurora (MIT)."""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn

from .encoder import init_weights
from .fourier import levels_expansion
from .perceiver import PerceiverResampler


def unpatchify(x: torch.Tensor, V: int, H: int, W: int, P: int) -> torch.Tensor:
    """``(B, (H/P)*(W/P), C, V*P*P) -> (B, V, C, H, W)``."""
    B, C = x.size(0), x.size(2)
    h, w = H // P, W // P
    assert x.size(1) == h * w and x.size(-1) == V * P**2
    x = x.reshape(B, h, w, C, P, P, V).permute(0, 6, 3, 1, 4, 2, 5)  # B V C h P1 w P2
    return x.reshape(B, V, C, h * P, w * P)


class Perceiver3DDecoder(nn.Module):
    def __init__(
        self,
        surf_vars: Sequence[str],
        atmos_vars: Sequence[str],
        patch_size: int = 4,
        embed_dim: int = 1024,
        depth: int = 1,
        head_dim: int = 64,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.0,
        perceiver_ln_eps: float = 1e-5,
    ):
        super().__init__()
        self.patch_size, self.embed_dim = patch_size, embed_dim
        self.surf_vars, self.atmos_vars = tuple(surf_vars), tuple(atmos_vars)
        self.level_decoder = PerceiverResampler(
            embed_dim, embed_dim, depth, head_dim, num_heads, mlp_ratio, drop_rate, True, perceiver_ln_eps
        )
        self.surf_heads = nn.ModuleDict({n: nn.Linear(embed_dim, patch_size**2) for n in self.surf_vars})
        self.atmos_heads = nn.ModuleDict({n: nn.Linear(embed_dim, patch_size**2) for n in self.atmos_vars})
        self.atmos_levels_embed = nn.Linear(embed_dim, embed_dim)
        self.apply(init_weights)

    def deaggregate_levels(self, level_embed: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """``level_embed (B, L, C_A, D)``, ``x (B, L, C, D)`` -> ``(B, L, C_A, D)``."""
        B, L, C, D = level_embed.shape
        out = self.level_decoder(level_embed.flatten(0, 1), x.flatten(0, 1))
        return out.reshape(B, L, C, D)

    def forward(
        self,
        x: torch.Tensor,
        lat: torch.Tensor,
        lon: torch.Tensor,
        atmos_levels: Sequence[float],
        patch_res: Tuple[int, int, int],
    ):
        """``x``: backbone output ``(B, C*H/P*W/P, D)`` -> ``(surf (B,V_S,H,W), atmos (B,V_A,C_A,H,W))``."""
        B = x.shape[0]
        H, W = lat.shape[0], lon.shape[-1]
        C, h, w = patch_res
        x = x.reshape(B, C, h * w, -1).permute(0, 2, 1, 3)  # (B, L, C, D)

        # surface: head per variable on latent level 0
        xs = torch.stack([self.surf_heads[n](x[..., :1, :]) for n in self.surf_vars], dim=-1)  # (B,L,1,P²,V)
        surf = unpatchify(xs.reshape(*xs.shape[:3], -1), len(self.surf_vars), H, W, self.patch_size).squeeze(2)

        # atmosphere: query with level embeddings, cross-attend to the C-1 atmospheric latents
        lv = torch.tensor(list(atmos_levels), device=x.device)
        lv_embed = self.atmos_levels_embed(levels_expansion(lv, self.embed_dim).to(dtype=x.dtype))  # (C_A, D)
        lv_embed = lv_embed.expand(B, x.size(1), -1, -1)
        xa = self.deaggregate_levels(lv_embed, x[..., 1:, :])  # (B, L, C_A, D)
        xa = torch.stack([self.atmos_heads[n](xa) for n in self.atmos_vars], dim=-1)  # (B,L,C_A,P²,V)
        atmos = unpatchify(xa.reshape(*xa.shape[:3], -1), len(self.atmos_vars), H, W, self.patch_size)
        return surf, atmos
