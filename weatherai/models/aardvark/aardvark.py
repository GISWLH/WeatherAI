"""Aardvark Weather — **processor (forecast) module only**, PyTorch re-implementation.

Aardvark Weather (Allen et al., *Nature* 641, 2025) is an end-to-end system: an *encoder*
(ConvCNP/set-conv + ViT that assimilates raw satellite / in-situ observations into a gridded
initial state), a *processor* (ViT that steps the 24-channel 1.5° state forward 24 h) and a
*decoder* (set-conv + MLP that produces station forecasts). Official code (PyTorch, CC0):
https://github.com/anna-allen/aardvark-weather-public, weights on HF ``av555/aardvark-weather``.

This module re-implements ONLY the **processor ViT** (``ConvCNPWeather(mode="forecast",
decoder="vit")``): per-variable patch embeddings → variable aggregation attention → ViT blocks with
lead-time embedding → patch head. Parameter names match the official ``decoder_lr.*`` keys, so the
official processor checkpoint loads with ``strict=True`` (``load_official_processor``), and the output
was compared numerically with the official ``vit.ViT`` on the official sample input (see
``tests/models/aardvark``). NOT implemented: the observation encoder (set-conv + ViT assimilation),
the station decoder, data loaders, end-to-end finetune.

Grid convention (official): input ``(B, 35, 240, 121)`` = 24 normalised state channels
(u10, v10, t2m, mslp, z/q/t/u/v at 200/500/700/850 hPa) + 11 auxiliary channels (lon 240 × lat 121,
1.5°); output ``(B, 121, 240, 24)`` = normalised 24-h *tendency*.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- pos-embeddings
def _sincos_1d(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float32) / (embed_dim / 2.0)
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def sincos_pos_embed_2d(embed_dim: int, grid_h: int, grid_w: int) -> np.ndarray:
    gh = np.arange(grid_h, dtype=np.float32)
    gw = np.arange(grid_w, dtype=np.float32)
    grid = np.stack(np.meshgrid(gw, gh), axis=0).reshape(2, 1, grid_h, grid_w)
    return np.concatenate([_sincos_1d(embed_dim // 2, grid[0]), _sincos_1d(embed_dim // 2, grid[1])], axis=1)


# ----------------------------------------------------------------------------- building blocks
class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class _Mlp(nn.Module):
    def __init__(self, dim: int, ratio: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, int(dim * ratio))
        self.fc2 = nn.Linear(int(dim * ratio), dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class _PatchEmbed(nn.Module):
    def __init__(self, patch: int, in_ch: int, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)


class MLP(nn.Module):
    """Official Aardvark MLP: Linear-ReLU, ``h_layers`` x (Linear-ReLU), Linear (``mlp.0``, ``mlp.2.0`` ... keys)."""

    def __init__(self, in_channels: int, out_channels: int, h_channels: int = 64, h_layers: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, h_channels), nn.ReLU(),
            *[nn.Sequential(nn.Linear(h_channels, h_channels), nn.ReLU()) for _ in range(h_layers)],
            nn.Linear(h_channels, out_channels),
        )

    def forward(self, x):
        return self.mlp(x)


class ViT(nn.Module):
    """Aardvark ViT (adapted from ClimaX). Same parameter layout as the official ``vit.ViT``.

    * ``per_var_embedding=True``  (processor): one patch embedding per input variable -> variable
      aggregation (cross-attention with a learned query) -> blocks.
    * ``per_var_embedding=False`` (encoder / "vit_assimilation"): pointwise MLP ``mlp`` (``mlp_in`` -> ``in_channels``) ->
      a single patch embedding of all channels -> blocks. (The unused ``var_embed`` / ``var_query`` / ``var_agg``
      parameters are kept because the official checkpoints contain them.)
    """

    def __init__(self, in_channels, out_channels, embed_dim, img_size, patch_size, depth, decoder_depth, num_heads,
                 mlp_ratio, per_var_embedding: bool = True, mlp_in: int = 277):
        super().__init__()
        self.img_size, self.p, self.out_dim, self.in_channels = tuple(img_size), patch_size, out_channels, in_channels
        self.per_var_embedding = per_var_embedding
        gh, gw = img_size[0] // patch_size, img_size[1] // patch_size
        if per_var_embedding:
            self.token_embeds = nn.ModuleList(_PatchEmbed(patch_size, 1, embed_dim) for _ in range(in_channels))
        else:
            self.token_embeds = nn.ModuleList([_PatchEmbed(patch_size, in_channels, embed_dim)])
        self.var_embed = nn.Parameter(torch.zeros(1, in_channels, embed_dim))
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, gh * gw, embed_dim))
        self.lead_time_embed = nn.Linear(1, embed_dim)
        self.blocks = nn.ModuleList(_Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth))
        self.norm = nn.LayerNorm(embed_dim)
        head = []
        for _ in range(decoder_depth):
            head += [nn.Linear(embed_dim, embed_dim), nn.GELU()]
        head.append(nn.Linear(embed_dim, out_channels * patch_size**2))
        self.head = nn.Sequential(*head)
        self.apply(self._init)
        self.pos_embed.data.copy_(torch.from_numpy(sincos_pos_embed_2d(embed_dim, gh, gw)).float().unsqueeze(0))
        self.var_embed.data.copy_(torch.from_numpy(_sincos_1d(embed_dim, np.arange(in_channels))).float().unsqueeze(0))
        if not per_var_embedding:  # created after init in the official code => default torch init
            self.mlp = MLP(mlp_in, in_channels)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight.view(m.weight.shape[0], -1), std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, lead_times: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x (B, C, H, W); lead_times (B, 1) → (B, H', W', out) with H'=H//p*p, W'=W//p*p."""
        B = x.shape[0]
        if lead_times is None:
            lead_times = torch.ones(B, 1, dtype=x.dtype, device=x.device)
        if self.per_var_embedding:
            toks = torch.stack([self.token_embeds[i](x[:, i : i + 1]) for i in range(self.in_channels)], dim=1)  # (B,V,L,D)
            toks = toks + self.var_embed.unsqueeze(2)
            b, _, L, D = toks.shape
            t = toks.permute(0, 2, 1, 3).flatten(0, 1)  # (B*L, V, D)
            q = self.var_query.expand(t.shape[0], -1, -1)
            t, _ = self.var_agg(q, t, t)  # aggregate the variables of every patch
            t = t.squeeze(1).unflatten(0, (b, L))  # (B, L, D)
        else:
            x = self.mlp(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # pointwise channel mixing 277 -> 256
            t = self.token_embeds[0](x)  # (B, L, D)
        t = t + self.pos_embed + self.lead_time_embed(lead_times[:, 0].unsqueeze(-1)).unsqueeze(1)
        for blk in self.blocks:
            t = blk(t)
        t = self.head(self.norm(t))  # (B, L, out*p*p)
        gh, gw, p, c = self.img_size[0] // self.p, self.img_size[1] // self.p, self.p, self.out_dim
        t = t.reshape(B, gh, gw, p, p, c)
        img = torch.einsum("nhwpqc->nchpwq", t).reshape(B, c, gh * p, gw * p)
        return img.permute(0, 2, 3, 1)


_ViT = ViT  # backwards-compatible alias


# ----------------------------------------------------------------------------- public module
class AardvarkProcessor(nn.Module):
    """Aardvark processor: normalised gridded state (+aux) → normalised 24 h tendency.

    ``forward(y_context, lead_time)``: ``y_context`` ``(B, in_channels, H, W)`` with ``H`` = lon
    nodes, ``W`` = lat nodes (official: 240 × 121); ``lead_time`` ``(B, 1)`` (official: 1.0).
    Returns ``(B, W, H, out_channels)`` (official layout, lat-major).
    Defaults reproduce the official processor configuration (see ``num_parameters``).
    """

    def __init__(
        self,
        in_channels: int = 35,
        out_channels: int = 24,
        img_size: Tuple[int, int] = (240, 121),
        embed_dim: int = 512,
        depth: int = 16,
        patch_size: int = 5,
        num_heads: int = 16,
        decoder_depth: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.img_size = tuple(img_size)
        self.out_channels = out_channels
        self.decoder_lr = ViT(in_channels, out_channels, embed_dim, img_size, patch_size, depth, decoder_depth, num_heads, mlp_ratio)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, y_context: torch.Tensor, lead_time: Optional[torch.Tensor] = None) -> torch.Tensor:
        if lead_time is None:
            lead_time = torch.ones(y_context.shape[0], 1, dtype=y_context.dtype, device=y_context.device)
        H, W = self.img_size
        if tuple(y_context.shape[-2:]) != (H, W):
            raise ValueError(f"expected spatial {(H, W)}, got {tuple(y_context.shape[-2:])}")
        x = self.decoder_lr(y_context, lead_time)  # (B, H', W', C), H' = H//p*p
        x = F.interpolate(x.permute(0, 3, 1, 2), size=(H, W))  # back to (B, C, H, W)
        return x.permute(0, 2, 3, 1).permute(0, 2, 1, 3)  # (B, W, H, C) — official layout

    def forecast_step(self, y_context, input_mean, input_std, diff_mean, diff_std, lead_time=None):
        """One official 24 h step incl. (un)normalisation (official ``process_forecast_output``).

        Means/stds: tensors of shape (out_channels,). Returns ``(forecast (B, W, H, C) physical units,
        next_y_context (B, C_in, H, W))`` where the aux channels (index ≥ out_channels) are carried over.
        """
        C = self.out_channels
        x = self(y_context, lead_time)
        base = y_context[:, :C].permute(0, 3, 2, 1) * input_std + input_mean  # (B, W, H, C)
        fc = base + diff_mean + x * diff_std
        nxt = ((fc - input_mean) / input_std).permute(0, 3, 2, 1)  # (B, C, H, W)
        return fc, torch.cat([nxt, y_context[:, C:]], dim=1)


def AardvarkProcessor_lite(
    img_size: Tuple[int, int] = (60, 31),
    in_channels: int = 35,
    out_channels: int = 24,
    embed_dim: int = 64,
    depth: int = 2,
    num_heads: int = 4,
    patch_size: int = 5,
    decoder_depth: int = 2,
) -> AardvarkProcessor:
    """Tiny random-init processor (≈0.8 M params at defaults) on a 60×31 (6°) grid for smoke tests."""
    return AardvarkProcessor(in_channels, out_channels, img_size, embed_dim, depth, patch_size, num_heads, decoder_depth)


def load_official_processor(path: str, strict: bool = True, device="cpu") -> AardvarkProcessor:
    """Build the full-size processor and load an official ``forecast_N/epoch_0`` checkpoint (HF
    ``av555/aardvark-weather`` dataset repo, ``trained_model/processor/forecast_1/epoch_0``, 648 MB).

    Official files are pickled training checkpoints; ``weights_only=False`` is required, so only load
    files obtained from the official source.
    """
    model = AardvarkProcessor()
    sd = torch.load(path, map_location=device, weights_only=False)["model_state_dict"]
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    extra = {k for k in sd if not k.startswith("decoder_lr.")}  # unused setconv / MLP params of the official wrapper
    sd = {k: v for k, v in sd.items() if k.startswith("decoder_lr.")}
    model.load_state_dict(sd, strict=strict)
    model.ignored_official_keys = sorted(extra)
    return model.to(device)
