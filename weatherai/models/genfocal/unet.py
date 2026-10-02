"""Lite 3-D U-ViT flow network in the spirit of GenFocal's ``RescaledUnet3d`` (conv residual blocks with noise-level FiLM, axial
spatial/temporal attention at coarse levels, skip connections, channel conditioning, periodic-longitude padding).

This is NOT weight-compatible with the official Flax ``UNet3d`` (no ``FilteredResize``, simplified blocks): it is a trainable
stand-in with the same call interface ``net(x, sigma, cond)`` and channels-last layout ``(batch, time, lon, lat, channels)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GenFocalNetConfig:
    out_channels: int = 10
    num_channels: Sequence[int] = (128,) * 6
    downsample_ratio: Sequence[int] = (2,) * 6
    num_blocks: int = 6
    noise_embed_dim: int = 256
    cond_keys: Sequence[str] = ("channel:mean", "channel:std")
    use_spatial_attention: Sequence[bool] = (False, False, False, False, True, True)
    use_temporal_attention: Sequence[bool] = (False, False, False, False, True, True)
    num_heads: int = 8
    normalize_qk: bool = True
    dropout_rate: float = 0.0
    time_rescale: float = 1000.0

    @classmethod
    def lite(cls, out_channels: int = 4) -> "GenFocalNetConfig":
        return cls(out_channels=out_channels, num_channels=(16, 32), downsample_ratio=(2, 2), num_blocks=1,
                   noise_embed_dim=32, use_spatial_attention=(False, True), use_temporal_attention=(False, True), num_heads=4)


class FourierEmbedding(nn.Module):
    """sin/cos features of the (rescaled) noise level followed by a 2-layer MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("freqs", torch.exp(torch.linspace(0.0, math.log(1000.0), dim // 2)))
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        a = s[:, None].float() * self.freqs[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], -1))


class Conv(nn.Module):
    """(1, 3, 3) conv over (time, lon, lat) with circular longitude and zero latitude padding (the official 'LONLAT' mode)."""

    def __init__(self, cin: int, cout: int, k: int = 3):
        super().__init__()
        self.k = k
        self.conv = nn.Conv3d(cin, cout, (1, k, k))

    def forward(self, x):                                  # (B, C, T, X, Y)
        p = self.k // 2
        x = torch.cat([x[:, :, :, -p:], x, x[:, :, :, :p]], 3) if p else x
        x = F.pad(x, (p, p, 0, 0, 0, 0))
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, emb: int, dropout: float):
        super().__init__()
        g = lambda c: nn.GroupNorm(min(c // 4, 32) or 1, c)
        self.n1, self.c1 = g(cin), Conv(cin, cout)
        self.film = nn.Linear(emb, 2 * cout)
        self.n2, self.c2 = g(cout), Conv(cout, cout)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        sc, sh = self.film(F.silu(emb))[:, :, None, None, None].chunk(2, 1)
        h = self.n2(h) * (1 + sc) + sh
        h = self.c2(self.drop(F.silu(h)))
        return self.skip(x) + h


class AxialAttention(nn.Module):
    """Self-attention along one axis (0: time, 1: lon, 2: lat) of a (B, C, T, X, Y) tensor, pre-norm + residual."""

    def __init__(self, c: int, axis: int, heads: int, normalize_qk: bool):
        super().__init__()
        self.axis, self.h = axis, heads
        self.norm = nn.GroupNorm(min(c // 4, 32) or 1, c)
        self.qkv, self.out = nn.Linear(c, 3 * c), nn.Linear(c, c)
        self.qn = nn.LayerNorm(c // heads) if normalize_qk else nn.Identity()
        self.kn = nn.LayerNorm(c // heads) if normalize_qk else nn.Identity()

    def forward(self, x):
        B, C, T, X, Y = x.shape
        h = self.norm(x).permute(0, 2, 3, 4, 1)               # (B, T, X, Y, C)
        ax = 1 + self.axis
        h = h.movedim(ax, -2)                                  # (..., L, C)
        lead = h.shape[:-2]
        h = h.reshape(-1, h.shape[-2], C)
        q, k, v = self.qkv(h).reshape(h.shape[0], h.shape[1], 3, self.h, C // self.h).unbind(2)
        q, k = self.qn(q), self.kn(k)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        o = self.out(o.transpose(1, 2).reshape(h.shape[0], h.shape[1], C))
        o = o.reshape(*lead, h.shape[1], C).movedim(-2, ax).permute(0, 4, 1, 2, 3)
        return x + o


class Level(nn.Module):
    def __init__(self, cin, cout, emb, cfg: GenFocalNetConfig, sp: bool, tp: bool):
        super().__init__()
        self.blocks = nn.ModuleList([ResBlock(cin if i == 0 else cout, cout, emb, cfg.dropout_rate) for i in range(cfg.num_blocks)])
        self.attn = nn.ModuleList([AxialAttention(cout, a, cfg.num_heads, cfg.normalize_qk)
                                   for a, on in ((1, sp), (2, sp), (0, tp)) if on])

    def forward(self, x, emb):
        for b in self.blocks:
            x = b(x, emb)
        for a in self.attn:
            x = a(x)
        return x


class GenFocalNet(nn.Module):
    def __init__(self, cfg: GenFocalNetConfig):
        super().__init__()
        self.cfg = cfg
        C, ch, r = cfg.out_channels, list(cfg.num_channels), list(cfg.downsample_ratio)
        emb = cfg.noise_embed_dim
        self.embed = FourierEmbedding(emb)
        self.inp = Conv(C * (1 + len(cfg.cond_keys)), ch[0])
        self.down = nn.ModuleList([Level(ch[max(i - 1, 0)] if i else ch[0], ch[i], emb, cfg, cfg.use_spatial_attention[i],
                                         cfg.use_temporal_attention[i]) for i in range(len(ch))])
        self.pool = nn.ModuleList([nn.Conv3d(ch[i], ch[i], (1, r[i], r[i]), stride=(1, r[i], r[i])) for i in range(len(ch) - 1)])
        self.up = nn.ModuleList()
        self.upconv = nn.ModuleList()
        for i in reversed(range(len(ch) - 1)):
            self.up.append(Level(ch[i + 1] + ch[i], ch[i], emb, cfg, cfg.use_spatial_attention[i], cfg.use_temporal_attention[i]))
            self.upconv.append(nn.Conv3d(ch[i + 1], ch[i + 1], 1))
        self.norm = nn.GroupNorm(min(ch[0] // 4, 32) or 1, ch[0])
        self.out = Conv(ch[0], C)
        nn.init.zeros_(self.out.conv.weight)                  # official: near-zero output kernel
        nn.init.zeros_(self.out.conv.bias)
        self.ratios = r

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, cond: "Mapping[str, torch.Tensor] | None" = None) -> torch.Tensor:
        if x.shape[-1] != self.cfg.out_channels:
            raise ValueError("channels of x must equal out_channels")
        if sigma.ndim < 1:
            sigma = sigma.expand(x.shape[0])
        cond = cond or {}
        h = torch.cat([x] + [cond[k] for k in self.cfg.cond_keys], -1).permute(0, 4, 1, 2, 3)   # (B, C, T, X, Y)
        emb = self.embed(sigma * self.cfg.time_rescale)
        h = self.inp(h)
        skips = []
        for i, lv in enumerate(self.down):
            h = lv(h, emb)
            if i < len(self.pool):
                skips.append(h)
                h = self.pool[i](h)
        for j, i in enumerate(reversed(range(len(self.pool)))):
            r = self.ratios[i]
            h = F.interpolate(self.upconv[j](h), scale_factor=(1, r, r), mode="nearest")
            h = self.up[j](torch.cat([h, skips[i]], 1), emb)
        h = self.out(F.silu(self.norm(h)))
        return h.permute(0, 2, 3, 4, 1)


__all__ = ["GenFocalNet", "GenFocalNetConfig"]
