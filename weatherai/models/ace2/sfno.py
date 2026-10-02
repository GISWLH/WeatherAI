"""Spherical Fourier Neural Operator as used by ACE2-ERA5 (``dhconv`` linear spectral filters), native PyTorch."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .sht import InverseRealSHT, RealSHT


@dataclass
class SFNOConfig:
    in_chans: int = 44
    out_chans: int = 50
    img_shape: tuple = (180, 360)
    embed_dim: int = 384
    num_layers: int = 8
    mlp_ratio: float = 2.0
    grid: str = "legendre-gauss"
    pos_embed: bool = True
    big_skip: bool = True


class DHConvFilter(nn.Module):
    """Driscoll-Healy convolution: SHT -> per-degree complex channel mixing (shared over order m) -> iSHT, plus a bias."""

    def __init__(self, sht: RealSHT, isht: InverseRealSHT, dim: int):
        super().__init__()
        self.sht, self.isht = sht, isht
        self.filter = nn.Module()
        self.filter.weight = nn.Parameter(torch.randn(dim, dim, isht.lmax, 2) / (dim * dim))
        self.filter.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        c = self.sht(x.float())
        w = torch.view_as_complex(self.filter.weight.contiguous())
        c = torch.einsum("bixy,iox->boxy", c, w)
        return self.isht(c) + self.filter.bias


class MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fwd = nn.Sequential(nn.Conv2d(dim, hidden, 1), nn.GELU(), nn.Conv2d(hidden, dim, 1))

    def forward(self, x):
        return self.fwd(x)


class FNOBlock(nn.Module):
    def __init__(self, sht, isht, dim, mlp_ratio):
        super().__init__()
        self.norm0 = nn.InstanceNorm2d(dim, eps=1e-6, affine=True, track_running_stats=False)
        self.filter = DHConvFilter(sht, isht, dim)
        self.inner_skip = nn.Conv2d(dim, dim, 1)
        self.norm1 = nn.InstanceNorm2d(dim, eps=1e-6, affine=True, track_running_stats=False)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x):
        res = self.norm0(x)                      # the filter input doubles as the (identity) outer skip
        y = self.filter(res) + self.inner_skip(res)
        y = self.norm1(torch.nn.functional.gelu(y))
        return self.mlp(y) + res


class SFNO(nn.Module):
    def __init__(self, cfg: SFNOConfig):
        super().__init__()
        self.cfg = cfg
        h, w = cfg.img_shape
        # one transform pair is shared by every block (same grid, lmax = nlat, mmax = nlon//2+1)
        self.sht = RealSHT(h, w, lmax=h, mmax=w // 2 + 1, grid=cfg.grid)
        self.isht = InverseRealSHT(h, w, lmax=h, mmax=w // 2 + 1, grid=cfg.grid)
        d = cfg.embed_dim
        self.encoder = nn.Sequential(nn.Conv2d(cfg.in_chans, d, 1), nn.GELU(), nn.Conv2d(d, d, 1, bias=False))
        self.pos_embed = nn.Parameter(torch.zeros(1, d, h, w)) if cfg.pos_embed else None
        self.blocks = nn.ModuleList([FNOBlock(self.sht, self.isht, d, cfg.mlp_ratio) for _ in range(cfg.num_layers)])
        self.decoder = nn.Sequential(nn.Conv2d(d + cfg.big_skip * cfg.in_chans, d, 1), nn.GELU(), nn.Conv2d(d, cfg.out_chans, 1, bias=False))

    def forward(self, x):
        skip = x
        x = self.encoder(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        if self.cfg.big_skip:
            x = torch.cat([x, skip], dim=1)
        return self.decoder(x)
