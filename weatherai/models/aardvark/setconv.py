"""Gaussian set-convolution (ConvDeepSet) used by Aardvark to map between off-grid observations and
grids. Explicit PyTorch port of the official ``aardvark/set_convs.py`` (CC0).

Inputs are *normalised coordinates* (degrees / 360). Three modes:

* ``OffToOn``  – scattered points -> grid (separable lon/lat Gaussian weights, einsum over points)
* ``OnToOn``   – grid -> grid (separable weights over lon and lat axes)
* ``OnToOff``  – grid -> scattered points

A density channel (1 where data, 0 where NaN) is prepended; outputs are divided by the smoothed
density (clamped to ``[1e-6, 1e5]``). NaNs in ``wt`` are zeroed. The official code mutates its
inputs in place; this port does not (it copies), numerics are identical.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class SetConv(nn.Module):
    def __init__(self, init_ls: float, mode: str, density_channel: bool = True, learnable: bool = True):
        super().__init__()
        assert mode in ("OffToOn", "OnToOn", "OnToOff")
        # In the official model only ``ascat_setconvs`` and ``sc_out`` are registered nn.Modules (so only
        # their length scale is in the checkpoints); the others are plain python lists, i.e. fixed at
        # 0.001 and never stored. ``learnable=False`` mimics that with a non-persistent buffer.
        if learnable:
            self.init_ls = nn.Parameter(torch.tensor([init_ls]))
        else:
            self.register_buffer("init_ls", torch.tensor([init_ls]), persistent=False)
        self.mode, self.density_channel = mode, density_channel

    def _weights(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        a, b = x1.unsqueeze(-1), x2.unsqueeze(-1)  # (..., N1, 1), (..., N2, 1)
        d2 = (a**2).sum(-1)[..., :, None] + (b**2).sum(-1)[..., None, :] - 2 * torch.matmul(a, b.permute(0, 2, 1))
        return torch.exp((-0.5 * d2) / self.init_ls.to(x1.device) ** 2)

    def forward(self, x_in: Sequence[torch.Tensor], wt: torch.Tensor, x_out: Sequence[torch.Tensor]) -> torch.Tensor:
        density = torch.ones_like(wt[:, 0:1])
        density[torch.isnan(wt[:, 0:1])] = 0
        wt = torch.cat([density, wt], dim=1)
        wt = torch.nan_to_num(wt, nan=0.0)

        if self.mode == "OffToOn":  # (B, C, N) points -> (B, C, X, Y)
            m0, m1 = ~torch.isnan(x_in[0]), ~torch.isnan(x_in[1])
            xi = [torch.where(m0, x_in[0], torch.zeros_like(x_in[0])), torch.where(m1, x_in[1], torch.zeros_like(x_in[1]))]
            w0 = self._weights(xi[0], x_out[0]) * m0.unsqueeze(-1).int()
            w1 = self._weights(xi[1], x_out[1]) * m1.unsqueeze(-1).int()
            ee = torch.einsum("...cw,...wx,...wy->...cxy", wt, w0, w1)
        elif self.mode == "OnToOn":  # (B, C, W, H) grid -> (B, C, X, Y) grid
            w0, w1 = self._weights(x_in[0], x_out[0]), self._weights(x_in[1], x_out[1])
            ee = torch.einsum("...cwh,...wx,...hy->...cxy", wt, w0, w1)
        else:  # OnToOff: grid -> (B, C, N) points
            m0, m1 = ~torch.isnan(x_out[0]), ~torch.isnan(x_out[1])
            xo = [torch.where(m0, x_out[0], torch.zeros_like(x_out[0])), torch.where(m1, x_out[1], torch.zeros_like(x_out[1]))]
            w0 = self._weights(x_in[0], xo[0]) * m0.unsqueeze(-2).int()
            w1 = self._weights(x_in[1], xo[1]) * m1.unsqueeze(-2).int()
            ee = torch.einsum("...cwh,...wx,...hx->...cx", wt, w0, w1)

        norm = ee[:, 1:] / torch.clamp(ee[:, 0:1], min=1e-6, max=1e5)
        return torch.cat([ee[:, 0:1], norm], dim=1) if self.density_channel else norm
