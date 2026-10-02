"""Real spherical-harmonic transforms on Gauss-Legendre / equiangular grids (native PyTorch).

Mathematically equivalent to ``torch_harmonics.RealSHT`` / ``InverseRealSHT`` (orthonormal, Condon-Shortley phase), written
from the definitions: rFFT over longitude, then a Legendre transform with quadrature weights.
Grid layout: latitude index 0 is the smallest colatitude (``theta = arccos(x)``, ascending), longitude uniform.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def legendre_gauss(n: int):
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


def clenshaw_curtis(n: int):
    """Clenshaw-Curtis nodes (ascending cos(theta) from -1 to 1) and weights on [-1, 1] (Waldvogel's FFT construction)."""
    assert n > 1
    x = np.cos(np.linspace(np.pi, 0, n))
    if n == 2:
        return x, np.array([1.0, 1.0])
    n1 = n - 1
    N = np.arange(1, n1, 2, dtype=np.float64)
    l = len(N)
    m = n1 - l
    v = np.concatenate([2 / N / (N - 2), 1 / N[-1:], np.zeros(m)])
    v = 0 - v[:-1] - v[1:][::-1]
    g0 = -np.ones(n1)
    g0[l] += n1
    g0[m] += n1
    g = g0 / (n1 ** 2 - 1 + (n1 % 2))
    w = np.fft.ifft(v + g).real
    return x, np.concatenate([w, w[:1]])


def legendre_table(mmax: int, lmax: int, x: np.ndarray, csphase: bool = True) -> np.ndarray:
    """Orthonormal associated Legendre functions ``sqrt((2l+1)/4pi (l-m)!/(l+m)!) P_l^m(x)`` as array [m, l, len(x)] (float64)."""
    nmax = max(mmax, lmax)
    v = np.zeros((nmax, nmax, len(x)))
    v[0, 0] = 1.0 / math.sqrt(4 * math.pi)
    for l in range(1, nmax):
        v[l - 1, l] = math.sqrt(2 * l + 1) * x * v[l - 1, l - 1]
        v[l, l] = np.sqrt((2 * l + 1) * (1 + x) * (1 - x) / 2 / l) * v[l - 1, l - 1]
    for l in range(2, nmax):
        for m in range(0, l - 1):
            v[m, l] = x * math.sqrt((2 * l - 1) / (l - m) * (2 * l + 1) / (l + m)) * v[m, l - 1] \
                - math.sqrt((l + m - 1) / (l - m) * (2 * l + 1) / (2 * l - 3) * (l - m - 1) / (l + m)) * v[m, l - 2]
    v = v[:mmax, :lmax]
    if csphase:
        v[1::2] *= -1
    return v


def _nodes(nlat: int, grid: str):
    if grid == "legendre-gauss":
        return legendre_gauss(nlat)
    if grid == "equiangular":
        return clenshaw_curtis(nlat)
    raise ValueError(f"unknown grid {grid}")


class RealSHT(nn.Module):
    """x[..., nlat, nlon] -> complex [..., lmax, mmax]."""

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="legendre-gauss"):
        super().__init__()
        self.nlat, self.nlon, self.grid = nlat, nlon, grid
        self.lmax = lmax or nlat
        self.mmax = mmax or nlon // 2 + 1
        cost, w = _nodes(nlat, grid)
        theta = np.arccos(cost)[::-1]
        pct = legendre_table(self.mmax, self.lmax, np.cos(theta)) * w[None, None, :]
        self.register_buffer("weights", torch.from_numpy(np.ascontiguousarray(pct)).float(), persistent=False)

    def forward(self, x):
        x = 2.0 * math.pi * torch.fft.rfft(x, dim=-1, norm="forward")[..., : self.mmax]
        xr = torch.view_as_real(x)
        out = torch.einsum("...kmr,mlk->...lmr", xr, self.weights.to(xr.dtype))
        return torch.view_as_complex(out.contiguous())


class InverseRealSHT(nn.Module):
    """complex [..., lmax, mmax] -> x[..., nlat, nlon]."""

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="legendre-gauss"):
        super().__init__()
        self.nlat, self.nlon, self.grid = nlat, nlon, grid
        self.lmax = lmax or nlat
        self.mmax = mmax or nlon // 2 + 1
        cost, _ = _nodes(nlat, grid)
        theta = np.arccos(cost)[::-1]
        pct = legendre_table(self.mmax, self.lmax, np.cos(theta))
        self.register_buffer("pct", torch.from_numpy(np.ascontiguousarray(pct)).float(), persistent=False)

    def forward(self, x):
        xr = torch.view_as_real(x)
        xs = torch.einsum("...lmr,mlk->...kmr", xr, self.pct.to(xr.dtype)).contiguous()
        xc = torch.view_as_complex(xs)
        xc = torch.cat([xc[..., :1].real.to(xc.dtype), xc[..., 1:]], dim=-1)  # the m=0 mode is real
        if self.nlon % 2 == 0 and self.nlon // 2 < self.mmax:
            k = self.nlon // 2
            xc = torch.cat([xc[..., :k], xc[..., k:k + 1].real.to(xc.dtype), xc[..., k + 1:]], dim=-1)  # Nyquist is real
        return torch.fft.irfft(xc, n=self.nlon, dim=-1, norm="forward")
