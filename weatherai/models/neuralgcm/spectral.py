"""Native PyTorch spherical-harmonic (SHT) layer for NeuralGCM's ``RealSphericalHarmonics`` grids.

This is a re-implementation (Apache-2.0, after google-research/dinosaur ``spherical_harmonic.py``,
``associated_legendre.py``, ``fourier.py``) of the pieces of NeuralGCM's spectral core that are
purely linear-algebraic and therefore tractable in PyTorch:

* real normalised Fourier x associated-Legendre basis on a Gauss-Legendre latitude grid,
* ``to_modal`` / ``to_nodal`` transforms (``transform`` / ``inverse_transform``),
* ``laplacian`` / ``inverse_laplacian``, ``d_dlon``, ``cos_lat_d_dlat``, ``sec_lat_d_dlat_cos2``,
  ``cos_lat_grad``, ``div_cos_lat``, ``curl_cos_lat`` with the top-wavenumber clip,
* ``uv_nodal_to_vor_div_modal`` / ``vor_div_to_uv_nodal``.

It is NOT the dynamical core: the sigma-coordinate primitive equations, the implicit-explicit
Runge-Kutta stepper, filters and the dycore corrector are not ported (see docs/model_status.md).
Layout is torch-native: nodal fields ``(..., lon, lat)``, modal ``(..., m, l)`` exactly as dinosaur.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------- basis construction (numpy, f64)
def _legendre_rhombus_triangle(n_l: int, n_m: int, x: np.ndarray) -> np.ndarray:
    y = np.sqrt(1 - x * x)
    p = np.zeros((n_l, n_m, len(x)))
    p[0, 0] = p[0, 0] + 1 / np.sqrt(2)
    for m in range(1, n_m):
        p[0, m] = -np.sqrt(1 + 1 / (2 * m)) * y * p[0, m - 1]
    for k in range(1, n_l):
        m_max = min(n_m, n_l - k)
        m = np.arange(m_max).reshape((-1, 1))
        m2, mk2, mkp2 = np.square(m), np.square(m + k), np.square(m + k - 1)
        a = np.sqrt((4 * mk2 - 1) / (mk2 - m2))
        b = np.sqrt((mkp2 - m2) / (4 * mkp2 - 1))
        p[k, :m_max] = a * (x * p[k - 1, :m_max] - b * p[k - 2, :m_max])
    return p


def associated_legendre(n_m: int, n_l: int, x: np.ndarray) -> np.ndarray:
    """p[m, i, l] = c_lm P_l^m(x_i), unit-L2 normalised; zero for l < m."""
    r = np.transpose(_legendre_rhombus_triangle(n_l, n_m, x), (1, 2, 0))
    p = np.zeros((n_m, len(x), n_l))
    for m in range(n_m):
        p[m, :, m:n_l] = r[m, :, 0 : n_l - m]
    return p


def real_fourier_basis(wavenumbers: int, nodes: int) -> np.ndarray:
    """F[i,0]=1/sqrt(2pi); F[i,2j-1]=cos(j x_i)/sqrt(pi); F[i,2j]=sin(j x_i)/sqrt(pi), x_i=2 pi i/nodes."""
    x = 2 * np.pi * np.arange(nodes) / nodes
    f = np.empty((nodes, 2 * wavenumbers - 1))
    f[:, 0] = 1 / np.sqrt(2 * np.pi)
    j = np.arange(1, wavenumbers)
    f[:, 1::2] = np.cos(np.outer(x, j)) / np.sqrt(np.pi)
    f[:, 2::2] = np.sin(np.outer(x, j)) / np.sqrt(np.pi)
    return f


@dataclass(frozen=True)
class SpectralGridConfig:
    longitude_wavenumbers: int
    total_wavenumbers: int
    longitude_nodes: int
    latitude_nodes: int
    radius: float = 1.0
    longitude_offset: float = 0.0

    @classmethod
    def TL63(cls) -> "SpectralGridConfig":
        """The 2.8 deg NeuralGCM grid (128 x 64 nodes, modal shape (127, 65))."""
        return cls(64, 65, 128, 64)


class SphericalHarmonicsGrid(torch.nn.Module):
    """Gaussian grid + real spherical-harmonic basis. All buffers are float64; ``dtype`` casts at use."""

    def __init__(self, cfg: SpectralGridConfig):
        super().__init__()
        from scipy.special import roots_legendre

        self.cfg = cfg
        x, wp = roots_legendre(cfg.latitude_nodes)  # sin(lat), Gauss-Legendre weights
        wf = 2 * np.pi / cfg.longitude_nodes
        f = real_fourier_basis(cfg.longitude_wavenumbers, cfg.longitude_nodes)
        p = associated_legendre(cfg.longitude_wavenumbers, cfg.total_wavenumbers, x)
        p = np.repeat(p, 2, axis=0)[1:]  # (2M-1, lat, l)
        self.register_buffer("f", torch.tensor(f), persistent=False)
        self.register_buffer("p", torch.tensor(p), persistent=False)
        self.register_buffer("w", torch.tensor(wf * wp), persistent=False)  # (lat,)
        self.register_buffer("sin_lat", torch.tensor(x), persistent=False)
        m_pos = np.arange(1, cfg.longitude_wavenumbers)
        m_axis = np.concatenate([[0], np.stack([m_pos, -m_pos], 1).ravel()])
        l_axis = np.arange(cfg.total_wavenumbers)
        self.register_buffer("m_axis", torch.tensor(m_axis), persistent=False)
        self.register_buffer("l_axis", torch.tensor(l_axis), persistent=False)
        mm, ll = np.meshgrid(m_axis, l_axis, indexing="ij")
        mask = np.abs(mm) <= ll
        self.register_buffer("mask", torch.tensor(mask), persistent=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.sqrt(mask * (ll**2 - mm**2) / (4 * ll**2 - 1))
        a = np.nan_to_num(a)
        a[:, 0] = 0
        b = np.sqrt(mask * ((ll + 1) ** 2 - mm**2) / (4 * (ll + 1) ** 2 - 1))
        b[:, -1] = 0
        self.register_buffer("_a", torch.tensor(a), persistent=False)
        self.register_buffer("_b", torch.tensor(b), persistent=False)
        self.register_buffer("_ll", torch.tensor(ll, dtype=torch.float64), persistent=False)
        self.register_buffer("_mm", torch.tensor(mm, dtype=torch.float64), persistent=False)

    # ------------------------------------------------------------------ shapes / axes
    @property
    def nodal_shape(self) -> Tuple[int, int]:
        return (self.cfg.longitude_nodes, self.cfg.latitude_nodes)

    @property
    def modal_shape(self) -> Tuple[int, int]:
        return (2 * self.cfg.longitude_wavenumbers - 1, self.cfg.total_wavenumbers)

    @cached_property
    def cos_lat(self) -> torch.Tensor:
        return torch.sqrt(1 - self.sin_lat**2)

    @property
    def lat(self) -> torch.Tensor:
        return torch.asin(self.sin_lat)

    @property
    def lon(self) -> torch.Tensor:
        return 2 * math.pi * torch.arange(self.cfg.longitude_nodes, dtype=torch.float64) / self.cfg.longitude_nodes + self.cfg.longitude_offset

    # ------------------------------------------------------------------ transforms
    def to_modal(self, z: torch.Tensor) -> torch.Tensor:
        """nodal (..., lon, lat) -> modal (..., m, l)."""
        dt = z.dtype
        z = z.to(self.f.dtype)
        fwx = torch.einsum("im,...ij->...mj", self.f, self.w * z)
        return torch.einsum("mjl,...mj->...ml", self.p, fwx).to(dt)

    def to_nodal(self, x: torch.Tensor) -> torch.Tensor:
        """modal (..., m, l) -> nodal (..., lon, lat)."""
        dt = x.dtype
        x = x.to(self.f.dtype)
        px = torch.einsum("mjl,...ml->...mj", self.p, x)
        return torch.einsum("im,...mj->...ij", self.f, px).to(dt)

    # ------------------------------------------------------------------ spectral operators
    @staticmethod
    def _shift(x: torch.Tensor, offset: int, dim: int) -> torch.Tensor:
        """Zero-padded shift (dinosaur ``jax_numpy_utils.shift``)."""
        n = x.shape[dim]
        if abs(offset) >= n:
            return torch.zeros_like(x)
        out = torch.zeros_like(x)
        if offset > 0:
            out.narrow(dim, offset, n - offset).copy_(x.narrow(dim, 0, n - offset))
        else:
            out.narrow(dim, 0, n + offset).copy_(x.narrow(dim, -offset, n + offset))
        return out

    def laplacian(self, x):
        eig = (-self._ll * (self._ll + 1) / self.cfg.radius**2).to(x.dtype)
        return x * eig

    def inverse_laplacian(self, x):
        l = self.l_axis.to(torch.float64)
        eig = -l * (l + 1) / self.cfg.radius**2
        inv = torch.where(l > 0, 1.0 / torch.where(l > 0, eig, torch.ones_like(eig)), torch.zeros_like(eig))
        return x * inv.to(x.dtype)

    def clip_wavenumbers(self, x, n: int = 1):
        mask = torch.ones(self.modal_shape[-1], dtype=x.dtype, device=x.device)
        mask[-n:] = 0
        return x * mask

    def d_dlon(self, x):
        """Derivative along longitude in the real Fourier basis (axis -2)."""
        n = x.shape[-2]
        i = torch.arange(n, device=x.device).reshape(-1, 1)
        j = ((i + 1) // 2).to(x.dtype)
        down = self._shift(x, -1, -2)
        up = self._shift(x, +1, -2)
        return j * torch.where((i % 2).bool(), down, -up)

    def cos_lat_d_dlat(self, x):
        l = self._ll.to(x.dtype)
        a, b = self._a.to(x.dtype), self._b.to(x.dtype)
        return self._shift((l + 1) * a * x, -1, -1) + self._shift((-l * b) * x, +1, -1)

    def sec_lat_d_dlat_cos2(self, x):
        l = self._ll.to(x.dtype)
        a, b = self._a.to(x.dtype), self._b.to(x.dtype)
        return self._shift((l - 1) * a * x, -1, -1) + self._shift((-(l + 2) * b) * x, +1, -1)

    def cos_lat_grad(self, x, clip: bool = True):
        r = self.cfg.radius
        raw = (self.d_dlon(x) / r, self.cos_lat_d_dlat(x) / r)
        return tuple(self.clip_wavenumbers(c) for c in raw) if clip else raw

    def div_cos_lat(self, v, clip: bool = True):
        raw = (self.d_dlon(v[0]) + self.sec_lat_d_dlat_cos2(v[1])) / self.cfg.radius
        return self.clip_wavenumbers(raw) if clip else raw

    def curl_cos_lat(self, v, clip: bool = True):
        raw = (self.d_dlon(v[1]) - self.sec_lat_d_dlat_cos2(v[0])) / self.cfg.radius
        return self.clip_wavenumbers(raw) if clip else raw

    # ------------------------------------------------------------------ wind <-> vorticity/divergence
    def uv_nodal_to_vor_div_modal(self, u, v, clip: bool = True):
        cl = self.cos_lat.to(u.dtype)
        uc, vc = self.to_modal(u / cl), self.to_modal(v / cl)
        return self.curl_cos_lat((uc, vc), clip), self.div_cos_lat((uc, vc), clip)

    def vor_div_to_uv_nodal(self, vor, div, clip: bool = True):
        psi = self.inverse_laplacian(vor)
        chi = self.inverse_laplacian(div)
        gc = self.cos_lat_grad(chi, clip)
        gp = self.cos_lat_grad(psi, clip)
        u_cos = gc[0] + (-gp[1])  # k x grad(psi) = (-d_lat psi, d_lon psi)
        v_cos = gc[1] + gp[0]
        cl = self.cos_lat.to(vor.dtype)
        return self.to_nodal(u_cos) / cl, self.to_nodal(v_cos) / cl
