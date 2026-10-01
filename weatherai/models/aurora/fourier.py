"""Fourier-series feature expansions used by Aurora (positions, patch areas, lead / absolute
time, pressure levels).

Re-written for WeatherAI from the public Aurora source (microsoft/aurora, MIT,
``aurora/model/fourier.py`` and ``aurora/area.py``); numerically identical to upstream.
"""
from __future__ import annotations

import math

import numpy as np
import torch

RADIUS_EARTH_KM = 6378137 / 1000


def polygon_area_km2(polygon: torch.Tensor) -> torch.Tensor:
    """Spherical-polygon area (km^2) for ``polygon`` of shape ``(..., n, 2)`` = (lat, lon) degrees.

    Port of the PyPI ``area`` package as used by Aurora (``aurora.area.area``).
    """
    polygon = torch.cat((polygon, polygon[..., -1:, :]), dim=-2)  # close the loop
    n = polygon.shape[-2]
    area = torch.zeros(polygon.shape[:-2], dtype=polygon.dtype, device=polygon.device)
    rad = torch.deg2rad
    if n > 2:
        for i in range(n):
            lon_lower = polygon[..., i, 1]
            lat_middle = polygon[..., (i + 1) % n, 0]
            lon_upper = polygon[..., (i + 2) % n, 1]
            area = area + (rad(lon_upper) - rad(lon_lower)) * torch.sin(rad(lat_middle))
    return torch.abs(area * RADIUS_EARTH_KM * RADIUS_EARTH_KM / 2)


def fourier_expansion(
    x: torch.Tensor, d: int, lower: float, upper: float, assert_range: bool = True
) -> torch.Tensor:
    """``sin``/``cos`` expansion of ``x`` (shape ``(..., n)``) into ``(..., n, d)``.

    ``d // 2`` log-spaced wavelengths between ``lower`` and ``upper``; computed in float64 and
    returned as float32. Zeros are always allowed; otherwise ``lower <= |x| <= upper`` is
    required when ``assert_range``.
    """
    if d % 2 != 0:
        raise ValueError("The dimensionality must be a multiple of two.")
    if assert_range:
        ax = x.abs()
        ok = torch.all((ax <= upper) & ((ax >= lower) | (x == 0)))
        if not ok:
            raise AssertionError(
                f"The input tensor is not within the configured range `[{lower}, {upper}]`."
            )
    x = x.double()
    wavelengths = torch.logspace(
        math.log10(lower), math.log10(upper), d // 2, base=10, device=x.device, dtype=x.dtype
    )
    prod = torch.einsum("...i,j->...ij", x, 2 * np.pi / wavelengths)
    return torch.cat((torch.sin(prod), torch.cos(prod)), dim=-1).float()


# --- the expansions used in Aurora (wavelength ranges as upstream) -----------------------
_DELTA = 0.01  # smallest lat/lon delta assumed for the scale embedding
_MIN_PATCH_AREA = polygon_area_km2(
    torch.tensor(
        [[90, 0], [90, _DELTA], [90 - _DELTA, _DELTA], [90 - _DELTA, 0]], dtype=torch.float64
    )
).item()
_AREA_EARTH = 4 * np.pi * RADIUS_EARTH_KM * RADIUS_EARTH_KM


def pos_expansion(x, d):
    """Latitude / longitude in degrees."""
    return fourier_expansion(x, d, _DELTA, 720)


def scale_expansion(x, d):
    """Square root of patch area in km."""
    return fourier_expansion(x, d, _MIN_PATCH_AREA, _AREA_EARTH)


def lead_time_expansion(x, d):
    """Lead time in hours (original pretrained checkpoints)."""
    return fourier_expansion(x, d, 1 / 60, 24 * 7 * 3)


def lead_time_expansion_v3(x, d):
    """Updated lead-time expansion: min wavelength 6 h, sin of the largest wavelength replaced by
    the linear feature ``x / upper``. Used when ``use_updated_lead_time_embedding=True``."""
    upper = 24 * 7 * 3
    enc = fourier_expansion(x, d, 6, upper, assert_range=False)
    linear = (x / upper).float().unsqueeze(-1)
    return torch.cat([enc[..., : d // 2 - 1], linear, enc[..., d // 2 :]], dim=-1)


def levels_expansion(x, d):
    """Pressure level in hPa."""
    return fourier_expansion(x, d, 0.01, 1e5)


def absolute_time_expansion(x, d):
    """Absolute time in hours since the epoch."""
    return fourier_expansion(x, d, 1, 24 * 365.25, assert_range=False)


def patch_root_area(lat_min, lon_min, lat_max, lon_max) -> torch.Tensor:
    """sqrt(area in km^2) of lat/lon rectangles (spherical formula, R = 6371 km as upstream)."""
    assert (lat_max > lat_min).all(), f"lat_max - lat_min: {torch.min(lat_max - lat_min)}."
    assert (lon_max > lon_min).all(), f"lon_max - lon_min: {torch.min(lon_max - lon_min)}."
    assert (abs(lat_max) <= 90.0).all() and (abs(lat_min) <= 90.0).all()
    assert (lon_max <= 360.0).all() and (lon_min <= 360.0).all()
    assert (lon_max >= 0.0).all() and (lon_min >= 0.0).all()
    area = (
        6371**2
        * torch.pi
        * (torch.sin(torch.deg2rad(lat_max)) - torch.sin(torch.deg2rad(lat_min)))
        * (torch.deg2rad(lon_max) - torch.deg2rad(lon_min))
    )
    assert (area > 0.0).all()
    return torch.sqrt(area)


def pos_scale_encoding(embed_dim: int, lat: torch.Tensor, lon: torch.Tensor, patch_size: int):
    """Position and scale encodings of the ``(H/P) * (W/P)`` patches of a lat/lon grid.

    ``lat`` (H,) decreasing, ``lon`` (W,) increasing (both vectors) or both ``(H, W)`` matrices.
    Returns two ``(H/P * W/P, embed_dim)`` tensors.
    """
    import torch.nn.functional as F

    assert embed_dim % 4 == 0
    if lat.dim() == lon.dim() == 1:
        grid = torch.stack(
            (lat[:, None].expand(-1, lon.shape[0]), lon[None, :].expand(lat.shape[0], -1)), dim=0
        )
    elif lat.dim() == lon.dim() == 2:
        grid = torch.stack((lat, lon), dim=0)
    else:
        raise ValueError("lat and lon must both be vectors or both be matrices.")
    grid = grid[None]  # (1, 2, H, W)
    p = (patch_size, patch_size)
    grid_h = F.avg_pool2d(grid[:, 0], p)  # patch-centre latitude
    grid_w = F.avg_pool2d(grid[:, 1], p)
    lat_max = F.max_pool2d(grid[:, 0], p)
    lat_min = -F.max_pool2d(-grid[:, 0], p)
    lon_max = F.max_pool2d(grid[:, 1], p)
    lon_min = -F.max_pool2d(-grid[:, 1], p)
    root_area = patch_root_area(lat_min, lon_min, lat_max, lon_max)

    enc_h = pos_expansion(grid_h.reshape(1, -1), embed_dim // 2)
    enc_w = pos_expansion(grid_w.reshape(1, -1), embed_dim // 2)
    pos = torch.cat((enc_h, enc_w), dim=-1)  # (1, L, D)
    scale = scale_expansion(root_area.reshape(1, -1), embed_dim)
    return pos.squeeze(0), scale.squeeze(0)
