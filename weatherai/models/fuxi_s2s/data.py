"""Input preparation for FuXi-S2S (mirrors the official ``data_util.make_input`` + ``land_to_nan``)."""
from __future__ import annotations

import os
import urllib.request

import numpy as np

from .model import CHANNELS

__all__ = ["make_input", "download_sample", "SAMPLE_FILES"]

_NAMES = [("geopotential", "z"), ("temperature", "t"), ("u_component_of_wind", "u"), ("v_component_of_wind", "v"),
          ("specific_humidity", "q"), ("2m_temperature", "t2m"), ("2m_dewpoint_temperature", "d2m"),
          ("sea_surface_temperature", "sst"), ("top_net_thermal_radiation", "ttr"), ("10m_u_component_of_wind", "10u"),
          ("10m_v_component_of_wind", "10v"), ("100m_u_component_of_wind", "100u"), ("100m_v_component_of_wind", "100v"),
          ("mean_sea_level_pressure", "msl"), ("total_column_water_vapour", "tcwv"), ("total_precipitation", "tp")]
SAMPLE_FILES = [f"{n}.nc" for n, _ in _NAMES]
RAW = "https://raw.githubusercontent.com/tpys/FuXi-S2S/main/data/{}"


def download_sample(root: str):
    """Fetch the official sample ERA5 inputs (16 small NetCDF files + land mask) from the authors' GitHub repo."""
    os.makedirs(os.path.join(root, "sample"), exist_ok=True)
    for f in SAMPLE_FILES:
        p = os.path.join(root, "sample", f)
        if not os.path.exists(p):
            urllib.request.urlretrieve(RAW.format("sample/" + f), p)
    mp = os.path.join(root, "mask.nc")
    if not os.path.exists(mp):
        urllib.request.urlretrieve(RAW.format("mask.nc"), mp)
    return root


def make_input(root: str):
    """Returns (array (2, 76, 121, 240) float32 in raw units, the two date strings); channels = ``CHANNELS``.

    tp: clip(m*1000, 0, 1000) (mm/day); ttr: /3600 (W m-2); sst: NaN over land (official mask.nc)."""
    import xarray as xr
    chans, init = [], None
    for long, short in _NAMES:
        v = xr.open_dataarray(os.path.join(root, "sample", f"{long}.nc"))
        if short == "tp":
            v = np.clip(v * 1000, 0, 1000)
        elif short == "ttr":
            v = v / 3600
        if v.level.values[0] != 1000:
            v = v.reindex(level=v.level[::-1])
        chans.append(v.values.astype(np.float32).reshape(v.shape[0], -1, *v.shape[-2:]))
        init = [str(t) for t in v.time.values]
    x = np.concatenate(chans, 1)
    assert x.shape[1] == len(CHANNELS)
    mp = os.path.join(root, "mask.nc")
    if os.path.exists(mp):
        mask = xr.open_dataarray(mp).values.astype(bool)
        i = CHANNELS.index("sst")
        x[:, i] = np.where(mask, x[:, i], np.nan)
    return x, init
