"""Input handling for ORCA-DL: GODAS-style monthly fields on the 128x360 grid, normalised with the official monthly statistics."""
from __future__ import annotations

import os

import numpy as np

OCEAN_VARS = ["salt", "pottmp", "sst", "ucur", "vcur", "sshg"]     # -> model so, thetao, tos, uo, vo, zos (16,16,1,16,16,1 channels)
ATMO_VARS = ["uflx", "vflx"]                                        # wind stress tauu, tauv (N/m^2)
UNITS = {"salt": "g/kg", "pottmp": "degC", "sst": "degC", "ucur": "m/s", "vcur": "m/s", "sshg": "m", "uflx": "N/m^2", "vflx": "N/m^2"}
EXAMPLE_URL = "https://raw.githubusercontent.com/OpenEarthLab/ORCA-DL/main/example_data/{v}.nc"   # repo has no LICENSE file: fetched, not committed


def load_stats(root: str):
    st = {k: {v: np.load(os.path.join(root, "stat", k, f"{v}.npy")) for v in OCEAN_VARS + ATMO_VARS} for k in ("mean", "std")}
    return st


def download_example(dst: str):
    import urllib.request
    os.makedirs(dst, exist_ok=True)
    for v in OCEAN_VARS + ATMO_VARS:
        p = os.path.join(dst, f"{v}.nc")
        if not os.path.exists(p):
            urllib.request.urlretrieve(EXAMPLE_URL.format(v=v), p)
    return dst


def load_example(example_dir: str, stat_root: str, month: int = 0):
    """Official demo input (initial month index ``month``, 0 = January statistics) -> (ocean (1,66,128,360), atmo (1,2,128,360)) float32, NaN->0."""
    import xarray as xr
    st = load_stats(stat_root)
    def norm(v):
        a = xr.open_dataset(os.path.join(example_dir, f"{v}.nc"))[v].values
        return (a - st["mean"][v][month]) / st["std"][v][month]
    ocean = [norm(v) if norm(v).ndim == 3 else norm(v)[None] for v in OCEAN_VARS]
    atmo = [norm(v)[None] for v in ATMO_VARS]
    f = lambda xs: np.nan_to_num(np.concatenate(xs, 0))[None].astype(np.float32)
    return f(ocean), f(atmo)


def denormalise(pred, st, lead_month_index: int, var: str):
    """pred: (...,H,W) normalised field of variable ``var`` at statistics-month index ``lead_month_index`` (0..11)."""
    return pred * st["std"][var][lead_month_index % 12] + st["mean"][var][lead_month_index % 12]
