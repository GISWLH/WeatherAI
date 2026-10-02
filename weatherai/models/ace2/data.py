"""Read the ACE2-ERA5 initial-condition / forcing NetCDF files shipped with the HF repo into tensors."""
from __future__ import annotations

import os

import numpy as np
import torch


def load_case(ckpt_dir: str, ic_index: int = 0, n_steps: int = 8, start_time=None):
    """Return (ic, forcing, times). ``ic``: dict name -> [1,H,W]; ``forcing``: dict name -> [1, n_steps+1, H, W]
    (static fields and the global-mean CO2 scalar are broadcast over space/time); ``times``: numpy datetime64 array."""
    import xarray as xr

    ic_ds = xr.open_dataset(os.path.join(ckpt_dir, "initial_conditions", "ic_2020.nc"))
    f_ds = xr.open_dataset(os.path.join(ckpt_dir, "forcing_data", "forcing_2020.nc"))
    t0 = ic_ds.time.values[ic_index]
    i0 = int(np.where(f_ds.time.values == t0)[0][0])
    fs = f_ds.isel(time=slice(i0, i0 + n_steps + 1))
    ic = {k: torch.from_numpy(ic_ds[k].isel(time=ic_index).values.astype("float32"))[None] for k in ic_ds.data_vars}
    hw = fs["land_fraction"].shape
    forcing = {}
    for k in ["DSWRFtoa", "ocean_fraction", "sea_ice_fraction", "surface_temperature"]:
        forcing[k] = torch.from_numpy(fs[k].values.astype("float32"))[None]
    nt = fs.time.size
    for k in ["HGTsfc", "land_fraction"]:
        forcing[k] = torch.from_numpy(fs[k].values.astype("float32"))[None, None].expand(1, nt, *hw).contiguous()
    forcing["global_mean_co2"] = torch.from_numpy(fs["global_mean_co2"].values.astype("float32")).view(1, nt, 1, 1).expand(1, nt, *hw).contiguous()
    return ic, forcing, fs.time.values
