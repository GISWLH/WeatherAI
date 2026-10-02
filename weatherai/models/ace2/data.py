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


def compare_with_wb2(outs, ckpt_dir: str, days=(1, 2, 3, 5, 7), steps_per_day: int = 4):
    """Lat-weighted RMSE of h500 / TMP850 / TMP2m of a rollout started 2020-06-01T00Z against WeatherBench2 ERA5 (1.5 deg; sample file
    written by ``scripts/fetch_wb2_ace2.py``), plus the persistence baseline. ``outs[i]`` is the output after step i+1."""
    import xarray as xr

    f = xr.open_dataset(os.path.join(ckpt_dir, "forcing_data", "forcing_2020.nc"))
    lat_f, lon_f = f.latitude.values, f.longitude.values
    era = xr.open_dataset(os.path.join(ckpt_dir, "wb2_sample_2020.nc"))
    g = 9.80665

    def to_era_grid(a):
        da = xr.DataArray(a, dims=("latitude", "longitude"), coords={"latitude": lat_f, "longitude": lon_f})
        left = da.isel(longitude=[-1]).assign_coords(longitude=[lon_f[-1] - 360])
        right = da.isel(longitude=[0]).assign_coords(longitude=[lon_f[0] + 360])
        da = xr.concat([left, da, right], "longitude")
        return da.interp(latitude=era.latitude, longitude=era.longitude, kwargs={"fill_value": "extrapolate"}).values

    w = np.cos(np.deg2rad(era.latitude.values))[:, None]
    w = w / w.mean()
    rmse = lambda a, b: float(np.sqrt((w * (a - b) ** 2).mean()))

    def fields(e):
        tr = lambda x: x.transpose("latitude", "longitude").values
        return {"h500": tr(e.geopotential.sel(level=500)) / g, "TMP850": tr(e.temperature.sel(level=850)), "TMP2m": tr(e["2m_temperature"])}

    pers = fields(era.sel(time="2020-06-01T00:00"))
    out = {}
    for d in days:
        ref = fields(era.sel(time=str(np.datetime64("2020-06-01") + np.timedelta64(d, "D")) + "T00:00"))
        o = outs[d * steps_per_day - 1]
        out[d] = {k: {"ace2": rmse(to_era_grid(o[k][0].float().cpu().numpy()), ref[k]), "persistence": rmse(pers[k], ref[k])} for k in ref}
    return out
