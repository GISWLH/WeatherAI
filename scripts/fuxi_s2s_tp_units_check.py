# global-mean part of the check (second run)
import sys, json, numpy as np, xarray as xr, gcsfs
S, OUT = sys.argv[1], sys.argv[2]
fs = gcsfs.GCSFileSystem(token="anon")
ds = xr.open_zarr(fs.get_mapper("weatherbench2/datasets/era5/1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr"), chunks=None)
lat = np.linspace(90, -90, 121); w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, 240)); w /= w.sum()
out = {}
for day in ("2020-06-01", "2020-06-02"):
    ref = xr.open_dataarray(f"{S}/sample/total_precipitation.nc").sel(time=day).squeeze().values.astype("f8")
    t = np.datetime64(day) + np.arange(1, 25).astype("timedelta64[h]")
    a = ds["total_precipitation"].sel(time=t).transpose("time", "latitude", "longitude").sortby("latitude", ascending=False).values.astype("f8")
    ttr_ref = xr.open_dataarray(f"{S}/sample/top_net_thermal_radiation.nc").sel(time=day).squeeze().values
    olr = ds["mean_top_net_long_wave_radiation_flux"].sel(time=t).mean("time").sortby("latitude", ascending=False).transpose("latitude","longitude").values
    out[day] = dict(tp_sample_globalmean_m=float((ref * w).sum()), tp_wb2_mean_hourly_accum_m=float((a.mean(0) * w).sum()),
                    tp_wb2_daily_total_m=float((a.sum(0) * w).sum()),
                    tp_sample_as_mm_per_day=float((ref * w).sum() * 1000 * 24),
                    ttr_sample_over_3600_globalmean_Wm2=float((ttr_ref / 3600 * w).sum()), olr_wb2_globalmean_Wm2=float((olr * w).sum()))
print(json.dumps(out, indent=1)); json.dump(out, open(OUT, "w"), indent=1)
