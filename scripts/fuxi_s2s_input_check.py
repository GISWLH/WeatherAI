"""Compare the official FuXi-S2S sample inputs (2020-06-01/02) with daily means built from WeatherBench2 hourly ERA5 (1.5 deg)."""
import json, sys, numpy as np, xarray as xr, gcsfs
S = sys.argv[1]; OUT = sys.argv[2]
fs = gcsfs.GCSFileSystem(token="anon")
ds = xr.open_zarr(fs.get_mapper("weatherbench2/datasets/era5/1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr"), chunks=None)
pairs = {"2m_temperature": "2m_temperature", "sea_surface_temperature": "sea_surface_temperature",
         "mean_sea_level_pressure": "mean_sea_level_pressure", "total_column_water_vapour": "total_column_water_vapour",
         "10m_u_component_of_wind": "10m_u_component_of_wind", "total_precipitation": "total_precipitation",
         "top_net_thermal_radiation": "mean_top_net_long_wave_radiation_flux", "2m_dewpoint_temperature": "2m_dewpoint_temperature"}
res = {}
for day in ("2020-06-01", "2020-06-02"):
    for name, wb in pairs.items():
        ref = xr.open_dataarray(f"{S}/sample/{name}.nc").sel(time=day).squeeze().values.astype("f8")   # (121,240) lat 90..-90
        cands = {"00-23": slice(f"{day}T00", f"{day}T23"), "01-24": (np.datetime64(day) + np.arange(1, 25).astype("timedelta64[h]"))}
        for lab, sl in cands.items():
            a = ds[wb].sel(time=sl).mean("time").transpose("latitude", "longitude").sortby("latitude", ascending=False).values.astype("f8")
            for scale_lab, k in (("x1", 1.0), ("x3600", 3600.0), ("x24", 24.0)):
                if wb != "mean_top_net_long_wave_radiation_flux" and scale_lab == "x3600": continue
                if wb != "total_precipitation" and scale_lab == "x24": continue
                b = a * k
                ok = np.isfinite(ref) & np.isfinite(b)
                rel = float(np.sqrt(np.mean((b[ok] - ref[ok]) ** 2)) / (np.std(ref[ok]) + 1e-30))
                res[f"{day}|{name}|{lab}|{scale_lab}"] = dict(rmse_over_std=rel, ref_max=float(np.nanmax(ref)), wb2_max=float(np.nanmax(b)),
                                                             nan_ref=int((~np.isfinite(ref)).sum()), nan_wb2=int((~np.isfinite(b)).sum()))
best = {}
for k, v in res.items():
    d, n, w, s = k.split("|")
    if (d, n) not in best or v["rmse_over_std"] < best[(d, n)][1]["rmse_over_std"]: best[(d, n)] = (f"{w} {s}", v)
for (d, n), (lab, v) in sorted(best.items()): print(d, n, "best:", lab, {k: round(x, 6) for k, x in v.items()})
json.dump({"all": res, "best": {f"{d}|{n}": dict(choice=lab, **v) for (d, n), (lab, v) in best.items()}}, open(OUT, "w"), indent=1)
