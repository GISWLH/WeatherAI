"""Write a small WeatherBench2 ERA5 (1.5 deg, public GCS bucket, anonymous) sample at 00Z used by scripts/ace2_real_data.py."""
import sys

import xarray as xr

out = sys.argv[1] if len(sys.argv) > 1 else "/workspace/ckpt/ace2/wb2_sample_2020.nc"
url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
ds = xr.open_zarr(url, storage_options={"token": "anon"}, chunks=None)
times = ["2020-06-01T00:00", "2020-06-02T00:00", "2020-06-03T00:00", "2020-06-04T00:00", "2020-06-06T00:00", "2020-06-08T00:00"]
sub = ds[["2m_temperature", "geopotential", "temperature"]].sel(time=times, level=[500, 850]).load()
sub.to_netcdf(out)
print(sub.dims)
