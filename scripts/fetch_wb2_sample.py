"""Write a 4-timestep WeatherBench2 ERA5 (1.5 deg, public GCS bucket, anonymous) sample used by scripts/arches_real_data.py."""
import sys

import xarray as xr

out = sys.argv[1] if len(sys.argv) > 1 else "/workspace/ckpt/arches/wb2_sample_2020.nc"
url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
ds = xr.open_zarr(url, storage_options={"token": "anon"}, chunks=None)
v = ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'mean_sea_level_pressure', 'geopotential', 'u_component_of_wind',
     'v_component_of_wind', 'temperature', 'specific_humidity', 'vertical_velocity']
ds[v].sel(time=["2020-05-31T12:00", "2020-06-01T12:00", "2020-06-02T12:00", "2020-06-03T12:00"],
          level=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]).load().to_netcdf(out)
