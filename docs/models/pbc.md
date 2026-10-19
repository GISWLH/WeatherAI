# PBC — official probabilistic subseasonal calibration

Guan et al., *Advancing subseasonal forecasting with machine learning*, Nature
2026. [Paper](https://doi.org/10.1038/s41586-026-11126-x) ·
[Author code](https://github.com/mouatadid/pbc/tree/d59e18f720bac712e18563bd469d4679c120f9e5) ·
[Archived release](https://zenodo.org/records/22238796).

`PBC` executes the unchanged official scripts pinned to
`d59e18f720bac712e18563bd469d4679c120f9e5`. The paper combines Debias++,
Persistence++, CDF projection and probabilistic ensembling. It is statistical
post-processing, not a new autoregressive neural forecaster.

```python
from weatherai import PBC
model = PBC(work_dir="pbc-work")
# Official baseline, runnable without reforecast archives:
result = model.run("climatology", "era5-f1_mslp", 19, target_dates="20200106")
# After preparing author-format historical data and component forecasts:
# model.run("ecmwfpp", "era5-f1_mslp", 19, target_dates="20200106", arguments=(...))
# model.run("perpp_ecmwf", "era5-f1_mslp", 19, target_dates="20200106", arguments=(...))
# model.run("projector", "era5-mslp", 19, target_dates="20200106", arguments=(...))
# model.run("pbc_ecmwf", "era5-f1_mslp", 19, target_dates="20200106")
```

There is **no universal pretrained checkpoint in the official release**.
Coefficients depend on variable, grid, target date and historical observations/
reforecasts. `PBC.from_pretrained` raises an explanatory error. Author code/data
layout and selection configuration are preserved; missing data never trigger a
lite substitute. Original scripts sometimes exit successfully after skipping
missing inputs, so inspect their logs and expected NetCDF outputs.

Recommended runtime: author's Linux/Python 3.13 environment; the tested CPU
stages also ran on Windows/Python 3.13. Dependencies include NumPy, pandas,
xarray, netCDF4, dask, scikit-learn, ttictoc and setuptools with pkg_resources.
The full archive preparation uses further tools listed upstream.

`python scripts/validate_official_pbc.py` actually runs the original climatology,
projector and equal-weight PBC scripts. A synthetic fixture verifies clipping,
isotonic projection and final CDF `[0.05, 0.35, 0.55, 0.95]`.
[Execution report](../results/pbc_official_execution.json). This is not a
historical fit or reproduction of the paper's skill scores.

`PBC_lite()` is the earlier independent tensor approximation, with `.fit`,
`.predict_proba` and `state_dict`. It requires explicitly prepared CDF histories;
it is not the official pipeline. Official target windows are days 19–25 and
26–32, distinct from WeatherPre's generic weekly windows. Code license: MIT.
