# PBC: probabilistic subseasonal calibration

[Model pages](README.md) · [Three-paper integration](../research_20261019.md)

Guan et al., **Advancing subseasonal forecasting with machine learning**, Nature,
14 October 2026. [Paper](https://doi.org/10.1038/s41586-026-11126-x).
[Official MIT code](https://github.com/mouatadid/pbc), inspected at
`d59e18f720bac712e18563bd469d4679c120f9e5`;
[archived release](https://zenodo.org/records/22238796).

PBC is statistical post-processing, not an autoregressive neural forecaster.
It combines Debias++ (historical CDF bias subtraction), Persistence++ (per-grid,
per-threshold least squares on intercept, rolling climatology, two observable
lag indicators and raw forecast CDF), clip-then-isotonic CDF projection, and
equal-weight averaging. Forecasts in the paper cover days 19-25 and 26-32.
These are target-relative lead windows, **not** WeatherPre's standard weeks 3/4
(days 14-21/21-28); do not interchange the labels.

```python
from weatherai import PBC, PBC_lite
model = PBC()  # PBC_lite() uses the same statistical algorithm
model.fit(raw_train, observed_train, clim_train, lag1_train, lag2_train,
          debias_raw=seasonal_raw, debias_observed=seasonal_observed)
cdf = model(raw, climatology, lag1, lag2, debias_target=raw_observed_thresholds)
probabilities = model.predict_proba(raw, climatology, lag1, lag2)
```

Tensor axes: `(sample, *grid, K-1)` in training; either that layout or `(*grid,
K-1)` in prediction. Four CDF thresholds produce five category probabilities.
`ensemble_cdf` uses strict `<`, including ties at zero precipitation. Save/load
the fitted parameters with `state_dict`; there is no universal pretrained
checkpoint in the official release. SVD least squares also handles collinear
predictors on CUDA. The lite variant reduces data/grid size, not the algorithm.

Scope: the numerical kernels are implemented. Users must prepare the paper's
20-year climatologies, model/observed quintile thresholds, hindcasts, seasonal
training windows and observable lags; the rolling three-year selection over
Debias++ spans 14/28/35 and issuance-date ensembles is not automated here.
Pass separately prepared Debias and Persistence CDFs when their thresholds
differ (as they do in several official experiments). Fit only on observations
available at issuance. No live data download, paper RPSS replication, POET,
AIFS-SUBS retraining, or MicroDuet operational product is claimed.

Tests compare the isotonic result to SciPy and the fitted coefficients to
independent NumPy least squares, and check category normalization, rank
deficiency, strict ties, invalid inputs and checkpoint round trips.
