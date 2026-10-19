from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _finite(x, name):
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError(f"{name} must contain finite floating-point values")


def project_cdf(x: torch.Tensor) -> torch.Tensor:
    """Clip, then L2 isotonic regression along the last (K-1 thresholds) axis.

    The min-max formula is equivalent to PAVA and stays on the input device.
    Clipping BEFORE projection matches the released PBC projector.
    """
    _finite(x, "CDF")
    if x.ndim == 0 or x.shape[-1] < 1:
        raise ValueError("CDF needs a nonempty threshold axis")
    x = x.clamp(0, 1)
    n = x.shape[-1]
    sums = torch.cat((torch.zeros_like(x[..., :1]), x.cumsum(-1)), -1)
    result = []
    for i in range(n):
        candidates = []
        for a in range(i + 1):
            means = torch.stack([(sums[..., b + 1] - sums[..., a]) / (b + 1 - a)
                                 for b in range(i, n)], -1)
            candidates.append(means.amin(-1))
        result.append(torch.stack(candidates, -1).amax(-1))
    return torch.stack(result, -1)


def ensemble_cdf(members: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    """Members (..., M), thresholds (..., K-1) -> empirical CDF (..., K-1).

    Strict less-than is intentional (paper convention, including dry ties).
    """
    _finite(members, "members")
    _finite(thresholds, "thresholds")
    if not members.shape[-1] or torch.any(thresholds[..., 1:] < thresholds[..., :-1]):
        raise ValueError("need nonempty members and ordered thresholds")
    return (members[..., :, None] < thresholds[..., None, :]).to(members.dtype).mean(-2)


def seasonal_mask(dates, target, span=28, years=20):
    """Debias++ Algorithm 1 seasonal window, excluding all future dates.

    Caller must additionally ensure observations have finished by forecast
    issuance; target dates alone cannot establish operational availability.
    """
    if not 0 <= span <= 182 or years < 1:
        raise ValueError("span must be 0..182 and years positive")
    delta = (np.datetime64(target, "D") - np.asarray(dates, dtype="datetime64[D]")).astype(int)
    year_diff = np.floor(delta / 365.242199)
    day_diff = 182.5 - np.abs(delta % 365.242199 - 182.5)
    return (delta > 0) & (year_diff <= years) & (day_diff <= span)


class PBC(nn.Module):
    """Fit per-grid/per-threshold corrections on historical CDFs.

    Arrays: (sample, *grid, K-1). Persistence predictors, in paper order:
    intercept, climatology, lag1, lag2, raw CDF. Debias training can use a
    separately selected seasonal sample set. This class does not download
    hindcasts or silently invent climatology/lag observations.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("bias", torch.empty(0))
        self.register_buffer("coefficients", torch.empty(0))

    @staticmethod
    def features(raw, climatology, lag1, lag2):
        for name, value in (("raw", raw), ("climatology", climatology), ("lag1", lag1), ("lag2", lag2)):
            _finite(value, name)
            if value.shape != raw.shape or torch.any((value < 0) | (value > 1)):
                raise ValueError(f"{name} must match raw shape and lie in [0, 1]")
        return torch.stack((torch.ones_like(raw), climatology, lag1, lag2, raw), -1)

    @torch.no_grad()
    def fit(self, raw, observed, climatology, lag1, lag2, *, debias_raw=None, debias_observed=None):
        x = self.features(raw, climatology, lag1, lag2)
        if raw.ndim < 2 or raw.shape[0] < 5:
            raise ValueError("need at least five historical samples and a threshold axis")
        _finite(observed, "observed")
        if observed.shape != raw.shape or torch.any((observed < 0) | (observed > 1)):
            raise ValueError("observed indicators must match raw and lie in [0, 1]")
        if (debias_raw is None) != (debias_observed is None):
            raise ValueError("provide both debias_raw and debias_observed")
        dr, do = (raw, observed) if debias_raw is None else (debias_raw, debias_observed)
        for value in (dr, do):
            _finite(value, "debias training")
        if dr.shape != do.shape or dr.shape[1:] != raw.shape[1:] or dr.shape[0] == 0:
            raise ValueError("debias training needs matching nonempty samples on the same grid")
        # Solve original least-squares problem by SVD, robust to collinear
        # predictors on both CPU and CUDA (CUDA gels requires full rank).
        design = x.movedim(0, -2)
        target = observed.movedim(0, -1).unsqueeze(-1)
        self.coefficients = (torch.linalg.pinv(design) @ target).squeeze(-1)
        self.bias = (do - dr).mean(0)
        return self

    def forward(self, raw, climatology, lag1, lag2, *, debias_target=None):
        if self.bias.numel() == 0:
            raise RuntimeError("PBC must be fitted on historical data before prediction")
        x = self.features(raw, climatology, lag1, lag2)
        if raw.shape[-self.bias.ndim:] != self.bias.shape:
            raise ValueError("prediction grid/thresholds differ from fitted model")
        dr = raw if debias_target is None else debias_target
        _finite(dr, "debias_target")
        if dr.shape != raw.shape or torch.any((dr < 0) | (dr > 1)):
            raise ValueError("debias_target must match raw and lie in [0, 1]")
        debias = project_cdf(dr + self.bias)
        persistence = project_cdf((x * self.coefficients).sum(-1))
        return (debias + persistence) / 2

    def predict_proba(self, *args, **kwargs):
        cdf = self(*args, **kwargs)
        return torch.diff(torch.cat((torch.zeros_like(cdf[..., :1]), cdf, torch.ones_like(cdf[..., :1])), -1), dim=-1)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for name in ("bias", "coefficients"):
            if prefix + name in state_dict:
                setattr(self, name, torch.empty_like(state_dict[prefix + name], device=getattr(self, name).device))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


def PBC_lite():
    """Same exact statistical algorithm; reduce the grid/history for a lite run."""
    return PBC()
