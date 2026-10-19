"""Probabilistic bias correction (Guan et al., Nature 2026).

Tensor implementation of Debias++, Persistence++, isotonic projection and
equal-weight ensembling. Inputs are prepared CDFs, not deterministic fields.
Calendar selection/validation must precede fit; no universal weights exist.
"""
from .model import PBC, PBC_lite, ensemble_cdf, project_cdf, seasonal_mask

__all__ = ["PBC", "PBC_lite", "ensemble_cdf", "project_cdf", "seasonal_mask"]
