"""Probabilistic bias correction (Guan et al., Nature 2026).

PBC executes the author's original scripts. PBCLite/PBC_lite is the separate
tensor approximation. Historical data are required; no universal weights exist.
"""
from .official import PBC
from .model import PBCLite, PBC_lite, ensemble_cdf, project_cdf, seasonal_mask

__all__ = ["PBC", "PBCLite", "PBC_lite", "ensemble_cdf", "project_cdf", "seasonal_mask"]
