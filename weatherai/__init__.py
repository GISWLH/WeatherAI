"""WeatherAI: weather models, official pretrained adapters and explicit lite models.

Independent project by Longhao Wang (GISWLH). Model implementations are
adapted from WeatherLearn (MIT) and paper-inspired skeletons (FengWu,
GraphCast). GenFocal, PBC and Translocation main entries use pinned author code.
"""

from .models import (
    GenFocal, GenFocal_lite, PBC, PBC_lite, Translocation, Translocation_lite,
    FengWu,
    FengWu_lite,
    FuXi,
    Fuxi,
    GraphCast,
    GraphCast_lite,
    Pangu,
    Pangu_lite,
)

__version__ = "0.1.0"
__all__ = [
    "GenFocal", "GenFocal_lite", "PBC", "PBC_lite", "Translocation", "Translocation_lite",
    "Pangu",
    "Pangu_lite",
    "FuXi",
    "Fuxi",
    "FengWu",
    "FengWu_lite",
    "GraphCast",
    "GraphCast_lite",
]
