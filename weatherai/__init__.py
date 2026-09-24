"""WeatherAI: personal PyTorch weather deep learning model zoo.

Independent project by Longhao Wang (GISWLH). Model implementations are
adapted from WeatherLearn (MIT) and paper-inspired skeletons (FengWu,
GraphCast).
"""

from .models import (
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
    "Pangu",
    "Pangu_lite",
    "FuXi",
    "Fuxi",
    "FengWu",
    "FengWu_lite",
    "GraphCast",
    "GraphCast_lite",
]
