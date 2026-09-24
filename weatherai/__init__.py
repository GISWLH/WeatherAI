"""WeatherAI: personal PyTorch weather deep learning model zoo.

Independent project by Longhao Wang (GISWLH). Model implementations are
adapted from WeatherLearn (MIT) and the FengWu paper (arXiv:2304.02948).
"""

from .models import FengWu, FengWu_lite, FuXi, Fuxi, Pangu, Pangu_lite

__version__ = "0.1.0"
__all__ = [
    "Pangu",
    "Pangu_lite",
    "FuXi",
    "Fuxi",
    "FengWu",
    "FengWu_lite",
]
