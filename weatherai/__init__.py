"""WeatherAI: personal PyTorch weather deep learning model zoo.

Independent project by Longhao Wang (GISWLH). Model implementations are
adapted from WeatherLearn (MIT) and paper-inspired skeletons (FengWu,
GraphCast).

The top-level names are resolved lazily (PEP 562): ``import weatherai`` / ``import weatherai.inference`` import no torch,
build no model and download nothing; ``from weatherai import Pangu`` works as before.
"""

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


def __getattr__(name):
    if name in __all__:
        from . import models
        return getattr(models, name)
    raise AttributeError(f"module 'weatherai' has no attribute {name!r}")
