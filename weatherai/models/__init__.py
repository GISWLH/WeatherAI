"""Model zoo: Pangu, FuXi, FengWu."""

from .pangu.pangu import Pangu, Pangu_lite
from .fuxi.fuxi import Fuxi
from .fengwu.fengwu import FengWu, FengWu_lite

# Preferred alias (paper-style capitalization)
FuXi = Fuxi

__all__ = [
    "Pangu",
    "Pangu_lite",
    "FuXi",
    "Fuxi",
    "FengWu",
    "FengWu_lite",
]
