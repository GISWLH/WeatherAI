"""Model zoo: Pangu, FuXi, FengWu, GraphCast, Aurora (wrapper) and more."""

from .pangu.pangu import Pangu, Pangu_lite
from .fuxi.fuxi import Fuxi
from .fengwu.fengwu import FengWu, FengWu_lite
from .graphcast.graphcast import GraphCast, GraphCast_lite
from .aurora.aurora import Aurora_lite, Aurora_small, AuroraWrapper
from .gencast.gencast import GenCast, GenCast_lite
from .neuralgcm.neuralgcm import NeuralGCM_lite, NeuralGCMWrapper

# Preferred alias (paper-style capitalization)
FuXi = Fuxi

__all__ = [
    "Pangu",
    "Pangu_lite",
    "FuXi",
    "Fuxi",
    "FengWu",
    "FengWu_lite",
    "GraphCast",
    "GraphCast_lite",
    "AuroraWrapper",
    "Aurora_lite",
    "Aurora_small",
    "NeuralGCMWrapper",
    "NeuralGCM_lite",
    "GenCast",
    "GenCast_lite",
]
