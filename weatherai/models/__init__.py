"""Model zoo: Pangu, FuXi, FengWu, GraphCast, Aurora (wrapper) and more."""

from .pangu.pangu import Pangu, Pangu_lite
from .fuxi.fuxi import Fuxi
from .fengwu.fengwu import FengWu, FengWu_lite
from .graphcast.graphcast import GraphCast, GraphCast_lite
from .aurora import Aurora, Aurora_lite, Aurora_small, AuroraWrapper
from .aardvark.aardvark import AardvarkProcessor, AardvarkProcessor_lite
from .gencast.gencast import GenCast, GenCast_lite
from .weathernext_cyclones.wnc import WeatherNextCyclonesWrapper, WeatherNextCyclones_lite
from .weathernext_cyclones import WeatherNextCyclonesNet, WeatherNextCyclonesNet_from_official, WeatherNextCyclonesNative_lite, WNCEnsemble
from .neuralgcm.neuralgcm import NeuralGCM_lite, NeuralGCMWrapper
from .fuxi_ens import FuXiENS, FuXiENS_lite, FuXiENSConfig, FuXiENSNoise

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
    "Aurora",
    "AuroraWrapper",
    "Aurora_lite",
    "Aurora_small",
    "NeuralGCMWrapper",
    "NeuralGCM_lite",
    "FuXiENS",
    "FuXiENS_lite",
    "GenCast",
    "GenCast_lite",
    "AardvarkProcessor",
    "AardvarkProcessor_lite",
    "WeatherNextCyclonesWrapper",
    "WeatherNextCyclones_lite",
    "WeatherNextCyclonesNet",
    "WeatherNextCyclonesNet_from_official",
    "WeatherNextCyclonesNative_lite",
    "WNCEnsemble",
]
