"""Model zoo: Pangu, FuXi, FengWu, GraphCast, Aurora (wrapper) and more."""

import importlib

# name -> submodule (resolved lazily, PEP 562): importing one model family (e.g. the JAX NeuralGCM wrapper, which must work in
# a torch-free environment) no longer imports every other family.
_LAZY = {
    "Pangu": ".pangu.pangu", "Pangu_lite": ".pangu.pangu", "Fuxi": ".fuxi.fuxi", "FuXi": ".fuxi.fuxi",
    "FengWu": ".fengwu.fengwu", "FengWu_lite": ".fengwu.fengwu", "GraphCast": ".graphcast.graphcast",
    "GraphCast_lite": ".graphcast.graphcast", "Aurora": ".aurora", "Aurora_lite": ".aurora", "Aurora_small": ".aurora",
    "AuroraWrapper": ".aurora", "AardvarkProcessor": ".aardvark.aardvark", "AardvarkProcessor_lite": ".aardvark.aardvark",
    "GenCast": ".gencast.gencast", "GenCast_lite": ".gencast.gencast",
    "WeatherNextCyclonesWrapper": ".weathernext_cyclones.wnc", "WeatherNextCyclones_lite": ".weathernext_cyclones.wnc",
    "WeatherNextCyclonesNet": ".weathernext_cyclones", "WeatherNextCyclonesNet_from_official": ".weathernext_cyclones",
    "WeatherNextCyclonesNative_lite": ".weathernext_cyclones", "WNCEnsemble": ".weathernext_cyclones",
    "NeuralGCM_lite": ".neuralgcm.neuralgcm", "NeuralGCMWrapper": ".neuralgcm.neuralgcm",
    "FuXiENS": ".fuxi_ens", "FuXiENS_lite": ".fuxi_ens", "FuXiENSConfig": ".fuxi_ens", "FuXiENSNoise": ".fuxi_ens",
}


def __getattr__(name):
    if name in _LAZY:
        mod = importlib.import_module(_LAZY[name], __name__)
        return getattr(mod, "Fuxi" if name == "FuXi" else name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
