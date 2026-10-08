"""NeuralGCM: official JAX wrapper (``neuralgcm.py``, torch-free) + partial native PyTorch port (``network.py``, ``spectral.py``).

The torch parts are resolved lazily so the JAX wrapper can be imported in an environment without torch."""
import importlib

from .neuralgcm import (
    CHECKPOINTS,
    NeuralGCMWrapper,
    NeuralGCM_lite,
    neuralgcm_available,
)

_TORCH = {"EpdTower": ".network", "NeuralGCMLearnedComponents": ".network", "NeuralGCMLearnedConfig": ".network",
          "SurfaceEmbedding": ".network", "VerticalConvTower": ".network", "convert_official_params": ".network",
          "SphericalHarmonicsGrid": ".spectral", "SpectralGridConfig": ".spectral"}

__all__ = ["NeuralGCMWrapper", "NeuralGCM_lite", "neuralgcm_available", "CHECKPOINTS"] + list(_TORCH)


def __getattr__(name):
    if name in _TORCH:
        return getattr(importlib.import_module(_TORCH[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
