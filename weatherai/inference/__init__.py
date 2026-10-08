"""Reproducible forecast API for the ocean / sub-seasonal / seasonal models of the zoo.

    from weatherai import inference as wi
    wi.card("orca-dl")                         # machine-readable checkpoint card (no heavy imports)
    f = wi.load("orca-dl", root="/ckpt/orca_dl", seeds=(1, 2), device="cuda")
    res = f.run(inputs, lead_months=6, out_path="orca.nc")      # ForecastResult + orca.nc.manifest.json

    f = wi.load("fuxi-s2s", path="/ckpt/fuxi_s2s")              # official ONNX, seeded noise
    idx = f.run(inputs, lead_days=42, members=11, seed=0, out_dir="run/")   # one file per member, resumable

Importing this package imports neither torch, onnxruntime nor jax and downloads nothing; ``load`` imports the backend it
needs. Results keep model-native arrays, coordinates and masks; WeatherPre converts them to its common schema.
"""
from __future__ import annotations

import importlib

from .cards import CARDS, LEVELS, get as card, summary_rows, to_json
from .errors import (BackendUnavailableError, HorizonError, InferenceError, MissingForcingError, MissingInputError, UnitError,
                     WeightsUnavailableError)

# model -> (module, class, {backend: constructor})
_FORECASTERS = {
    "orca-dl": ("orca_dl", "ORCADLForecaster", {"torch": "from_checkpoint"}),
    "fuxi-s2s": ("fuxi_s2s", "FuXiS2SForecaster", {"onnx": "from_onnx", "torch": "from_native"}),
    "ace2": ("ace2", "ACE2Forecaster", {"torch": "from_checkpoint"}),
    "neuralgcm": ("neuralgcm", "NeuralGCMForecaster", {"jax": "from_pretrained"}),
    "unicm": ("unicm", "UniCMForecaster", {"none": "from_checkpoint"}),
}
DEFAULT_BACKEND = {"orca-dl": "torch", "fuxi-s2s": "onnx", "ace2": "torch", "neuralgcm": "jax", "unicm": "none"}


def forecaster_class(model: str):
    mod, cls, _ = _FORECASTERS[card(model).model]
    return getattr(importlib.import_module(f".{mod}", __name__), cls)


def load(model: str, backend: str | None = None, **kwargs):
    """Construct a forecaster. ``backend`` defaults to the reference path of the card (FuXi-S2S: official ONNX)."""
    key = card(model).model
    b = backend or DEFAULT_BACKEND[key]
    ctors = _FORECASTERS[key][2]
    if b not in ctors:
        raise BackendUnavailableError(f"{key}: backend {b!r} not offered (available: {sorted(ctors)})")
    return getattr(forecaster_class(key), ctors[b])(**kwargs)


__all__ = ["CARDS", "LEVELS", "card", "summary_rows", "to_json", "load", "forecaster_class", "InferenceError",
           "WeightsUnavailableError", "BackendUnavailableError", "MissingInputError", "UnitError", "MissingForcingError",
           "HorizonError"]
