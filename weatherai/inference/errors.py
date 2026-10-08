"""Explicit failure types of the inference API (callers such as WeatherPre map them to availability reasons)."""
from __future__ import annotations


class InferenceError(RuntimeError):
    """Base class. ``reason`` is a stable machine-readable code."""
    reason = "error"


class WeightsUnavailableError(InferenceError):
    """The checkpoint is not on disk or not published at all (UniCM)."""
    reason = "weights_missing"


class BackendUnavailableError(InferenceError):
    """A required package (onnxruntime, jax/neuralgcm, xarray) is not installed, or a backend is deliberately not offered."""
    reason = "not_installed"


class MissingInputError(InferenceError):
    """A required input variable / level / history step / mask is missing or malformed."""
    reason = "input_missing"


class UnitError(MissingInputError):
    """Values are outside the physical range expected for the documented unit (usually a unit-conversion slip)."""
    reason = "unit_mismatch"


class MissingForcingError(InferenceError):
    """An external forcing needed for some forecast step is not provided (e.g. ACE2 future SST)."""
    reason = "forcing_missing"


class HorizonError(InferenceError):
    """Requested lead exceeds the checkpoint's technical limit (e.g. FuXi-S2S > 42 days)."""
    reason = "beyond_technical_horizon"
