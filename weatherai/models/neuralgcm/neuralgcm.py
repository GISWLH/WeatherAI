"""Thin wrapper around Google's official **JAX** NeuralGCM (the oracle; the native PyTorch learned components are in network.py, spectral.py).

NeuralGCM (Kochkov et al., *Nature* 632, 2024) couples a differentiable spectral dynamical core
(``dinosaur``: primitive equations on sigma levels, spherical-harmonic transforms) with learned
Haiku/JAX physics and encoders/decoders. Re-implementing that in PyTorch (spherical harmonics,
semi-implicit time stepping, trained corrector networks, checkpoint conversion) is a multi-month
project and **is not attempted here**. Instead:

* ``NeuralGCMWrapper`` runs the *official* code/checkpoints via ``neuralgcm`` (JAX) and
  returns results as ``xarray`` / ``torch`` tensors, so it can sit beside the zoo's PyTorch models;
* ``NeuralGCM_lite()`` loads the smallest official checkpoint (2.8°, deterministic, 58 MB,
  CC-BY-SA-4.0 weights, Apache-2.0 code) that runs in seconds on CPU.

No gradients flow from torch into JAX: this wrapper is inference-only. Because it executes the
upstream implementation unchanged, it is numerically the official model; WeatherAI adds only I/O
conversion (tests check wrapper == direct upstream call).

Install: ``pip install neuralgcm dinosaur`` (JAX; ``jax[cpu]`` is enough for 2.8°).
"""
from __future__ import annotations

import os
import pickle
import urllib.request
from typing import Dict, Optional

import numpy as np

_BASE = "https://storage.googleapis.com/neuralgcm/models/"
CHECKPOINTS = {
    "deterministic_2_8_deg": "v1/deterministic_2_8_deg.pkl",
    "deterministic_1_4_deg": "v1/deterministic_1_4_deg.pkl",
    "deterministic_0_7_deg": "v1/deterministic_0_7_deg.pkl",
    "stochastic_1_4_deg": "v1/stochastic_1_4_deg.pkl",
    "stochastic_precip_2_8_deg": "v1_precip/stochastic_precip_2_8_deg.pkl",
    "stochastic_evap_2_8_deg": "v1_precip/stochastic_evap_2_8_deg.pkl",
}


def neuralgcm_available() -> bool:
    try:
        import dinosaur  # noqa: F401
        import jax  # noqa: F401
        import neuralgcm  # noqa: F401

        return True
    except Exception:
        return False


def _cache_dir() -> str:
    d = os.environ.get("WEATHERAI_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "weatherai"))
    d = os.path.join(d, "neuralgcm")
    os.makedirs(d, exist_ok=True)
    return d


def download_checkpoint(name: str = "deterministic_2_8_deg") -> str:
    """Download an official NeuralGCM checkpoint (public GCS bucket) and return the local path."""
    if name not in CHECKPOINTS:
        raise KeyError(f"unknown checkpoint {name!r}; choose from {sorted(CHECKPOINTS)}")
    path = os.path.join(_cache_dir(), os.path.basename(CHECKPOINTS[name]))
    if not os.path.exists(path):
        tmp = path + ".part"
        urllib.request.urlretrieve(_BASE + CHECKPOINTS[name], tmp)
        os.replace(tmp, path)
    return path


class NeuralGCMWrapper:
    """Inference wrapper around ``neuralgcm.PressureLevelModel``."""

    def __init__(self, model, name: str = "custom"):
        self.model = model
        self.name = name

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_checkpoint_file(cls, path: str, name: Optional[str] = None) -> "NeuralGCMWrapper":
        if not neuralgcm_available():
            raise ImportError("needs `pip install neuralgcm dinosaur jax` (official NeuralGCM, JAX)")
        import neuralgcm

        with open(path, "rb") as f:
            ckpt = pickle.load(f)  # official checkpoints are pickles — only load trusted files
        return cls(neuralgcm.PressureLevelModel.from_checkpoint(ckpt), name or os.path.basename(path))

    @classmethod
    def from_pretrained(cls, name: str = "deterministic_2_8_deg") -> "NeuralGCMWrapper":
        return cls.from_checkpoint_file(download_checkpoint(name), name)

    # ------------------------------------------------------------------ info
    @property
    def input_variables(self):
        return list(self.model.input_variables)

    @property
    def forcing_variables(self):
        return list(self.model.forcing_variables)

    @property
    def grid_shape(self):
        """(n_lon, n_lat) of the model's data grid."""
        return tuple(self.model.data_coords.horizontal.nodal_shape)

    @property
    def levels(self):
        return np.asarray(self.model.data_coords.vertical.centers)

    def demo_data(self):
        """One ERA5 snapshot (1959-01-02 00Z, TL31 regridded to this model's grid; shipped with
        the official package). Good for smoke tests only."""
        import neuralgcm

        return neuralgcm.demo.load_data(self.model.data_coords)

    # ------------------------------------------------------------------ inference
    def forecast(self, ds, steps: int = 4, step_hours: int = 6, seed: int = 0):
        """Run an autoregressive forecast from the first time of ``ds`` (xarray, as the official
        ``inference_demo``: input + forcing variables on this model's grid and levels).

        Returns an ``xarray.Dataset`` with ``steps`` outputs at 0, step_hours, ... hours
        (``start_with_input=True``, so index 0 is the encoded/decoded initial state).
        Forcings (SST, sea ice) are held at their initial values (persistence).
        """
        import jax

        m = self.model
        first = ds.isel(time=0)
        inputs = m.inputs_from_xarray(first)
        forcings0 = m.forcings_from_xarray(first)
        state = m.encode(inputs, forcings0, jax.random.key(seed))
        all_forcings = m.forcings_from_xarray(ds.head(time=1))
        _, preds = m.unroll(
            state,
            all_forcings,
            steps=steps,
            timedelta=np.timedelta64(step_hours, "h"),
            start_with_input=True,
        )
        return m.data_to_xarray(preds, times=np.arange(steps) * step_hours)

    @staticmethod
    def to_torch(ds, variables=None, dtype=None) -> Dict[str, "torch.Tensor"]:  # noqa: F821
        """Convert an output ``xarray.Dataset`` to ``{var: torch.Tensor}`` (dims as in ``ds``)."""
        import torch

        out = {}
        for k in variables or list(ds.data_vars):
            t = torch.from_numpy(np.asarray(ds[k].values).copy())
            out[k] = t.to(dtype) if dtype is not None else t
        return out


def NeuralGCM_lite() -> NeuralGCMWrapper:
    """Smallest official checkpoint: deterministic, 2.8° (128×64), 37 pressure levels, 58 MB."""
    return NeuralGCMWrapper.from_pretrained("deterministic_2_8_deg")
