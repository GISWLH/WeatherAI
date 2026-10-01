"""Thin wrapper around Google DeepMind's official **JAX** WeatherNext Cyclones (WN-C) code.

WeatherNext Cyclones (Alet et al., *Nature* 2026) is a Functional Generative Network (FGN): a
GraphCast-style grid→mesh→grid model whose mesh processor is a 16-layer sparse transformer, fed with
learned noise to produce ensemble members (``weathernext.weathernext2.fgn``), plus a direct cyclone
tracker. Everything is JAX/Haiku, with TPU-oriented sparse attention. A PyTorch port would require
re-deriving the whole typed-graph/transformer stack and converting Haiku ``.npz`` weights with no
reference other than the JAX code itself; **it is not attempted**. This module instead

* loads the official config + checkpoint of the smallest model, *WeatherNextCyclones_Mini* (1°,
  227 MB ``.npz`` on the public bucket ``gs://dm_graphcast``; weights CC-BY-4.0, code Apache-2.0),
* runs the official predictor on CPU/GPU (switching the TPU-only ``splash_mha`` attention to plain
  ``mha``, as the official demo notebook does for non-TPU backends), and
* exposes results as ``xarray`` datasets (and torch tensors via ``to_torch``).

Inference only; requires Python ≥ 3.12 and ``pip install git+https://github.com/google-deepmind/weathernext``
(plus ``h5netcdf``/``h5py``). The cyclone *tracker* (needs IBTrACS initial storms) is not wrapped.
"""
from __future__ import annotations

import dataclasses
import os
import urllib.parse
import urllib.request
from typing import Dict, Optional

import numpy as np

_BUCKET = "https://storage.googleapis.com/dm_graphcast/"
CHECKPOINTS = {
    "WeatherNextCyclones_Mini_<2024": "weathernext2/params/WeatherNextCyclones_Mini_<2024.npz",
    "WeatherNextCyclones_Mini_<2023": "weathernext2/params/WeatherNextCyclones_Mini_<2023.npz",
}
SAMPLE_DATA = {  # HRES-initialised 1° sample, init 2024-10-07 00Z (6 frames incl. 2 input frames), 159 MB
    "1.0-steps04": "weathernext2/dataset/source-hres_forecast_init-2024-10-07 00:00:00_res-1.0_levels-13_steps-04.nc",
}


def weathernext_available() -> bool:
    try:
        import haiku  # noqa: F401
        import jax  # noqa: F401
        import xarray_jax  # noqa: F401
        from weathernext.weathernext2 import fgn  # noqa: F401

        return True
    except Exception:
        return False


def _cache() -> str:
    d = os.path.join(os.environ.get("WEATHERAI_CACHE", os.path.expanduser("~/.cache/weatherai")), "weathernext")
    os.makedirs(d, exist_ok=True)
    return d


def _fetch(key: str) -> str:
    path = os.path.join(_cache(), os.path.basename(key))
    if not os.path.exists(path):
        tmp = path + ".part"
        urllib.request.urlretrieve(_BUCKET + urllib.parse.quote(key), tmp)
        os.replace(tmp, path)
    return path


def download_checkpoint(name: str = "WeatherNextCyclones_Mini_<2024") -> str:
    if name not in CHECKPOINTS:
        raise KeyError(f"unknown checkpoint {name!r}; choose from {sorted(CHECKPOINTS)}")
    return _fetch(CHECKPOINTS[name])


def download_sample_data(name: str = "1.0-steps04") -> str:
    return _fetch(SAMPLE_DATA[name])


class WeatherNextCyclonesWrapper:
    """Official WN-C (Mini) predictor with a small xarray-in / xarray-out API."""

    CONFIG = "weathernext2/configs/WeatherNextCyclones_Mini"

    def __init__(self, checkpoint_path: str, attention_type: Optional[str] = None):
        if not weathernext_available():
            raise ImportError("needs Python>=3.12 and `pip install git+https://github.com/google-deepmind/weathernext`")
        import haiku as hk
        import jax
        from weathernext.utils import checkpoint, fiddle_config_io
        from weathernext.weathernext2 import fgn

        self.config = fiddle_config_io.get_fiddle_config_by_name(self.CONFIG)
        with open(checkpoint_path, "rb") as f:
            self.ckpt = checkpoint.load(f, fgn.CheckPoint)
        tk = self.config.predictor_kwargs["noisy_function_kwargs"]["mesh_model_ctor"].keywords["transformer_kwargs"]
        backend = jax.default_backend()
        # Official README/notebook: splash attention is TPU-only; GPUs need 'triblockdiag_mha'; CPU: 'mha'.
        tk["attention_type"] = attention_type or {"tpu": tk["attention_type"], "gpu": "triblockdiag_mha"}.get(backend, "mha")
        self.attention_type = tk["attention_type"]
        cfg = fgn.PredictorConfig(
            task=self.config.task,
            predictor_constructor=self.config.predictor_constructor,
            predictor_kwargs=self.config.predictor_kwargs,
            predictor_wrappers=self.config.predictor_wrappers[:-1],  # drop the training-time ensemble wrapper
        )

        @hk.transform
        def _forward(inputs, targets_template, forcings):
            return fgn.construct_predictor(cfg)(inputs, targets_template=targets_template, forcings=forcings)

        self._apply = jax.jit(
            lambda rng, inputs, targets_template, forcings: _forward.apply(
                self.ckpt.params, rng, inputs, targets_template, forcings
            )
        )

    @classmethod
    def from_pretrained(cls, name: str = "WeatherNextCyclones_Mini_<2024", **kw) -> "WeatherNextCyclonesWrapper":
        return cls(download_checkpoint(name), **kw)

    @property
    def task(self):
        return self.config.task

    def split(self, example_batch, max_steps: Optional[int] = None):
        """``example_batch`` (official sample-data layout) → (inputs, targets, forcings) for 6 h steps."""
        from weathernext.utils import data_utils

        n = example_batch.sizes["time"] - 2
        n = n if max_steps is None else min(n, max_steps)
        return data_utils.extract_inputs_targets_forcings(
            example_batch, target_lead_times=slice("6h", f"{n * 6}h"), **dataclasses.asdict(self.task)
        )

    def forecast(self, example_batch, steps: int = 2, num_members: int = 2, seed: int = 0):
        """Autoregressive 6 h-step ensemble forecast from the first 2 frames of ``example_batch``.

        Returns ``xarray.Dataset`` with dims ``(sample, time, batch, [level,] lat, lon)``. Members use
        ``fold_in(PRNGKey(seed), i)`` keys, as in the official demo.
        """
        import jax
        import xarray
        from weathernext.utils import rollout

        inputs, targets, forcings = self.split(example_batch, steps)
        rngs = np.stack([jax.random.fold_in(jax.random.PRNGKey(seed), i) for i in range(num_members)], axis=0)
        chunks = list(
            rollout.chunked_prediction_generator_multiple_runs(
                predictor_fn=self._apply,
                rngs=rngs,
                inputs=inputs,
                targets_template=targets * np.nan,
                forcings=forcings,
                num_steps_per_chunk=1,
                num_samples=num_members,
            )
        )
        # Without pmap the generator yields one chunk per (sample, step) with a *scalar* "sample"
        # coordinate, so combine per member over time, then stack members along a "sample" dim.
        members = []
        for i in range(num_members):
            mine = [c for c in chunks if int(c.coords["sample"]) == i]
            members.append(xarray.combine_by_coords([c.drop_vars("sample") for c in mine]))
        return xarray.concat(members, dim=xarray.DataArray(np.arange(num_members), dims="sample", name="sample"))

    @staticmethod
    def to_torch(ds, variables=None) -> Dict[str, "torch.Tensor"]:  # noqa: F821
        import torch

        return {k: torch.from_numpy(np.asarray(ds[k].values).copy()) for k in (variables or list(ds.data_vars))}


def WeatherNextCyclones_lite(**kw) -> WeatherNextCyclonesWrapper:
    """Smallest official model: WeatherNextCyclones_Mini (1°, trained through 2023)."""
    return WeatherNextCyclonesWrapper.from_pretrained("WeatherNextCyclones_Mini_<2024", **kw)
