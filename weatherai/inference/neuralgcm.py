"""NeuralGCM forecaster: the complete official JAX model (``neuralgcm`` + ``dinosaur``), unchanged.

The native PyTorch port in ``weatherai.models.neuralgcm`` covers the learned components and the spherical-harmonic layer
only (no dynamical core, no feature builders) and is **not** offered as a forecaster: ``backend="torch"`` raises.

Boundary conditions (SST, sea ice) must be chosen explicitly:

    forcing_policy="persist_initial"  forcings held at their initial values for the whole run (the official demo choice);
                                      fine for weather-range checks, a strong assumption beyond ~2 weeks, recorded in the output
    forcing_policy="provided"         the dataset must contain the forcing variables for every output time; a gap stops the
                                      run with ``MissingForcingError``

Install the JAX stack in its own environment: ``pip install "weatherai[neuralgcm]"`` (jax[cpu] is enough for 2.8 deg).
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .cards import NEURALGCM as CARD
from .errors import BackendUnavailableError, MissingForcingError, MissingInputError
from .manifest import RunManifest, file_sha256
from .result import ForecastResult

POLICIES = ("persist_initial", "provided")


class NeuralGCMForecaster:
    card = CARD

    def __init__(self, wrapper, checkpoint: str, path: Optional[str] = None):
        self.wrapper, self.checkpoint, self.path = wrapper, checkpoint, path

    @classmethod
    def from_pretrained(cls, name: str = "deterministic_2_8_deg", backend: str = "jax") -> "NeuralGCMForecaster":
        if backend != "jax":
            raise BackendUnavailableError("NeuralGCM: only the official JAX model is a complete forecaster; the PyTorch port is partial "
                                          "(learned components + spherical harmonics, no dycore) and is not exposed for forecasting")
        from ..models.neuralgcm.neuralgcm import NeuralGCMWrapper, download_checkpoint, neuralgcm_available
        if not neuralgcm_available():
            raise BackendUnavailableError("NeuralGCM needs `pip install neuralgcm dinosaur jax` (separate JAX environment recommended)")
        p = download_checkpoint(name)
        return cls(NeuralGCMWrapper.from_checkpoint_file(p, name), name, p)

    @property
    def stochastic(self) -> bool:
        return "stochastic" in self.checkpoint

    def run(self, ds, n_steps: int, step_hours: int = 6, forcing_policy: str = "persist_initial", members: int = 1,
            seed: int = 0, variables=None, out_path: Optional[str] = None) -> ForecastResult:
        """``ds``: xarray on the model grid/levels (``wrapper.model.data_coords``) with input + forcing variables; the first time
        is the initial condition. Returns (member, lead[h], level, lat, lon) arrays in the official output units."""
        import jax
        if forcing_policy not in POLICIES:
            raise ValueError(f"forcing_policy must be one of {POLICIES}")
        if members > 1 and not self.stochastic:
            raise ValueError(f"{self.checkpoint} is deterministic: members > 1 would be identical copies")
        m = self.wrapper.model
        miss = [v for v in list(m.input_variables) + list(m.forcing_variables) if v not in ds]
        if miss:
            raise MissingInputError(f"NeuralGCM {self.checkpoint}: dataset lacks {miss}")
        t0 = time.time()
        init = np.datetime64(ds.time.values[0], "ns")
        need = init + np.arange(n_steps) * np.timedelta64(step_hours, "h").astype("timedelta64[ns]")
        if forcing_policy == "provided":
            have = ds.time.values.astype("datetime64[ns]")
            if need[-1] > have.max():
                raise MissingForcingError(f"forcings end {str(have.max())[:16]} but the run needs them to {str(need[-1])[:16]} "
                                          f"({list(m.forcing_variables)}); provide them or use forcing_policy='persist_initial'")
            fsrc = ds
        else:
            fsrc = ds.head(time=1)
        first = ds.isel(time=0)
        inputs = m.inputs_from_xarray(first)
        f0 = m.forcings_from_xarray(first)
        allf = m.forcings_from_xarray(fsrc)
        outs = []
        lead = np.arange(n_steps) * step_hours
        for k in range(members):
            state = m.encode(inputs, f0, jax.random.key(seed + k))
            _, preds = m.unroll(state, allf, steps=n_steps, timedelta=np.timedelta64(step_hours, "h"), start_with_input=True)
            outs.append(m.data_to_xarray(preds, times=lead))
        import xarray as xr
        out = xr.concat(outs, "member")
        keep = variables or [v for v in out.data_vars]
        data, dims = {}, {}
        for v in keep:
            a = out[v]
            data[v] = np.asarray(a.values, dtype=np.float32)
            dims[v] = tuple("lead" if d == "time" else d for d in a.dims)
        coords = {d: np.asarray(out[d].values) for d in out.dims if d not in ("time", "member")}
        coords.update(member=np.arange(members), lead=lead, init_time=init,
                      valid_time=(("lead",), init + lead.astype("timedelta64[h]").astype("timedelta64[ns]")))
        attrs = dict(model="neuralgcm", checkpoint=self.checkpoint, product=CARD.product, forcing_policy=forcing_policy,
                     lead_units="hours", note="lead 0 = encoded/decoded initial state (official start_with_input=True), NOT the analysis: on the official demo snapshot it reaches 157 K at 10 hPa where the input minimum is 183 K; use leads >= 1 step",
                     sst_is_prescribed="yes" + (" (held at initial values)" if forcing_policy == "persist_initial" else ""))
        man = RunManifest("neuralgcm", self.checkpoint, "jax-official",
                          config=dict(n_steps=n_steps, step_hours=step_hours, forcing_policy=forcing_policy, members=members,
                                      seed=seed, init=str(init)), card=CARD.key)
        if self.path:
            man.weights = [file_sha256(self.path)]
        man.timings_s = {"total_s": time.time() - t0}
        man.finish()
        res = ForecastResult("neuralgcm", data, dims, coords, {}, {}, attrs, man).hash_outputs()
        if out_path:
            res.save(out_path)
        return res
