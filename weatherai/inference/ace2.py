"""ACE2-ERA5 forecaster: an atmosphere emulator driven by prescribed SST / sea ice / insolation / CO2.

A run is an *atmospheric response to the boundary conditions you supply*. It is never an SST forecast, and with observed
future SST it is a conditional hindcast, not a forecast. ``forcing_provenance`` is therefore mandatory and is written into
the output (``experiment_type``):

    observed     -> conditional_hindcast   (future observations used: NOT usable as a forecast / for skill claims)
    forecast     -> forced_forecast        (boundary conditions from another forecast available at init time)
    scenario     -> scenario               (explicit what-if boundary conditions)
    persistence  -> persisted_boundary     (initial anomalies held; a forecast assumption, state it)

Every forcing must cover every step; a gap stops the run with ``MissingForcingError`` naming the variables and steps.
"""
from __future__ import annotations

import os
import time
from typing import Dict, Optional, Sequence

import numpy as np

from .cards import ACE2_ERA5 as CARD
from .errors import MissingForcingError, MissingInputError, WeightsUnavailableError
from .manifest import RunManifest, array_sha256, file_sha256
from .result import ForecastResult

PROVENANCE = {"observed": "conditional_hindcast", "forecast": "forced_forecast", "scenario": "scenario",
              "persistence": "persisted_boundary"}
STEP = np.timedelta64(6, "h")


def gauss_legendre_lat(n: int = 180) -> np.ndarray:
    x, _ = np.polynomial.legendre.leggauss(n)
    return np.rad2deg(np.arcsin(x))            # south -> north


class ACE2Forecaster:
    card = CARD

    def __init__(self, model, checkpoint: str = "in-memory", files=(), device: str = "cpu"):
        self.model, self.checkpoint, self.files, self.device = model, checkpoint, list(files), device
        c = model.cfg
        self.prognostic = tuple(model.prognostic)
        self.forcing_names = tuple(model.forcing_names)
        self.next_names = (c.ocean_fraction_name, c.sst_name)

    @classmethod
    def from_checkpoint(cls, ckpt_dir: str, device: str = "cpu") -> "ACE2Forecaster":
        p = os.path.join(ckpt_dir, "ace2_era5_ckpt.tar")
        if not os.path.exists(p):
            raise WeightsUnavailableError(f"{p} missing; download HF allenai/ACE2-ERA5 (Apache-2.0) with "
                                          "`from weatherai.models.ace2 import download; download(dir)`")
        from ..models.ace2 import load_official
        return cls(load_official(ckpt_dir, device=device), p, [p], device)

    def required_forcings(self):
        """Names needed at every step (time-varying), including the next-step ocean fraction and SST."""
        return tuple(dict.fromkeys(self.forcing_names + self.next_names))

    def check(self, ic: Dict[str, np.ndarray], forcing: Dict[str, np.ndarray], n_steps: int, init=None, forcing_times=None):
        miss_ic = [k for k in self.prognostic if k not in ic]
        if miss_ic:
            raise MissingInputError(f"ACE2 initial condition lacks {miss_ic[:6]}{' ...' if len(miss_ic) > 6 else ''}")
        miss = [k for k in self.required_forcings() if k not in forcing]
        if miss:
            raise MissingForcingError(f"ACE2 needs forcing {miss} for steps 0..{n_steps}; none given")
        short = {}
        for k in self.required_forcings():
            a = np.asarray(forcing[k])
            if a.ndim == 2:                                            # static field: valid for all steps
                continue
            if a.shape[0] < n_steps + 1:
                short[k] = f"{a.shape[0]} of {n_steps + 1} times"
            elif not np.isfinite(a[: n_steps + 1]).all():
                short[k] = "non-finite values"
        if forcing_times is not None and init is not None:
            ft = np.asarray(forcing_times, dtype="datetime64[ns]")
            need = np.datetime64(init, "ns") + np.arange(n_steps + 1) * STEP.astype("timedelta64[ns]")
            gaps = np.setdiff1d(need, ft)
            if gaps.size:
                short["<time axis>"] = f"missing {gaps.size} times, first {str(gaps[0])[:16]}, last {str(gaps[-1])[:16]}"
        if short:
            raise MissingForcingError(f"ACE2 forcing does not cover the requested {n_steps} steps ({n_steps * 6} h): {short}. "
                                      "Supply forecast or scenario boundary conditions for the whole period, or shorten the run.")

    def run(self, ic: Dict[str, np.ndarray], forcing: Dict[str, np.ndarray], n_steps: int, forcing_provenance: str, init=None,
            forcing_times=None, outputs: Optional[Sequence[str]] = None, return_every: int = 1, out_path: Optional[str] = None,
            lat=None, lon=None, forcing_source: str = "") -> ForecastResult:
        import torch
        if forcing_provenance not in PROVENANCE:
            raise ValueError(f"forcing_provenance must be one of {sorted(PROVENANCE)}")
        self.check(ic, forcing, n_steps, init, forcing_times)
        t0 = time.time()
        H, W = np.asarray(ic[self.prognostic[0]]).shape[-2:]

        def tens(a):
            a = np.asarray(a, dtype=np.float32)
            if a.ndim == 1:                                             # scalar series (e.g. global_mean_co2)
                a = np.broadcast_to(a[:, None, None], (a.shape[0], H, W))
            elif a.ndim == 2:
                a = np.broadcast_to(a[None], (n_steps + 1, H, W))
            return torch.from_numpy(np.ascontiguousarray(a[: n_steps + 1]))[None].to(self.device)

        ic_t = {k: torch.from_numpy(np.asarray(ic[k], dtype=np.float32).reshape(1, H, W)).to(self.device) for k in self.prognostic}
        f_t = {k: tens(forcing[k]) for k in self.required_forcings()}
        outs = self.model.rollout(ic_t, f_t, n_steps, return_every=return_every)
        names = list(outputs or self.model.cfg.out_names)
        data = {k: np.stack([o[k][0].float().cpu().numpy() for o in outs])[None].astype(np.float32) for k in names}
        leads = np.arange(1, len(outs) + 1) * return_every
        coords = {"member": np.array([0]), "lead": leads * 6, "lat": gauss_legendre_lat(H) if lat is None else lat,
                  "lon": (np.arange(W) + 0.5) * 360.0 / W if lon is None else lon}
        if init is not None:
            it = np.datetime64(init, "ns")
            coords["init_time"] = it
            coords["valid_time"] = (("lead",), it + (leads * 6).astype("timedelta64[h]").astype("timedelta64[ns]"))
        attrs = dict(model="ace2", product=CARD.product, experiment_type=PROVENANCE[forcing_provenance],
                     forcing_provenance=forcing_provenance, forcing_source=forcing_source, lead_units="hours",
                     sst_is_prescribed="yes: surface_temperature over ocean is the supplied forcing, not a prediction",
                     usable_as_forecast="no" if forcing_provenance == "observed" else "only with the stated boundary assumption",
                     time_semantics=CARD.time_semantics)
        man = RunManifest("ace2", self.checkpoint, "torch-native",
                          config=dict(n_steps=n_steps, return_every=return_every, device=self.device,
                                      forcing_provenance=forcing_provenance, forcing_source=forcing_source,
                                      init=str(init) if init is not None else None), card=CARD.key)
        man.weights = [file_sha256(f) for f in self.files]
        man.inputs = {**{f"ic:{k}": array_sha256(ic[k]) for k in self.prognostic},
                      **{f"forcing:{k}": array_sha256(np.asarray(forcing[k])[: n_steps + 1]) for k in self.required_forcings()}}
        man.timings_s = {"total_s": time.time() - t0}
        man.finish()
        dims = {k: ("member", "lead", "lat", "lon") for k in data}
        res = ForecastResult("ace2", data, dims, coords, {}, {}, attrs, man).hash_outputs()
        if out_path:
            res.save(out_path)
        return res
