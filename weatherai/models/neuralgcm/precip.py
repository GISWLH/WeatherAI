"""NeuralGCM with precipitation / evaporation (Yuval et al., Sci. Adv. 12, eadv6891, 2026): load, forecast and TRAIN (JAX).

Two official 2.8 deg stochastic checkpoints (Zenodo 10.5281/zenodo.17109230, CC-BY-4.0; Apache-2.0 code):
  * ``stochastic_precip_2_8_deg.pkl`` - the network *predicts precipitation*; evaporation is diagnosed from the column
    water budget (P - E = -(vertically integrated moisture physics tendency), ``PrecipitationDiagnosticsConstrained``);
  * ``stochastic_evap_2_8_deg.pkl``   - the network *predicts evaporation*; precipitation is diagnosed.
Both expose the model diagnostics ``precipitation_cumulative_mean`` (accumulated precipitation in metres of water since the
encoded state) and ``evaporation`` (kg m-2 s-1, negative = evaporation, ERA5 sign convention) next to the usual state.

This module only adds what the inference package lacks: helpers to read precipitation rates out of a rollout, a
differentiable precipitation loss through the dycore and a small optax fit loop (JAX path; the dynamical core is JAX-only,
see ``train.py``). Nothing of the dycore or the learned towers is re-implemented here, so the forecast is the official model.
"""
from __future__ import annotations

import os
import pickle
import urllib.request
from typing import Any, Mapping, Optional, Sequence

import numpy as np

ZENODO = "https://zenodo.org/records/17109230/files/"
FILES = {"precip": "stochastic_precip_2_8_deg.pkl", "evap": "stochastic_evap_2_8_deg.pkl"}
PRECIP_KEY = "precipitation_cumulative_mean"   # metres of water, cumulative since the encoded state
EVAP_KEY = "evaporation"                       # kg m-2 s-1 (negative = evaporation)


def download(dest: str, which: Sequence[str] = ("precip", "evap")) -> dict:
    """Download the checkpoints (45 MB each, CC-BY-4.0) from Zenodo; returns {name: path}."""
    os.makedirs(dest, exist_ok=True)
    out = {}
    for k in which:
        p = os.path.join(dest, FILES[k])
        if not os.path.exists(p):
            urllib.request.urlretrieve(ZENODO + FILES[k] + "?download=1", p + ".part")
            os.replace(p + ".part", p)
        out[k] = p
    return out


def load_checkpoint(path: str) -> dict:
    """Official checkpoint dict (``params``, ``model_config_str``, ``aux_ds_dict``). Pickle: only load trusted files."""
    with open(path, "rb") as f:
        return pickle.load(f)


def build(ckpt: Mapping[str, Any]):
    """``neuralgcm.PressureLevelModel`` with the checkpoint's own parameters."""
    import neuralgcm

    return neuralgcm.PressureLevelModel.from_checkpoint(dict(ckpt))


def rollout(model, ds, steps: int, hours: int = 1, seed: int = 0, start_index: int = 0):
    """Official-model rollout from ``ds`` time ``start_index`` (dataset like ``train.fetch_era5_window``). Returns the raw prediction
    dict ``{name: (steps, ...)}`` for 1..steps (``start_with_input=False``); forcings (SST, sea ice) are persisted."""
    import jax

    first = ds.isel(time=start_index)
    inputs, forcings = model.inputs_from_xarray(first), model.forcings_from_xarray(first)
    state = model.encode(inputs, forcings, jax.random.key(seed))
    fa = model.forcings_from_xarray(ds.isel(time=slice(start_index, start_index + 1)))
    _, pred = model.unroll(state, fa, steps=steps, timedelta=np.timedelta64(hours, "h"), start_with_input=False)
    return pred


def precip_rate_mm_per_hour(pred_or_cum, hours: int = 1) -> np.ndarray:
    """Per-step precipitation rate (mm/h) from the cumulative diagnostic, shape ``(steps, lon, lat)``."""
    cum = pred_or_cum[PRECIP_KEY] if isinstance(pred_or_cum, Mapping) else pred_or_cum
    cum = np.asarray(cum)[:, 0] if np.ndim(cum) == 4 else np.asarray(cum)
    prev = np.concatenate([np.zeros_like(cum[:1]), cum[:-1]], 0)
    return (cum - prev) * 1000.0 / hours


def era5_precip_target(ds, steps: int, name: str = "total_precipitation") -> np.ndarray:
    """ERA5 hourly accumulation (m) at times 1..steps -> mm/h, ``(steps, lon, lat)`` (needs ``fetch_era5_window(..., extra_vars=[name])``)."""
    return np.asarray(ds[name].isel(time=slice(1, steps + 1)).values) * 1000.0


# --------------------------------------------------------------------------- training (JAX)
def _transform(rate_mm_h, kind: str):
    import jax.numpy as jnp

    if kind == "sqrt":      # variance-stabilising for the heavy-tailed rates
        return jnp.sqrt(jnp.maximum(rate_mm_h, 0.0) + 1e-3)
    if kind == "linear":
        return rate_mm_h
    raise ValueError(kind)


def precip_loss(params, model_template, inputs, forcings, target_mm_h, steps: int, hours: int = 1, rng=None,
                lat_weights=None, kind: str = "sqrt"):
    """Mean squared error between the model's hourly precipitation rate (mm/h, from the cumulative diagnostic) after 1..steps steps and
    ``target_mm_h`` of shape ``(steps, lon, lat)``; differentiable w.r.t. ``params`` through the dycore and all learned towers."""
    import jax
    import jax.numpy as jnp
    from neuralgcm.legacy import api

    model = api.PressureLevelModel(model_template._structure, params, model_template.gin_config)
    rng = jax.random.key(0) if rng is None else rng
    state = model.encode(inputs, forcings, rng)
    fa = {k: jnp.asarray(v)[None] for k, v in forcings.items()}
    _, pred = model.unroll(state, fa, steps=steps, timedelta=np.timedelta64(hours, "h"), start_with_input=False)
    cum = pred[PRECIP_KEY][:, 0]
    prev = jnp.concatenate([jnp.zeros_like(cum[:1]), cum[:-1]], 0)
    rate = (cum - prev) * 1000.0 / hours
    err = jnp.square(_transform(rate, kind) - _transform(jnp.asarray(target_mm_h), kind))
    if lat_weights is not None:
        err = err * lat_weights
    return jnp.mean(err)


def make_precip_train_step(model_template, optimizer, steps: int, hours: int = 1, freeze: Sequence[str] = (), lat_weights=None,
                           kind: str = "sqrt"):
    """jitted ``(params, opt_state, inputs, forcings, target, rng) -> (params, opt_state, loss, grad_norm)``; ``freeze`` = regexes over
    Haiku module paths whose parameters stay fixed (e.g. ``["decoder", "encoder"]`` to train only physics/precipitation towers)."""
    import jax
    import optax

    from .train import _freeze_mask

    def step(params, opt_state, inputs, forcings, target, rng):
        loss, grads = jax.value_and_grad(precip_loss)(params, model_template, inputs, forcings, target, steps, hours, rng, lat_weights, kind)
        if freeze:
            mask = _freeze_mask(params, freeze)
            grads = jax.tree.map(lambda g, m: g if m else g * 0.0, grads, mask)
        gn = optax.global_norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        if freeze:
            updates = jax.tree.map(lambda u, m: u if m else u * 0.0, updates, mask)
        return optax.apply_updates(params, updates), opt_state, loss, gn

    return jax.jit(step)


def fit_precip(model, inputs, forcings, target_mm_h, n_updates: int = 5, steps: int = 3, hours: int = 1, lr: float = 1e-5,
               clip: float = 1.0, freeze: Sequence[str] = (), seed: int = 0, kind: str = "sqrt", log=print):
    """Few-step precipitation fine-tuning on one window (mechanics demo, not the paper's multi-year recipe).
    Returns ``(new_params, history)`` with history = [(loss, grad_norm), ...]."""
    import jax
    import optax

    from .train import latitude_weights

    opt = optax.chain(optax.clip_by_global_norm(clip), optax.adam(lr))
    params = model.params
    opt_state = opt.init(params)
    lw = latitude_weights(model)
    step_fn = make_precip_train_step(model, opt, steps, hours, freeze, lw, kind)
    hist = []
    for i in range(n_updates):
        params, opt_state, loss, gn = step_fn(params, opt_state, inputs, forcings, target_mm_h[:steps], jax.random.key(seed + i))
        hist.append((float(loss), float(gn)))
        log(f"  update {i}: loss {hist[-1][0]:.5f} grad_norm {hist[-1][1]:.4f}")
    return params, hist
