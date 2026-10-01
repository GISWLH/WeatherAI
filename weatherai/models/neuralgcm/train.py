"""JAX/Haiku workflow to CONFIGURE, TRAIN and FINE-TUNE NeuralGCM (official ``neuralgcm`` + ``dinosaur`` code).

NeuralGCM's dynamical core is JAX-only, so training happens in JAX (the torch-native learned components in
``network.py`` / ``spectral.py`` are for inspection, porting and tower-level experiments; they do not forecast).
This module adds the pieces the official *inference* package does not ship:

* ``make_config_str`` / ``build_model``: re-configure the learned parts through gin overrides (tower width,
  number of residual blocks, hidden layers per block, CNN features, positional-feature channels, dycore substeps, ...)
  and create a ``neuralgcm.PressureLevelModel`` whose parameters are either the official ones, freshly
  initialised (from scratch) or transferred shape-wise from a checkpoint (``transfer_params``).
* ``rollout_loss``: differentiable multi-step loss (encode -> ``unroll`` -> decode) against target snapshots
  (optionally with latitude weights and per-variable/level normalisation). ``steps=0`` is an encode/decode
  reconstruction loss.
* ``make_train_step`` / ``fit``: ``optax`` Adam + global-norm clipping, with optional freezing of parameter
  subtrees by regex (e.g. train only the physics tower).
* data helpers: ``demo_snapshot`` (bundled ERA5 snapshot), ``fetch_era5_window`` (public ARCO-ERA5 on GCS,
  conservatively regridded to the model grid), ``teacher_targets`` (rollout of a reference model).

Honest scope: this is a *workflow* on the official code, not a re-implementation. Training recipes here are
short-horizon examples that demonstrate gradients and loss decrease; they are not the paper's training setup
(multi-year ERA5, 1-4 day rollouts with MSE + spectral losses, multi-stage schedule, TPUs).
"""
from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

# gin keys of the official 2.8 deg config that control the learned parts (macros defined at the top of the config)
CONFIG_KEYS = {
    "LATENT_SIZE": "width of every main EPD tower (decoder, encoders, physics) -- default 384",
    "LAYER_SIZE": "hidden width inside the process/decode MLPs of the main towers -- default 384",
    "NUM_BLOCKS": "number of residual process blocks in the main EPD towers -- default 5",
    "N_CNN_FEATURES": "output channels of the vertical 1-D CNN embedding -- default 32",
    "POSITIONAL_LATENT_SIZE": "channels of the learned positional features -- default 8",
    "SURFACE_MODEL_LATENT_SIZE": "width of the land/sea/sea-ice surface towers -- default 8",
    "SURFACE_MODEL_LAYER_SIZE": "hidden width in the surface towers -- default 8",
    "SURFACE_MODEL_OUTPUT_SIZE": "surface embedding channels fed to the physics tower -- default 8",
    "N_INNER_DYCORE_STEPS": "dycore substeps per physics step -- default 5",
    "process/MlpUniform.num_hidden_layers": "layers inside each residual process block -- default 3 (4 linears)",
}


def _require_jax():
    try:
        import jax  # noqa: F401
        import haiku  # noqa: F401
        import neuralgcm  # noqa: F401
        import optax  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError("training needs `pip install jax dm-haiku optax neuralgcm dinosaur`") from e


# --------------------------------------------------------------------------- configuration / construction
def make_config_str(base: str, overrides: Optional[Mapping[str, Any]] = None, extra_bindings: Sequence[str] = ()) -> str:
    """Append gin bindings to ``base`` (later bindings win). ``overrides`` maps a CONFIG_KEYS name (or any gin
    ``scope/Configurable.param`` / ``MACRO``) to a value; ``extra_bindings`` are raw gin lines."""
    lines = [base.rstrip(), "", "# --- WeatherAI overrides"]
    for k, v in (overrides or {}).items():
        lines.append(f"{k} = {v!r}" if isinstance(v, str) else f"{k} = {v}")
    lines += list(extra_bindings)
    return "\n".join(lines) + "\n"


def build_model(
    checkpoint: Mapping[str, Any],
    overrides: Optional[Mapping[str, Any]] = None,
    extra_bindings: Sequence[str] = (),
    params: str = "official",
    seed: int = 0,
    sample: Optional[Tuple[dict, dict]] = None,
):
    """Create a ``PressureLevelModel`` from an official checkpoint dict (``pickle.load``) with modified config.

    params='official'  keep checkpoint params (requires unchanged shapes, i.e. no shape-changing overrides)
    params='init'      Haiku random init (from scratch) for the (possibly modified) architecture
    params='transfer'  random init, then copy every official array whose path+shape still matches
    ``sample`` = (inputs, forcings) as returned by ``demo_snapshot`` (needed for init; created automatically).
    Returns ``(model, report)`` where report lists transferred / re-initialised parameter counts.
    """
    _require_jax()
    import jax
    import neuralgcm
    from neuralgcm.legacy import api

    cfg = make_config_str(checkpoint["model_config_str"], overrides, extra_bindings)
    ck = dict(checkpoint)
    ck["model_config_str"] = cfg
    ck["params"] = {}
    skeleton = neuralgcm.PressureLevelModel.from_checkpoint(ck)
    report = {"mode": params, "config_overrides": dict(overrides or {}), "extra_bindings": list(extra_bindings)}
    if params == "official":
        new_params = checkpoint["params"]
    else:
        if sample is None:
            sample = demo_snapshot(skeleton)
        new_params = init_params(skeleton, sample, jax.random.key(seed))
        if params == "transfer":
            new_params, tr = transfer_params(checkpoint["params"], new_params)
            report.update(tr)
        elif params != "init":
            raise ValueError(params)
    model = api.PressureLevelModel(skeleton._structure, new_params, skeleton.gin_config)
    report["n_params"] = count_params(new_params)
    return model, report


def init_params(skeleton, sample: Tuple[dict, dict], rng) -> dict:
    """Haiku-initialise all parameters of ``skeleton``'s architecture (encode -> advance -> decode)."""
    import haiku as hk
    from neuralgcm.legacy import api, gin_utils

    inputs, forcings = sample
    st = skeleton._structure
    inp = api._prepend_dummy_time_axis(skeleton._to_abbreviated_names_and_tracers(inputs))
    f = api._prepend_dummy_time_axis(skeleton._squeeze_level_from_forcings(forcings))

    def fwd(inp, f, sim_time):
        model = st.model_cls()
        ff = model.forcing_fn(f, sim_time)
        s = model.encode(inp, ff)
        s = model.advance(s, ff)
        return model.decode(s, ff)

    with gin_utils.specific_config(skeleton.gin_config):
        return hk.transform(fwd).init(rng, inp, f, inputs["sim_time"])


def count_params(params) -> int:
    import jax

    return int(sum(int(np.size(x)) for x in jax.tree_util.tree_leaves(params)))


def transfer_params(src: Mapping, dst: Mapping) -> Tuple[dict, dict]:
    """Copy arrays of ``src`` into ``dst`` wherever module path, name and shape agree (others stay as initialised)."""
    import jax.numpy as jnp

    out, copied, kept = {}, 0, 0
    for mod, ps in dst.items():
        out[mod] = {}
        for name, v in ps.items():
            s = src.get(mod, {}).get(name)
            if s is not None and tuple(np.shape(s)) == tuple(np.shape(v)):
                out[mod][name] = jnp.asarray(s)
                copied += int(np.size(v))
            else:
                out[mod][name] = v
                kept += int(np.size(v))
    return out, {"transferred_weights": copied, "reinitialised_weights": kept}


# --------------------------------------------------------------------------- data helpers
def demo_snapshot(model, time_index: int = 0):
    """Bundled ERA5 snapshot (1959-01-02 00Z, TL31 regridded). Returns ``(inputs, forcings)`` model dicts."""
    import neuralgcm

    ds = neuralgcm.demo.load_data(model.data_coords)
    first = ds.isel(time=time_index)
    return model.inputs_from_xarray(first), model.forcings_from_xarray(first)


_ARCO_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
_ERA5_VARS = [
    "u_component_of_wind", "v_component_of_wind", "geopotential", "temperature", "specific_humidity",
    "specific_cloud_ice_water_content", "specific_cloud_liquid_water_content", "sea_surface_temperature", "sea_ice_cover",
]


def fetch_era5_window(model, start: str, steps: int, hours: int = 1, cache_dir: Optional[str] = None):
    """Public ARCO-ERA5 (0.25 deg, 37 levels, hourly) at ``start + k*hours``, k=0..steps, conservatively regridded to
    the model's data grid -> an ``xarray.Dataset`` like ``neuralgcm.demo.load_data`` (needs ``gcsfs zarr``, network).

    Land SST is NaN in ERA5; it is filled with the global mean of the valid (ocean) values and sea-ice NaN with 0
    (NeuralGCM only uses SST/ice over the ocean). Results are cached as .nc in ``cache_dir``.
    """
    import os

    import xarray as xr
    from dinosaur import horizontal_interpolation as hi
    from dinosaur import spherical_harmonic as sh

    cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "weatherai", "era5")
    os.makedirs(cache_dir, exist_ok=True)
    key = f"era5_{start}_{steps}x{hours}h_{model.data_coords.horizontal.nodal_shape[0]}x{model.data_coords.horizontal.nodal_shape[1]}.nc".replace(":", "")
    path = os.path.join(cache_dir, key)
    if os.path.exists(path):
        return xr.load_dataset(path)
    ds = xr.open_zarr(_ARCO_URL, chunks=None, storage_options={"token": "anon"})
    times = np.datetime64(start) + np.arange(steps + 1) * np.timedelta64(hours, "h")
    src = sh.Grid(longitude_wavenumbers=2, total_wavenumbers=3, longitude_nodes=1440, latitude_nodes=721, latitude_spacing="equiangular_with_poles")
    tgt = model.data_coords.horizontal
    reg = hi.ConservativeRegridder(src, tgt)
    levels = model.data_coords.vertical.centers
    data = {}
    for v in _ERA5_VARS:
        frames = []
        for t in times:
            a = ds[v].sel(time=t)
            if "level" in a.dims:
                a = a.sel(level=levels)
            a = np.asarray(a.values)[..., ::-1, :]  # latitude ascending (equiangular_with_poles source grid)
            a = np.swapaxes(a, -1, -2)  # (..., lon, lat)
            if v == "sea_surface_temperature":
                a = np.where(np.isnan(a), np.nanmean(a), a)
            if v == "sea_ice_cover":
                a = np.nan_to_num(a, nan=0.0)
            frames.append(np.asarray(reg(a), dtype=np.float32))
        arr = np.stack(frames)
        dims = ("time", "level", "longitude", "latitude") if arr.ndim == 4 else ("time", "longitude", "latitude")
        data[v] = (dims, arr)
    out = xr.Dataset(
        data,
        coords={"time": times.astype("datetime64[ns]"), "longitude": np.rad2deg(tgt.longitudes), "latitude": np.rad2deg(tgt.latitudes), "level": levels},
    )
    out.to_netcdf(path)
    return out


def teacher_targets(teacher, ds, steps: int, timedelta=np.timedelta64(1, "h")):
    """Roll out ``teacher`` (e.g. the official model) from ``ds`` time 0 and return a Dataset of targets at 1..steps."""
    import jax

    first = ds.isel(time=0)
    state = teacher.encode(teacher.inputs_from_xarray(first), teacher.forcings_from_xarray(first), jax.random.key(0))
    _, preds = teacher.unroll(state, teacher.forcings_from_xarray(ds.head(time=1)), steps=steps, timedelta=timedelta, start_with_input=False)
    return preds


def stack_targets(model, ds, n: int) -> Dict[str, Any]:
    """Model-format target dict ``{var: (n, level, lon, lat)}`` for dataset times 1..n."""
    import jax.numpy as jnp

    frames = [model.inputs_from_xarray(ds.isel(time=k)) for k in range(1, n + 1)]
    return {k: jnp.stack([f[k] for f in frames]) for k in frames[0] if k != "sim_time"}


# --------------------------------------------------------------------------- loss / training
def variable_scales(inputs: Mapping[str, Any]) -> Dict[str, Any]:
    """Per-variable standard deviation over (level, lon, lat) of the initial snapshot (loss normalisation).

    A single scale per variable is used on purpose: per-level scales blow up at the model top (1 hPa humidity has
    std ~1e-7), where even the official checkpoint has large encode/decode error (see README)."""
    import jax.numpy as jnp

    return {k: jnp.maximum(jnp.std(v), 1e-12) for k, v in inputs.items() if k != "sim_time"}


def level_weights(model, min_hpa: float = 50.0) -> Any:
    """0/1 mask over pressure levels (shape ``(level, 1, 1)``): levels with p >= ``min_hpa`` are trained on."""
    import jax.numpy as jnp

    c = np.asarray(model.data_coords.vertical.centers, dtype=np.float32)
    w = (c >= min_hpa).astype(np.float32)
    return jnp.asarray((w / w.mean())[:, None, None])


def rollout_loss(
    params,
    model_template,
    inputs: Mapping[str, Any],
    forcings: Mapping[str, Any],
    targets: Mapping[str, Any],
    scales: Mapping[str, Any],
    steps: int,
    timedelta=np.timedelta64(1, "h"),
    variables: Sequence[str] = ("temperature", "u_component_of_wind", "v_component_of_wind", "specific_humidity", "geopotential"),
    rng=None,
    lat_weights: Optional[Any] = None,
    level_w: Optional[Any] = None,
):
    """Mean squared error (per-variable/level normalised, optionally latitude weighted) between the model's decoded
    predictions after 1..``steps`` steps of ``timedelta`` and ``targets[var][k]``.

    steps=0 -> reconstruction: ``decode(encode(inputs))`` vs ``targets[var][0]`` (``targets`` must hold the input).
    Differentiable w.r.t. ``params`` (dycore, SHT and all learned towers).
    """
    import jax
    import jax.numpy as jnp
    from neuralgcm.legacy import api

    model = api.PressureLevelModel(model_template._structure, params, model_template.gin_config)
    rng = jax.random.key(0) if rng is None else rng
    state = model.encode(inputs, forcings, rng)
    if steps == 0:
        pred = {k: v[None] for k, v in model.decode(state, forcings).items() if hasattr(v, "shape")}
        n = 1
    else:
        # time-leading forcings with a single entry: persistence of the initial SST / sea ice
        fa = {k: jnp.asarray(v)[None] for k, v in forcings.items()}
        _, pred = model.unroll(state, fa, steps=steps, timedelta=timedelta, start_with_input=False)
        n = steps
    total, cnt = 0.0, 0
    for k in variables:
        if k not in pred or k not in targets:
            continue
        err = jnp.square((pred[k][:n] - targets[k][:n]) / scales[k])
        if lat_weights is not None:
            err = err * lat_weights
        if level_w is not None and err.ndim >= 4:
            err = err * level_w
        total = total + jnp.mean(err)
        cnt += 1
    return total / max(cnt, 1)


def latitude_weights(model) -> Any:
    """cos(lat) weights normalised to mean 1, shaped to broadcast against ``(..., lon, lat)``."""
    import jax.numpy as jnp

    w = np.asarray(model.data_coords.horizontal.cos_lat)
    return jnp.asarray(w / w.mean(), dtype=jnp.float32)


def _freeze_mask(params, freeze: Sequence[str]):
    pats = [re.compile(p) for p in freeze]
    return {mod: {n: (not any(p.search(mod) for p in pats)) for n in ps} for mod, ps in params.items()}


def make_train_step(model_template, optimizer, steps: int, timedelta=np.timedelta64(1, "h"), variables=None, freeze: Sequence[str] = (), lat_weights=None, level_w=None):
    """Returns a jitted ``step(params, opt_state, inputs, forcings, targets, scales) -> (params, opt_state, loss, grad_norm)``.

    ``freeze``: regexes over Haiku module paths whose parameters are NOT updated (gradients are zeroed)."""
    import jax
    import jax.numpy as jnp
    import optax

    kw = {} if variables is None else {"variables": tuple(variables)}

    def loss_fn(p, inputs, forcings, targets, scales):
        return rollout_loss(p, model_template, inputs, forcings, targets, scales, steps, timedelta, lat_weights=lat_weights, level_w=level_w, **kw)

    @jax.jit
    def step(params, opt_state, inputs, forcings, targets, scales):
        loss, grads = jax.value_and_grad(loss_fn)(params, inputs, forcings, targets, scales)
        if freeze:
            m = _freeze_mask(params, freeze)
            grads = jax.tree_util.tree_map(lambda g, keep: g * keep, grads, m)
        gn = optax.global_norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        if freeze:
            updates = jax.tree_util.tree_map(lambda u, keep: u * keep, updates, m)
        return optax.apply_updates(params, updates), opt_state, loss, gn

    return step


def fit(
    model,
    inputs,
    forcings,
    targets,
    steps: int,
    n_iters: int = 10,
    lr: float = 1e-3,
    clip: float = 1.0,
    timedelta=np.timedelta64(1, "h"),
    freeze: Sequence[str] = (),
    variables=None,
    lat_weighted: bool = True,
    min_hpa: float = 50.0,
    log: Optional[Callable[[int, float, float], None]] = None,
):
    """Minimal training loop on one (inputs, targets) window; returns ``(trained_model, history)`` with history of
    ``{"loss": [...], "grad_norm": [...], "seconds": ...}`` (loss[0] is evaluated before the first update)."""
    _require_jax()
    import optax
    from neuralgcm.legacy import api

    opt = optax.chain(optax.clip_by_global_norm(clip), optax.adam(lr))
    params = model.params
    opt_state = opt.init(params)
    sc = variable_scales(inputs)
    lw = latitude_weights(model) if lat_weighted else None
    step = make_train_step(model, opt, steps, timedelta, variables, freeze, lw, level_weights(model, min_hpa))
    hist = {"loss": [], "grad_norm": []}
    t0 = time.time()
    for i in range(n_iters):
        params, opt_state, loss, gn = step(params, opt_state, inputs, forcings, targets, sc)
        hist["loss"].append(float(loss))
        hist["grad_norm"].append(float(gn))
        if log:
            log(i, float(loss), float(gn))
    hist["seconds"] = time.time() - t0
    return api.PressureLevelModel(model._structure, params, model.gin_config), hist


def evaluate(model, inputs, forcings, targets, steps: int, timedelta=np.timedelta64(1, "h"), variables=None, lat_weighted: bool = True, min_hpa: float = 50.0) -> float:
    """Loss value of ``model`` on a window (same definition as the training loss)."""
    kw = {} if variables is None else {"variables": tuple(variables)}
    return float(rollout_loss(model.params, model, inputs, forcings, targets, variable_scales(inputs), steps, timedelta,
                              lat_weights=latitude_weights(model) if lat_weighted else None, level_w=level_weights(model, min_hpa), **kw))
