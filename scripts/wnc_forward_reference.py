"""Reference outputs of the OFFICIAL WeatherNext Cyclones network (JAX/Haiku) for the native PyTorch port.

Runs ``weathernext.weathernext2.architecture.ForwardPass`` (the neural network inside ``fgn.Predictor``; its
inputs are the already-normalised fields plus the 32-channel ``noise`` input) with the official
``WeatherNextCyclones_Mini_<2024`` parameters on random inputs.  Needs the python>=3.12 env (jax, haiku, chex,
xarray, xarray_jax, trimesh, ...) and a clone of google-deepmind/weathernext.

Cases
  small : mesh_num_splits=2, 13x24 grid, k_hop=3, batch 2    -> tests/models/weathernext_cyclones/data/forward_ref_small.npz
  full  : mesh_num_splits=5 (trained), 181x360 grid, k_hop=16, batch 1 -> $WNC_REF_DIR/forward_ref_full.npz (not committed)
Attention is the official plain ``mha`` (TPU ``splash_mha`` is unavailable on CPU); the mask is identical.
Also records the (static) edge index arrays the official GNN code builds, to compare the native graph construction.
Run:  python scripts/wnc_forward_reference.py small|full
"""
import os
import sys

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import xarray
import xarray_jax

sys.path.insert(0, os.environ.get("WEATHERNEXT_SRC", "/workspace/sources/weathernext"))
from weathernext.utils import checkpoint, deep_gnn, fiddle_config_io  # noqa: E402
from weathernext.weathernext2 import architecture, fgn  # noqa: E402

CKPT = os.environ.get("WNC_MINI_CKPT", "/workspace/ckpt/wn/WNC_Mini_2024.npz")
CONFIG = "weathernext2/configs/WeatherNextCyclones_Mini"
CASES = {
    "small": dict(splits=2, nlat=13, nlon=24, k_hop=3, batch=2, out="tests/models/weathernext_cyclones/data/forward_ref_small.npz"),
    "full": dict(splits=5, nlat=181, nlon=360, k_hop=16, batch=1,
                 out=os.path.join(os.environ.get("WNC_REF_DIR", "/workspace/ckpt/wn"), "forward_ref_full.npz")),
}
GLOBAL_VARS = ("year_progress_sin", "year_progress_cos")  # (batch, time)
LON_VARS = ("day_progress_sin", "day_progress_cos")  # (batch, time, lon)  (official sample layout)
STATIC_VARS = ("geopotential_at_surface", "land_sea_mask")  # (lat, lon)


def main(case):
    c = CASES[case]
    x64 = os.environ.get("WNC_X64") == "1"  # float64 reference (JAX_ENABLE_X64=1): separates float32 rounding from structural errors
    if x64:
        c = dict(c, out=c["out"].replace(".npz", "_f64.npz"))
    dt = np.float64 if x64 else np.float32
    cfg = fiddle_config_io.get_fiddle_config_by_name(CONFIG)
    with open(CKPT, "rb") as f:
        ckpt = checkpoint.load(f, fgn.CheckPoint)
    task = cfg.task
    kw = dict(cfg.predictor_kwargs["noisy_function_kwargs"])
    kw["mesh_num_splits"] = c["splits"]
    tk = kw["mesh_model_ctor"].keywords["transformer_kwargs"]
    tk["attention_type"], tk["attention_k_hop"] = "mha", c["k_hop"]

    if x64:  # the official attention softmax is up-cast to float32 whenever dtype != float32 (meant for bf16): disable for the f64 run
        import functools
        from weathernext.utils import sparse_transformer_utils as stu
        from weathernext.utils import sparse_transformer as st
        st.utils.wrap_fn_for_upcast_downcast = functools.partial(stu.wrap_fn_for_upcast_downcast, f32_upcast=False)
    if x64:  # official float32 edge aggregation would down-cast a float64 run; switch it off for the float64 reference
        kw["points_to_mesh_model_ctor"].keywords["deep_gnn_kwargs"]["f32_aggregation"] = False
    levels = np.asarray(task.pressure_levels)
    lat = np.linspace(-90, 90, c["nlat"]).astype(np.float32)  # float32 coordinates, like the official data
    lon = np.linspace(0, 360, c["nlon"], endpoint=False).astype(np.float32)
    B, H, W, L = c["batch"], c["nlat"], c["nlon"], len(levels)
    rng = np.random.default_rng(0)
    r = lambda *s: rng.standard_normal(s).astype(dt)  # noqa: E731
    t_in = np.array([-6, 0], dtype="timedelta64[h]").astype("timedelta64[ns]")
    t_out = np.array([6], dtype="timedelta64[h]").astype("timedelta64[ns]")
    atmos = {"temperature", "geopotential", "u_component_of_wind", "v_component_of_wind", "vertical_velocity", "specific_humidity"}
    raw = {}

    def mk(name, T, tag):
        if name in STATIC_VARS:
            a, dims = r(H, W), ("lat", "lon")
        elif name in GLOBAL_VARS:
            a, dims = r(B, T), ("batch", "time")
        elif name in LON_VARS:
            a, dims = r(B, T, W), ("batch", "time", "lon")
        elif name in atmos:
            a, dims = r(B, T, L, H, W), ("batch", "time", "level", "lat", "lon")
        else:
            a, dims = r(B, T, H, W), ("batch", "time", "lat", "lon")
        raw[f"{tag}/{name}"] = a
        return xarray_jax.Variable(dims, jnp.asarray(a))

    def ds(names, T, tag, times):
        return xarray.Dataset({n: mk(n, T, tag) for n in names}, coords=dict(lat=lat, lon=lon, level=levels, time=times))

    inputs = ds(task.input_variables, 2, "inputs", t_in)
    forcings = ds(task.forcing_variables, 1, "forcings", t_out)
    templ = {}
    for n in task.target_variables:
        dims = ("batch", "time", "level", "lat", "lon") if n in atmos else ("batch", "time", "lat", "lon")
        shape = (B, 1, L, H, W) if n in atmos else (B, 1, H, W)
        templ[n] = xarray_jax.Variable(dims, jnp.zeros(shape, dt))
    targets_template = xarray.Dataset(templ, coords=dict(lat=lat, lon=lon, level=levels, time=t_out))
    noise = r(B, 32)
    raw["noise"] = noise
    inputs["noise"] = xarray_jax.Variable(("batch", "noise_channels"), jnp.asarray(noise))

    edges = {}
    orig_call = deep_gnn.DeepGNN.__call__

    def spy(self, input_graph, *a, **k):
        for key, es in input_graph.edges.items():
            edges[key.name] = (np.asarray(es.indices.senders), np.asarray(es.indices.receivers))
        return orig_call(self, input_graph, *a, **k)

    deep_gnn.DeepGNN.__call__ = spy

    @hk.transform
    def fwd(inputs, targets_template, forcings):
        return architecture.ForwardPass(**kw)(inputs=inputs, targets_template=targets_template, forcings=forcings)

    if os.environ.get("WNC_DEBUG"):
        init = fwd.init(jax.random.PRNGKey(0), inputs, targets_template, forcings)
        flat = lambda p: {f"{m}/{n}" for m, d in p.items() for n in d}
        print("MISSING in ckpt:", sorted(flat(init) - flat(ckpt.params))[:20]); print("EXTRA in ckpt:", sorted(flat(ckpt.params) - flat(init))[:20]); return
    params = jax.tree.map(lambda a: np.asarray(a, dt), ckpt.params)
    # (official split-matmul layers draw throw-away initial values even in apply, so an rng key is required)
    run = (lambda fn: fn) if os.environ.get("WNC_NOJIT") else jax.jit
    out = run(lambda i, t, f: fwd.apply(params, jax.random.PRNGKey(0), i, t, f))(inputs, targets_template, forcings)
    out_np = {f"out/{n}": np.asarray(xarray_jax.unwrap_data(out[n])) for n in task.target_variables}
    for n in task.target_variables:
        assert out_np[f"out/{n}"].shape == templ[n].shape, (n, out_np[f"out/{n}"].shape)
    ed = {}
    for name, (s, rcv) in edges.items():
        ed[f"edges/{name}/senders"], ed[f"edges/{name}/receivers"] = s, rcv
    meta = dict(
        splits=c["splits"], k_hop=c["k_hop"], lat=lat, lon=lon, levels=levels,
        input_variables=np.asarray(task.input_variables), target_variables=np.asarray(task.target_variables),
        forcing_variables=np.asarray(task.forcing_variables),
    )
    os.makedirs(os.path.dirname(c["out"]) or ".", exist_ok=True)
    np.savez_compressed(c["out"], **raw, **out_np, **ed, **meta)
    print("wrote", c["out"], os.path.getsize(c["out"]) / 1e6, "MB", {k: v[0].shape for k, v in edges.items()})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "small")
