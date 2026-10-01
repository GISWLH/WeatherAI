"""Reference outputs of the OFFICIAL GenCast denoiser (JAX/Haiku) for the native PyTorch port.

Runs ``weathernext.weathernext1_gen.denoiser.Denoiser`` with the official GenCast 1p0deg Mini
parameters (``/workspace/ckpt/gencast/mini.npz`` <- gs://dm_graphcast/gencast/params/) on random inputs.
Needs python>=3.12 env (jax, haiku, chex, xarray, xarray_jax, trimesh, ...) and the weathernext clone.

Cases
  small : mesh_size=2, 13x24 grid, attention_k_hop=3, batch 2    -> tests/models/gencast/data/denoiser_ref_small.npz
  full  : mesh_size=4 (trained), 181x360 grid, k_hop=16, batch 1 -> $GENCAST_REF_DIR/denoiser_ref_full.npz (not committed)
Attention is the official plain ``mha`` (the TPU ``splash_mha`` kernel is not available on CPU); the mask is
identical. Run:  python scripts/gencast_denoiser_reference.py small|full
"""
import dataclasses
import os
import sys

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import xarray
import xarray_jax

sys.path.insert(0, os.environ.get("WEATHERNEXT_SRC", "/workspace/sources/weathernext"))
from weathernext.utils import checkpoint, model_utils  # noqa: E402
from weathernext.weathernext1_gen import denoiser, gencast  # noqa: E402

CKPT = os.environ.get("GENCAST_CKPT", "/workspace/ckpt/gencast/mini.npz")
CASES = {
    "small": dict(mesh_size=2, nlat=13, nlon=24, k_hop=3, batch=2, out="tests/models/gencast/data/denoiser_ref_small.npz"),
    "full": dict(mesh_size=4, nlat=181, nlon=360, k_hop=16, batch=1,
                 out=os.path.join(os.environ.get("GENCAST_REF_DIR", "/workspace/ckpt/gencast"), "denoiser_ref_full.npz")),
}


def main(case):
    c = CASES[case]
    with open(CKPT, "rb") as f:
        ckpt = checkpoint.load(f, gencast.CheckPoint)
    task = ckpt.task_config
    cfg = ckpt.denoiser_architecture_config
    cfg.mesh_size = c["mesh_size"]
    cfg.sparse_transformer_config.attention_k_hop = c["k_hop"]
    cfg.sparse_transformer_config.attention_type = "mha"
    cfg.node_output_size = 84
    levels = np.asarray(task.pressure_levels)
    lat = np.linspace(-90, 90, c["nlat"]).astype(np.float32)
    lon = np.linspace(0, 360, c["nlon"], endpoint=False).astype(np.float32)
    B, H, W, L = c["batch"], c["nlat"], c["nlon"], len(levels)
    rng = np.random.default_rng(0)
    r = lambda *s: rng.standard_normal(s).astype(np.float32)  # noqa: E731

    from weathernext.utils import variables as V
    atmos = set(V.ALL_ATMOSPHERIC_VARS)
    coords = dict(lat=lat, lon=lon, level=levels)

    def mk(name, time):
        if name in V.STATIC_VARS:
            return xarray_jax.Variable(("lat", "lon"), jnp.asarray(r(H, W)))
        if name in V.TIME_FORCING_VARS:
            return xarray_jax.Variable(("batch", "time"), jnp.asarray(r(B, time)))
        if name in atmos:
            return xarray_jax.Variable(("batch", "time", "level", "lat", "lon"), jnp.asarray(r(B, time, L, H, W)))
        return xarray_jax.Variable(("batch", "time", "lat", "lon"), jnp.asarray(r(B, time, H, W)))

    raw = {}

    def ds(names, time, tag):
        d = {}
        for n in names:
            v = mk(n, time)
            raw[f"{tag}/{n}"] = np.asarray(xarray_jax.unwrap_data(v))
            d[n] = v
        return xarray.Dataset(d, coords=coords)

    forcing_names = list(task.forcing_variables)
    inputs = ds(task.input_variables, 2, "inputs")
    forcings = ds(forcing_names, 1, "forcings")
    noisy = ds(task.target_variables, 1, "noisy_targets")
    sigma = np.asarray([0.7, 23.0][:B], dtype=np.float32)
    raw["noise_levels"] = sigma

    @hk.transform
    def fwd(inputs, noisy_targets, noise_levels, forcings):
        den = denoiser.Denoiser(ckpt.noise_encoder_config, cfg)
        return den(inputs, noisy_targets, xarray_jax.DataArray(noise_levels, dims=("batch",)), forcings)

    out = jax.jit(lambda i, n, s, f: fwd.apply(ckpt.params, None, i, n, s, f))(inputs, noisy, jnp.asarray(sigma), forcings)
    out_np = {}
    for n in task.target_variables:
        out_np[f"out/{n}"] = np.asarray(xarray_jax.unwrap_data(out[n]))
        assert out_np[f"out/{n}"].shape[:1] == (B,)
    meta = dict(
        mesh_size=c["mesh_size"], k_hop=c["k_hop"], lat=lat, lon=lon, levels=levels,
        input_variables=np.asarray(task.input_variables), target_variables=np.asarray(task.target_variables),
        forcing_variables=np.asarray(forcing_names),
    )
    os.makedirs(os.path.dirname(c["out"]) or ".", exist_ok=True)
    np.savez_compressed(c["out"], **raw, **out_np, **meta)
    print("wrote", c["out"], os.path.getsize(c["out"]) / 1e6, "MB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "small")
