"""Generate tests/models/neuralgcm/data/tower_ref.npz from the official JAX NeuralGCM (needs `neuralgcm`, `dinosaur`).

Runs one official ``encode -> advance -> decode`` step of the deterministic 2.8 deg model on the demo ERA5
snapshot with jit disabled and Haiku method interception, and stores for every learned column tower
(EpdTower / VerticalConvTower) its *real* packed input and output on a fixed random subset of columns
(towers act independently on each (lon, lat) column), plus the learned-orography modal outputs and masks.
"""
import argparse
import pickle

import haiku as hk
import jax
import neuralgcm
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/workspace/ckpt/ngcm_det_2_8.pkl")
ap.add_argument("--out", default="tests/models/neuralgcm/data/tower_ref.npz")
ap.add_argument("--ncols", type=int, default=24)
a = ap.parse_args()

ck = pickle.load(open(a.ckpt, "rb"))
m = neuralgcm.PressureLevelModel.from_checkpoint(ck)
ds = neuralgcm.demo.load_data(m.data_coords)
first = ds.isel(time=0)
rec = {}
orog = {}
surf = {}


def interceptor(next_fun, args, kwargs, ctx):
    mod = ctx.module
    out = next_fun(*args, **kwargs)
    if ctx.method_name == "__call__":
        n = type(mod).__name__
        if n in ("EpdTower", "VerticalConvTower"):
            rec[mod.module_name] = (n, np.asarray(args[0]), np.asarray(out))
        if n == "LearnedOrography":
            base = np.asarray(mod.base_orography_fn())
            orog[mod.module_name] = (np.asarray(out), np.asarray(mod.coords.horizontal.mask), base, float(mod.scale))
        if n == "NodalLandSeaIceEmbedding":
            leaves = jax.tree_util.tree_leaves(out)
            surf["out"] = np.concatenate([np.asarray(x) for x in leaves], axis=-3)
            surf["mask"] = np.asarray(mod.land_sea_mask)
            surf["ice"] = np.asarray(args[4]["sea_ice_cover"])
    return out


with jax.disable_jit(), hk.experimental.intercept_methods(interceptor):
    state = m.encode(m.inputs_from_xarray(first), m.forcings_from_xarray(first), jax.random.key(0))
    f = m.forcings_from_xarray(first)
    st2 = m.advance(state, f)
    m.decode(st2, f)

rng = np.random.RandomState(0)
ncol = a.ncols
H, W = 128, 64
idx = rng.choice(H * W, ncol, replace=False)
out = {}
for i, (name, (kind, x, y)) in enumerate(sorted(rec.items())):
    # x: (C, H, W) for EPD, (Cin, L, H, W) for vertical conv; columns are the trailing (H, W)
    xs = x.reshape(x.shape[:-2] + (H * W,))[..., idx]
    ys = y.reshape(y.shape[:-2] + (H * W,))[..., idx]
    out[f"t{i}_name"] = np.array(name)
    out[f"t{i}_kind"] = np.array(kind)
    out[f"t{i}_x"] = xs.astype(np.float32)
    out[f"t{i}_y"] = ys.astype(np.float32)
    print(i, kind, xs.shape, ys.shape, name.split("/~/")[-3:])
for i, (name, (val, mask, base, scale)) in enumerate(sorted(orog.items())):
    out[f"o{i}_name"] = np.array(name)
    out[f"o{i}_val"] = val.astype(np.float32)
    out[f"o{i}_base"] = base.astype(np.float32)
    out[f"o{i}_scale"] = np.float32(scale)
    out[f"o{i}_mask"] = mask
    print("orog", name.split("/~/")[-3:], val.shape, mask.shape, int(mask.sum()))
sf = lambda a: np.asarray(a).reshape(np.asarray(a).shape[:-2] + (H * W,))[..., idx].astype(np.float32)
out["surf_out"] = sf(surf["out"])
out["surf_mask"] = sf(np.broadcast_to(surf["mask"], (H, W)))
out["surf_ice"] = sf(np.broadcast_to(surf["ice"], (H, W)))
print("surface", out["surf_out"].shape)
out["n_towers"] = len(rec)
out["n_orog"] = len(orog)
out["columns"] = idx
np.savez_compressed(a.out, **out)
