"""Fine-tune the OFFICIAL NeuralGCM 2.8 deg checkpoint for a few Adam steps on real ERA5 (1 h rollout loss).

Mechanics demonstration (gradients flow through the dycore, loss goes down on the training snapshot and is
re-evaluated on a held-out time) -- NOT a skill claim.  Needs network access to the public ARCO-ERA5 zarr (gcsfs).
Peak RSS ~7 GB on CPU, ~50 s/step on 8 cores.   Usage: python scripts/neuralgcm_finetune_era5.py [--steps 6]
"""
import argparse, json, os, pickle, time

import numpy as np

from weatherai.models.neuralgcm import train as T
from weatherai.models.neuralgcm.neuralgcm import download_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=os.environ.get("NGCM_CKPT"))
ap.add_argument("--train-time", default="2020-01-01T00")
ap.add_argument("--heldout-time", default="2020-07-01T12")
ap.add_argument("--steps", type=int, default=6)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--cache", default="/workspace/ckpt/era5cache")
ap.add_argument("--out", default=None)
a = ap.parse_args()

ck = pickle.load(open(a.ckpt or download_checkpoint("deterministic_2_8_deg"), "rb"))
model, rep = T.build_model(ck, params="official")


def window(t):
    ds = T.fetch_era5_window(model, t, steps=1, hours=1, cache_dir=a.cache)
    return model.inputs_from_xarray(ds.isel(time=0)), model.forcings_from_xarray(ds.isel(time=0)), T.stack_targets(model, ds, 1)


x, f, tg = window(a.train_time)
xh, fh, tgh = window(a.heldout_time)
res = {"n_params": rep["n_params"], "lr": a.lr, "steps": a.steps}
res["train_loss_before"] = T.evaluate(model, x, f, tg, 1)
res["heldout_loss_before"] = T.evaluate(model, xh, fh, tgh, 1)
t0 = time.time()
m2, h = T.fit(model, x, f, tg, steps=1, n_iters=a.steps, lr=a.lr)
res["loss_history"], res["grad_norm_history"], res["fit_seconds"] = h["loss"], h["grad_norm"], round(time.time() - t0, 1)
res["train_loss_after"] = T.evaluate(m2, x, f, tg, 1)
res["heldout_loss_after"] = T.evaluate(m2, xh, fh, tgh, 1)
print(json.dumps(res, indent=1))
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
