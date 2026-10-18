"""CAS-Canglong: the released figshare checkpoint on released data, native model vs the official code.

Both files come from figshare 10.6084/m9.figshare.26779969 (``canglong3_0005_600ep_base.tar``,
``normalized_data.pt``, ``ocean_mask.npy``). The official model is the unchanged ``notebook/model.ipynb`` cell of
GISWLH/CAS-Canglong (as in ``canglong_parity.py``); the native model is ``load_official``. Both receive the same
16-month window of real normalised ERA5 fields and the outputs are compared; the 3 predicted SST months are also
scored against the following 3 months of the same file (one window: a sanity check, not a skill estimate).

    python scripts/canglong_official_checkpoint.py --repo CAS-Canglong --checkpoint canglong3_0005_600ep_base.tar \
        --data normalized_data.pt --mask ocean_mask.npy --device cuda:0 --out docs/results/canglong_official_checkpoint.json
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import types

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="/workspace/scratch/CAS-Canglong")
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--mask", required=True)
ap.add_argument("--window", type=int, default=-1, help="first input month; default: the last window with 3 target months")
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="")
a = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from weatherai.models.canglong import load_official  # noqa: E402
from weatherai.models.canglong.convert import _pickle_module, official_state_dict  # noqa: E402

SST_MEAN, SST_STD = 286.79, 11.627                          # channel-0 normalisation of notebook/Infer.ipynb (mean_all/std_all)


def official_module(repo):
    """The unchanged notebook cell with ``utils/`` as the ``canglong`` package (same as canglong_parity.py)."""
    tmp = tempfile.mkdtemp()
    shutil.copytree(os.path.join(repo, "utils"), os.path.join(tmp, "canglong"),
                    ignore=shutil.ignore_patterns("*.zip", "__pycache__", ".ipynb_checkpoints", "plot.py"))
    open(os.path.join(tmp, "canglong", "__init__.py"), "w").close()
    sys.path.insert(0, tmp)
    nb = json.load(open(os.path.join(repo, "notebook", "model.ipynb")))
    src = next("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "class Canglong" in "".join(c["source"]))
    mod = types.ModuleType("official_canglong")
    exec(compile(src, "notebook/model.ipynb", "exec"), mod.__dict__)
    return mod


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


dev = torch.device(a.device)
res = {"checkpoint_md5": md5(a.checkpoint), "data_md5": md5(a.data), "mask_md5": md5(a.mask),
       "torch": torch.__version__, "device": str(dev), "dtype": "float32"}
if dev.type == "cuda":
    res["gpu"] = torch.cuda.get_device_name(dev)
data = torch.load(a.data, mmap=True, weights_only=True)
w = a.window if a.window >= 0 else data.shape[0] - 19
x = data[w:w + 16].permute(1, 0, 2, 3).unsqueeze(0).float().contiguous()
target = data[w + 16:w + 19, 0].float().numpy()
res.update(data_shape=list(data.shape), window_first_index=w, input_shape=list(x.shape))
land = np.load(a.mask).astype(bool)                          # True on land (the notebook sets these cells to NaN)

nat = load_official(a.checkpoint).to(dev)
t0 = time.time()
with torch.no_grad():
    yn = nat(x.to(dev)).float().cpu()
res["native_forward_s"] = round(time.time() - t0, 2)
res["params"] = sum(p.numel() for p in nat.parameters())
del nat
torch.cuda.empty_cache() if dev.type == "cuda" else None

O = official_module(a.repo)
off = O.Canglong().eval()
sd = official_state_dict(torch.load(a.checkpoint, map_location="cpu", pickle_module=_pickle_module, weights_only=False))
res["official_strict_load"] = str(off.load_state_dict(sd, strict=True))
off = off.to(dev)
t0 = time.time()
with torch.no_grad():
    yo = off(x.to(dev)).float().cpu()
res["official_forward_s"] = round(time.time() - t0, 2)
res["out_shape"] = list(yo.shape)
res["max_abs"] = float((yn - yo).abs().max())
res["rel_l2"] = float((yn - yo).norm() / yo.norm())
res["finite"] = bool(torch.isfinite(yo).all())

pred = yo[0, 0].numpy() * SST_STD + SST_MEAN
obs = target * SST_STD + SST_MEAN
pers = x[0, 0, -1].numpy() * SST_STD + SST_MEAN
ocean = ~land
wts = np.cos(np.deg2rad(np.linspace(90, -90, 721)))[:, None] * ocean


def rmse(p, o):
    return float(np.sqrt(np.sum(wts * (p - o) ** 2) / np.sum(wts)))


res["sst_rmse_K_by_lead"] = [round(rmse(pred[k], obs[k]), 3) for k in range(3)]
res["persistence_rmse_K_by_lead"] = [round(rmse(pers, obs[k]), 3) for k in range(3)]
res["note"] = "one window of the released data; RMSE is area-weighted over ocean cells of ocean_mask.npy"
print(json.dumps(res, indent=1))
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
