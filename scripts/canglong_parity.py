"""CAS-Canglong: native model vs the official code (random weights; the figshare checkpoint is not reachable here).

The official model is defined in a notebook cell of GISWLH/CAS-Canglong (``notebook/model.ipynb``) and imports
``canglong.*`` helpers that the repository ships as ``utils/*.py``. This script copies ``utils`` into a temporary
``canglong`` package, executes the unchanged cell, strict-loads the official ``state_dict`` into the native
model and compares the forward pass and gradients.

    git clone https://github.com/GISWLH/CAS-Canglong /workspace/scratch/CAS-Canglong
    python scripts/canglong_parity.py                     # full size (1, 8, 16, 721, 1440), ~10 GB RAM
    python scripts/canglong_parity.py --lite --save-ref tests/models/canglong/data/ref_lite.npz
"""
import argparse
import json
import os
import resource
import shutil
import sys
import tempfile
import time
import types

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="/workspace/scratch/CAS-Canglong")
ap.add_argument("--out", default="")
ap.add_argument("--lite", action="store_true")
ap.add_argument("--save-ref", default="")
a = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def official_module(repo):
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


def official_lite(O, cfg):
    """Official classes and official ``Canglong.forward`` at the lite sizes (``Canglong.__init__`` hard-codes 721x1440)."""
    m = O.Canglong.__new__(O.Canglong)
    torch.nn.Module.__init__(m)
    C, (H, W) = cfg.embed_dim, cfg.img_size
    fine, coarse = cfg.token_grid, cfg.coarse_grid
    d1, d2, d3, d4 = cfg.depths
    dp = np.linspace(0, cfg.drop_path_rate, max(d1, d4) + max(d2, d3)).tolist()
    m.patchembed_wlh = O.ImageToPatch3D(img_dims=(cfg.n_months, H, W), patch_dims=cfg.patch_size, in_channels=cfg.in_chans, out_channels=C)
    m.layer1 = O.BasicLayer(dim=C, input_resolution=fine, depth=d1, num_heads=cfg.num_heads[0], window_size=cfg.window_size, drop_path=dp[:d1])
    m.downsample = O.DownSample(in_dim=C, input_resolution=fine, output_resolution=coarse)
    m.layer2 = O.BasicLayer(dim=2 * C, input_resolution=coarse, depth=d2, num_heads=cfg.num_heads[1], window_size=cfg.window_size, drop_path=dp[d1:d1 + d2])
    m.layer3 = O.BasicLayer(dim=2 * C, input_resolution=coarse, depth=d3, num_heads=cfg.num_heads[2], window_size=cfg.window_size, drop_path=dp[d1:d1 + d3])
    m.upsample = O.UpSample(2 * C, C, coarse, fine)
    m.layer4 = O.BasicLayer(dim=C, input_resolution=fine, depth=d4, num_heads=cfg.num_heads[3], window_size=cfg.window_size, drop_path=dp[:d4])
    m.patchrecovery2d = O.RecoveryImage2D((H, W), (8, 8), 2 * C, 4)          # computed and discarded by the official forward
    m.patchrecovery3d = O.RecoveryImage3D((13, H, W), (2, 8, 8), 2 * C, 5)
    m.patchrecovery3d_wlh = O.RecoveryImage3D((cfg.recovery_depth, H, W), cfg.recovery_patch, 2 * C, 1)
    return m


O = official_module(a.repo)
from weatherai.models.canglong import Canglong, CanglongConfig, canglong_loss  # noqa: E402

cfg = CanglongConfig.lite() if a.lite else CanglongConfig.official()
torch.manual_seed(0)
off = (official_lite(O, cfg) if a.lite else O.Canglong()).eval()
nat = Canglong(cfg).eval()
sd = off.state_dict()
if not cfg.legacy_heads:                                     # lite native model omits the discarded heads
    sd = {k: v for k, v in sd.items() if not k.startswith(("patchrecovery2d.", "patchrecovery3d."))}
res = {"config": "lite" if a.lite else "official", "params": sum(p.numel() for p in nat.parameters()),
       "params_official": sum(p.numel() for p in off.parameters()),
       "strict_load": str(nat.load_state_dict(sd, strict=True)),
       "torch": torch.__version__, "device": "cpu", "dtype": "float32"}
B = 2 if a.lite else 1
g = torch.Generator().manual_seed(1)
x = torch.randn(B, cfg.in_chans, cfg.n_months, *cfg.img_size, generator=g)
t0 = time.time()
with torch.no_grad():
    yo = off(x)
res["official_forward_s"] = round(time.time() - t0, 2)
t0 = time.time()
with torch.no_grad():
    yn = nat(x)
res["native_forward_s"] = round(time.time() - t0, 2)
res["out_shape"] = list(yo.shape)
res["max_abs"] = float((yn - yo).abs().max())
res["rel_l2"] = float((yn - yo).norm() / yo.norm())
res["out_std"] = float(yo.std())
if a.save_ref:
    np.savez_compressed(a.save_ref, x=x.numpy(), y=yo.numpy(), **{"sd/" + k: v.numpy() for k, v in sd.items()})
if a.lite:                                                   # gradients through the official training loss
    tgt = torch.randn(cfg.out_months, *cfg.img_size, generator=g)
    box = (slice(8, 20), slice(16, 40))
    off.train(); nat.train()
    for m in (off, nat):
        for mod in m.modules():                              # identical stochastic depth is not reproducible; switch it off
            if mod.__class__.__name__ == "DropPath":
                mod.drop_prob = 0.0
    lo = canglong_loss(off(x[:1])[0, 0], tgt, box=box)
    ln = canglong_loss(nat(x[:1])[0, 0], tgt, box=box)
    lo.backward(); ln.backward()
    res["loss"] = [float(lo), float(ln)]
    npn = dict(nat.named_parameters())
    rels = [float((npn[k].grad - p.grad).norm() / (p.grad.norm() + 1e-12)) for k, p in off.named_parameters() if p.grad is not None]
    res["grad_max_rel_l2"] = max(rels)
    res["grad_params_compared"] = len(rels)
res["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)   # both models + both forwards
print(json.dumps(res, indent=1))
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
