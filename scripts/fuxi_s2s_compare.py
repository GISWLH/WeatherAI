"""Native FuXi-S2S (official weights) vs the saved onnxruntime reference (``fuxi_s2s_ort_ref.py``)."""
import argparse, json, sys, time
import numpy as np, torch
from weatherai.models.fuxi_s2s import FuXiS2SNoise
from weatherai.models.fuxi_s2s.convert import load_official

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/workspace/ckpt/fuxi_s2s")
ap.add_argument("--ref", default="/workspace/ckpt/fuxi_s2s/ref/step0.npz")
ap.add_argument("--out", default="docs/results/fuxi_s2s_parity.json")
a = ap.parse_args()
torch.set_num_threads(8)
t0 = time.time()
m = load_official(f"{a.root}/model-1.0/fuxi_s2s.onnx")
print("loaded", round(time.time() - t0, 1), flush=True)
r = np.load(a.ref)
x = torch.from_numpy(r["input"])
mask = None
noise = FuXiS2SNoise(torch.from_numpy(r["eps1"])[None], torch.from_numpy(r["eps2"])[None])
t0 = time.time()
with torch.no_grad():
    out, taps = m(x, torch.from_numpy(r["step"]), noise=noise, return_taps=True)
print("forward", round(time.time() - t0, 1), flush=True)
rel = lambda u, v: float(np.linalg.norm(u - v) / (np.linalg.norm(v) + 1e-30))
res = {}
tp = {k: v.float().numpy() for k, v in taps.items()}
for k in ["xn", "emb", "L0", "L1", "fpn", "mean", "cov", "diag", "z", "dec_in", "D0", "D1", "head"]:
    u, v = tp[k].reshape(r[k].shape) if tp[k].size == r[k].size else tp[k], r[k]
    res[k] = {"rel_l2": rel(u, v), "max_abs": float(np.abs(u - v).max()), "ref_std": float(v.std())}
    print(k, res[k], flush=True)
o = out.numpy()
ro = r["out"]
res["out_nan_mask_identical"] = bool(np.array_equal(np.isnan(o), np.isnan(ro)))   # sst is NaN over land in both
res["cov_weight_abs_max"] = float(m.dist_p.cov.weight.abs().max())
fin = np.isfinite(ro) & np.isfinite(o)
o = np.where(fin, o, 0.0); ro = np.where(fin, ro, 0.0)
r = dict(r); r["out"] = ro
res["out_rel_l2"] = rel(o, r["out"]); res["out_max_abs"] = float(np.abs(o - r["out"]).max())
res["out_rel_l2_per_channel_max"] = float(max(rel(o[0, 1, c], r["out"][0, 1, c]) for c in range(76)))
print({k: v for k, v in res.items() if k.startswith("out")})
json.dump(res, open(a.out, "w"), indent=1)

# ---- autoregressive chain (own outputs fed back) vs the onnxruntime chain
r = np.load(a.ref)
ks = sorted(int(k.split("_")[1]) for k in r.files if k.startswith("out_"))
cur = out.detach()
res["chain"] = {}


def fin_rel(u, v):
    f = np.isfinite(u) & np.isfinite(v)
    return rel(np.where(f, u, 0), np.where(f, v, 0))
for k in ks:
    if k > 0:
        nz = FuXiS2SNoise(torch.from_numpy(r[f"eps1_{k}"])[None], torch.from_numpy(r[f"eps2_{k}"])[None])
        st = torch.tensor([float(r["step"][0]) + k])
        with torch.no_grad():
            cur = m(cur, st, noise=nz)                                   # own state fed back
            one = m(torch.from_numpy(r[f"out_{k-1}"]), st, noise=nz)     # onnxruntime state as input
        res["chain"][f"step{k}_onestep_rel_l2"] = fin_rel(one.numpy(), r[f"out_{k}"])
    ro, oc = r[f"out_{k}"], cur.numpy()
    res["chain"][f"step{k}_rollout_rel_l2"] = fin_rel(oc, ro)
    res["chain"][f"step{k}_rollout_rel_l2_worst_channel"] = float(max(fin_rel(oc[0, 1, c], ro[0, 1, c]) for c in range(76)))
    print(k, res["chain"], flush=True)
json.dump(res, open(a.out, "w"), indent=1)
