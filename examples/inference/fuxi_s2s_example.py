"""FuXi-S2S through the stable API: official ONNX, seeded noise, N members x up to 42 days, one file per member (resumable).

    python examples/inference/fuxi_s2s_example.py --model /ckpt/fuxi_s2s --sample /data/FuXi-S2S/data --members 11 --days 42 \
        --device cuda --out runs/fuxi_2020-06-02

Inputs: the official sample (ERA5 2020-06-01/02, GitHub tpys/FuXi-S2S data/sample + data/mask.nc). Weights: Zenodo 15718402
model-1.0.tar (CC-BY-NC-ND-4.0, download yourself). Peak memory / per-step time land in each member's manifest."""
import argparse
import os

from weatherai import inference as wi
from weatherai.inference.fuxi_s2s import FuXiS2SInputs

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--sample", required=True)
ap.add_argument("--members", type=int, default=11); ap.add_argument("--days", type=int, default=42)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="cpu")
ap.add_argument("--backend", default="onnx", choices=["onnx", "torch"]); ap.add_argument("--out", required=True)
ap.add_argument("--keep", default="", help="comma list of channels to store (default all 76)")
a = ap.parse_args()
kw = dict(path=a.model, device=a.device)
f = wi.load("fuxi-s2s", backend=a.backend, **kw)
inp = FuXiS2SInputs.from_official_files(os.path.join(a.sample, "sample"), os.path.join(a.sample, "mask.nc"), "official sample")
idx = f.run(inp, a.days, members=a.members, seed=a.seed, keep=a.keep.split(",") if a.keep else None, out_dir=a.out)
print("members:", idx.members, "files:", idx.files[:2], "...", "index:", idx.manifest)
