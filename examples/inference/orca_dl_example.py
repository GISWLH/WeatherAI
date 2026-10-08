"""ORCA-DL through the stable API: official GODAS example (Jan 1980) -> 11 monthly leads, seeds as members, manifest.

    python examples/inference/orca_dl_example.py --root /ckpt/orca_dl --example /data/orca_example --seeds 1 --out orca_1980-01.nc

Reproduction target (seed 1, CPU fp32; docs/results/orca_dl_real_case.json): tos vs GODAS 5 m, lat-weighted RMSE 0.43 degC at +1
month ... 0.96 degC at +11 months. The script prints the domain-mean tos per lead and writes <out> + <out>.manifest.json.
Downloads: weights + stats from HF JayKuo/ORCA-DL-data (weatherai.models.orca_dl.download), example inputs from GitHub
(weatherai.models.orca_dl.data.download_example)."""
import argparse

import numpy as np

from weatherai import inference as wi
from weatherai.inference.orca_dl import ORCADLInputs

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--example", required=True)
ap.add_argument("--seeds", default="1"); ap.add_argument("--leads", type=int, default=11)
ap.add_argument("--device", default="cpu"); ap.add_argument("--out", default="orca_dl_1980-01.nc")
a = ap.parse_args()
f = wi.load("orca-dl", root=a.root, seeds=[int(s) for s in a.seeds.split(",")], device=a.device)
inp = ORCADLInputs.from_official_example(a.example, "1980-01")
r = f.run(inp, lead_months=a.leads, out_path=a.out)
w = np.cos(np.deg2rad(r.coords["lat"]))[:, None]
for k in range(a.leads):
    x = r.data["tos"][:, k]
    print(f"lead {k + 1:2d}  valid {str(r.coords['valid_start'][1][k])[:7]}  domain-mean tos (members) "
          f"{[round(float(np.nansum(m * w) / np.nansum(np.isfinite(m) * w)), 3) for m in x]}")
print("wrote", a.out, "and", a.out + ".manifest.json")
