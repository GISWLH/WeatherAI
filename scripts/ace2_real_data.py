"""ACE2-ERA5 native port: 8-day rollout from the shipped 2020-06-01T00Z initial condition, compared with WeatherBench2 ERA5.
Usage: python scripts/ace2_real_data.py [out.json] [device]. Metrics are latitude-weighted RMSE on the 1.5 deg ERA5 grid."""
import json, sys, time
import numpy as np
import torch
import xarray as xr
from weatherai.models.ace2 import load_official
from weatherai.models.ace2.data import load_case

CK = "/workspace/ckpt/ace2"
out = sys.argv[1] if len(sys.argv) > 1 else "/workspace/WeatherAI/tests/models/ace2/data/real_case_report.json"
dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"
N = 32                                                     # 8 days of 6 h steps
G = 9.80665
m = load_official(CK, dev)
ic, forcing, times = load_case(CK, 5, N)                   # ic index 5 = 2020-06-01T00:00
assert str(times[0]).startswith("2020-06-01T00")
ic = {k: v.to(dev) for k, v in ic.items()}
forcing = {k: v.to(dev) for k, v in forcing.items()}
t0 = time.time()
outs = m.rollout(ic, forcing, N)
elapsed = time.time() - t0


rep = {"ic": "2020-06-01T00:00", "device": dev, "seconds_for_32_steps": elapsed}
from weatherai.models.ace2.data import compare_with_wb2
rep["lead_days"] = compare_with_wb2(outs, CK)
names = ("PRESsfc",) + tuple(f"specific_total_water_{i}" for i in range(8))
dry = [m.global_dry_air({k: g[k] for k in names}).item() for g in outs]
rep["global_dry_air_Pa"] = {"first": dry[0], "last": dry[-1], "max_abs_dev": max(abs(d - dry[0]) for d in dry)}
last = outs[-1]
rep["finite"] = all(bool(torch.isfinite(v).all()) for v in last.values())
rep["global_mean_t2m_K_last"] = float(m.area_mean(last["TMP2m"].double()).item())
rep["global_mean_precip_mm_day_last"] = float(m.area_mean(last["PRATEsfc"].double()).item() * 86400)
json.dump(rep, open(out, "w"), indent=1)
print(json.dumps(rep, indent=1))
