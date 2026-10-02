"""NeuralGCM precipitation / evaporation checkpoints: 24 h forecast from ERA5 2020-01-01T00Z vs ERA5 total precipitation (2.8 deg).
Usage: python scripts/neuralgcm_precip_eval.py [out.json]   (needs the Zenodo pickles in /workspace/ckpt/ngcm_precip and the ARCO window cache)"""
import json, sys
import numpy as np
from weatherai.models.neuralgcm import precip as P
from weatherai.models.neuralgcm.train import fetch_era5_window

out = sys.argv[1] if len(sys.argv) > 1 else "/workspace/WeatherAI/docs/results/neuralgcm_precip_eval.json"
D = "/workspace/ckpt/ngcm_precip"
STEPS, SEEDS = 24, 4
rep = {"start": "2020-01-01T00Z", "steps": STEPS, "seeds": SEEDS, "models": {}}
for name in ("precip", "evap"):
    ck = P.load_checkpoint(f"{D}/{P.FILES[name]}")
    m = P.build(ck)
    ds = fetch_era5_window(m, "2020-01-01T00", STEPS, 1, cache_dir="/workspace/ckpt/era5cache", extra_vars=["total_precipitation"])
    w = np.asarray(m.data_coords.horizontal.cos_lat)[None, :]
    w = w / w.mean()
    tgt = P.era5_precip_target(ds, STEPS)                        # (steps, lon, lat) mm/h
    tgt24 = tgt.sum(0)
    gm = lambda a: float((a * w).mean())
    rmse = lambda a, b: float(np.sqrt((w * (a - b) ** 2).mean()))
    def corr(a, b):
        a0, b0 = a - gm(a), b - gm(b)
        return float((w * a0 * b0).mean() / np.sqrt((w * a0 ** 2).mean() * (w * b0 ** 2).mean()))
    members = []
    for s in range(SEEDS):
        pred = P.rollout(m, ds, STEPS, 1, seed=s)
        rate = P.precip_rate_mm_per_hour(pred)                   # (steps, lon, lat)
        members.append(rate)
    members = np.stack(members)
    ens = members.mean(0).sum(0)
    r = {"era5_24h_global_mean_mm": gm(tgt24), "member_24h_global_mean_mm": [gm(x.sum(0)) for x in members],
         "member_rmse_24h_mm": [rmse(x.sum(0), tgt24) for x in members], "member_pattern_corr": [corr(x.sum(0), tgt24) for x in members],
         "ensemble_mean_rmse_24h_mm": rmse(ens, tgt24), "ensemble_mean_pattern_corr": corr(ens, tgt24),
         "baseline_rmse_constant_global_mean_mm": rmse(np.full_like(tgt24, gm(tgt24)), tgt24),
         "member_min_rate_mm_h": float(members.min()), "negative_fraction": float((members < -1e-6).mean()),
         "spread_across_members_mm": float(np.sqrt(((members.sum(1) - members.sum(1).mean(0)) ** 2 * w).mean()))}
    rep["models"][name] = r
    print(name, json.dumps(r, indent=1), flush=True)
json.dump(rep, open(out, "w"), indent=1)
