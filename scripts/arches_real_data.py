"""Real-data sanity check of the native ArchesWeather(Gen) on WeatherBench2 ERA5 (1.5 degree), 2020-06-01 12Z -> +24 h / +48 h.

Needs the HF checkpoints (/workspace/ckpt/arches) and a small ERA5 netcdf sample written by scripts/fetch_wb2_sample.py.
Reports RMSE of the ensemble-mean / deterministic forecast against ERA5 truth vs persistence (and vs climatology-free skill)."""
import json
import sys

import torch
import xarray as xr

from weatherai.models.arches import load_official_gen, normalize, denormalize, wb2_to_state

ck = sys.argv[1] if len(sys.argv) > 1 else "/workspace/ckpt/arches"
ds = xr.open_dataset(f"{ck}/wb2_sample_2020.nc")
dev = "cuda" if torch.cuda.is_available() else "cpu"
gen, st = load_official_gen(ck, device=dev)
raw = {t: wb2_to_state(ds, t) for t in ds.time.values}
ts = list(ds.time.values)
x = lambda t: {k: v.to(dev) for k, v in normalize(raw[t], st).items()}
w = torch.cos(torch.linspace(torch.pi / 2, -torch.pi / 2, 121))[None, None, None, :, None].to(dev)
w = w / w.mean()


def _r(pred, truth, key, var, lev=None):
    a, b = (pred[key][:, var], truth[key][:, var]) if lev is None else (pred[key][:, var, lev], truth[key][:, var, lev])
    a, b = (a[:, 0], b[:, 0]) if lev is None else (a, b)
    return float(((a - b).pow(2) * w[:, 0]).mean().sqrt())


def report(pred, truth):  # physical units, latitude-weighted RMSE: z500 (m^2/s^2), t850 (K), t2m (K), u10 (m/s)
    return {"z500": _r(pred, truth, "level", 0, 7), "t850": _r(pred, truth, "level", 3, 10), "t2m": _r(pred, truth, "surface", 2),
            "u10": _r(pred, truth, "surface", 0)}


t_prev, t0, t1, t2 = ts
res = {"init": str(t0), "device": dev}
truth1 = {k: v.to(dev) for k, v in raw[t1].items()}
truth2 = {k: v.to(dev) for k, v in raw[t2].items()}
init = {k: v.to(dev) for k, v in raw[t0].items()}
month, hour = torch.tensor([6]), torch.tensor([12])
ts_sec = torch.tensor([int(t0.astype("datetime64[s]").astype("int64"))])
res["persistence_24h"] = report(init, truth1)
res["persistence_48h"] = report(init, truth2)
with torch.no_grad():
    s0, sp = x(t0), x(t_prev)
    det24 = gen.det_model(s0, sp, month, hour)
    res["det_ensemble_mean_24h"] = report(denormalize(det24, st), truth1)
    members = []
    for seed in range(3):
        members.append(gen.sample(s0, sp, month, hour, timestamp=ts_sec, seed=seed, num_steps=25))
    mean = {k: torch.stack([m[k] for m in members]).mean(0) for k in members[0]}
    res["gen_member0_24h"] = report(denormalize(members[0], st), truth1)
    res["gen_3member_mean_24h"] = report(denormalize(mean, st), truth1)
    spread = {k: torch.stack([m[k] for m in members]).std(0).mean().item() for k in members[0]}
    res["gen_member_spread_normalised_units"] = spread
    # 48 h rollout of member 0 (feed back sample, previous = the initial state)
    m48 = gen.sample(members[0], s0, month, hour, timestamp=ts_sec + 86400, seed=0, num_steps=25)
    res["gen_member0_48h"] = report(denormalize(m48, st), truth2)
print(json.dumps(res, indent=1))
