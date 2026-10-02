"""ORCA-DL real case: GODAS January-1980 initial state (official demo input), native model, 11-month forecast of tos vs GODAS 5 m potential
temperature (the top level of the public GODAS pottmp file, K -> degC, bilinear to the model grid), compared with persistence and the monthly
climatology that ships with the model (stat/mean). Writes docs/results/orca_dl_real_case.json. One initial month, `SEEDS` checkpoints."""
import json, os, sys
import numpy as np
import torch
import xarray as xr

from weatherai.models.orca_dl import load_official
from weatherai.models.orca_dl.data import denormalise, download_example, load_example, load_stats

CK = "/workspace/ckpt/orca_dl"
SEEDS = [int(s) for s in os.environ.get("SEEDS", "1").split(",")]
STEPS = 11
torch.set_num_threads(4)
ex = download_example("/workspace/scratch/orca_example")
o, a = load_example(ex, CK, month=0)
st = load_stats(CK)
lat = np.linspace(-63.5, 63.5, 128)
lon = np.linspace(0.5, 359.5, 360)
truth_ds = xr.open_dataset("/workspace/scratch/orca_example/pottmp.1980.nc").pottmp.isel(level=0)
truth = truth_ds.interp(lat=lat, lon=lon).values - 273.15            # (12, 128, 360), NaN over land
clim = np.stack([st["mean"]["sst"][m] for m in range(12)])[:, 0] if st["mean"]["sst"].ndim == 4 else np.stack([st["mean"]["sst"][m] for m in range(12)])
clim = clim.reshape(12, 128, 360)
w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, 360))

def wrmse(f, t):
    ok = np.isfinite(t) & np.isfinite(f)
    return float(np.sqrt((w[ok] * (f[ok] - t[ok]) ** 2).sum() / w[ok].sum()))

def nino34(x):
    la, lo = (lat >= -5) & (lat <= 5), (lon >= 190) & (lon <= 240)
    return float(np.nanmean(x[np.ix_(la, lo)]))

# sanity of the truth preprocessing: GODAS Jan top level vs the official demo 'sst' input
demo_sst = xr.open_dataset(f"{ex}/sst.nc").sst.values
res = {"init": "1980-01 (GODAS)", "seeds": SEEDS, "truth_vs_demo_sst_jan_wrmse_degC": wrmse(truth[0], demo_sst)}
preds = []
for s in SEEDS:
    m = load_official(CK, seed=s)
    with torch.no_grad():
        p = m(torch.from_numpy(o), torch.from_numpy(a), predict_time_steps=STEPS)[0, :, 32].numpy()     # tos channel (index 32 of 66: so 0-15, thetao 16-31, tos 32)
    preds.append(np.stack([denormalise(p[t], st, t + 1, "sst") for t in range(STEPS)]))
    del m
pred = np.stack(preds)
ens = pred.mean(0)
land = ~np.isfinite(truth[0])
rows = []
for t in range(STEPS):
    tm = t + 1
    pers = truth[0]
    n_clim = nino34(clim[tm]) 
    rows.append({"lead_months": tm,
                 "rmse_orca_dl": wrmse(ens[t], truth[tm]), "rmse_persistence": wrmse(pers, truth[tm]), "rmse_climatology": wrmse(clim[tm], truth[tm]),
                 "nino34_truth_anom": nino34(truth[tm]) - n_clim, "nino34_orca_dl_anom": nino34(ens[t]) - n_clim, "nino34_persistence_anom": nino34(pers) - n_clim})
res["leads"] = rows
res["note"] = "RMSE: latitude-weighted over GODAS ocean points (|lat|<=63.5); Nino3.4 anomaly = box mean minus the shipped monthly climatology"
print(json.dumps(res, indent=1))
json.dump(res, open("docs/results/orca_dl_real_case.json", "w"), indent=1)
