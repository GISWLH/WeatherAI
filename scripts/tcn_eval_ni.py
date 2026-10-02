"""TCN_M on the North-Indian-Ocean test split of TCND (20 storms 2017-2022, all 6-hourly windows with 8 observed steps -> 4 forecast steps = 24 h),
native model, official checkpoint. Reports track error (km) at 6/12/18/24 h as in the official script (best of 6 members per window, selected by the
summed 4-step track error) and also the mean over members, plus constant-velocity and persistence baselines, and pressure/wind MAE (best-of-6 by track).
NB: our dataset is the public TCND Zenodo release (15009527) NI split; the official script's quoted numbers are for all six basins."""
import json, os
import torch

from weatherai.models.tropicyclonenet import load_official, make_sample, sequences, relative_to_abs, to_physical, track_error_km

ROOT = "/workspace/ckpt/tcn/ni_root"
m = load_official("/workspace/ckpt/tcn/checkpoint_with_model_16000.pt")
files = sorted(os.listdir(f"{ROOT}/BST_data/NI/test"))
samples = [make_sample(ROOT, f"{ROOT}/BST_data/NI/test/{f}", s) for f in files for s in sequences(f"{ROOT}/BST_data/NI/test/{f}")]
N = len(samples)
cat = lambda k: torch.cat([s[k] for s in samples], 1)
obs, rel, gt = cat("obs_traj"), cat("obs_traj_rel"), cat("gt")
img = torch.cat([s["image_obs"] for s in samples], 0)
env = {k: torch.cat([s["env"][k] for s in samples], 0) for k in samples[0]["env"]}
seeds = [0, 1, 2]
res = {"windows": N, "storms": len(files), "seeds": seeds}
tde_best, tde_mean, p_mae, w_mae = [], [], [], []
gtp = to_physical(gt[..., :2], gt[..., 2:])
with torch.no_grad():
    for sd in seeds:
        torch.manual_seed(sd)
        out = []
        for i in range(0, N, 64):
            sl = slice(i, i + 64)
            o, _ = m.sample(rel[:, sl], img[sl], {k: v[sl] for k, v in env.items()}, num_samples=6)
            out.append(o)
        inc = torch.cat(out, 2)                                      # (4, 6, N, 4)
        ab = relative_to_abs(inc, obs[-1])
        pl, pm = to_physical(ab[..., :2], ab[..., 2:])
        err = track_error_km(pl, gtp[0][:, None].expand_as(pl))      # (4, 6, N)
        best = err.sum(0).argmin(0)
        ar = torch.arange(N)
        tde_best.append(err[:, best, ar].mean(1))
        tde_mean.append(err.mean((1, 2)))
        p_mae.append((pm[..., 0] - gtp[1][:, None, :, 0]).abs()[:, best, ar].mean(1))
        w_mae.append((pm[..., 1] - gtp[1][:, None, :, 1]).abs()[:, best, ar].mean(1))
    cv = relative_to_abs(obs.new_zeros(4, 1, N, 4) + rel[-1][None, None], obs[-1])
    cl, _ = to_physical(cv[..., :2], cv[..., 2:])
    cv_err = track_error_km(cl, gtp[0][:, None].expand_as(cl))[:, 0].mean(1)
    per = to_physical(obs[-1:, :, :2].expand(4, N, 2), obs[-1:, :, 2:].expand(4, N, 2))[0]
    pers_err = track_error_km(per, gtp[0]).mean(1)
fmt = lambda t: [round(float(x), 1) for x in torch.stack(t).mean(0)]
res["track_error_km_best_of_6"] = fmt(tde_best)
res["track_error_km_best_of_6_per_seed"] = [[round(float(x), 1) for x in t] for t in tde_best]
res["track_error_km_member_mean"] = fmt(tde_mean)
res["track_error_km_constant_velocity"] = [round(float(x), 1) for x in cv_err]
res["track_error_km_persistence"] = [round(float(x), 1) for x in pers_err]
res["pressure_mae_hPa_best_track_member"] = fmt(p_mae)
res["wind_mae_ms_best_track_member"] = fmt(w_mae)
res["official_script_quoted_all_basins_ckpt5100_km"] = [23.9360, 47.1055, 72.3829, 103.1691]
res["lead_hours"] = [6, 12, 18, 24]
print(json.dumps(res, indent=1))
json.dump(res, open("docs/results/tcn_eval_ni.json", "w"), indent=1)
