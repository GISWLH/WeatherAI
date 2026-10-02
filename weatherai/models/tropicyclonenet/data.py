"""Data handling for TCN_M, re-implemented from the official loader without OpenCV: BST track files (``*.txt``), the per-step environment dictionaries
(``Env_data/<area>/<year>/<TC>/<date>.npy``) and 500 hPa geopotential patches (``ERA5_gph500/<area>/<year>/<TC>/<date>.npy``, 81x81, m2/s2).
Use ``scripts/tcn_prepare_ni.py`` to build that layout from the public TCND Zenodo record."""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from .model import ENV_KEYS

GPH_RANGE = (44490.578125, 58768.4486860389)


def read_bst(path):
    """Rows: frame, id, lon, lat, pressure, wind (normalised), date(YYYYMMDDHH), name."""
    rows = [l.strip().split("\t") for l in open(path)]
    num = np.array([[float(x) for x in r[:-2]] for r in rows], dtype=np.float64)
    return num, [r[-2] for r in rows], rows[0][-1]


def sequences(path, obs_len=8, pred_len=4):
    """All sliding windows of a storm: dict with obs/pred (n,4) arrays (lon, lat, pressure, wind), dates."""
    num, dates, name = read_bst(path)
    traj = np.around(num[:, 2:6], 4)
    out = []
    for i in range(0, len(traj) - obs_len - pred_len + 1):
        out.append({"obs": traj[i:i + obs_len], "pred": traj[i + obs_len:i + obs_len + pred_len], "dates": dates[i:i + obs_len + pred_len], "name": name})
    return out


def rel(obs):
    """Official relative increments: first step 0, then differences."""
    r = np.zeros_like(obs)
    r[1:] = obs[1:] - obs[:-1]
    return r


def _find(root, area, year, name):
    for n in (name, name.lower().capitalize(), name.upper()):
        if os.path.isdir(os.path.join(root, area, year, n)):
            return os.path.join(root, area, year, n)
    raise FileNotFoundError(os.path.join(root, area, year, name))


def load_gph(path):
    """81x81 geopotential -> 64x64 bilinear (== cv2.INTER_LINEAR), scaled to [0,1] with the official range, clipped."""
    g = torch.from_numpy(np.load(path).astype(np.float32))[None, None]
    g = F.interpolate(g, size=(64, 64), mode="bilinear", align_corners=False, antialias=False)[0, 0].numpy()
    lo, hi = GPH_RANGE
    return np.clip((g - lo) / (hi - lo), 0, 1)


def load_env(root, area, year, name, dates):
    d = _find(os.path.join(root, "Env_data"), area, year, name)
    items = [np.load(os.path.join(d, f"{dt}.npy"), allow_pickle=True).item() for dt in dates]
    env = {}
    for k in ENV_KEYS:
        vals = [it[k] for it in items]
        fill = next((v for v in vals if not (np.isscalar(v) and v == -1)), None)     # -1 marks missing: take the first valid value (official rule)
        vals = [fill if (np.isscalar(v) and v == -1) else v for v in vals]
        env[k] = np.asarray(vals, dtype=np.float32).reshape(len(dates), -1)
    return env


def make_sample(root, bst_file, seq, obs_len=8):
    """One forecast sample from a window of ``sequences`` -> tensors for TCNM (batch of 1)."""
    base = os.path.basename(bst_file)
    area, year, name = base[:2], base[2:6], base[:-4].split("BST", 1)[1]
    gdir = _find(os.path.join(root, "ERA5_gph500"), area, year, name)
    img = np.stack([load_gph(os.path.join(gdir, f"{d}.npy")) for d in seq["dates"][:obs_len]])        # (T,64,64)
    env = load_env(root, area, year, name, seq["dates"][:obs_len])
    obs = seq["obs"]
    return {
        "obs_traj": torch.tensor(obs, dtype=torch.float32)[:, None],                      # (T,1,4)
        "obs_traj_rel": torch.tensor(rel(obs), dtype=torch.float32)[:, None],
        "image_obs": torch.tensor(img, dtype=torch.float32)[None, None],                  # (1,1,T,64,64)
        "env": {k: torch.tensor(v)[None] for k, v in env.items()},                        # (1,T,d)
        "gt": torch.tensor(seq["pred"], dtype=torch.float32)[:, None],                    # (P,1,4)
        "start_date": seq["dates"][obs_len - 1],
    }
