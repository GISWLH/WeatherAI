"""What WeatherPre receives from WeatherAI (no weights needed): model cards + a ForecastResult saved as netCDF + manifest.

    python examples/inference/weatherpre_handoff.py

WeatherPre (``weatherpre.adapters.weatherai_run``) calls ``weatherai.inference.load(...).run(...)`` and converts the
model-native result to its common schema; it never imports research scripts. This demo uses a stand-in network returning
zeros so it runs anywhere."""
import json
import os
import tempfile

import numpy as np
import torch

from weatherai import inference as wi
from weatherai.inference import orca_dl as O
from weatherai.models.orca_dl import ORCADLConfig

print(json.dumps({k: wi.card(k).to_dict()["status"]["real_data_inference"] for k in wi.CARDS}, indent=1)[:600], "...")
cfg = ORCADLConfig(lat_space=(-31.5, 31.5, 32), lon_space=(0.5, 359.5, 60), in_chans=[2, 2, 1, 2, 2, 1], out_chans=[2, 2, 1, 2, 2, 1],
                   embed_dim=24, lg_hidden_dim=192, enc_heads=(2, 4, 8), lg_heads=(8, 8), window_size=(4, 5), max_t=3)


class Zero(torch.nn.Module):
    def forward(self, o, a, predict_time_steps=1):
        return torch.zeros(o.shape[0], predict_time_steps, *o.shape[1:])


st = {k: {v: np.full((12, 1 if v in ("tos", "zos", "tauu", "tauv") else 2, 32, 60), 20.0 if (k == "mean" and v in ("so", "thetao", "tos")) else
                     (0.0 if k == "mean" else 1.0)) for v in O.OCEAN_VARS + O.ATMO_VARS} for k in ("mean", "std")}
f = O.ORCADLForecaster.from_modules({1: Zero()}, st, cfg, (10.0, 50.0))
fields = {v: np.full((1 if v in ("tos", "zos", "tauu", "tauv") else 2, 32, 60), 20.0 if v in ("so", "thetao", "tos") else 0.1) for v in
          O.OCEAN_VARS + O.ATMO_VARS}
r = f.run(O.ORCADLInputs({k: v.squeeze() if v.shape[0] == 1 else v for k, v in fields.items()}, "2026-12", "demo"), lead_months=6,
          out_path=os.path.join(tempfile.mkdtemp(), "demo.nc"))
print(r.to_xarray())
