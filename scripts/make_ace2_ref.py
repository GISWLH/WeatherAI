"""CPU fp32 reference slices (1 and 4 steps from the 2020-06-01 IC) used by the GPU smoke test."""
import numpy as np
import torch
from weatherai.models.ace2 import load_official
from weatherai.models.ace2.data import load_case

CK = "/workspace/ckpt/ace2"
m = load_official(CK)
ic, forcing, _ = load_case(CK, 5, 4)
outs = m.rollout(ic, forcing, 4)
d = {}
for s, key in [(0, "step1"), (3, "step4")]:
    for n in ("TMP2m", "PRESsfc", "PRATEsfc", "air_temperature_3", "eastward_wind_5"):
        d[f"{key}_{n}"] = outs[s][n][0, ::5, ::7].numpy()
np.savez_compressed("/workspace/WeatherAI/tests/models/ace2/data/ref_cpu_fp32.npz", **d)
print({k: v.shape for k, v in d.items()})
