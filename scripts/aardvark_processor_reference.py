"""Make the numerical reference for tests/models/aardvark from the OFFICIAL aardvark code + checkpoint.

    AARDVARK_SRC=/path/to/aardvark-weather-public AARDVARK_PROC_CKPT=/path/to/forecast_1/epoch_0 \
        python scripts/aardvark_processor_reference.py

Runs official ``ConvCNPWeather(mode="forecast", decoder="vit")`` (cpu, eval) on seeded random input at
lead_time 0 and 1 and stores a [::6, ::6] crop of the outputs. (Needs a timm shim: the official ``vit.py``
passes ``Block(drop=...)`` which recent timm renamed to ``proj_drop``; the shim only renames that kwarg.)
"""
import os
import sys

import numpy as np
import torch

SRC = os.environ.get("AARDVARK_SRC", "/workspace/sources/aardvark-weather-public")
CK = os.environ["AARDVARK_PROC_CKPT"]
sys.path.insert(0, os.path.join(SRC, "aardvark"))
import timm.models.vision_transformer as tv  # noqa: E402

_B = tv.Block
tv.Block = type("Block", (_B,), {"__init__": lambda self, *a, drop=0.0, **k: _B.__init__(self, *a, proj_drop=drop, **k)})
torch.Tensor.cuda = lambda self, *a, **k: self  # official code hard-codes .cuda() for constant grids
import models as om  # noqa: E402

m = om.ConvCNPWeather(35, 24, 24, device="cpu", res=1, data_path=os.path.join(SRC, "data") + "/", decoder="vit", mode="forecast")
sd = torch.load(CK, map_location="cpu", weights_only=False)["model_state_dict"]
m.load_state_dict({k[7:]: v for k, v in sd.items()})
m.eval()
g = torch.Generator().manual_seed(0)
x = torch.randn(1, 35, 240, 121, generator=g)
out = {}
for lt in (0.0, 1.0):
    with torch.no_grad():
        y = m({"y_context": x.clone(), "lt": torch.full((1, 1), lt)}, None)
    out[f"y_lt{int(lt)}"] = y[:, ::6, ::6].numpy()
dst = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests/models/aardvark/data/processor_ref.npz")
np.savez_compressed(dst, **out)
print("wrote", dst, {k: v.shape for k, v in out.items()})
