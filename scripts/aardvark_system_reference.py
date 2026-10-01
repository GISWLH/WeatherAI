"""Numerical reference for the Aardvark encoder / decoder / end-to-end system from the OFFICIAL code.

    AARDVARK_SRC=/path/to/aardvark-weather-public AARDVARK_CKPT=/path/to/trained_model \
        python scripts/aardvark_system_reference.py

Runs the official ``ConvCNPWeather`` (assimilation), ``ConvCNPWeather`` (forecast) and
``ConvCNPWeatherOnToOff`` + ``ConvCNPWeatherE2E`` on the official ``sample_data_final.pkl`` on CPU and
stores (sub-sampled) outputs in ``tests/models/aardvark/data/system_ref.npz``.
CPU shims (they do not change any numerics): ``Tensor.cuda`` -> identity, ``torch.load`` -> CPU
``map_location``, ``device="cuda"`` strings -> "cpu", and timm ``Block(drop=...)`` -> ``proj_drop``.
Official code also mutates the sample dict in place, so a fresh copy is loaded for every call.
"""
import copy
import io
import os
import pickle
import sys

import numpy as np
import torch
import torch.storage as ts

SRC = os.environ.get("AARDVARK_SRC", "/workspace/sources/aardvark-weather-public")
CK = os.environ.get("AARDVARK_CKPT", "/workspace/ckpt/aardvark/trained_model")
DEC_EPOCH = os.environ.get("AARDVARK_DEC", "decoder/tas/lt_1/epoch_18")
sys.path.insert(0, os.path.join(SRC, "aardvark"))

ts._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
_orig_load = torch.load
torch.load = lambda f, *a, **k: _orig_load(f, *a, **{**k, "map_location": "cpu", "weights_only": False})
torch.Tensor.cuda = lambda self, *a, **k: self
_mto = torch.nn.Module.to
torch.nn.Module.to = lambda self, *a, **k: _mto(self, *[("cpu" if a_ == "cuda" else a_) for a_ in a], **k)  # E2E loader does .to("cuda")
import timm.models.vision_transformer as tv  # noqa: E402

_B = tv.Block
tv.Block = type("Block", (_B,), {"__init__": lambda self, *a, drop=0.0, **k: _B.__init__(self, *a, proj_drop=drop, **k)})
import models as om  # noqa: E402
import misc_downscaling_functionality as md  # noqa: E402

sample0 = pickle.load(open(os.path.join(SRC, "data/sample_data_final.pkl"), "rb"))
fresh = lambda: copy.deepcopy(sample0)  # noqa: E731
DATA = os.path.join(SRC, "data") + "/"

def sd_of(path):
    sd = torch.load(path)["model_state_dict"]
    return {k[7:]: v for k, v in sd.items()}

out = {}
# --- encoder
enc = om.ConvCNPWeather(277, 24, 24, device="cpu", res=1, data_path=DATA, decoder="vit_assimilation", mode="assimilation")
enc.load_state_dict(sd_of(f"{CK}/encoder/epoch_96")); enc.eval()
with torch.no_grad():
    x0 = enc(fresh()["assimilation"], None)  # (1, 121, 240, 24)
out["encoder_state"] = x0.numpy()
# --- processor (lead 1) on the encoder output, via the official E2E plumbing below
# --- decoder alone on the sample's own downscaling context
dec = md.ConvCNPWeatherOnToOff(36, 1, 24, device="cpu", res=1, data_path=DATA, decoder="base", mode="downscaling", film=False)
dec.load_state_dict(sd_of(f"{CK}/{DEC_EPOCH}")); dec.eval()
with torch.no_grad():
    st = dec(fresh()["downscaling"], None)
out["decoder_station"] = st.numpy()
# --- full E2E (encoder -> 1 processor -> decoder), official class
from e2e_model import ConvCNPWeatherE2E  # noqa: E402

_Mdl = om.ConvCNPWeather
class _P(_Mdl):  # official E2E hard-codes device="cuda" at construction
    def __init__(self, *a, **k):
        k["device"] = "cpu"; k["data_path"] = DATA; super().__init__(*a, **k)
om.ConvCNPWeather = _P
import e2e_model as em  # noqa: E402
em.ConvCNPWeather = _P
class _D(md.ConvCNPWeatherOnToOff):
    def __init__(self, *a, **k):
        k["device"] = "cpu"; k["data_path"] = DATA; super().__init__(*a, **k)
em.ConvCNPWeatherOnToOff = _D
# official loader picks the best epoch from losses_0.npy; same files are used here
e2e = ConvCNPWeatherE2E("cpu", 1, f"{CK}/encoder", f"{CK}/processor", f"{CK}/decoder/tas/", return_gridded=True, aux_data_path=DATA)
e2e.eval()
with torch.no_grad():
    station, forecast, init = e2e(fresh())
out["e2e_station"] = station.numpy()
out["e2e_forecast"] = forecast.numpy()
out["e2e_init"] = init.numpy()
dst = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests/models/aardvark/data/system_ref.npz")
# keep the repo small: sub-sample the gridded fields (stride 4 in lat/lon); station outputs are kept in full
for k in ("encoder_state", "e2e_forecast", "e2e_init"):
    out[k] = out[k][:, ::4, ::4]
np.savez_compressed(dst, **out)
print("wrote", dst, {k: v.shape for k, v in out.items()})
