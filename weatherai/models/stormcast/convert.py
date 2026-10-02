"""Load the official NVIDIA StormCast v1 (ERA5 -> HRRR) checkpoint into the native modules.

Weights: https://huggingface.co/nvidia/stormcast-v1-era5-hrrr (Apache-2.0, not gated). Never committed to this repo.
The two ``*.mdlus`` files are tar archives containing ``args.json``, ``metadata.json`` and ``model.pt``; ``model.pt`` is a plain
state dict. Strict loading only drops the two *empty* ``device_buffer`` tensors that PhysicsNeMo adds to every module.
"""
from __future__ import annotations

import io
import json
import os
import tarfile

import torch

from .model import StormCast, StormCastConfig

HF_REPO = "nvidia/stormcast-v1-era5-hrrr"
FILES = ["StormCastUNet.0.0.mdlus", "EDMPrecond.0.0.mdlus", "model.yaml", "metadata.zarr.zip", "config.json"]


def download(local_dir: str, token: str | None = None) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(HF_REPO, local_dir=local_dir, allow_patterns=FILES, token=token or os.environ.get("HF_TOKEN"))


def read_mdlus(path: str) -> tuple[dict, dict[str, torch.Tensor]]:
    """Return (args.json, state_dict) from a PhysicsNeMo ``.mdlus`` tar."""
    with tarfile.open(path) as tf:
        args = json.load(tf.extractfile("args.json"))
        sd = torch.load(io.BytesIO(tf.extractfile("model.pt").read()), map_location="cpu", weights_only=True)
    return args, sd


def _clean(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        if k.endswith("device_buffer"):
            assert v.numel() == 0, k
            continue
        out[k] = v
    return out


def official_config() -> StormCastConfig:
    return StormCastConfig()


def load_metadata(path: str) -> dict:
    """Read means/stds/invariants from ``metadata.zarr.zip`` (needs ``zarr`` + ``xarray``)."""
    import xarray as xr
    import zarr
    store = zarr.storage.ZipStore(path, mode="r")
    try:
        ds = xr.open_zarr(store, zarr_format=2)  # zarr>=3
    except TypeError:                            # zarr 2.x (python 3.10)
        ds = xr.open_zarr(store)
    return {k: ds[k].values for k in ["variable", "conditioning_variable", "means", "stds", "conditioning_means",
                                      "conditioning_stds"]} | {"invariants": ds["invariants"].sel(invariant=["lsm", "orography"]).values}


def load_official(ckpt_dir: str, device: str = "cpu", with_metadata: bool = True) -> StormCast:
    m = StormCast(official_config())
    _, reg = read_mdlus(os.path.join(ckpt_dir, "StormCastUNet.0.0.mdlus"))
    _, dif = read_mdlus(os.path.join(ckpt_dir, "EDMPrecond.0.0.mdlus"))
    m.regression.load_state_dict(_clean(reg), strict=True)
    m.diffusion.load_state_dict(_clean(dif), strict=True)
    if with_metadata:
        md = load_metadata(os.path.join(ckpt_dir, "metadata.zarr.zip"))
        t = lambda a: torch.as_tensor(a, dtype=torch.float32)
        m.means.copy_(t(md["means"])[None, :, None, None]); m.stds.copy_(t(md["stds"])[None, :, None, None])
        m.cond_means.copy_(t(md["conditioning_means"])[None, :, None, None]); m.cond_stds.copy_(t(md["conditioning_stds"])[None, :, None, None])
        m.invariants.copy_(t(md["invariants"]).reshape(m.invariants.shape))
    return m.to(device).eval()
