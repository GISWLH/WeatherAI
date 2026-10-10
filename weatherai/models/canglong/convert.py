"""Load official CAS-Canglong checkpoints into the native model.

The released checkpoint (figshare 10.6084/m9.figshare.26779969, e.g. ``canglong3_0005_600ep_base.tar``) was
written with ``torch.save(model)``: a pickled *module* whose classes live in the training notebook's
``__main__`` (and ``canglong.*`` / ``weatherlearn.*`` helpers). Those classes are not needed here: they are
unpickled as empty ``nn.Module`` stand-ins, only their parameters/buffers are read, and the tensors are
strict-loaded into ``Canglong``. Unpickling runs code from the file, so only load files you trust.
"""
from __future__ import annotations

import pickle
import types

import torch
from torch import nn

from .model import Canglong, CanglongConfig

_FOREIGN = ("__main__", "canglong", "weatherlearn", "utils")


class _Stub(nn.Module):
    """Stand-in for a class from the training notebook; holds whatever state the pickle restores."""

    def forward(self, *a, **k):  # pragma: no cover - never called
        raise RuntimeError("stub module from an official pickle; load its state_dict into Canglong instead")


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in _FOREIGN:
            return _Stub
        return super().find_class(module, name)


_pickle_module = types.SimpleNamespace(Unpickler=_Unpickler, load=pickle.load, __name__="canglong_stub_pickle")


def official_state_dict(obj) -> dict:
    """Plain ``state_dict`` from a checkpoint object (module, ``{'state_dict'|'model': ...}`` or a state dict)."""
    if isinstance(obj, nn.Module):
        obj = obj.state_dict()
    elif isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "model"):
            if k in obj and not torch.is_tensor(obj[k]):
                return official_state_dict(obj[k])
    sd = {}
    for k, v in obj.items():
        k = k[len("module."):] if k.startswith("module.") else k          # nn.DataParallel
        # WeatherLearn-named variant (Infer.ipynb `Pangu_lite`): PatchEmbed.proj / PatchRecovery.conv
        k = k.replace("patchembed_wlh.proj.", "patchembed_wlh.projection.").replace("patchembed2d.proj.", "patchembed2d.projection.")
        k = k.replace("patchembed3d.proj.", "patchembed3d.projection.")
        for r in ("patchrecovery2d", "patchrecovery3d", "patchrecovery3d_wlh"):
            k = k.replace(f"{r}.conv.", f"{r}.transposed_conv.")
        sd[k] = v
    return sd


def load_official(path: str, cfg: CanglongConfig | None = None, map_location="cpu") -> Canglong:
    """Strict-load an official checkpoint (trusted file) into ``Canglong(cfg)`` (default: official config)."""
    obj = torch.load(path, map_location=map_location, pickle_module=_pickle_module, weights_only=False)
    sd = official_state_dict(obj)
    cfg = cfg or CanglongConfig.official(legacy_heads=any(k.startswith("patchembed2d.") for k in sd))
    model = Canglong(cfg)
    model.load_state_dict(sd, strict=True)
    return model.eval()
