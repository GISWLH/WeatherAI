"""Official ORCA-DL checkpoint (HF dataset ``JayKuo/ORCA-DL-data``, ``model_weights/seed_N.bin``, plain ``state_dict``) -> native model."""
from __future__ import annotations

import os

import torch

from .model import ORCADL, ORCADLConfig

HF_REPO = "JayKuo/ORCA-DL-data"
_ROTARY = ("pos_embed.sin1", "pos_embed.cos1", "pos_embed.sin2", "pos_embed.cos2")


def download(root: str, seeds=(1,), stats: bool = True) -> str:
    """Fetch seed_N.bin (2.16 GB each; seeds 1..8 are on the hub) and the monthly mean/std statistics (0.23 GB). Licence of the HF dataset
    is not stated; the files are never committed."""
    from huggingface_hub import snapshot_download
    pats = [f"model_weights/seed_{s}.bin" for s in seeds] + (["stat/*/*.npy"] if stats else [])
    snapshot_download(HF_REPO, repo_type="dataset", local_dir=root, allow_patterns=pats)
    return root


def convert_state_dict(sd: dict, model: ORCADL) -> dict:
    """Drops the deterministic rotary tables (after checking they equal the recomputed ones), keeps everything else by name."""
    own = model.state_dict()
    ref = dict(model.named_buffers())
    out = {}
    for k, v in sd.items():
        if k.endswith(_ROTARY):
            continue
        out[k] = v
    for k, v in sd.items():
        if k.endswith(_ROTARY):
            mod = model.get_submodule(k.rsplit(".", 1)[0])
            name = k.rsplit(".", 1)[1]
            assert torch.allclose(getattr(mod, name), v, atol=1e-6), f"rotary table mismatch at {k}"
    assert set(out) == set(own), (sorted(set(out) ^ set(own))[:5])
    return out


def load_official(root: str, seed: int = 1, cfg: ORCADLConfig | None = None, device="cpu") -> ORCADL:
    model = ORCADL(cfg or ORCADLConfig.official())
    sd = torch.load(os.path.join(root, "model_weights", f"seed_{seed}.bin"), map_location="cpu")
    model.load_state_dict(convert_state_dict(sd, model), strict=True)
    return model.to(device).eval()
