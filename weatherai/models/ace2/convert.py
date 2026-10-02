"""Load the official ACE2-ERA5 checkpoint (``ace2_era5_ckpt.tar``, a torch file written by ``fme``)."""
from __future__ import annotations

import os

import torch

from .model import ACE2, ACE2Config
from .sfno import SFNOConfig

REPO = "allenai/ACE2-ERA5"


def download(dest: str, with_data: bool = True, token: str | None = None):
    from huggingface_hub import snapshot_download

    pats = ["ace2_era5_ckpt.tar", "inference_config.yaml", "README.md"]
    if with_data:
        pats += ["initial_conditions/*", "forcing_data/*"]
    return snapshot_download(REPO, local_dir=dest, allow_patterns=pats, token=token)


def read_checkpoint(ckpt_dir: str):
    # trusted file from the official repo: nested dicts of tensors and plain python values
    return torch.load(os.path.join(ckpt_dir, "ace2_era5_ckpt.tar"), map_location="cpu", weights_only=False)


def load_official(ckpt_dir: str, device="cpu") -> ACE2:
    st = read_checkpoint(ckpt_dir)["stepper"]
    c = st["config"]
    b = c["builder"]["config"]
    assert b["operator_type"] == "dhconv" and b["filter_type"] == "linear" and b["normalization_layer"] == "instance_norm"
    assert not c["ocean"]["interpolate"] and c["ocean"]["slab"] is None
    in_names, out_names = tuple(c["in_names"]), tuple(c["out_names"])
    h, w = st["img_shape"]
    sf = SFNOConfig(in_chans=len(in_names), out_chans=len(out_names), img_shape=(h, w), embed_dim=b["embed_dim"],
                    num_layers=b["num_layers"], pos_embed=b["pos_embed"], big_skip=b.get("big_skip", True),
                    grid="legendre-gauss")
    cor = c["corrector"]
    cfg = ACE2Config(sfno=sf, in_names=in_names, out_names=out_names, force_positive=tuple(cor["force_positive_names"]),
                     conserve_dry_air=cor["conserve_dry_air"], moisture_budget=cor["moisture_budget_correction"],
                     timestep_seconds=st["encoded_timestep"] / 1e6, next_step_forcing=tuple(c["next_step_forcing_names"]),
                     sst_name=c["ocean"]["surface_temperature_name"], ocean_fraction_name=c["ocean"]["ocean_fraction_name"])
    nm = st["normalizer"]
    model = ACE2(cfg, nm["means"], nm["stds"], st["area"], st["sigma_coordinates"]["ak"], st["sigma_coordinates"]["bk"])
    sd = {k[len("module."):]: v for k, v in st["module"].items()}
    model.net.load_state_dict(sd, strict=True)
    return model.to(device).eval()
