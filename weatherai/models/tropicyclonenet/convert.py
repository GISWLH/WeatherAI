"""Official TCN_M checkpoint (Zenodo 10.5281/zenodo.15024028, CC-BY-4.0, ``checkpoint_with_model_16000.pt``) -> native :class:`TCNM`."""
from __future__ import annotations

import os

import torch

from .model import TCNM, TCNMConfig

ZENODO = "https://zenodo.org/records/15024028/files/checkpoint_with_model_16000.pt"


def download(root: str) -> str:
    import urllib.request
    os.makedirs(root, exist_ok=True)
    p = os.path.join(root, "checkpoint_with_model_16000.pt")
    if not os.path.exists(p):
        urllib.request.urlretrieve(ZENODO, p)
    return p


def load_official(path: str, which: str = "g_state", device="cpu") -> TCNM:
    """``which``: ``g_state`` (what the official evaluation script loads), ``g_best_state`` or ``g_best_nl_state``. The pickle also holds
    optimiser states and the discriminator; it is a trusted file from the authors' Zenodo record (``weights_only=False``)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    cfg = TCNMConfig(obs_len=a["obs_len"], pred_len=a["pred_len"], embedding_dim=a["embedding_dim"], encoder_h_dim=a["encoder_h_dim_g"],
                     decoder_h_dim=a["decoder_h_dim_g"], mlp_dim=a["mlp_dim"], noise_dim=a["noise_dim"][0])
    m = TCNM(cfg)
    m.load_state_dict(ck[which], strict=True)
    return m.to(device).eval()
