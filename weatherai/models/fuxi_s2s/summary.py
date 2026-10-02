"""Scalar fingerprint of one forecast step (used to compare CPU and GPU runs without storing model output)."""
from __future__ import annotations

import numpy as np
import torch

from .model import CHANNELS

PROBE = ["z500", "t850", "u850", "q700", "t2m", "sst", "msl", "tp"]


def fixed_noise(model, seed: int = 0):
    from .model import FuXiS2SNoise
    g = torch.Generator().manual_seed(seed)
    c = model.cfg
    return FuXiS2SNoise(torch.randn(1, c.latent, c.rank, generator=g), torch.randn(1, c.latent, c.n_tokens, generator=g))


def summarise(out: torch.Tensor):
    """out (1,2,C,H,W) -> {channel: [mean, std]} of the forecast frame (NaN-aware)."""
    y = out[0, 1].float().cpu().numpy()
    return {k: [float(np.nanmean(y[CHANNELS.index(k)])), float(np.nanstd(y[CHANNELS.index(k)]))] for k in PROBE}
