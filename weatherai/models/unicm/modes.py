"""Climate-mode indices from the field grid and the UniCM training loss (official ``Trainer.py``)."""
from __future__ import annotations

import torch

from .model import UniCM, UniCMConfig, RESO

__all__ = ["wwv_box", "climate_modes", "load_official", "unicm_loss"]


def wwv_box(reso: int = RESO, lat_start: int = 45):
    L = lat_start // reso
    return (70 // reso - L, 80 // reso - L, 120 // reso, 280 // reso)


def climate_modes(x: torch.Tensor, cfg: UniCMConfig) -> torch.Tensor:
    """x (B,T,C,H,W) -> (B, n_modes + t20d, T): SST (channel 0) box means of the ten modes (dipole modes
    IOD, SIOD: box 1 minus box 2) followed by the warm-water-volume proxy (last channel, T20D, WWV box)."""
    sst, t20 = x[:, :, 0], x[:, :, -1]
    out = []
    for i, reg in enumerate(cfg.val_relative):
        if i in cfg.special_index:
            a, b = reg
            out.append(sst[:, :, a[0]:a[1], a[2]:a[3]].mean((2, 3)) - sst[:, :, b[0]:b[1], b[2]:b[3]].mean((2, 3)))
        else:
            out.append(sst[:, :, reg[0]:reg[1], reg[2]:reg[3]].mean((2, 3)))
    if cfg.t20d_mode:
        w = wwv_box()
        out.append(t20[:, :, w[0]:w[1], w[2]:w[3]].mean((2, 3)))
    return torch.stack(out, 1)


def load_official(path: str, cfg: UniCMConfig | None = None, device="cpu") -> UniCM:
    """Strict-load an officially trained ``model_save/model_best.pkl`` (plain state_dict).
    No pretrained UniCM weights are public; this is for checkpoints trained with ``app_train.py``."""
    m = UniCM(cfg or UniCMConfig.official())
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return m.to(device).eval()


def unicm_loss(model: UniCM, x: torch.Tensor, months: torch.Tensor, lambda1=1.0, lambda2=0.01, sv_ratio=0.0):
    """Teacher-forced training loss: lambda1 * SST MSE + lambda2 * mode-series MSE.  The official loss also
    has a lambda3 skill-score term (``climate_mode_sc``) which is not ported."""
    cfg = model.cfg
    modes = climate_modes(x, cfg)
    pred, pmode = model(x, modes, months, train=True, sv_ratio=sv_ratio)
    tgt = x[:, -cfg.pred_len:]
    l_field = ((pred[:, :, 0] - tgt[:, :, 0]) ** 2).mean()
    l_mode = ((pmode - modes[:, :, -cfg.pred_len:]) ** 2).mean()
    return lambda1 * l_field + lambda2 * l_mode, {"field": float(l_field.detach()), "mode": float(l_mode.detach())}
