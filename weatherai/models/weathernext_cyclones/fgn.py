"""Trainable ensemble wrapper around the native WN-C network: noise sampling + CRPS loss (official ``fgn.Predictor``).

``WNCEnsemble`` draws ``S`` independent 32-channel N(0,1) noise vectors per sample (official
``gaussian_noise_generator``) and runs :class:`WeatherNextCyclonesNet` once per noise draw; the training loss is
the official *fair* (unbiased) CRPS, ``mean|x_i − y| − Σ_{i≠j}|x_i − x_j| / (2 S (S−1))``, with per-variable weights
(``fgn_loss_function_kwargs.per_variable_weights``). NOT ported: official level weighting / NaN-target masking
(``vars_with_nan_targets``) beyond the simple mask below, normalisation wrappers, autoregressive rollout.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from ..gencast.denoiser import SparseTransformerConfig
from .network import WNCConfig, WeatherNextCyclonesNet


def fair_crps(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Per-element fair CRPS. ``pred (S,B,N,C)``, ``target (B,N,C)`` → ``(B,N,C)`` (NaN targets masked to 0 if ``mask``)."""
    S = pred.shape[0]
    err = (pred - target.unsqueeze(0)).abs().mean(0)
    if S > 1:
        diff = (pred.unsqueeze(0) - pred.unsqueeze(1)).abs().sum((0, 1)) / (2 * S * (S - 1))
        err = err - diff
    if mask is not None:
        err = err * mask
    return err


class WNCEnsemble(nn.Module):
    def __init__(self, net: WeatherNextCyclonesNet, channel_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.net = net
        w = torch.ones(net.cfg.out_channels) if channel_weights is None else channel_weights
        self.register_buffer("channel_weights", w, persistent=False)

    def sample(self, grid_x, mesh_x, num_samples: int = 2, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """``(S, B, Ng, C_out)`` ensemble of forecasts (one forward pass per noise draw)."""
        B = grid_x.shape[0]
        outs = []
        for _ in range(num_samples):
            noise = torch.randn(B, self.net.cfg.noise_channels, generator=generator, device=grid_x.device, dtype=grid_x.dtype)
            outs.append(self.net(grid_x, mesh_x, noise))
        return torch.stack(outs, 0)

    def loss(self, grid_x, mesh_x, target, num_samples: int = 2, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        pred = self.sample(grid_x, mesh_x, num_samples, generator)
        mask = torch.isfinite(target).to(pred.dtype)
        crps = fair_crps(pred, torch.nan_to_num(target), mask)
        return (crps * self.channel_weights).sum() / (mask * self.channel_weights).sum().clamp(min=1)


def layout_from_variables(surface: Sequence[str] = (), atmospheric: Sequence[str] = (), num_levels: int = 4) -> Tuple[Tuple[str, int], ...]:
    """Decoder layout in the official (alphabetical) order."""
    return tuple(sorted([(n, 1) for n in surface] + [(n, num_levels) for n in atmospheric]))


def WeatherNextCyclonesNative_lite(img_size: Tuple[int, int] = (16, 32), mesh_splits: int = 2, latent: int = 64,
                                   num_layers: int = 2, num_heads: int = 4, k_hop: int = 4, grid_in_channels: int = 12,
                                   mesh_in_channels: int = 2, target_layout: Optional[Sequence[Tuple[str, int]]] = None,
                                   attention_type: str = "dense") -> WNCEnsemble:
    """Small, config-driven, randomly initialised native WN-C (same code path as the official network; ~0.5 M params).

    Use :func:`WeatherNextCyclonesNet_from_official` for the released 56.7 M-parameter Mini weights.
    """
    from ..graphcast.mesh import default_grid_lat_lon

    layout = tuple(target_layout) if target_layout is not None else (("cyclone_exists_gaussian_unit_mode", 1), ("mean_sea_level_pressure", 1), ("temperature", 4))
    cfg = WNCConfig(
        transformer=SparseTransformerConfig(attention_k_hop=k_hop, d_model=latent, num_layers=num_layers, num_heads=num_heads,
                                            ffw_hidden=4 * latent, attention_type=attention_type),
        mesh_splits=mesh_splits, latent_size=latent, edge_latent_size=16, noise_channels=16,
        grid_in_channels=grid_in_channels, mesh_in_channels=mesh_in_channels, target_layout=layout,
    )
    lat, lon = default_grid_lat_lon(img_size)
    return WNCEnsemble(WeatherNextCyclonesNet(cfg, lat, lon))
