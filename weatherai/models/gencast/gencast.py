"""GenCast: EDM diffusion wrapper + DPM-Solver++ 2S sampler around a **native PyTorch port of the official denoiser**.

GenCast (Price et al., *Nature* 637, 2025) is a conditional diffusion model (Karras et al. "EDM")
whose denoiser is a GraphCast-style grid→mesh→grid network with a 16-layer k-hop mesh transformer,
conditioned on the noise level. Official code (JAX/Haiku): ``google-deepmind/weathernext``
(``weathernext1_gen``).

* :mod:`weatherai.models.gencast.denoiser` — explicit ``nn.Module`` port of the official denoiser
  (noise Fourier-MLP, grid2mesh GNN, mesh transformer, mesh2grid GNN). Parameter names mirror the Haiku
  names; ``GenCastDenoiser_from_official`` loads the official ``.npz`` strictly. Numerically checked
  against the official JAX denoiser *with the official GenCast-1p0deg-Mini weights* (see
  ``tests/models/gencast`` and ``docs/model_status.md`` for the exact scope).
* This module — EDM preconditioning ``D = c_skip·x + c_out·F(c_in·x, σ)``, loss weighting, noise-level
  schedule and the DPM-Solver++ 2S sampler with stochastic churn (checked against the official JAX
  sampler with a toy denoiser), plus ``DenoiserNet``/``GenCast_lite``: a config-driven small instance of
  the *same* native architecture for training experiments (random init).

Not covered: sampling noise is i.i.d. Gaussian on the grid, not the official spherical-harmonic
isotropic white noise (needs the dinosaur SHT); no ERA5 normalisation/NaN-cleaning/forcing pipeline
(``InputsAndResiduals``); no end-to-end 12-step ensemble comparison against the official model.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- schedules (official math)


def rho_inverse_cdf(min_value: float, max_value: float, rho: float, cdf):
    """Quantiles of the rho distribution (Karras et al. Eqn 5, ascending). numpy/torch/float."""
    return (min_value ** (1 / rho) + cdf * (max_value ** (1 / rho) - min_value ** (1 / rho))) ** rho


def noise_schedule(
    max_noise_level: float = 80.0,
    min_noise_level: float = 0.03,
    num_noise_levels: int = 20,
    rho: float = 7.0,
) -> np.ndarray:
    """Descending noise levels with a trailing 0 (length ``num_noise_levels + 1``)."""
    levels = rho_inverse_cdf(min_noise_level, max_noise_level, rho, np.linspace(1, 0, num_noise_levels))
    return np.append(levels, 0.0)


def stochastic_churn_rate_schedule(
    noise_levels: np.ndarray,
    stochastic_churn_rate: float = 0.0,
    churn_min_noise_level: float = 0.05,
    churn_max_noise_level: float = 50.0,
) -> np.ndarray:
    n = len(noise_levels) - 1
    per_step = min(stochastic_churn_rate / n, math.sqrt(2) - 1)
    return ((churn_min_noise_level <= noise_levels[:-1]) & (noise_levels[:-1] <= churn_max_noise_level)) * per_step


@dataclasses.dataclass(frozen=True)
class SamplerConfig:
    """Defaults = official ``SamplerConfig`` (GenCast paper)."""

    max_noise_level: float = 80.0
    min_noise_level: float = 0.03
    num_noise_levels: int = 20
    rho: float = 7.0
    stochastic_churn_rate: float = 2.5
    churn_min_noise_level: float = 0.75
    churn_max_noise_level: float = float("inf")
    noise_level_inflation_factor: float = 1.05


@dataclasses.dataclass(frozen=True)
class NoiseConfig:
    """Training noise distribution; defaults = official ``NoiseConfig``."""

    training_noise_level_rho: float = 7.0
    training_max_noise_level: float = 88.0
    training_min_noise_level: float = 0.02


NoiseFn = Callable[[torch.Tensor], torch.Tensor]
Denoise = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # (x, sigma[B]) -> x0_hat


def _gauss(template: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    return torch.randn(template.shape, dtype=template.dtype, device=template.device, generator=generator)


@torch.no_grad()
def dpm_solver_pp_2s_sample(
    denoise: Denoise,
    template: torch.Tensor,
    cfg: SamplerConfig = SamplerConfig(),
    noise_fn: Optional[NoiseFn] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """DPM-Solver++ 2S with stochastic churn (port of official ``dpm_solver_plus_plus_2s.Sampler``).

    ``denoise(x, sigma)`` must return the *denoised* estimate D(x, sigma) for ``sigma`` of shape (B,).
    ``noise_fn(template)`` draws unit white noise (default: iid Gaussian).
    """
    noise_fn = noise_fn or (lambda t: _gauss(t, generator))
    dtype, dev = template.dtype, template.device
    levels = noise_schedule(cfg.max_noise_level, cfg.min_noise_level, cfg.num_noise_levels, cfg.rho)
    churn = stochastic_churn_rate_schedule(
        levels, cfg.stochastic_churn_rate, cfg.churn_min_noise_level, cfg.churn_max_noise_level
    )
    stochastic = cfg.stochastic_churn_rate > 0
    levels_t = torch.as_tensor(levels, dtype=dtype, device=dev)
    churn_t = torch.as_tensor(churn, dtype=dtype, device=dev)
    B = template.shape[0]

    def D(x, s):
        return denoise(x, s.expand(B))

    x = torch.zeros_like(template)
    for i in range(len(levels) - 1):
        if i == 0:
            x = x + levels_t[0] * noise_fn(x)
        sigma = levels_t[i]
        if stochastic:
            new_sigma = sigma * (1.0 + churn_t[i])
            diff = torch.clamp(new_sigma**2 - sigma**2, min=0)
            x = x + noise_fn(x) * torch.sqrt(diff) * cfg.noise_level_inflation_factor
            sigma = new_sigma
        nxt = levels_t[i + 1]
        mid = torch.sqrt(sigma * nxt)
        x_den = D(x, sigma)
        if float(levels[i + 1]) == 0.0:  # final step: official code returns the denoised estimate
            x = x_den
            continue
        r_mid = mid / sigma
        x_mid = r_mid * x + (1 - r_mid) * x_den
        x_mid_den = D(x_mid, mid)
        r_next = nxt / sigma
        x = r_next * x + (1 - r_next) * x_mid_den
    return x


# --------------------------------------------------------------------------- denoiser network


class DenoiserNet(nn.Module):
    """``(B,C,H,W)`` adapter around the native official-architecture :class:`GenCastDenoiser`.

    ``forward(noisy, cond, sigma)`` stacks ``[cond ‖ noisy]`` (the official order is inputs ‖ forcings ‖
    noisy targets) into node features, runs the explicit grid2mesh GNN → k-hop mesh transformer →
    mesh2grid GNN of :mod:`.denoiser` and returns the raw (un-preconditioned) prediction.
    Every size is a config argument, so the architecture can be modified and retrained freely; with
    the official config (+ :func:`GenCastDenoiser_from_official`) the identical module runs the official
    weights.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (16, 32),
        out_channels: int = 4,
        cond_channels: int = 8,
        mesh_level: int = 1,
        latent: int = 32,
        transformer_layers: int = 2,
        num_heads: int = 4,
        attention_k_hop: int = 4,
        ffw_hidden: Optional[int] = None,
        hidden_layers: int = 1,
        radius_query_fraction_edge_length: float = 0.6,
        attention_type: str = "dense",
        grid_lat: Optional[Sequence[float]] = None,
        grid_lon: Optional[Sequence[float]] = None,
        noise_out: Tuple[int, int] = (32, 16),
        noise_base_period: float = 16.0,
        noise_freqs: int = 32,
    ):
        super().__init__()
        from .denoiser import DenoiserConfig, GenCastDenoiser, SparseTransformerConfig
        from ..graphcast.mesh import default_grid_lat_lon

        self.img_size = tuple(img_size)
        self.out_channels, self.cond_channels = out_channels, cond_channels
        if grid_lat is None or grid_lon is None:
            grid_lat, grid_lon = default_grid_lat_lon(self.img_size)
        cfg = DenoiserConfig(
            transformer=SparseTransformerConfig(
                attention_k_hop=attention_k_hop, d_model=latent, num_layers=transformer_layers, num_heads=num_heads,
                ffw_hidden=ffw_hidden or 2 * latent, attention_type=attention_type),
            mesh_size=mesh_level, latent_size=latent, hidden_layers=hidden_layers,
            radius_query_fraction_edge_length=radius_query_fraction_edge_length,
            node_output_size=out_channels, in_channels=cond_channels + out_channels,
            noise_base_period=noise_base_period, noise_num_frequencies=noise_freqs, noise_mlp_sizes=tuple(noise_out),
        )
        self.net = GenCastDenoiser(cfg, np.asarray(grid_lat), np.asarray(grid_lon))

    def forward(self, noisy: torch.Tensor, cond: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        B, C, H, W = noisy.shape
        if (H, W) != self.img_size:
            raise ValueError(f"grid {(H, W)} != img_size {self.img_size}")
        x = torch.cat([cond, noisy], 1).reshape(B, -1, H * W).transpose(1, 2)
        y = self.net(x, sigma.to(noisy.dtype))
        return y.transpose(1, 2).reshape(B, self.out_channels, H, W)


# --------------------------------------------------------------------------- GenCast (EDM wrapper)


class GenCast(nn.Module):
    """EDM-preconditioned diffusion forecaster around a :class:`DenoiserNet`.

    * ``denoise(noisy, cond, sigma)`` – official preconditioned denoiser D (Eqn 7).
    * ``loss(target, cond)`` – noise-weighted MSE (one random σ per sample, official rho-distribution).
    * ``sample(cond, num_members)`` – ensemble of samples via DPM-Solver++ 2S with churn.

    ``target`` is whatever the caller wants to diffuse (typically the normalised residual
    ``x_{t+12h} − x_t``).
    """

    def __init__(
        self,
        net: DenoiserNet,
        sampler: SamplerConfig = SamplerConfig(),
        noise: NoiseConfig = NoiseConfig(),
    ):
        super().__init__()
        self.net, self.sampler_cfg, self.noise_cfg = net, sampler, noise

    @staticmethod
    def c_in(s):
        return (s**2 + 1) ** -0.5

    @staticmethod
    def c_out(s):
        return s * (s**2 + 1) ** -0.5

    @staticmethod
    def c_skip(s):
        return 1 / (s**2 + 1)

    @classmethod
    def loss_weighting(cls, s):
        return cls.c_out(s) ** -2

    def denoise(self, noisy: torch.Tensor, cond: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        s = sigma.view(-1, 1, 1, 1).to(noisy.dtype)
        raw = self.net(noisy * self.c_in(s), cond, sigma)
        return raw * self.c_out(s) + noisy * self.c_skip(s)

    def loss(self, target: torch.Tensor, cond: torch.Tensor, generator: Optional[torch.Generator] = None):
        B = target.shape[0]
        u = torch.rand(B, dtype=target.dtype, device=target.device, generator=generator)
        c = self.noise_cfg
        sigma = rho_inverse_cdf(c.training_min_noise_level, c.training_max_noise_level, c.training_noise_level_rho, u)
        noisy = target + _gauss(target, generator) * sigma.view(-1, 1, 1, 1)
        den = self.denoise(noisy, cond, sigma)
        per_sample = ((den - target) ** 2).mean(dim=(1, 2, 3))
        return (per_sample * self.loss_weighting(sigma)).mean()

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        num_members: int = 1,
        generator: Optional[torch.Generator] = None,
        noise_fn: Optional[NoiseFn] = None,
    ) -> torch.Tensor:
        """Returns ``(num_members, B, C, H, W)`` samples of the diffused target given ``cond``."""
        B, _, H, W = cond.shape
        outs = []
        for _ in range(num_members):
            tmpl = cond.new_zeros(B, self.net.out_channels, H, W)
            outs.append(
                dpm_solver_pp_2s_sample(
                    lambda x, s: self.denoise(x, cond, s), tmpl, self.sampler_cfg, noise_fn, generator
                )
            )
        return torch.stack(outs, 0)


def GenCast_lite(
    img_size: Tuple[int, int] = (16, 32),
    out_channels: int = 4,
    cond_channels: int = 8,
    num_noise_levels: int = 6,
    **net_kwargs,
) -> GenCast:
    """Tiny random-init GenCast (≈0.1 M params at defaults, 6 solver steps) for CPU/GPU smoke tests."""
    net = DenoiserNet(img_size=img_size, out_channels=out_channels, cond_channels=cond_channels, **net_kwargs)
    return GenCast(net, sampler=SamplerConfig(num_noise_levels=num_noise_levels))
