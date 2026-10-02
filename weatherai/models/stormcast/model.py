"""StormCast: regression network predicts the next-hour mean state, a conditional EDM diffusion network adds the residual."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .sampler import edm_heun_sample
from .unet import EDMPrecond, SongUNetConfig, StormCastUNet


@dataclass
class StormCastConfig:
    img_resolution: tuple = (512, 640)
    n_state: int = 99          # HRRR hybrid-level state channels (u,v,T,q,Z,p on 14 model levels + u10m,v10m,t2m,msl + refc)
    n_cond: int = 26           # coarse (GFS/ERA5-like) boundary conditioning channels
    n_inv: int = 2             # static invariants: land-sea mask, orography
    model_channels: int = 128
    channel_mult: tuple = (1, 2, 2, 2, 2)
    num_blocks: int = 4
    sigma_data: float = 0.5
    sampler: dict = field(default_factory=lambda: dict(num_steps=18, sigma_min=0.002, sigma_max=800.0, rho=7.0,
                                                       S_churn=0.0, S_min=0.0, S_max=float("inf"), S_noise=1.0))

    def regression_cfg(self) -> SongUNetConfig:
        return SongUNetConfig(self.img_resolution, self.n_state + self.n_cond + self.n_inv, self.n_state, self.model_channels,
                              self.channel_mult, num_blocks=self.num_blocks, embedding_type="zero", additive_pos_embed=False)

    def diffusion_cfg(self) -> SongUNetConfig:
        return SongUNetConfig(self.img_resolution, 3 * self.n_state + self.n_inv, self.n_state, self.model_channels,
                              self.channel_mult, num_blocks=self.num_blocks, embedding_type="positional", additive_pos_embed=True)


class StormCast(nn.Module):
    def __init__(self, cfg: StormCastConfig | None = None):
        super().__init__()
        self.cfg = cfg or StormCastConfig()
        self.regression = StormCastUNet(self.cfg.regression_cfg())
        self.diffusion = EDMPrecond(self.cfg.diffusion_cfg(), self.cfg.sigma_data)
        H, W = self.cfg.img_resolution
        c = self.cfg
        self.register_buffer("means", torch.zeros(1, c.n_state, 1, 1))
        self.register_buffer("stds", torch.ones(1, c.n_state, 1, 1))
        self.register_buffer("cond_means", torch.zeros(1, c.n_cond, 1, 1))
        self.register_buffer("cond_stds", torch.ones(1, c.n_cond, 1, 1))
        self.register_buffer("invariants", torch.zeros(1, c.n_inv, H, W))

    @torch.no_grad()
    def forward(self, x: torch.Tensor, conditioning: torch.Tensor, generator: torch.Generator | None = None,
                sampler: dict | None = None, deterministic_only: bool = False) -> torch.Tensor:
        """One 1-hour step. ``x``: (B,99,H,W) physical units; ``conditioning``: (B,26,H,W) interpolated to the HRRR grid.

        Returns the next state in physical units. ``deterministic_only`` returns the regression mean (no diffusion)."""
        s = {**self.cfg.sampler, **(sampler or {})}
        xn = (x - self.means) / self.stds
        cn = (conditioning - self.cond_means) / self.cond_stds
        inv = self.invariants.expand(x.shape[0], -1, -1, -1)
        mean = self.regression(torch.cat([xn, cn, inv], dim=1))
        if deterministic_only:
            return mean * self.stds + self.means
        cond = torch.cat([xn, mean, inv], dim=1)
        lat = s["sigma_max"] * torch.randn(xn.shape, generator=generator, device=x.device, dtype=xn.dtype)
        res = edm_heun_sample(lambda z, sig: self.diffusion(z, sig, cond), lat, generator=generator, **s)
        return (mean + res) * self.stds + self.means


def StormCast_lite() -> StormCast:
    """Tiny trainable config for tests / CPU smoke (same code path, 3 levels, 32 channels, 64x64)."""
    return StormCast(StormCastConfig(img_resolution=(64, 64), n_state=8, n_cond=4, n_inv=2, model_channels=32,
                                     channel_mult=(1, 2, 2), num_blocks=1, sampler=dict(
                                         num_steps=4, sigma_min=0.002, sigma_max=800.0, rho=7.0, S_churn=0.0, S_min=0.0,
                                         S_max=float("inf"), S_noise=1.0)))
