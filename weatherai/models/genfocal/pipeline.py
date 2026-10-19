"""Two-stage GenFocal research interface: ReFlow + conditional EDM residuals.

Native networks are architecture approximations, NOT Orbax-weight compatible.
Sampling here is a single-segment Heun ODE, not upstream multi-device SDE.
"""
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from .flow import sample_flow
from .unet import GenFocalNet, GenFocalNetConfig


def resize_field(x, size):
    """(B,T,X,Y,C) -> requested (T,X,Y); linear space, replicated time."""
    if x.ndim != 5 or len(size) != 3 or any(v < 1 for v in size):
        raise ValueError("expected B,T,X,Y,C and positive T,X,Y output size")
    b, t, _, _, c = x.shape
    spatial = F.interpolate(x.permute(0, 1, 4, 2, 3).reshape(b * t, c, *x.shape[2:4]),
                            size=size[1:], mode="bilinear", align_corners=False)
    spatial = spatial.reshape(b, t, c, *size[1:]).permute(0, 2, 1, 3, 4)
    return F.interpolate(spatial, size=size, mode="nearest").permute(0, 2, 3, 4, 1)


class GenFocalSuperResolution(nn.Module):
    """EDM preconditioned conditional 3-D U-ViT on normalized residuals.

    Conditioning must already be resized/normalized to the output grid.
    The inherited U-ViT is lite/approximate (see unet.py), not the official
    Flax UNet3d. sigma_data=1 follows released normalized residual data.
    """
    def __init__(self, config=None, sigma_data=1.0):
        super().__init__()
        if sigma_data <= 0:
            raise ValueError("sigma_data must be positive")
        config = config or GenFocalNetConfig(
            out_channels=4, num_channels=(128, 256, 384, 512), downsample_ratio=(3, 2, 2, 2),
            num_blocks=4, noise_embed_dim=128, use_spatial_attention=(False, False, True, True),
            use_temporal_attention=(False, False, True, True), num_heads=8)
        self.config = replace(config, cond_keys=("channel:lowres",), time_rescale=1.0)
        self.network = GenFocalNet(self.config)
        self.sigma_data = sigma_data

    def forward(self, noisy, sigma, lowres):
        sigma = torch.as_tensor(sigma, device=noisy.device, dtype=noisy.dtype).expand(noisy.shape[0])
        if torch.any(sigma <= 0) or not torch.isfinite(sigma).all():
            raise ValueError("sigma must be positive and finite")
        if lowres.shape != noisy.shape:
            raise ValueError("lowres condition must have the same resized shape as noisy residuals")
        s = sigma.reshape(-1, 1, 1, 1, 1)
        data2 = self.sigma_data ** 2
        denom = s.square() + data2
        raw = self.network(noisy / denom.sqrt(), sigma.log() / 4, {"channel:lowres": lowres})
        return data2 / denom * noisy + s * self.sigma_data / denom.sqrt() * raw

    def loss(self, residual, lowres, sigma, noise=None):
        sigma = torch.as_tensor(sigma, device=residual.device, dtype=residual.dtype).expand(residual.shape[0])
        s = sigma.reshape(-1, 1, 1, 1, 1)
        noise = torch.randn_like(residual) if noise is None else noise
        prediction = self(residual + s * noise, sigma, lowres)
        weight = (s.square() + self.sigma_data ** 2) / (s * self.sigma_data).square()
        return (weight * (prediction - residual).square()).mean()

    @torch.no_grad()
    def sample(self, lowres, *, steps=16, sigma_min=0.002, sigma_max=80.0, rho=7.0, generator=None):
        if steps < 2 or not 0 < sigma_min < sigma_max or rho <= 0:
            raise ValueError("need steps>=2, 0<sigma_min<sigma_max and rho>0")
        ramp = torch.linspace(0, 1, steps, device=lowres.device, dtype=lowres.dtype)
        sigmas = (sigma_max ** (1 / rho) + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        sigmas = torch.cat((sigmas, sigmas.new_zeros(1)))
        x = torch.randn(lowres.shape, device=lowres.device, dtype=lowres.dtype, generator=generator) * sigmas[0]
        for i in range(steps):
            s, nxt = sigmas[i], sigmas[i + 1]
            d = (x - self(x, s, lowres)) / s
            euler = x + (nxt - s) * d
            if i < steps - 1:
                d_next = (euler - self(euler, nxt, lowres)) / nxt
                x = x + (nxt - s) * (d + d_next) / 2
            else:
                x = euler
        return x


class GenFocal(nn.Module):
    """Compose explicit debiasing and super-resolution modules.

    Inputs and conditions are normalized tensors, not raw GCM fields.
    Output remains normalized unless a physical baseline and residual
    climatology are supplied. Channel selection/regions are caller-owned.
    """
    def __init__(self, debiasing: nn.Module, super_resolution: GenFocalSuperResolution):
        super().__init__()
        self.debiasing = debiasing
        self.super_resolution = super_resolution

    def forward(self, x, sigma, cond=None):
        return self.debiasing(x, sigma, cond)

    @torch.no_grad()
    def downscale(self, x, cond, *, output_size, flow_steps=8, diffusion_steps=16,
                  baseline=None, residual_mean=None, residual_std=None, generator=None):
        if flow_steps < 2:
            raise ValueError("flow_steps must be >=2")
        debiased = sample_flow(self.debiasing, x, cond, num_steps=flow_steps)
        lowres = resize_field(debiased, output_size)
        residual = self.super_resolution.sample(lowres, steps=diffusion_steps, generator=generator)
        supplied = (baseline is not None, residual_mean is not None, residual_std is not None)
        if any(supplied) and not all(supplied):
            raise ValueError("physical output requires baseline, residual_mean and residual_std together")
        if all(supplied):
            if baseline.shape != residual.shape or torch.any(torch.as_tensor(residual_std) <= 0):
                raise ValueError("baseline must match output; residual_std must be positive")
            output = baseline + residual_mean + residual_std * residual
        else:
            output = lowres + residual
        return {"debiased": debiased, "residual": residual, "output": output}


def GenFocal_lite(channels=4):
    """Random-weight two-stage U-ViT; train before interpreting predictions."""
    cfg = GenFocalNetConfig.lite(channels)
    return GenFocal(GenFocalNet(cfg), GenFocalSuperResolution(cfg))
