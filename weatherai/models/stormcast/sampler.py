"""EDM sampler (Karras et al. 2022, Alg. 2: 2nd-order Heun with optional stochastic churn)."""
from __future__ import annotations

import math
from typing import Callable

import torch


def edm_sigmas(num_steps: int = 18, sigma_min: float = 0.002, sigma_max: float = 800.0, rho: float = 7.0,
               device=None, dtype=torch.float64) -> torch.Tensor:
    """rho-schedule ``(smax^(1/rho) + i/(N-1) (smin^(1/rho) - smax^(1/rho)))^rho`` followed by a final 0 (N+1 values)."""
    i = torch.arange(num_steps, dtype=dtype, device=device)
    s = (sigma_max ** (1 / rho) + i / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return torch.cat([s, torch.zeros_like(s[:1])])


@torch.no_grad()
def edm_heun_sample(denoise: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], latents: torch.Tensor, num_steps: int = 18,
                    sigma_min: float = 0.002, sigma_max: float = 800.0, rho: float = 7.0, S_churn: float = 0.0,
                    S_min: float = 0.0, S_max: float = float("inf"), S_noise: float = 1.0,
                    generator: torch.Generator | None = None) -> torch.Tensor:
    """``latents`` must already be scaled to ``sigma_max`` (``sigma_max * randn``). ``denoise(x, sigma[B]) -> D(x; sigma)``."""
    t = edm_sigmas(num_steps, sigma_min, sigma_max, rho, device=latents.device).to(latents.dtype)
    x = latents
    B = x.shape[0]
    for i in range(num_steps):
        t_cur, t_next = t[i], t[i + 1]
        gamma = min(S_churn / num_steps, math.sqrt(2) - 1) if S_min <= float(t_cur) <= S_max else 0.0
        t_hat = t_cur + gamma * t_cur
        x_hat = x
        if gamma > 0:
            z = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
            x_hat = x + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * z
        d = (x_hat - denoise(x_hat, t_hat.expand(B))) / t_hat
        x = x_hat + (t_next - t_hat) * d
        if float(t_next) != 0.0:
            d2 = (x - denoise(x, t_next.expand(B))) / t_next
            x = x_hat + (t_next - t_hat) * (0.5 * d + 0.5 * d2)
    return x
