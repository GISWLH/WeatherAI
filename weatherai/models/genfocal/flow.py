"""Conditional rectified-flow (ReFlow) core of GenFocal's debiasing stage, ported from
``swirl_dynamics.projects.genfocal.debiasing`` (``ConditionalReFlowModel.loss_fn``, ``inference_utils.sampling_from_batch``).

Layout is channels-last like the official code: ``x`` is ``(batch, time, lon, lat, channels)``.  A flow network is any callable
``net(x, sigma, cond) -> v`` with ``sigma`` of shape ``(batch,)`` in [0, 1] and ``cond`` a dict of tensors shaped like ``x``.
"""
from __future__ import annotations

import math
from typing import Callable, Mapping

import torch

Net = Callable[[torch.Tensor, torch.Tensor, "Mapping[str, torch.Tensor] | None"], torch.Tensor]

MIN_TIME = 1e-4
MAX_TIME = 1.0 - 1e-4


def logistic_time_sampler(u: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    """Official ``lognormal_sampler``: ``logistic(std * u + mean)`` with ``u ~ U(0, 1)`` (despite the name it is not log-normal)."""
    return torch.sigmoid(std * u + mean)


def interpolate(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """``x_t = t x_1 + (1 - t) x_0`` with ``t`` of shape ``(batch,)``."""
    t = t.reshape(-1, *([1] * (x0.ndim - 1)))
    return x1 * t + x0 * (1 - t)


def train_times(u: torch.Tensor, min_time: float = MIN_TIME, max_time: float = MAX_TIME) -> torch.Tensor:
    """Official time mapping ``(max - min) * u + min`` of a ``(batch,)`` draw ``u`` (``U(0, 1)`` or `logistic_time_sampler`)."""
    return (max_time - min_time) * u + min_time


def reflow_loss(net: Net, x0: torch.Tensor, x1: torch.Tensor, cond: "Mapping[str, torch.Tensor] | None" = None,
                u: "torch.Tensor | None" = None, weighted_norm: "torch.Tensor | None" = None,
                min_time: float = MIN_TIME, max_time: float = MAX_TIME) -> torch.Tensor:
    """Flow-matching loss ``mean(w * ((x_1 - x_0) - v(x_t, t, cond))^2)``; ``u`` is the (batch,) uniform draw (random if None)."""
    if u is None:
        u = torch.rand(x0.shape[0], device=x0.device, dtype=x0.dtype)
    t = train_times(u, min_time, max_time)
    v = net(interpolate(x0, x1, t), t, cond)
    err = (x1 - x0) - v
    w = 1.0 if weighted_norm is None else weighted_norm
    return (w * err.square()).mean()


def _rk4_step(f, x, t, dt):
    k1 = f(x, t)
    k2 = f(x + dt * k1 / 2, t + dt / 2)
    k3 = f(x + dt * k2 / 2, t + dt / 2)
    k4 = f(x + dt * k3, t + dt)
    return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def _euler_step(f, x, t, dt):
    return x + dt * f(x, t)


@torch.no_grad()
def sample_flow(net: Net, x0: torch.Tensor, cond: "Mapping[str, torch.Tensor] | None" = None, num_steps: int = 8,
                solver: str = "rk4", reverse: bool = False, to_end: bool = False) -> torch.Tensor:
    """Integrates ``dx/dt = v(x, t, cond)`` from ``x0``.

    ``to_end=False`` reproduces the official schedule exactly: ``tspan = arange(0, 1, 1/num_steps)`` has ``num_steps`` points, so the
    solver takes ``num_steps - 1`` steps and the returned state is at ``t = 1 - 1/num_steps``, not t = 1 (an apparent off-by-one in
    the released code).  ``to_end=True`` uses ``linspace(0, 1, num_steps + 1)`` and integrates to t = 1.
    ``reverse=True`` integrates the reversed flow, ``v_rev(x, t) = -v(x, 1 - t)``.
    """
    step = {"rk4": _rk4_step, "euler": _euler_step}[solver]

    def f(x, t):
        tt = torch.full((x.shape[0],), float(t), dtype=x.dtype, device=x.device)
        if reverse:
            return -net(x, 1.0 - tt, cond)
        return net(x, tt, cond)

    if to_end:
        ts = torch.linspace(0.0, 1.0, num_steps + 1, dtype=torch.float32)
    else:
        ts = torch.arange(0.0, 1.0, 1.0 / num_steps, dtype=torch.float32)
    x = x0
    for i in range(len(ts) - 1):
        t0 = float(ts[i])
        x = step(f, x, t0, float(ts[i + 1]) - t0)
    return x


def denormalise(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std + mean


__all__ = ["logistic_time_sampler", "interpolate", "train_times", "reflow_loss", "sample_flow", "denormalise"]
