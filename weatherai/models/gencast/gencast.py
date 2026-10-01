"""GenCast-lite: a small **PyTorch re-implementation** of the GenCast diffusion forecaster.

GenCast (Price et al., *Nature* 637, 2025) is a conditional diffusion model (Karras et al.
"EDM" framework) whose denoiser is a GraphCast-style grid→mesh→grid network with a mesh
transformer processor, conditioned on the noise level. The official code (JAX/Haiku) lives in
``google-deepmind/weathernext`` (``weathernext1_gen``); its weights are ``.npz`` Haiku parameters.

What this module is
-------------------
* EDM preconditioning ``D = c_skip·x + c_out·F(c_in·x, σ)``, loss weighting ``c_out^-2``,
  noise-level schedule (``rho`` quantiles) and the **DPM-Solver++ 2S sampler with stochastic
  churn**, written from the official source (``denoiser`` / ``gencast`` / ``samplers_utils`` /
  ``dpm_solver_plus_plus_2s``). The sampler and schedules are checked numerically against the
  official JAX code (see ``tests/models/gencast`` and ``scripts/gencast_sampler_reference.py``).
* A *small* denoiser network ``DenoiserNet`` in the same spirit as the official one (noise-level
  Fourier-MLP encoding → conditional LayerNorm; grid2mesh GNN → mesh transformer with k-hop
  attention mask → mesh2grid GNN), built on WeatherAI's GraphCast graphs. Random init.

What it is NOT
--------------
* Not weight-compatible with the official checkpoints and not numerically equal to the official
  denoiser network (different layer layout, conditioning details, no banded sparse attention, no
  per-variable normalisation / forcings pipeline). **No official weights are loaded.**
* Noise is i.i.d. Gaussian on the grid, not the official spherical-harmonic isotropic white noise.
* Targets are generic ``(B, C, H, W)`` tensors (residual to the previous state in normalised units);
  there is no ERA5 data pipeline.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..graphcast.mesh import build_graphs

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


def fourier_features(values: torch.Tensor, base_period: float, num_frequencies: int) -> torch.Tensor:
    """sin/cos features at integer multiples of 1/base_period (official ``fourier_features``)."""
    freqs = torch.arange(1, num_frequencies + 1, dtype=values.dtype, device=values.device) / base_period
    ang = 2 * math.pi * freqs
    v = values[..., None] * ang
    return torch.cat([torch.sin(v), torch.cos(v)], dim=-1)


class _CondLN(nn.Module):
    """LayerNorm whose scale/offset come from the noise-level encoding (norm conditioning)."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.to_so = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.to_so.weight)
        nn.init.zeros_(self.to_so.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:  # x (B,N,D) cond (B,Dc)
        s, o = self.to_so(cond).unsqueeze(1).chunk(2, dim=-1)
        return self.norm(x) * (1 + s) + o


class _CMLP(nn.Module):
    def __init__(self, i: int, o: int, h: int, cond_dim: int, ln: bool = True):
        super().__init__()
        self.l1, self.l2 = nn.Linear(i, h), nn.Linear(h, o)
        self.ln = _CondLN(o, cond_dim) if ln else None

    def forward(self, x, cond):
        y = self.l2(F.silu(self.l1(x)))
        return self.ln(y, cond) if self.ln is not None else y


def _seg_sum(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = src.new_zeros((src.shape[0], n) + src.shape[2:])
    return out.index_add_(1, index, src)


class _Bipartite(nn.Module):
    """One message-passing step src→dst with conditioned MLPs (GraphCast-style residual deltas)."""

    def __init__(self, d: int, cond_dim: int, update_src: bool):
        super().__init__()
        self.edge = _CMLP(3 * d, d, d, cond_dim)
        self.dst = _CMLP(2 * d, d, d, cond_dim)
        self.src = _CMLP(d, d, d, cond_dim) if update_src else None

    def forward(self, src, dst, e, s_idx, r_idx, cond):
        de = self.edge(torch.cat([e, src[:, s_idx], dst[:, r_idx]], -1), cond)
        agg = _seg_sum(de, r_idx, dst.shape[1])
        dst = dst + self.dst(torch.cat([dst, agg], -1), cond)
        if self.src is not None:
            src = src + self.src(src, cond)
        return src, dst, e + de


class _MeshTransformerBlock(nn.Module):
    def __init__(self, d: int, heads: int, ffw: int, cond_dim: int):
        super().__init__()
        assert d % heads == 0
        self.h = heads
        self.ln1, self.ln2 = _CondLN(d, cond_dim), _CondLN(d, cond_dim)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ffw), nn.GELU(), nn.Linear(ffw, d))

    def forward(self, x, cond, mask):
        B, N, D = x.shape
        q, k, v = self.qkv(self.ln1(x, cond)).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # mask: (N,N) bool, True = attend
        x = x + self.proj(a.transpose(1, 2).reshape(B, N, D))
        return x + self.ff(self.ln2(x, cond))


class DenoiserNet(nn.Module):
    """Raw network F(c_in·x_noisy, cond, σ) → same shape as the target.

    ``forward(noisy, cond, sigma)``: ``noisy`` (B, C, H, W), ``cond`` (B, Cc, H, W) conditioning
    channels (previous states / forcings, already stacked), ``sigma`` (B,) *raw* noise level.
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
        noise_base_period: float = 16.0,
        noise_freqs: int = 32,
        noise_out: Tuple[int, int] = (32, 16),
    ):
        super().__init__()
        self.img_size = tuple(img_size)
        self.out_channels, self.cond_channels = out_channels, cond_channels
        self.noise_base_period, self.noise_freqs = noise_base_period, noise_freqs
        g = build_graphs(img_size=self.img_size, mesh_level=mesh_level, use_multi_mesh=False)
        self.Nm, self.Ng = g.mesh_nodes.shape[0], g.grid_nodes.shape[0]
        t = lambda a, dt: torch.as_tensor(a, dtype=dt)
        for k in ("g2m", "mesh", "m2g"):
            self.register_buffer(f"{k}_s", t(getattr(g, f"{k}_senders"), torch.long), persistent=False)
            self.register_buffer(f"{k}_r", t(getattr(g, f"{k}_receivers"), torch.long), persistent=False)
            self.register_buffer(f"{k}_ea", t(getattr(g, f"{k}_edge_attr"), torch.float32), persistent=False)
        self.register_buffer("grid_nf", t(g.grid_node_features, torch.float32), persistent=False)
        self.register_buffer("mesh_nf", t(g.mesh_node_features, torch.float32), persistent=False)
        # k-hop attention mask on the mesh graph (official uses banded sparse attention, k-hop).
        A = torch.eye(self.Nm, dtype=torch.bool)
        A[self.mesh_s, self.mesh_r] = True
        A[self.mesh_r, self.mesh_s] = True
        M, Af = A.clone(), A.float()
        for _ in range(attention_k_hop - 1):
            M = (M.float() @ Af > 0)
        self.register_buffer("attn_mask", M, persistent=False)

        dc = noise_out[-1]
        layers, i = [], 2 * noise_freqs
        for j, o in enumerate(noise_out):
            layers += [nn.Linear(i, o)] + ([nn.GELU()] if j < len(noise_out) - 1 else [])
            i = o
        self.noise_mlp = nn.Sequential(*layers)

        D, nf, ef = latent, self.grid_nf.shape[1], self.g2m_ea.shape[1]
        self.grid_embed = _CMLP(out_channels + cond_channels + nf, D, D, dc)
        self.mesh_embed = _CMLP(nf, D, D, dc)
        self.g2m_embed, self.mesh_e_embed, self.m2g_embed = (_CMLP(ef, D, D, dc) for _ in range(3))
        self.g2m = _Bipartite(D, dc, update_src=True)
        self.blocks = nn.ModuleList(_MeshTransformerBlock(D, num_heads, 2 * D, dc) for _ in range(transformer_layers))
        self.m2g = _Bipartite(D, dc, update_src=False)
        self.out = _CMLP(D, out_channels, D, dc, ln=False)

    def noise_encoding(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.noise_mlp(fourier_features(torch.log(sigma), self.noise_base_period, self.noise_freqs))

    def forward(self, noisy: torch.Tensor, cond: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        B, C, H, W = noisy.shape
        if (H, W) != self.img_size:
            raise ValueError(f"grid {(H, W)} != img_size {self.img_size}")
        enc = self.noise_encoding(sigma.to(noisy.dtype))
        grid_in = torch.cat([noisy, cond], 1).reshape(B, C + cond.shape[1], H * W).transpose(1, 2)
        gnf = self.grid_nf.unsqueeze(0).expand(B, -1, -1)
        grid = self.grid_embed(torch.cat([grid_in, gnf], -1), enc)
        mesh = self.mesh_embed(self.mesh_nf.unsqueeze(0).expand(B, -1, -1), enc)
        emb = lambda m, ea: m(ea.unsqueeze(0).expand(B, -1, -1), enc)
        grid, mesh, _ = self.g2m(grid, mesh, emb(self.g2m_embed, self.g2m_ea), self.g2m_s, self.g2m_r, enc)
        for blk in self.blocks:
            mesh = blk(mesh, enc, self.attn_mask)
        _, grid, _ = self.m2g(mesh, grid, emb(self.m2g_embed, self.m2g_ea), self.m2g_s, self.m2g_r, enc)
        return self.out(grid, enc).transpose(1, 2).reshape(B, self.out_channels, H, W)


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
