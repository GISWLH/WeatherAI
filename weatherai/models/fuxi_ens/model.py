"""Native PyTorch FuXi-ENS (reverse-engineered from the official ``fuxi_ens.onnx``).

FuXi-ENS (Zhong et al., 2024; weights: Zenodo 10.5281/zenodo.15124541, CC-BY-NC-4.0) is a
*conditional-VAE-style* ensemble forecaster.  The official release only ships an ONNX graph
(26,018 nodes, 498 initialisers, 2.486 B parameters); there is no official PyTorch source.
This file is an explicit, readable PyTorch re-implementation whose structure and parameter
names were read out of that graph and which is verified numerically against onnxruntime
(see ``tests/models/fuxi_ens`` and docs/model_status.md).

Data flow of one forward call (one 6-hour step; the ONNX returns the *next two-step state*)::

    input  [B, 2, C, H, W]  (C = 78 = 5 vars x 13 levels + 13 surface; H x W = 721 x 1440)
      |  (x - mean) / std ; NaN -> 0 ; +-inf -> +-fp32max ; accumulated channels (last 5) := 0
      v
    xn [B, 2, C, H, W]
      |  dist_p  : 8x8 patch embed (+ static `const` field) -> LN -> dropout(p=.2, "drop0")
      |            -> layers.0 (7 AdaLN shifted-window RoPE-attention blocks) -> dropout("drop1")
      |            -> layers.1 (7 blocks) -> final AdaLN (+ residual of the patch embedding)
      |            -> (mean, logvar) heads (ConvTranspose 8x8)
      |  sample = mean + exp(logvar/2) * eps          <-- random op #3 (RandomNormalLike)
      v
    z = xn + sample ; accumulated channels := 0
      |  decoder.0 : same 8x8 patch embed -> LN -> 6x7 AdaLN blocks -> final AdaLN + residual
      |              -> pl_head (65 ch) || sf_head (13 ch)   (ConvTranspose 8x8)
      v
    y = out * std + mean ; precipitation channel: y = exp(clip(y, 0, 7)) - 1
    output [B, 2, C, H, W] = concat(input[:, -1:], y)

Random ops baked into the ONNX graph (all three are *active* in the exported graph, there is
no eval switch): ``drop0`` and ``drop1`` (two ``RandomUniformLike`` dropout masks, keep prob
0.8, scale 1/0.8) and the ``RandomNormalLike`` latent noise.  ``FuXiENS.forward`` accepts them
explicitly through :class:`FuXiENSNoise` so the comparison with onnxruntime is deterministic.

Quirks that are reproduced bit-for-bit (all verified against the graph):
  * LayerNorm uses the *unbiased* variance, ``var = mean((x-mu)^2) * n/(n-1)``, eps = 1e-6.
  * RoPE is a 1-D rotary embedding on the *window-flattened* token index (0..16199) with
    interleaved (even, odd) pairs, base 10000, head_dim 64 -- not 2-D.
  * Odd blocks (1, 3, 5) of every layer use a cyclic shift of 9 and the stored additive mask
    (-100 between the wrapped-around bottom rows and the rest); even blocks do neither.
  * q and k are each scaled by head_dim**-0.25 (== head_dim**-0.5 overall).
  * The gated-GELU MLP computes ``a * gelu(b)`` with ``a, b = fc1(x).chunk(2)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "FuXiENSConfig",
    "FuXiENSNoise",
    "FuXiENS",
    "FuXiENS_lite",
    "unbiased_layer_norm",
    "sinusoidal_embedding",
    "make_shift_mask",
    "make_rope_table",
]


# ----------------------------------------------------------------------------- config
@dataclass
class FuXiENSConfig:
    in_steps: int = 2                 # the model sees 2 consecutive 6-h states
    n_pl_vars: int = 5                # z, t, u, v, q
    n_levels: int = 13
    n_surface: int = 13               # t2m d2m sst u10m v10m u100m v100m msl ssr ssrd fdir ttr tp
    n_accum: int = 5                  # last channels (ssr ssrd fdir ttr tp) are accumulations
    n_static: int = 6                 # static "const" field channels
    img_size: Tuple[int, int] = (721, 1440)
    patch: int = 8
    dim: int = 1536
    heads: int = 24
    window: Tuple[int, int] = (18, 18)
    shift: Tuple[int, int] = (9, 9)
    dist_layers: Tuple[int, ...] = (7, 7)         # blocks per layer in the noise-distribution net
    decoder_layers: Tuple[int, ...] = (7, 7, 7, 7, 7, 7)
    mlp_hidden: int = 8192            # fc1 output (gated => 2 x 4096)
    embed_freqs: int = 128            # sin/cos embedding -> 256 dims
    dropout_p: float = 0.2            # drop0 / drop1 in dist_p
    precip_clip: float = 7.0          # tp = exp(clip(y, 0, 7)) - 1
    rope_base: float = 10000.0
    rope_table_len: Optional[int] = None  # defaults to number of tokens

    # ----- derived
    @property
    def channels(self) -> int:
        return self.n_pl_vars * self.n_levels + self.n_surface

    @property
    def n_pl(self) -> int:
        return self.n_pl_vars * self.n_levels

    @property
    def grid(self) -> Tuple[int, int]:
        return (self.img_size[0] // self.patch, self.img_size[1] // self.patch)

    @property
    def n_tokens(self) -> int:
        return self.grid[0] * self.grid[1]

    @property
    def out_rows(self) -> int:
        return math.ceil(self.img_size[0] / self.patch)

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    @classmethod
    def official(cls) -> "FuXiENSConfig":
        return cls()

    @classmethod
    def lite(cls) -> "FuXiENSConfig":
        """Small trainable configuration (same code path as the official one)."""
        return cls(n_levels=3, n_surface=7, n_accum=3, n_static=2, img_size=(33, 64), dim=96, heads=4,
                   window=(4, 4), shift=(2, 2), dist_layers=(2,), decoder_layers=(2, 2), mlp_hidden=256,
                   embed_freqs=16)

    def replace(self, **kw) -> "FuXiENSConfig":
        return replace(self, **kw)


@dataclass
class FuXiENSNoise:
    """Explicit random inputs (so the forward pass is deterministic / comparable with ONNX).

    drop0, drop1: uniform(0,1) samples of shape [B, N, dim]; element kept iff u < 1 - p.
    eps:          standard normal of shape [B, in_steps*C, H, W].
    """
    drop0: Optional[torch.Tensor] = None
    drop1: Optional[torch.Tensor] = None
    eps: Optional[torch.Tensor] = None

    @staticmethod
    def sample(cfg: FuXiENSConfig, batch: int = 1, generator: Optional[torch.Generator] = None,
               device="cpu", dtype=torch.float32) -> "FuXiENSNoise":
        n, d = cfg.n_tokens, cfg.dim
        h, w = cfg.img_size
        gdev = generator.device if generator is not None else torch.device(device)
        kw = dict(generator=generator, device=gdev, dtype=dtype)       # draw where the generator lives, then move
        out = (torch.rand(batch, n, d, **kw), torch.rand(batch, n, d, **kw),
               torch.randn(batch, cfg.in_steps * cfg.channels, h, w, **kw))
        return FuXiENSNoise(*(t.to(device) for t in out))


# ----------------------------------------------------------------------------- helpers
def unbiased_layer_norm(x: torch.Tensor, weight: Optional[torch.Tensor] = None, eps: float = 1e-6) -> torch.Tensor:
    """LayerNorm exactly as in the ONNX graph (fp32 math, *unbiased* variance, no bias)."""
    dt = x.dtype
    x = x.float()
    n = x.shape[-1]
    mu = x.mean(-1, keepdim=True)
    xc = x - mu
    var = (xc * xc).mean(-1, keepdim=True) * (n / (n - 1))
    y = xc / torch.sqrt(var + eps)
    if weight is not None:
        y = y * weight.float()
    return y.to(dt)


def sinusoidal_embedding(t: torch.Tensor, n_freqs: int = 128) -> torch.Tensor:
    """[B] -> [B, 2*n_freqs]: cat(sin(t f), cos(t f)), f_i = 1 / 10000**(i / n_freqs)."""
    i = torch.arange(n_freqs, dtype=torch.float32, device=t.device) / n_freqs
    f = 1.0 / torch.pow(torch.tensor(10000.0, device=t.device), i)
    a = t.float().reshape(-1, 1) * f[None, :]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


def make_shift_mask(grid: Tuple[int, int], window: Tuple[int, int], shift: Tuple[int, int]) -> torch.Tensor:
    """Additive attention mask [nW, T, T] for the cyclically shifted windows.

    Longitude is periodic so wrapping columns is legal; latitude is not, so the rows that wrap
    around (the last ``shift[0]`` rows after the roll) are masked (-100) against all others.
    This equals the ``attn_mask`` initialisers of the ONNX graph (checked in the tests).
    """
    H, W = grid
    wh, ww = window
    img = torch.zeros(H, W)
    if shift[0] > 0:
        img[H - shift[0]:, :] = 1
    win = img.view(H // wh, wh, W // ww, ww).permute(0, 2, 1, 3).reshape(-1, wh * ww)
    diff = win[:, :, None] != win[:, None, :]
    return diff.float() * -100.0


def make_rope_table(n: int, head_dim: int, base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables [n, head_dim/2] in fp32 (1-D RoPE over the flattened index)."""
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ang = torch.outer(torch.arange(n).float(), freqs)
    return torch.cos(ang), torch.sin(ang)


# ----------------------------------------------------------------------------- modules
class ConditionEmbed(nn.Module):
    """sinusoidal(t) -> Linear -> SiLU -> Linear  (``mlp.0`` / ``mlp.2`` as in the checkpoint)."""

    def __init__(self, dim: int, n_freqs: int):
        super().__init__()
        self.n_freqs = n_freqs
        self.mlp = nn.Sequential(nn.Linear(2 * n_freqs, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal_embedding(t, self.n_freqs).to(self.mlp[0].weight.dtype))


class PatchEmbed(nn.Module):
    """8x8 non-overlapping patches of [state ; static const] -> tokens [B, N, dim] -> LN(weight)."""

    def __init__(self, cfg: FuXiENSConfig):
        super().__init__()
        self.input_proj = nn.Conv2d(cfg.in_steps * cfg.channels, cfg.dim, cfg.patch, cfg.patch)
        self.const_proj = nn.Conv2d(cfg.n_static, cfg.dim, cfg.patch, cfg.patch)
        self.norm = _WeightOnlyNorm(cfg.dim)

    def forward(self, x: torch.Tensor, const: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.const_proj(const)          # [B, dim, h, w] (partial last row dropped)
        h = h.permute(0, 2, 3, 1).flatten(1, 2)                  # [B, N, dim]
        return self.norm(h)


class _WeightOnlyNorm(nn.Module):
    """LayerNorm with a learnable scale only (``patch_embed.norm.weight``), unbiased variance."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return unbiased_layer_norm(x, self.weight, self.eps)


class WindowRoPEAttention(nn.Module):
    """Multi-head window attention with 1-D RoPE on the window-flattened token index."""

    def __init__(self, cfg: FuXiENSConfig, shifted: bool):
        super().__init__()
        self.cfg = cfg
        self.heads, self.hd = cfg.heads, cfg.head_dim
        self.wq = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.shifted = shifted
        self.grid = cfg.grid
        self.window = cfg.window
        self.shift = cfg.shift if shifted else (0, 0)

    # ---- window helpers
    def _partition(self, x):
        B, H, W, C = x.shape
        wh, ww = self.window
        x = x.view(B, H // wh, wh, W // ww, ww, C).permute(0, 1, 3, 2, 4, 5)
        return x.reshape(-1, wh * ww, C)

    def _reverse(self, x, B):
        H, W = self.grid
        wh, ww = self.window
        C = x.shape[-1]
        x = x.view(B, H // wh, W // ww, wh, ww, C).permute(0, 1, 3, 2, 4, 5)
        return x.reshape(B, H, W, C)

    @staticmethod
    def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x [B, L, heads, hd] interleaved pairs (x0, x1) -> (x0 c - x1 s, x0 s + x1 c)
        x = x.float().reshape(*x.shape[:-1], -1, 2)
        x0, x1 = x[..., 0], x[..., 1]
        c, s = cos[None, :, None, :], sin[None, :, None, :]
        return torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor], cos: torch.Tensor, sin: torch.Tensor):
        B, N, C = x.shape
        H, W = self.grid
        x = x.view(B, H, W, C)
        if self.shifted:
            x = torch.roll(x, (-self.shift[0], -self.shift[1]), (1, 2))
        xw = self._partition(x)                                   # [B*nW, T, C]
        BW, T, _ = xw.shape
        nW = BW // B
        dt = xw.dtype
        # rope over the (window-major) flattened index of each batch element
        q = self._rope(self.wq(xw).reshape(B, nW * T, self.heads, self.hd), cos[: nW * T], sin[: nW * T])
        k = self._rope(self.wk(xw).reshape(B, nW * T, self.heads, self.hd), cos[: nW * T], sin[: nW * T])
        q = q.to(dt).reshape(BW, T, self.heads, self.hd).transpose(1, 2)
        k = k.to(dt).reshape(BW, T, self.heads, self.hd).transpose(1, 2)
        v = self.wv(xw).reshape(BW, T, self.heads, self.hd).transpose(1, 2)
        am = None
        if mask is not None:
            am = mask.to(dt)[None, :, None].expand(B, nW, 1, T, T).reshape(BW, 1, T, T)
        # q,k each scaled by hd^-1/4 in the graph == SDPA scale hd^-1/2
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=am, scale=self.hd ** -0.5)
        o = self.wo(o.transpose(1, 2).reshape(BW, T, C))
        o = self._reverse(o, B)
        if self.shifted:
            o = torch.roll(o, (self.shift[0], self.shift[1]), (1, 2))
        return o.reshape(B, N, C)


class GatedMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden // 2, dim, bias=False)

    def forward(self, x):
        a, b = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(a * F.gelu(b))        # exact (erf) GELU


class AdaLNBlock(nn.Module):
    """x += g1 * Attn(LN(x)*(1+s1)+b1);  x += g2 * MLP(LN(x)*(1+s2)+b2);  (s,b,g) = Linear(SiLU(cond))."""

    def __init__(self, cfg: FuXiENSConfig, shifted: bool):
        super().__init__()
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(cfg.dim, 6 * cfg.dim))
        self.attn = WindowRoPEAttention(cfg, shifted)
        self.mlp = GatedMLP(cfg.dim, cfg.mlp_hidden)
        self.shifted = shifted
        if shifted:   # derived, not stored in the checkpoint (equality with the ONNX tensor is tested)
            self.register_buffer("attn_mask", make_shift_mask(cfg.grid, cfg.window, cfg.shift), persistent=False)
        else:
            self.attn_mask = None

    def forward(self, x, cond, cos, sin):
        s1, b1, g1, s2, b2, g2 = self.adaln(cond)[:, None].chunk(6, dim=-1)
        h = unbiased_layer_norm(x) * (1 + s1) + b1
        x = x + g1 * self.attn(h, self.attn_mask, cos, sin)
        h = unbiased_layer_norm(x) * (1 + s2) + b2
        return x + g2 * self.mlp(h)


class Layer(nn.Module):
    def __init__(self, cfg: FuXiENSConfig, n_blocks: int):
        super().__init__()
        self.blocks = nn.ModuleList([AdaLNBlock(cfg, shifted=(i % 2 == 1)) for i in range(n_blocks)])

    def forward(self, x, cond, cos, sin):
        for blk in self.blocks:
            x = blk(x, cond, cos, sin)
        return x


class FinalNorm(nn.Module):
    """AdaLN output norm: LN(x) * (1 + scale) + shift, (scale, shift) = Linear(SiLU(cond))."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale_shift = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, cond):
        s, b = self.scale_shift(cond)[:, None].chunk(2, dim=-1)
        return unbiased_layer_norm(x) * (1 + s) + b


class _Backbone(nn.Module):
    """patch embed + (step, hour, doy) condition + AdaLN window-attention stack (shared by both nets)."""

    def __init__(self, cfg: FuXiENSConfig, layers: Sequence[int]):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        self.step_embed = ConditionEmbed(cfg.dim, cfg.embed_freqs)
        self.hour_embed = ConditionEmbed(cfg.dim, cfg.embed_freqs)
        self.doy_embed = ConditionEmbed(cfg.dim, cfg.embed_freqs)
        self.layers = nn.ModuleList([Layer(cfg, n) for n in layers])
        self.norm_layer = FinalNorm(cfg.dim)
        n = cfg.rope_table_len or cfg.n_tokens
        cos, sin = make_rope_table(n, cfg.head_dim, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def condition(self, step, hour, doy):
        return self.step_embed(step) + self.hour_embed(hour) + self.doy_embed(doy)

    def tokens_to_map(self, x):
        """[B, N, dim] -> [B, dim, h+pad, w] (the bottom partial patch row is zero padded)."""
        cfg = self.cfg
        h, w = cfg.grid
        x = x.view(x.shape[0], h, w, cfg.dim).permute(0, 3, 1, 2)
        return F.pad(x, (0, 0, 0, cfg.out_rows - h))

    def crop(self, y):
        H, W = self.cfg.img_size
        return y[..., :H, :W]


class DistNet(_Backbone):
    """``dist_p``: predicts mean / log-variance of the latent perturbation (same shape as the input)."""

    def __init__(self, cfg: FuXiENSConfig):
        super().__init__(cfg, cfg.dist_layers)
        d, o = cfg.dim, cfg.in_steps * cfg.channels
        self.mean = nn.ConvTranspose2d(d, o, cfg.patch, cfg.patch)
        self.logvar = nn.ConvTranspose2d(d, o, cfg.patch, cfg.patch)

    def forward(self, xn, const, step, hour, doy, drop0=None, drop1=None):
        cfg = self.cfg
        cond = self.condition(step, hour, doy)
        emb = self.patch_embed(xn, const)
        x = _dropout_from_uniform(emb, drop0, cfg.dropout_p)
        for i, layer in enumerate(self.layers):
            x = layer(x, cond, self.rope_cos, self.rope_sin)
            if i == 0:     # drop1 sits between the 2 layers of dist_p (after layers.0)
                x = _dropout_from_uniform(x, drop1, cfg.dropout_p)
        x = self.norm_layer(x, cond) + emb
        m = self.tokens_to_map(x)
        return self.crop(self.mean(m)), self.crop(self.logvar(m))


class Decoder(_Backbone):
    """``decoder.0``: noisy state -> next 6-h state (normalised space), 65 pressure-level + 13 surface channels."""

    def __init__(self, cfg: FuXiENSConfig):
        super().__init__(cfg, cfg.decoder_layers)
        self.pred_layer = nn.Module()
        self.pred_layer.pl_head = nn.ConvTranspose2d(cfg.dim, cfg.n_pl, cfg.patch, cfg.patch)
        self.pred_layer.sf_head = nn.ConvTranspose2d(cfg.dim, cfg.n_surface, cfg.patch, cfg.patch)

    def forward(self, z, const, step, hour, doy):
        cond = self.condition(step, hour, doy)
        emb = self.patch_embed(z, const)
        x = emb
        for layer in self.layers:
            x = layer(x, cond, self.rope_cos, self.rope_sin)
        x = self.norm_layer(x, cond) + emb
        m = self.tokens_to_map(x)
        return self.crop(torch.cat([self.pred_layer.pl_head(m), self.pred_layer.sf_head(m)], dim=1))


def _dropout_from_uniform(x, u, p):
    """Dropout driven by an explicit uniform sample (keep iff u < 1-p, scale 1/(1-p)) -- as in the ONNX graph."""
    if u is None:
        return x
    keep = 1.0 - p
    return (x.float() / torch.tensor(keep, dtype=torch.float32, device=x.device) * (u < keep).float()).to(x.dtype)


# ----------------------------------------------------------------------------- model
class FuXiENS(nn.Module):
    """FuXi-ENS: one 6-hour step of a latent-noise ensemble forecaster (see module docstring)."""

    def __init__(self, cfg: Optional[FuXiENSConfig] = None):
        super().__init__()
        cfg = cfg or FuXiENSConfig.official()
        self.cfg = cfg
        C, (H, W) = cfg.channels, cfg.img_size
        self.register_buffer("mean", torch.zeros(C, 1, 1))
        self.register_buffer("std", torch.ones(C, 1, 1))
        self.register_buffer("const", torch.zeros(cfg.n_static, H, W))
        self.dist_p = DistNet(cfg)
        self.decoder = nn.ModuleList([Decoder(cfg)])      # name `decoder.0.*` as in the checkpoint

    # ---- pieces (public so they can be tested / reused / fine-tuned separately)
    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        """(x-mean)/std, NaN->0, +-inf->+-fp32max, accumulated channels zeroed.  x: [B, T, C, H, W]."""
        fmax = torch.finfo(torch.float32).max
        xn = torch.nan_to_num((x.float() - self.mean) / self.std, nan=0.0, posinf=fmax, neginf=-fmax)
        if self.cfg.n_accum:
            xn = torch.cat([xn[:, :, : -self.cfg.n_accum], torch.zeros_like(xn[:, :, -self.cfg.n_accum:])], dim=2)
        return xn

    def perturb(self, xn, step, hour, doy, noise: Optional[FuXiENSNoise] = None,
                generator: Optional[torch.Generator] = None):
        """Returns (z, mean, logvar): z = xn + mean + exp(logvar/2) * eps (accumulated channels zeroed)."""
        cfg = self.cfg
        B, T, C, H, W = xn.shape
        noise = noise or FuXiENSNoise.sample(cfg, B, generator, xn.device)
        mean, logvar = self.dist_p(xn.flatten(1, 2).to(self.const.dtype), self.const[None], step, hour, doy,
                                   noise.drop0, noise.drop1)
        eps = noise.eps if noise.eps is not None else torch.randn(
            mean.shape, generator=generator, device=generator.device if generator is not None else mean.device).to(mean.device)
        sample = mean.float() + torch.exp(0.5 * logvar.float()) * eps.float()
        z = xn + sample.view(B, T, C, H, W)
        if cfg.n_accum:
            z = torch.cat([z[:, :, : -cfg.n_accum], torch.zeros_like(z[:, :, -cfg.n_accum:])], dim=2)
        return z, mean, logvar

    def decode(self, z, step, hour, doy) -> torch.Tensor:
        """z [B, T, C, H, W] -> next state in normalised space [B, C, H, W]."""
        return self.decoder[0](z.flatten(1, 2).to(self.const.dtype), self.const[None], step, hour, doy)

    def denormalise(self, y: torch.Tensor) -> torch.Tensor:
        y = y.float() * self.std + self.mean
        tp = torch.exp(torch.clamp(y[:, -1:], 0.0, self.cfg.precip_clip)) - 1.0     # log1p-transformed precip
        return torch.cat([y[:, :-1], tp], dim=1)

    def forward(self, x, step, hour, doy, noise: Optional[FuXiENSNoise] = None,
                generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """x: [B, 2, C, H, W] raw physical units (z in m2/s2, t in K, ...); step/hour/doy: [B].

        Returns [B, 2, C, H, W] = concat(x[:, -1:], next state) exactly like the ONNX graph.
        Pass ``noise`` for a deterministic call; otherwise noise is drawn from ``generator``.
        """
        xn = self.normalise(x)
        z, _, _ = self.perturb(xn, step, hour, doy, noise, generator)
        y = self.denormalise(self.decode(z, step, hour, doy))
        return torch.cat([x[:, -1:].to(y.dtype), y[:, None]], dim=1)

    # ---- helpers
    @torch.no_grad()
    def reset_derived_buffers(self):
        """Re-create non-persistent buffers (RoPE tables, shift masks) -- needed after meta-device init."""
        dev = self.const.device
        cfg = self.cfg
        for m in self.modules():
            if isinstance(m, _Backbone):
                n = cfg.rope_table_len or cfg.n_tokens
                cos, sin = make_rope_table(n, cfg.head_dim, cfg.rope_base)
                m.rope_cos, m.rope_sin = cos.to(dev), sin.to(dev)
            if isinstance(m, AdaLNBlock) and m.shifted:
                m.attn_mask = make_shift_mask(cfg.grid, cfg.window, cfg.shift).to(dev)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def FuXiENS_lite(**overrides) -> FuXiENS:
    """Small randomly-initialised FuXi-ENS (same code, ~1M params) for tests, training demos and debugging."""
    return FuXiENS(FuXiENSConfig.lite().replace(**overrides))
