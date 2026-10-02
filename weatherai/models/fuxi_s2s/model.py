"""Native PyTorch FuXi-S2S (reverse-engineered from the official ``fuxi_s2s.onnx``).

FuXi-S2S (Chen et al., *Nature Communications* 2024; weights: Zenodo 10.5281/zenodo.15718402,
**CC-BY-NC-ND-4.0**) is a Swin-V2 U-less transformer that maps two consecutive daily states (76 channels on a
121x240 1.5-degree grid) to the next daily state, with a *stochastic latent* that gives ensemble members.
Only an ONNX graph (9,068 nodes, 576 initialisers, 1.04 B fp16 parameters) is public; the structure below was
read out of it and is checked numerically against onnxruntime (``tests/models/fuxi_s2s``, docs/model_status.md).

One call (``step`` = lead-day index, 0..41)::

    input [B,2,76,121,240]   (x-mean)/std, NaN->0, inf->+-fp32max, precipitation channel zeroed
      | bilinear 121->120 rows, (t,c)->152 channels, 2x2 patch conv, LayerNorm, + MLP(sin/cos(step*f_k))
      v   dist_p: Swin-V2 layer0 (6 blocks) + layer1 (6 blocks), window 10x10 on a 60x120 token grid, shift 5 on odd
      |   blocks, pre-norm, cosine attention (learned logit scale, continuous relative-position bias),
      |   gated-GELU MLP;  fpn(Linear(cat(norm0(L0), norm1(L1))) + GELU)
      |   heads: mean (128), log-variance (128), low-rank factor (128 x 12) per token
      |   z[c,n] = mean + sum_k factor[c,n,k] eps1[c,k] + sqrt(var) eps2[c,n]      <- 2 random draws
      v
    decoder: fpn + z @ sigmoid(alpha[128,1536]) -> 2 Swin-V2 layers (12 blocks each) -> norms -> fpn -> 2-layer head
      | -> 2x2 pixel shuffle -> bilinear 120->121 rows -> + last normalised input frame -> *std + mean
      | precipitation: exp(clip(y,0,7)) - 1
    output [B,2,76,121,240] = concat(input[:, -1], prediction)    (feed back as the next input)

Differences from FuXi-ENS (a different graph): standard biased LayerNorm (eps 1e-6), cosine attention with
relative-position MLP (Swin-V2), no RoPE, no AdaLN, 128-channel latent with low-rank-plus-diagonal Gaussian.
The ONNX computes matmuls in fp16; this module runs in the dtype you give it (fp32 for CPU parity).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FuXiS2SConfig", "FuXiS2SNoise", "FuXiS2S", "make_shift_mask", "make_relative_tables", "CHANNELS"]

LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
CHANNELS = ([f"{v}{l}" for v in "ztuvq" for l in LEVELS]
            + ["t2m", "d2m", "sst", "ttr", "10u", "10v", "100u", "100v", "msl", "tcwv", "tp"])
# official time-encoding frequencies (fp32 values stored in the graph: 10**(-0.8 k), the last one rounded)
FREQS = (1.0, 0.15848933160305023, 0.025118865072727203, 0.0039810724556446075, 0.000630957365501672, 9.99999901978299e-05)


@dataclass
class FuXiS2SConfig:
    in_steps: int = 2
    n_channels: int = 76
    img_size: Tuple[int, int] = (121, 240)
    proc_size: Tuple[int, int] = (120, 240)     # the input is resampled to this before the 2x2 patch conv
    patch: int = 2
    dim: int = 1536
    heads: int = 24
    window: int = 10
    shift: int = 5
    mlp_hidden: int = 4096                      # fc1 outputs 2*mlp_hidden (gated GELU)
    dist_depths: Tuple[int, ...] = (6, 6)
    dec_depths: Tuple[int, ...] = (12, 12)
    latent: int = 128
    rank: int = 12
    cpb_hidden: int = 512
    precip_channel: int = -1

    @classmethod
    def official(cls):
        return cls()

    @classmethod
    def lite(cls, **kw):
        base = dict(n_channels=8, img_size=(33, 64), proc_size=(32, 64), dim=48, heads=4, window=4, shift=2,
                    mlp_hidden=96, dist_depths=(2, 2), dec_depths=(2, 2), latent=8, rank=3, cpb_hidden=16)
        base.update(kw)
        return cls(**base)

    @property
    def grid(self):
        return self.proc_size[0] // self.patch, self.proc_size[1] // self.patch

    @property
    def n_tokens(self):
        g = self.grid
        return g[0] * g[1]


@dataclass
class FuXiS2SNoise:
    """The two latent draws (the ONNX graph samples them with ``RandomNormalLike``)."""
    eps1: torch.Tensor      # (B, latent, rank)
    eps2: torch.Tensor      # (B, latent, n_tokens)


# ----------------------------------------------------------------------------- derived tables
def make_shift_mask(grid, window, shift, device=None):
    """Shifted-window mask, (nW, w*w, w*w) with 0 / -100.0 (equals the stored ``attn_mask``).

    Only the *latitude* roll is masked: longitude is periodic, so tokens that wrap around the date line
    legitimately attend to each other (verified against the stored tensors: a standard Swin mask, which also
    masks the longitude wrap, differs in the 6 right-most windows)."""
    H, W = grid
    img = torch.zeros(1, H, W, 1, device=device)
    for cnt, h in enumerate((slice(0, -window), slice(-window, -shift), slice(-shift, None))):
        img[:, h, :, :] = cnt
    mw = img.reshape(1, H // window, window, W // window, window, 1).permute(0, 1, 3, 2, 4, 5).reshape(-1, window * window)
    m = mw[:, None, :] - mw[:, :, None]
    return m.masked_fill(m != 0, -100.0).masked_fill(m == 0, 0.0)


def make_relative_tables(window):
    """(relative_coords_table (1, 2w-1, 2w-1, 2), relative_position_index (w*w, w*w)) as in Swin-V2.

    The table is the normalised (x8/log2(8+1)-signed-log) coordinate grid; the ONNX stores it as an fp16 tensor."""
    r = torch.arange(-(window - 1), window, dtype=torch.float32)
    t = torch.stack(torch.meshgrid(r, r, indexing="ij"), -1)[None]
    t = t / (window - 1) * 8
    t = torch.sign(t) * torch.log2(t.abs() + 1.0) / math.log2(8)
    c = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij")).flatten(1)
    rel = (c[:, :, None] - c[:, None, :]).permute(1, 2, 0) + (window - 1)
    idx = rel[..., 0] * (2 * window - 1) + rel[..., 1]
    return t, idx


# ----------------------------------------------------------------------------- blocks
class WindowAttention(nn.Module):
    def __init__(self, dim, heads, window, cpb_hidden):
        super().__init__()
        self.dim, self.h, self.w = dim, heads, window
        self.logit_scale = nn.Parameter(torch.zeros(heads, 1, 1))
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.cpb_mlp = nn.Sequential(nn.Linear(2, cpb_hidden), nn.ReLU(), nn.Linear(cpb_hidden, heads, bias=False))
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        t, idx = make_relative_tables(window)
        self.register_buffer("relative_coords_table", t, persistent=False)
        self.register_buffer("relative_position_index", idx, persistent=False)

    def forward(self, x, mask=None):                       # x (nW*B, N, C)
        Bn, N, C = x.shape
        bias = torch.cat([self.q_bias, torch.zeros_like(self.v_bias), self.v_bias])
        qkv = F.linear(x, self.qkv.weight, bias.to(x.dtype)).reshape(Bn, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = F.normalize(q.float(), dim=-1, eps=1e-12).to(x.dtype)
        k = F.normalize(k.float(), dim=-1, eps=1e-12).to(x.dtype)
        attn = (q @ k.transpose(-2, -1)).float()
        attn = attn * self.logit_scale.float().clamp(max=math.log(100.0)).exp()
        tab = self.cpb_mlp(self.relative_coords_table.to(x.dtype)).view(-1, self.h)
        rpb = tab[self.relative_position_index.view(-1)].view(N, N, self.h).permute(2, 0, 1)
        attn = attn + (16 * torch.sigmoid(rpb.float()))[None]
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(Bn // nW, nW, self.h, N, N) + mask[None, :, None].to(attn.dtype)
            attn = attn.view(Bn, self.h, N, N)
        attn = attn.softmax(-1).to(x.dtype)
        out = (attn @ v).transpose(1, 2).reshape(Bn, N, C)
        return self.out_proj(out)


class GatedMLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, 2 * hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        a, b = self.fc1(x).chunk(2, -1)
        return self.fc2(a * F.gelu(b))


class Block(nn.Module):
    def __init__(self, cfg: FuXiS2SConfig, shifted: bool):
        super().__init__()
        self.cfg, self.shifted = cfg, shifted
        self.norm1 = nn.LayerNorm(cfg.dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(cfg.dim, eps=1e-6)
        self.attn = WindowAttention(cfg.dim, cfg.heads, cfg.window, cfg.cpb_hidden)
        self.mlp = GatedMLP(cfg.dim, cfg.mlp_hidden)
        if shifted:
            self.register_buffer("attn_mask", make_shift_mask(cfg.grid, cfg.window, cfg.shift), persistent=False)

    def forward(self, x):                                   # x (B, H*W, C)
        c = self.cfg
        B, L, C = x.shape
        H, W = c.grid
        w, s = c.window, c.shift
        y = self.norm1(x).view(B, H, W, C)
        if self.shifted:
            y = torch.roll(y, (-s, -s), (1, 2))
        y = y.view(B, H // w, w, W // w, w, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, w * w, C)
        y = self.attn(y, self.attn_mask if self.shifted else None)
        y = y.view(B, H // w, W // w, w, w, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        if self.shifted:
            y = torch.roll(y, (s, s), (1, 2))
        x = x + y.reshape(B, L, C)
        return x + self.mlp(self.norm2(x))


class Layer(nn.Module):
    def __init__(self, cfg, depth):
        super().__init__()
        self.blocks = nn.ModuleList([Block(cfg, i % 2 == 1) for i in range(depth)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.proj = nn.Conv2d(cfg.in_steps * cfg.n_channels, cfg.dim, cfg.patch, cfg.patch)
        self.norm = nn.LayerNorm(cfg.dim, eps=1e-6)

    def forward(self, x):                                   # (B, 2C, H, W)
        return self.norm(self.proj(x).flatten(2).transpose(1, 2))


def _fpn(dim):
    return nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU())          # state-dict key ``fpn.0``


class DistP(nn.Module):
    """Encoder + latent-distribution network."""

    def __init__(self, cfg: FuXiS2SConfig):
        super().__init__()
        d = cfg.dim
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        self.time_embed = nn.Sequential(nn.Linear(2 * len(FREQS), d), nn.SiLU(), nn.Linear(d, d))
        self.layers = nn.ModuleList([Layer(cfg, n) for n in cfg.dist_depths])
        self.norm0 = nn.LayerNorm(d, eps=1e-6)
        self.norm1 = nn.LayerNorm(d, eps=1e-6)
        self.fpn = _fpn(d)
        self.mean = nn.Linear(d, cfg.latent)
        self.diag = nn.Linear(d, cfg.latent)
        self.cov = nn.Linear(d, cfg.latent * cfg.rank)
        self.register_buffer("freqs", torch.tensor(FREQS), persistent=False)

    def time_encoding(self, step):                          # (B,) float
        a = step.float()[:, None] * self.freqs.float()[None]
        return torch.cat([a.sin(), a.cos()], -1)

    def forward(self, x, step):
        h = self.patch_embed(x)
        h = h + self.time_embed(self.time_encoding(step).to(h.dtype))[:, None]
        l0 = self.layers[0](h)
        l1 = self.layers[1](l0)
        f = self.fpn(torch.cat([self.norm0(l0), self.norm1(l1)], -1))
        return f, self.mean(f), self.diag(f), self.cov(f), (h, l0, l1)


class Decoder(nn.Module):
    def __init__(self, cfg: FuXiS2SConfig):
        super().__init__()
        d = cfg.dim
        self.cfg = cfg
        self.alpha = nn.Parameter(torch.zeros(cfg.latent, d))
        self.layers = nn.ModuleList([Layer(cfg, n) for n in cfg.dec_depths])
        self.norm0 = nn.LayerNorm(d, eps=1e-6)
        self.norm1 = nn.LayerNorm(d, eps=1e-6)
        self.fpn = _fpn(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, cfg.patch ** 2 * cfg.n_channels))

    def forward(self, base, z):                              # base (B,N,d), z (B,latent,N)
        x = base + z.transpose(1, 2).to(base.dtype) @ torch.sigmoid(self.alpha).to(base.dtype)
        l0 = self.layers[0](x)
        l1 = self.layers[1](l0)
        f = self.fpn(torch.cat([self.norm0(l0), self.norm1(l1)], -1))
        return self.head(f), (x, l0, l1)


class FuXiS2S(nn.Module):
    def __init__(self, cfg: Optional[FuXiS2SConfig] = None):
        super().__init__()
        cfg = cfg or FuXiS2SConfig.official()
        self.cfg = cfg
        self.register_buffer("mean", torch.zeros(cfg.n_channels, 1, 1))
        self.register_buffer("std", torch.ones(cfg.n_channels, 1, 1))
        self.dist_p = DistP(cfg)
        self.decoder = nn.ModuleList([Decoder(cfg)])         # official key prefix ``decoder.0``

    # -- pre/post processing -------------------------------------------------
    def normalise(self, x):
        """(B,2,C,H,W) raw -> normalised input: NaN->0, +-inf -> +-fp32max, precipitation channel := 0."""
        xn = (x.float() - self.mean) / self.std
        xn = torch.nan_to_num(xn, nan=0.0, posinf=3.4028234663852886e38, neginf=-3.4028234663852886e38)
        xn = xn.clone()
        xn[:, :, self.cfg.precip_channel] = 0.0
        return xn

    def sample_noise(self, B, generator=None, device=None, dtype=torch.float32):
        c = self.cfg
        kw = dict(generator=generator, device=device, dtype=dtype)
        return FuXiS2SNoise(torch.randn(B, c.latent, c.rank, **kw), torch.randn(B, c.latent, c.n_tokens, **kw))

    def forward(self, x, step, noise: Optional[FuXiS2SNoise] = None, generator=None, return_taps=False):
        """x (B,2,C,H,W) raw physical units (NaN over land for sst); step (B,) float lead-day index.
        returns (B,2,C,H,W): [:,0] = x[:,-1], [:,1] = forecast one day later."""
        c = self.cfg
        dt = self.dist_p.mean.weight.dtype
        B = x.shape[0]
        xn = self.normalise(x).to(dt)
        # (B,2,C,H,W) -> bilinear to proc_size -> (B, 2C, H', W')
        xr = F.interpolate(xn.float().flatten(0, 1), size=c.proc_size, mode="bilinear", align_corners=False)
        xr = xr.to(dt).reshape(B, c.in_steps * c.n_channels, *c.proc_size)
        f, mean, diag, cov, taps_d = self.dist_p(xr, step)
        if noise is None:
            noise = self.sample_noise(B, generator, x.device)
        N = c.n_tokens
        mean = mean.float().transpose(1, 2)                                  # (B,latent,N)
        var = diag.float().transpose(1, 2).exp()
        fac = cov.float().view(B, N, c.latent, c.rank).permute(0, 2, 1, 3)  # (B,latent,N,rank)
        z = mean + (fac @ noise.eps1.float().unsqueeze(-1).to(fac.dtype)).squeeze(-1) \
            + var.sqrt() * noise.eps2.float()
        raw, taps_s = self.decoder[0](f, z.to(dt))
        h = raw.view(B, *c.grid, c.patch, c.patch, c.n_channels).permute(0, 5, 1, 3, 2, 4)
        h = h.reshape(B, c.n_channels, *c.proc_size).float()
        h = F.interpolate(h, size=c.img_size, mode="bilinear", align_corners=False)
        y = (h + xn[:, -1].to(dt).float()) * self.std + self.mean
        pc = c.precip_channel
        y[:, pc] = torch.exp(y[:, pc].clamp(0.0, 7.0)) - 1.0
        out = torch.stack([x[:, -1].float(), y], 1)
        if return_taps:
            return out, dict(xn=xn, emb=taps_d[0], L0=taps_d[1], L1=taps_d[2], fpn=f, mean=mean, cov=fac, diag=var, z=z,
                             dec_in=taps_s[0], D0=taps_s[1], D1=taps_s[2], head=raw)
        return out


def FuXiS2S_lite(**kw) -> FuXiS2S:
    return FuXiS2S(FuXiS2SConfig.lite(**kw))
