"""ORCA-DL (Guo et al., Sci. Adv. 11, eadu2488, 2025): data-driven global ocean model for seasonal-to-decadal prediction.

Native, dependency-light PyTorch re-implementation (no ``transformers`` / ``timm`` / ``fairscale`` / ``global_land_mask``).
Parameter names equal the official ``OpenEarthLab/ORCA-DL`` module names, so the official ``seed_N.bin`` state dict loads with
only the deterministic rotary tables dropped (``convert.py`` checks them against the recomputed tables).

Architecture (monthly step, 128x360 grid, lat -63.5..63.5, lon 0.5..359.5):
  * six ocean encoders (so, thetao, tos, uo, vo, zos; 16/16/1/16/16/1 channels) = patch-embed 2x3 + 3 Swin stages (96/192/384 ch);
  * one atmosphere encoder for the wind stress (tauu, tauv), same structure;
  * a fusion module: the encoded wind-stress field gates the ocean latent (``x + <a,x> a`` after a lead-time rotary rotation),
    then 2 global-window Swin blocks on the 1152-channel latent;
  * six decoders (Swin stages + patch expansion + skip concatenation, ConvTranspose2d head);
  * the MLPs of all Swin blocks except the atmosphere encoder are a *mixture of lead-time experts*: one MLP per lead month
    (``max_t`` = 6), selected by ``lead_time % max_t``.
All six lead months are predicted *directly* from the same initial state; every ``max_t`` steps the last prediction becomes the new input.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import accumulate

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ORCADLConfig:
    lat_space: tuple = (-63.5, 63.5, 128)
    lon_space: tuple = (0.5, 359.5, 360)
    patch_size: tuple = (2, 3)
    in_chans: list = field(default_factory=lambda: [16, 16, 1, 16, 16, 1])
    out_chans: list = field(default_factory=lambda: [16, 16, 1, 16, 16, 1])
    embed_dim: int = 96
    lg_hidden_dim: int = 1152
    enc_depths: tuple = (2, 2, 2)
    enc_heads: tuple = (3, 6, 12)
    lg_depths: tuple = (2, 2)
    lg_heads: tuple = (12, 12)
    window_size: tuple = (8, 15)
    mlp_ratio: float = 4.0
    atmo_dims: int = 2
    max_t: int = 6
    layer_norm_eps: float = 1e-5
    var_list: tuple = ("so", "thetao", "tos", "uo", "vo", "zos")

    @property
    def input_shape(self):
        return (self.lat_space[-1], self.lon_space[-1])

    @classmethod
    def official(cls):
        return cls()

    @classmethod
    def from_json(cls, path):
        import json
        d = json.load(open(path))
        keys = cls.__dataclass_fields__.keys()
        d = {k: (tuple(v) if isinstance(v, list) and k not in ("in_chans", "out_chans") else v) for k, v in d.items() if k in keys}
        return cls(**d)


def window_partition(x, ws):
    """(B,H,W,C) -> (B*nW, ws0*ws1, C)."""
    B, H, W, C = x.shape
    x = x.view(B, H // ws[0], ws[0], W // ws[1], ws[1], C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws[0] * ws[1], C)


def window_reverse(w, ws, B, H, W):
    x = w.view(B, H // ws[0], W // ws[1], ws[0], ws[1], -1)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)


def effective_window(hw, window, shift):
    """Swin rule: a window never exceeds the feature map; a window covering the whole map is not shifted."""
    ws, ss = list(window), list(shift)
    for i in range(2):
        if hw[i] <= window[i]:
            ws[i], ss[i] = hw[i], 0
    return tuple(ws), tuple(ss)


def shift_attention_mask(Hp, Wp, ws, ss, device=None):
    """Swin shifted-window mask [nW, N, N]: -100 between tokens that came from different un-shifted regions."""
    img = torch.zeros(1, Hp, Wp, 1, device=device)
    cnt = 0
    for h in (slice(-ws[0]), slice(-ws[0], -ss[0]), slice(-ss[0], None)):
        for w in (slice(-ws[1]), slice(-ws[1], -ss[1]), slice(-ss[1], None)):
            img[:, h, w, :] = cnt
            cnt += 1
    mw = window_partition(img, ws).squeeze(-1)
    diff = mw.unsqueeze(1) - mw.unsqueeze(2)
    return torch.zeros_like(diff).masked_fill(diff != 0, -100.0)


class RotaryPosEmbed2D(nn.Module):
    """2-D rotary position embedding of the query/key inside one attention window (tables are deterministic, not learned)."""

    def __init__(self, shape, dim):
        super().__init__()
        half = dim // 2
        self.d1, self.d2 = half // 2, half - half // 2
        ii, jj = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]), indexing="ij")
        f1 = 10000 ** -(torch.arange(self.d1) / self.d1)
        f2 = 10000 ** -(torch.arange(self.d2) / self.d2)
        s1, s2 = ii[..., None] * f1, jj[..., None] * f2
        for n, v in (("sin1", torch.sin(s1)), ("cos1", torch.cos(s1)), ("sin2", torch.sin(s2)), ("cos2", torch.cos(s2))):
            self.register_buffer(n, v, persistent=False)

    def forward(self, x):                                           # x: (..., h, w, dim)
        x11, x21, x12, x22 = x.split([self.d1, self.d2, self.d1, self.d2], dim=-1)
        return torch.cat([x11 * self.cos1 - x12 * self.sin1, x21 * self.cos2 - x22 * self.sin2,
                          x12 * self.cos1 + x11 * self.sin1, x22 * self.cos2 + x21 * self.sin2], dim=-1)


class WindowAttention(nn.Module):
    def __init__(self, dim, window, heads):
        super().__init__()
        self.window, self.heads, self.hd = tuple(window), heads, dim // heads
        self.scale = self.hd ** -0.5
        self.pos_embed = RotaryPosEmbed2D(self.window, self.hd)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):                                # x: (B*nW, N, C)
        B_, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B_, N, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        q = self.pos_embed(q.reshape(-1, *self.window, self.hd)).reshape(B_, self.heads, N, self.hd)
        k = self.pos_embed(k.reshape(-1, *self.window, self.hd)).reshape(B_, self.heads, N, self.hd)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if mask is not None:
            nW = mask.shape[0]
            attn = (attn.view(B_ // nW, nW, self.heads, N, N) + mask[None, :, None]).view(B_, self.heads, N, N)
        return self.proj((attn.softmax(-1) @ v).transpose(1, 2).reshape(B_, N, C))


class Mlp(nn.Module):
    def __init__(self, dim, ratio):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, int(dim * ratio)), nn.Linear(int(dim * ratio), dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class MlpMoe(nn.Module):
    """One MLP per lead month; sample b uses expert ``lead[b] % max_t``."""

    def __init__(self, dim, ratio, max_t):
        super().__init__()
        self.mlps = nn.ModuleList([Mlp(dim, ratio) for _ in range(max_t)])
        self.max_t = max_t

    def forward(self, x, lead):
        lead = lead % self.max_t
        out = torch.zeros_like(x)
        for i in lead.unique().tolist():
            sel = lead == i
            out[sel] = self.mlps[i](x[sel])
        return out


class SwinBlock(nn.Module):
    """x + attn(norm1(x)) ; x + mlp(norm2(x)) (windowed, optionally cyclically shifted). No stochastic depth at inference."""

    def __init__(self, cfg, window, dim, heads, shift, moe):
        super().__init__()
        self.window, self.shift, self.moe = tuple(window), tuple(shift), moe
        self.norm1 = nn.LayerNorm(dim, eps=cfg.layer_norm_eps)
        self.attn = WindowAttention(dim, window, heads)
        self.norm2 = nn.LayerNorm(dim, eps=cfg.layer_norm_eps)
        self.mlp = MlpMoe(dim, cfg.mlp_ratio, cfg.max_t) if moe else Mlp(dim, cfg.mlp_ratio)

    def forward(self, x, lead):
        B, H, W, C = x.shape
        ws, ss = effective_window((H, W), self.window, self.shift)
        y = self.norm1(x)
        pb, pr = (-H) % ws[0], (-W) % ws[1]
        y = F.pad(y, (0, 0, 0, pr, 0, pb))
        Hp, Wp = y.shape[1:3]
        shifted = any(s > 0 for s in ss)
        mask = None
        if shifted:
            y = torch.roll(y, (-ss[0], -ss[1]), (1, 2))
            mask = shift_attention_mask(Hp, Wp, ws, ss, x.device)
        y = window_reverse(self.attn(window_partition(y, ws), mask), ws, B, Hp, Wp)
        if shifted:
            y = torch.roll(y, ss, (1, 2))
        x = x + y[:, :H, :W]
        z = self.norm2(x)
        return x + (self.mlp(z, lead) if self.moe else self.mlp(z))


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape
        if H % 2 or W % 2:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1)
        return self.reduction(self.norm(x))


class PatchExpanding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x):
        x = self.expand(x)                                           # channels ordered (p2, p1, c); out[h*2+p1, w*2+p2]
        B, H, W, C2 = x.shape
        x = x.view(B, H, W, 2, 2, C2 // 4).permute(0, 1, 4, 2, 3, 5).reshape(B, 2 * H, 2 * W, C2 // 4)
        return self.norm(x)


class Stage(nn.Module):
    """Encoder stage: Swin blocks (alternating shift) then optional patch merging; also returns the pre-merge features (skip)."""

    def __init__(self, cfg, window, dim, depth, heads, moe, down=None):
        super().__init__()
        shift = tuple(w // 2 for w in window)
        self.blocks = nn.ModuleList([SwinBlock(cfg, window, dim, heads, (0, 0) if i % 2 == 0 else shift, moe) for i in range(depth)])
        self.downsample = down(dim) if down is not None else None

    def forward(self, x, lead):
        for b in self.blocks:
            x = b(x, lead)
        return (self.downsample(x) if self.downsample is not None else x), x


class DecoderStage(nn.Module):
    def __init__(self, cfg, window, dim, depth, heads, moe, up=None):
        super().__init__()
        shift = tuple(w // 2 for w in window)
        self.blocks = nn.ModuleList([SwinBlock(cfg, window, dim, heads, (0, 0) if i % 2 == 0 else shift, moe) for i in range(depth)])
        self.upsample = up(dim) if up is not None else None

    def forward(self, x, lead):
        for b in self.blocks:
            x = b(x, lead)
        return self.upsample(x) if self.upsample is not None else x


class PatchEmbed(nn.Module):
    def __init__(self, input_shape, patch, in_chans, dim):
        super().__init__()
        self.patch, self.dim = patch, dim
        self.proj = nn.Conv2d(in_chans, dim, patch, patch)
        self.norm = nn.LayerNorm(dim)
        n = math.ceil(input_shape[0] / patch[0]) * math.ceil(input_shape[1] / patch[1])
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, n, dim))

    def forward(self, x):
        _, _, H, W = x.shape
        x = F.pad(x, (0, (-W) % self.patch[1], 0, (-H) % self.patch[0]))
        x = self.proj(x)
        Wh, Ww = x.shape[2:]
        x = self.norm(x.flatten(2).transpose(1, 2)) + self.absolute_pos_embed
        return x.transpose(1, 2).reshape(-1, self.dim, Wh, Ww)


class Encoder(nn.Module):
    def __init__(self, cfg, in_chans, moe):
        super().__init__()
        self.patch_embed = PatchEmbed(cfg.input_shape, cfg.patch_size, in_chans, cfg.embed_dim)
        n = len(cfg.enc_depths)
        self.stages = nn.ModuleList([Stage(cfg, cfg.window_size, cfg.embed_dim * 2 ** i, cfg.enc_depths[i], cfg.enc_heads[i], moe,
                                           PatchMerging if i < n - 1 else None) for i in range(n)])
        self.norm = nn.LayerNorm(cfg.embed_dim * 2 ** (n - 1), eps=cfg.layer_norm_eps)

    def forward(self, x, lead):
        x = self.patch_embed(x).permute(0, 2, 3, 1)
        skips = []
        for st in self.stages:
            x, before = st(x, lead)
            skips.append(before)
        return self.norm(x), tuple(skips)


class Decoder(nn.Module):
    def __init__(self, cfg, out_chans, moe):
        super().__init__()
        n = len(cfg.enc_depths)
        self.n = n
        self.stages, self.concat_proj_layers = nn.ModuleList(), nn.ModuleList()
        for i in range(n - 1, -1, -1):
            self.stages.append(DecoderStage(cfg, cfg.window_size, cfg.embed_dim * 2 ** i, cfg.enc_depths[i], cfg.enc_heads[i], moe,
                                            PatchExpanding if i > 0 else None))
            if i < n - 1:
                self.concat_proj_layers.append(nn.Linear(cfg.embed_dim * 2 ** i * 2, cfg.embed_dim * 2 ** i))
        self.norm = nn.LayerNorm(cfg.embed_dim, eps=cfg.layer_norm_eps)
        self.final_proj = nn.ConvTranspose2d(cfg.embed_dim, out_chans, cfg.patch_size, cfg.patch_size)

    def forward(self, x, lead, skips):
        for i in range(self.n):
            if i > 0:
                x = self.concat_proj_layers[i - 1](torch.cat([x, skips[self.n - 1 - i]], -1))
            x = self.stages[i](x, lead)
        return self.final_proj(self.norm(x).permute(0, 3, 1, 2))


class OceanEncoders(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.in_chans = list(cfg.in_chans)
        self.encoder_list = nn.ModuleList([Encoder(cfg, c, moe=True) for c in cfg.in_chans])
        self.proj = nn.Linear(cfg.embed_dim * 2 ** (len(cfg.enc_depths) - 1) * len(cfg.in_chans), cfg.lg_hidden_dim)

    def forward(self, x, lead):
        outs = [enc(xi, lead) for enc, xi in zip(self.encoder_list, torch.split(x, self.in_chans, 1))]
        return self.proj(torch.cat([o[0] for o in outs], -1)), [o[1] for o in outs]


class OceanDecoders(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.decoder_list = nn.ModuleList([Decoder(cfg, c, moe=True) for c in cfg.out_chans])
        self.split = cfg.embed_dim * 2 ** (len(cfg.enc_depths) - 1)
        self.proj = nn.Linear(cfg.lg_hidden_dim, self.split * len(cfg.in_chans))

    def forward(self, x, lead, skips):
        parts = torch.split(self.proj(x), self.split, -1)
        return torch.cat([d(p, lead, s) for d, p, s in zip(self.decoder_list, parts, skips)], 1)


class RotaryTimeEmbed(nn.Module):
    """Rotates the (channel-paired) wind-stress latent by an angle ``lead * theta_c``, theta_c = 10000^(2*floor(c/2)/dim)."""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("theta", 10000 ** (torch.arange(dim).div(2, rounding_mode="floor") * 2 / dim))

    def forward(self, x, t):                                         # x: (B,C,H,W), t: (B,)
        ang = torch.einsum("i,j->ij", t.to(self.theta.dtype), self.theta)
        cos, sin = torch.cos(ang)[:, :, None, None], torch.sin(ang)[:, :, None, None]
        x_rot = torch.stack([-x[:, 1::2], x[:, ::2]], dim=1).reshape_as(x)
        return x * cos + x_rot * sin


class Fusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n = len(cfg.enc_depths)
        win = (cfg.input_shape[0] // cfg.patch_size[0] // 2 ** (n - 1), cfg.input_shape[1] // cfg.patch_size[1] // 2 ** (n - 1))
        self.stages = nn.ModuleList([Stage(cfg, win, cfg.lg_hidden_dim, cfg.lg_depths[i], cfg.lg_heads[i], moe=True) for i in range(len(cfg.lg_depths))])
        self.pos_embed = nn.Parameter(torch.zeros(1, win[0] * win[1], cfg.lg_hidden_dim))
        self.rotate_atmo = RotaryTimeEmbed(cfg.lg_hidden_dim)

    def forward(self, x, atmo, lead):
        B, H, W, C = x.shape
        a = self.rotate_atmo(atmo.permute(0, 3, 1, 2), lead).permute(0, 2, 3, 1)
        x = x + (a * x).sum(-1, keepdim=True) * a                    # wind-stress-gated residual
        x = (x.reshape(B, -1, C) + self.pos_embed).reshape(B, H, W, C)
        for st in self.stages:
            x, _ = st(x, lead)
        return x


class AtmoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(cfg, cfg.atmo_dims, moe=False)
        self.proj = nn.Linear(cfg.embed_dim * 2 ** (len(cfg.enc_depths) - 1), cfg.lg_hidden_dim)

    def forward(self, x, lead=None):
        return self.proj(self.encoder(x, lead)[0])


class ORCADL(nn.Module):
    def __init__(self, cfg: ORCADLConfig | None = None):
        super().__init__()
        self.cfg = cfg or ORCADLConfig()
        self.enc_ocean, self.fusion = OceanEncoders(self.cfg), Fusion(self.cfg)
        self.dec_ocean, self.enc_atmo = OceanDecoders(self.cfg), AtmoEncoder(self.cfg)
        self.register_buffer("land_mask", torch.ones(self.cfg.input_shape, dtype=torch.bool))   # True = ocean; the checkpoint carries the real mask
        self.split_chans = list(accumulate(self.cfg.out_chans))
        self.max_t = self.cfg.max_t

    def single_step(self, ocean, atmo_latent, lead):
        x, skips = self.enc_ocean(ocean, lead)
        x = self.fusion(x, atmo_latent, lead)
        return self.dec_ocean(x, lead, skips)

    def forward(self, ocean_vars, atmo_vars, predict_time_steps: int = 1):
        """ocean_vars (B,66,128,360) and atmo_vars (B,2,128,360) are normalised; returns (B, steps, 66, 128, 360) normalised predictions
        of months +1..+steps. Direct multi-lead prediction; the input is replaced by the prediction of lead ``max_t`` every ``max_t`` steps;
        the wind-stress input is held at its initial value."""
        B = ocean_vars.shape[0]
        atmo_latent = self.enc_atmo(atmo_vars)
        outs = []
        for t in range(predict_time_steps):
            lead = torch.full((B,), t, dtype=torch.long, device=ocean_vars.device)
            outs.append(self.single_step(ocean_vars, atmo_latent, lead))
            if (t + 1) % self.max_t == 0:
                ocean_vars = outs[-1]
        return torch.stack(outs, 1)
