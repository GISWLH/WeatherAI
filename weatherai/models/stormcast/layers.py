"""Building blocks of the StormCast networks (EDM/DDPM++ "SongUNet" family), written natively.

These reproduce the layer semantics of NVIDIA PhysicsNeMo's ``Conv2d`` / ``GroupNorm`` / ``Linear`` / ``UNetAttention``
(Apache-2.0) with explicit, readable code and *no* dependency on physicsnemo. Parameter / buffer names are identical to the
official checkpoint so that ``load_state_dict(strict=True)`` works after dropping the two empty ``device_buffer`` entries.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class ResampleConv2d(nn.Module):
    """Convolution with optional fused anti-aliased 2x up/down-sampling (EDM ``Conv2d``).

    ``kernel=0`` means "resampling only" (no weight / bias), used by the skip path of down/up blocks.
    Up: depth-wise transposed conv with ``4 * f`` (stride 2) *then* the conv. Down: depth-wise conv with ``f`` (stride 2)
    *then* the conv. ``f`` is the outer product of ``resample_filter`` normalised to sum 1 (stored as buffer
    ``resample_filter`` exactly like the checkpoint).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel: int, bias: bool = True, up: bool = False, down: bool = False,
                 resample_filter: Sequence[float] = (1, 1)):
        super().__init__()
        assert not (up and down)
        self.in_channels, self.out_channels, self.up, self.down = in_channels, out_channels, up, down
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel, kernel)) if kernel else None
        self.bias = nn.Parameter(torch.zeros(out_channels)) if (kernel and bias) else None
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer("resample_filter", f if (up or down) else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = None if self.weight is None else self.weight.to(x.dtype)
        b = None if self.bias is None else self.bias.to(x.dtype)
        f = None if self.resample_filter is None else self.resample_filter.to(x.dtype)
        f_pad = 0 if f is None else (f.shape[-1] - 1) // 2
        if self.up:
            x = F.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
        if self.down:
            x = F.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
        if w is not None:
            x = F.conv2d(x, w, b, padding=w.shape[-1] // 2)
        return x


class GroupNorm(nn.Module):
    """GroupNorm (32 groups, >=4 channels/group) with optional fused SiLU.

    Official quirk reproduced on purpose: in *eval* mode PhysicsNeMo normalises with the **unbiased** variance
    (``Tensor.var`` default, divisor N-1); in *train* mode it uses ``F.group_norm`` (biased). Both paths are kept so that
    inference matches the released model bit-for-bit and training matches the library's training behaviour.
    """

    def __init__(self, num_channels: int, num_groups: int = 32, min_channels_per_group: int = 4, eps: float = 1e-5, act: bool = False):
        super().__init__()
        self.num_groups = min(num_groups, (num_channels + min_channels_per_group - 1) // min_channels_per_group)
        assert num_channels % self.num_groups == 0
        self.eps, self.act = eps, act
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w, b = self.weight.to(x.dtype), self.bias.to(x.dtype)
        if self.training:
            x = F.group_norm(x, self.num_groups, w, b, self.eps)
        else:
            B, C, H, W = x.shape
            g = x.reshape(B, self.num_groups, C // self.num_groups, H, W)
            mean = g.mean(dim=(2, 3, 4), keepdim=True)
            var = g.var(dim=(2, 3, 4), keepdim=True)  # unbiased, as in the official eval path
            g = (g - mean) * (var + self.eps).rsqrt()
            x = g.reshape(B, C, H, W) * w.view(1, C, 1, 1) + b.view(1, C, 1, 1)
        return F.silu(x) if self.act else x


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.weight.to(x.dtype).t()
        return x if self.bias is None else x + self.bias.to(x.dtype)


class PositionalEmbedding(nn.Module):
    """Sinusoidal embedding of the (log-)noise level: ``[cos(t f), sin(t f)]`` with ``f_i = (1/10000)^(i/(C/2-1))``."""

    def __init__(self, num_channels: int, max_positions: int = 10000, endpoint: bool = True):
        super().__init__()
        freqs = torch.arange(num_channels // 2, dtype=torch.float32) / (num_channels // 2 - (1 if endpoint else 0))
        self.register_buffer("freqs", (1.0 / max_positions) ** freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        a = torch.outer(t.to(self.freqs.dtype), self.freqs)
        return torch.cat([a.cos(), a.sin()], dim=1)


class UNetBlock(nn.Module):
    """DDPM++ residual block ("adaptive_scale=False" variant used by StormCast) with optional self-attention.

    ``x -> conv0(silu(norm0 x)) [+up/down] -> +affine(emb) -> silu(norm1) -> dropout -> conv1``; skip = identity or
    1x1 resampling conv; output scaled by ``skip_scale`` (1/sqrt(2)). Attention (``norm2``/``qkv``/``proj``, names kept
    from the checkpoint) is a single-/multi-head spatial self-attention applied after the residual sum.
    """

    def __init__(self, in_channels: int, out_channels: int, emb_channels: int, up: bool = False, down: bool = False,
                 attention: bool = False, num_heads: int = 1, dropout: float = 0.0, skip_scale: float = 0.7071067811865476,
                 eps: float = 1e-6, resample_filter: Sequence[float] = (1, 1)):
        super().__init__()
        self.in_channels, self.out_channels, self.dropout, self.skip_scale = in_channels, out_channels, dropout, skip_scale
        self.num_heads = num_heads if attention else 0
        self.norm0 = GroupNorm(in_channels, eps=eps, act=True)
        self.conv0 = ResampleConv2d(in_channels, out_channels, 3, up=up, down=down, resample_filter=resample_filter)
        self.affine = Linear(emb_channels, out_channels)
        self.norm1 = GroupNorm(out_channels, eps=eps, act=True)
        self.conv1 = ResampleConv2d(out_channels, out_channels, 3)
        self.skip = None
        if out_channels != in_channels or up or down:
            self.skip = ResampleConv2d(in_channels, out_channels, 1, up=up, down=down, resample_filter=resample_filter)
        if attention:
            self.norm2 = GroupNorm(out_channels, eps=eps)
            self.qkv = ResampleConv2d(out_channels, out_channels * 3, 1)
            self.proj = ResampleConv2d(out_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        orig = x
        x = self.conv0(self.norm0(x))
        x = self.norm1(x + self.affine(emb)[:, :, None, None].to(x.dtype))
        x = self.conv1(F.dropout(x, p=self.dropout, training=self.training))
        x = (x + (self.skip(orig) if self.skip is not None else orig)) * self.skip_scale
        if self.num_heads:
            B, C, H, W = x.shape
            qkv = self.qkv(self.norm2(x)).reshape(B, self.num_heads, C // self.num_heads, 3, H * W).permute(0, 1, 4, 3, 2)
            q, k, v = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]
            a = F.scaled_dot_product_attention(q, k, v, scale=1 / math.sqrt(k.shape[-1])).transpose(-1, -2)
            x = (self.proj(a.reshape(B, C, H, W)) + x) * self.skip_scale
        return x
