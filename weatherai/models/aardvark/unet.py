"""Aardvark's decoder backbone: a 4-level U-Net with cylindrical boundary handling.

Explicit port of the official ``aardvark/unet_wrap_padding.py`` (CC0): ``CylindricalConv2D`` /
``CylindricalConvTranspose2D`` (zero padding along dim -2, wrap-around padding along dim -1 --
exactly as the official code applies it to its ``(B, C, 240, 121)`` tensors), ``Down`` (conv-BN-GELU
twice, optional stride-2), ``Up`` (transposed conv + ``Down`` block + skip concat) and ``Unet``.
Parameter names equal the official ones so the official downscaling-decoder checkpoints load
with ``strict=True``. FiLM conditioning and the attention option of the official code are not
ported (the released checkpoints use neither).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def cylindrical_conv_pad(x: torch.Tensor, w_pad: int) -> torch.Tensor:
    return torch.cat([x[..., -w_pad:], x, x[..., :w_pad]], dim=-1)


class CylindricalConv2D(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__(in_channels, out_channels, kernel_size, stride)
        assert self.kernel_size[0] % 2 == 1 and self.kernel_size[1] % 2 == 1
        self.h_pad = self.kernel_size[0] // 2
        self.w_pad = self.kernel_size[1] // 2

    def forward(self, x):
        x = F.pad(x, (0, 0, self.h_pad, self.h_pad))
        return super().forward(cylindrical_conv_pad(x, self.w_pad))


class CylindricalConvTranspose2D(nn.ConvTranspose2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__(in_channels, out_channels, kernel_size, stride)
        assert self.kernel_size[0] % 2 == 1 and self.kernel_size[1] % 2 == 1
        self.sh, self.sw = self.stride
        self.kh, self.kw = self.kernel_size
        self.h_pad = math.ceil(((self.sh - 1) + 2 * (self.kh // 2)) / self.sh)
        self.w_pad = math.ceil(((self.sw - 1) + 2 * (self.kw // 2)) / self.sw)
        self.h0 = self.sh * self.h_pad - (self.sh - 1) + (self.kh // 2)
        self.w0 = self.sw * self.w_pad - (self.sw - 1) + (self.kw // 2)
        self._bias = nn.Parameter(1e-3 * torch.randn(out_channels))  # present (unused) in official checkpoints

    def forward(self, x):
        Nh, Nw = x.shape[2] * self.sh, x.shape[3] * self.sw
        x = cylindrical_conv_pad(x, self.w_pad)
        x = F.pad(x, (0, 0, self.h_pad, self.h_pad))
        x = super().forward(x)
        return x[:, :, self.h0 : self.h0 + Nh, self.w0 : self.w0 + Nw]


class Down(nn.Module):
    """conv_1 (s=1) -> BN -> GELU -> conv_2 (s=2 if ``down`` else 1) -> BN -> GELU."""

    def __init__(self, in_channels: int, out_channels: int, down: bool = True):
        super().__init__()
        self.conv_1 = CylindricalConv2D(in_channels, out_channels, 3, 1)
        self.conv_2 = CylindricalConv2D(out_channels, out_channels, 3, 2 if down else 1)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.activation(self.bn_1(self.conv_1(x)))
        return self.activation(self.bn_2(self.conv_2(x)))


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = CylindricalConvTranspose2D(in_channels, out_channels, 3, 2)
        self.conv = Down(out_channels, out_channels, down=False)

    def forward(self, x1, x2):
        x1 = self.conv(self.up(x1))
        if x1.shape[-1] != x2.shape[-1]:
            x1 = x1[..., :, :-1]
        if x1.shape[-2] != x2.shape[-2]:
            x1 = x1[..., :-1, :]
        return torch.cat([x2, x1], dim=1)


class Unet(nn.Module):
    """``(B, C_in, H, W) -> (B, H, W, C_out)`` (channels last, as the official code)."""

    def __init__(self, in_channels: int, out_channels: int, div_factor: int = 1):
        super().__init__()
        d = div_factor
        self.n_channels = in_channels
        self.variances = nn.Parameter(torch.zeros([out_channels]))  # unused by forward, kept for checkpoints
        self.down1 = Down(in_channels, 128 // d)
        self.down2 = Down(128 // d, 256 // d)
        self.down3 = Down(256 // d, 512 // d)
        self.down4 = Down(512 // d, 512 // d)
        self.up1 = Up(512 // d, 512 // d)
        self.up2 = Up(1024 // d, 256 // d)
        self.up3 = Up(512 // d, 128 // d)
        self.up4 = Up(256 // d, 64 // d)
        self.out = nn.Conv2d(64 // d + in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1 = x.contiguous()
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x).permute(0, 2, 3, 1)
