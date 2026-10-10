"""CAS-Canglong: 3-D Earth-specific Transformer for sub-seasonal to seasonal global SST prediction.

Native port of the official code in GISWLH/CAS-Canglong (``notebook/model.ipynb`` + ``utils/*.py``, MIT).
Parameter names equal the official ``Canglong`` module, so an official ``state_dict`` strict-loads.

Input: 16 monthly frames of 8 normalised variables on the 0.25° ERA5 grid, ``(B, 8, 16, 721, 1440)``
(channel 0 = SST, land set to 0 before normalisation; channel 7 = net solar radiation in the notebooks).
Output: ``(B, 1, 3, 721, 1440)`` = normalised SST for the next 3 months.

The transformer body (``BasicLayer``/``DownSample``/``UpSample``) is the Pangu code already in
``weatherai.models.pangu`` (identical maths and parameter names); only the patch embedding / recovery
modules differ in naming (``projection``/``transposed_conv``) and are defined here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ..pangu.pangu import BasicLayer, DownSample, UpSample


def _centre_pad(size, patch):
    """(before, after) zero padding per axis so that ``size`` becomes a multiple of ``patch``."""
    out = []
    for s, p in zip(size, patch):
        r = (p - s % p) % p
        out.append((r // 2, r - r // 2))
    return out


def _centre_crop_slices(have, want):
    """Official centre crop: ``x[d//2 : n - (d - d//2)]`` with ``d = have - want`` (``d`` may be negative)."""
    sl = []
    for n, m in zip(have, want):
        d = n - m
        sl.append(slice(d // 2, n - (d - d // 2)))
    return sl


class ImageToPatch2D(nn.Module):
    """2-D patch embedding (official ``canglong.embed.ImageToPatch2D``)."""

    def __init__(self, img_dims, patch_dims, in_channels, out_channels):
        super().__init__()
        self.img_dims = tuple(img_dims)
        (pt, pb), (pl, pr) = _centre_pad(img_dims, patch_dims)
        self.padding = nn.ZeroPad2d((pl, pr, pt, pb))
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=patch_dims, stride=patch_dims)

    def forward(self, x):
        assert tuple(x.shape[-2:]) == self.img_dims, f"expected {self.img_dims}, got {tuple(x.shape[-2:])}"
        return self.projection(self.padding(x))


class ImageToPatch3D(nn.Module):
    """3-D patch embedding over (time/level, lat, lon) (official ``canglong.embed.ImageToPatch3D``)."""

    def __init__(self, img_dims, patch_dims, in_channels, out_channels):
        super().__init__()
        self.img_dims = tuple(img_dims)
        (pf, pk), (pt, pb), (pl, pr) = _centre_pad(img_dims, patch_dims)
        self.padding = nn.ZeroPad3d((pl, pr, pt, pb, pf, pk))
        self.projection = nn.Conv3d(in_channels, out_channels, kernel_size=patch_dims, stride=patch_dims)

    def forward(self, x):
        assert tuple(x.shape[-3:]) == self.img_dims, f"expected {self.img_dims}, got {tuple(x.shape[-3:])}"
        return self.projection(self.padding(x))


class RecoveryImage2D(nn.Module):
    """Transposed-conv patch recovery + centre crop (official ``canglong.recovery.RecoveryImage2D``)."""

    def __init__(self, image_size, patch_size, input_channels, output_channels):
        super().__init__()
        self.image_size = tuple(image_size)
        self.transposed_conv = nn.ConvTranspose2d(input_channels, output_channels, patch_size, stride=patch_size)

    def forward(self, x):
        x = self.transposed_conv(x)
        return x[(...,) + tuple(_centre_crop_slices(x.shape[-2:], self.image_size))]


class RecoveryImage3D(nn.Module):
    """Transposed-conv patch recovery + centre crop (official ``canglong.recovery.RecoveryImage3D``)."""

    def __init__(self, image_size, patch_size, input_channels, output_channels):
        super().__init__()
        self.image_size = tuple(image_size)
        self.transposed_conv = nn.ConvTranspose3d(input_channels, output_channels, patch_size, stride=patch_size)

    def forward(self, x):
        x = self.transposed_conv(x)
        return x[(...,) + tuple(_centre_crop_slices(x.shape[-3:], self.image_size))]


@dataclass
class CanglongConfig:
    img_size: tuple = (721, 1440)          # (lat, lon)
    n_months: int = 16                     # input frames
    in_chans: int = 8                      # variables per frame (channel 0 = SST)
    embed_dim: int = 72
    depths: tuple = (2, 6, 6, 2)
    num_heads: tuple = (6, 12, 12, 6)
    window_size: tuple = (2, 6, 12)        # (time, lat, lon) tokens; the official block hard-codes the shift (1, 3, 6) = window // 2
    patch_size: tuple = (2, 4, 4)          # input embedding patch
    recovery_patch: tuple = (1, 8, 8)      # official output patch (2x the embedding patch, see `forward`)
    recovery_depth: int = 13               # official crop depth; with 8 time tokens it keeps the last 3
    drop_path_rate: float = 0.2
    legacy_heads: bool = True              # official modules whose outputs are discarded; kept so checkpoints strict-load
    extra: dict = field(default_factory=dict)

    @classmethod
    def official(cls, **kw):
        return cls(**kw)

    @classmethod
    def lite(cls, **kw):
        """CPU-sized config with the same code path: 8 months x 8 vars on 32x64, 3 output months."""
        d = dict(img_size=(32, 64), n_months=8, embed_dim=16, depths=(2, 2, 2, 2), num_heads=(2, 4, 4, 2),
                 recovery_depth=9, legacy_heads=False)
        d.update(kw)
        return cls(**d)

    @property
    def token_grid(self):
        return tuple(-(-s // p) for s, p in zip((self.n_months,) + tuple(self.img_size), self.patch_size))

    @property
    def coarse_grid(self):
        t, h, w = self.token_grid
        return t, -(-h // 2), -(-w // 2)

    @property
    def out_months(self):
        t = self.token_grid[0] * self.recovery_patch[0]
        return len(range(t)[_centre_crop_slices((t,), (self.recovery_depth,))[0]])


class Canglong(nn.Module):
    """CAS-Canglong (official layout: Pangu-style 3-D Earth-specific transformer, U-shaped, skip concat)."""

    def __init__(self, cfg: CanglongConfig | None = None):
        super().__init__()
        cfg = cfg or CanglongConfig.official()
        self.cfg = cfg
        C, (H, W) = cfg.embed_dim, cfg.img_size
        fine, coarse = cfg.token_grid, cfg.coarse_grid
        d1, d2, d3, d4 = cfg.depths
        o = max(d1, d4)                                               # official: linspace(0, 0.2, 8); stages 2/3 share dp[2:]
        dp = np.linspace(0, cfg.drop_path_rate, o + max(d2, d3)).tolist()
        if cfg.legacy_heads:          # never called (embeddings) or computed and discarded (recoveries) by the official forward
            self.patchembed2d = ImageToPatch2D(cfg.img_size, cfg.patch_size[1:], 4 + 3, C)
            self.patchembed3d = ImageToPatch3D((13, H, W), cfg.patch_size, 5, C)
        self.patchembed_wlh = ImageToPatch3D((cfg.n_months, H, W), cfg.patch_size, cfg.in_chans, C)
        self.layer1 = BasicLayer(C, fine, d1, cfg.num_heads[0], cfg.window_size, drop_path=dp[:d1])
        self.downsample = DownSample(C, fine, coarse)
        self.layer2 = BasicLayer(2 * C, coarse, d2, cfg.num_heads[1], cfg.window_size, drop_path=dp[o:o + d2])
        self.layer3 = BasicLayer(2 * C, coarse, d3, cfg.num_heads[2], cfg.window_size, drop_path=dp[o:o + d3])
        self.upsample = UpSample(2 * C, C, coarse, fine)
        self.layer4 = BasicLayer(C, fine, d4, cfg.num_heads[3], cfg.window_size, drop_path=dp[:d4])
        if cfg.legacy_heads:
            self.patchrecovery2d = RecoveryImage2D(cfg.img_size, (8, 8), 2 * C, 4)
            self.patchrecovery3d = RecoveryImage3D((13, H, W), (2, 8, 8), 2 * C, 5)
        self.patchrecovery3d_wlh = RecoveryImage3D((cfg.recovery_depth, H, W), cfg.recovery_patch, 2 * C, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_chans, n_months, H, W) normalised -> (B, 1, out_months, H, W) normalised SST.

        As in the official code the output patch (1, 8, 8) is twice the embedding patch (4, 4), and the
        recovered (8*lat_tokens, 8*lon_tokens) field is centre-cropped to (H, W): output pixels are decoded
        from the central half of the token grid in each direction (rows 45-135 of 181, cols 90-269 of 360).
        The time axis keeps the last 3 of the 8 time tokens. Both are reproduced on purpose.
        """
        x = self.patchembed_wlh(x)
        B, C, T, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)
        x = self.layer1(x)
        skip = x
        x = self.downsample(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.upsample(x)
        x = self.layer4(x)
        x = torch.cat([x, skip], dim=-1).transpose(1, 2).reshape(B, -1, T, Lat, Lon)
        return self.patchrecovery3d_wlh(x)


def Canglong_lite(**kw) -> Canglong:
    return Canglong(CanglongConfig.lite(**kw))


def penalty_box(img_size=(721, 1440)):
    """Index box of the official correlation penalty: rows 288:336, cols 480:640 on 721x1440, scaled to ``img_size``.

    The notebook calls it ``nino``; on the 90N->90S, 0->359.75E ERA5 grid it covers 18N-6N, 120E-160E
    (not the Nino-3.4 region). Kept as in the official code."""
    h, w = img_size
    return slice(round(288 * h / 721), round(336 * h / 721)), slice(round(480 * w / 1440), round(640 * w / 1440))


def canglong_loss(output, target, spatial_weights=None, time_weights=(1.0, 1.5, 2.0), box=None, corr_floor=0.5):
    """Official training loss: lead-weighted MSE + pattern-correlation penalty on ``penalty_box``.

    output, target: (leads, H, W) normalised SST (the notebook squeezes batch=1 and the channel axis).
    As in the notebook the lead weights multiply inside the square, and the penalty is
    ``sum(clamp(corr_floor - r_lead, 0)) / leads`` over the box.
    """
    n = output.shape[0]
    tw = torch.as_tensor(time_weights, dtype=output.dtype, device=output.device).view(n, 1, 1)
    sw = 1.0 if spatial_weights is None else spatial_weights
    mse = torch.mean(sw * (output * tw - target * tw) ** 2)
    rows, cols = box if box is not None else penalty_box(output.shape[-2:])
    o, t = output[:, rows, cols], target[:, rows, cols]
    om, tm = o.mean(dim=(1, 2), keepdim=True), t.mean(dim=(1, 2), keepdim=True)
    cov = ((o - om) * (t - tm)).mean(dim=(1, 2))
    r = cov / (o.std(dim=(1, 2), unbiased=False) * t.std(dim=(1, 2), unbiased=False))
    return mse + torch.clamp(corr_floor - r, min=0.0).sum() / n
