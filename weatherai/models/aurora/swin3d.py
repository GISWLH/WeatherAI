"""Aurora's 3-D Swin-Transformer U-Net backbone, written out explicitly.

Structure, tensor layouts and parameter names follow microsoft/aurora (MIT)
``aurora/model/swin3d.py`` / ``film.py`` (adapted there from Swin-Transformer-v2), so official
checkpoints load with ``strict=True``. Tokens are laid out as ``(B, C*H*W, D)`` where ``C`` is
the (never-halved) latent-level axis and ``H x W`` the patch grid. Differences to upstream:
no LoRA, no stochastic / noise-injection mode, no activation-checkpoint hooks.
"""
from __future__ import annotations

import itertools
from functools import lru_cache
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import DropPath
except ImportError:  # pragma: no cover
    from timm.models.layers import DropPath

from .fourier import lead_time_expansion, lead_time_expansion_v3

Res3 = Tuple[int, int, int]


# ----------------------------------------------------------------------------------------
# helpers: padding, window partition, shifted-window mask
# ----------------------------------------------------------------------------------------
def maybe_adjust_windows(window_size: Res3, shift_size: Res3, res: Res3) -> Tuple[Res3, Res3]:
    """If the input is not larger than the window along an axis, use one window and no shift."""
    ws, ss = list(window_size), list(shift_size)
    for i in range(3):
        if res[i] <= window_size[i]:
            ss[i], ws[i] = 0, res[i]
    return tuple(ws), tuple(ss)


def _two_sided(total: int) -> Tuple[int, int]:
    first = total // 2 if total else 0
    return first, total - first


def pad_3d(x: torch.Tensor, pad: Res3, value: float = 0.0) -> torch.Tensor:
    """Pad ``(B, C, H, W, D)`` on both sides of the C/H/W axes (extra element goes to the end)."""
    (c0, c1), (h0, h1), (w0, w1) = (_two_sided(p) for p in pad)
    return F.pad(x, (0, 0, w0, w1, h0, h1, c0, c1), value=value)


def crop_3d(x: torch.Tensor, pad: Res3) -> torch.Tensor:
    """Undo :func:`pad_3d`."""
    _, C, H, W, _ = x.shape
    (c0, c1), (h0, h1), (w0, w1) = (_two_sided(p) for p in pad)
    return x[:, c0 : C - c1, h0 : H - h1, w0 : W - w1, :]


def window_partition_3d(x: torch.Tensor, ws: Res3) -> torch.Tensor:
    """``(B, C, H, W, D) -> (nW*B, Wc, Wh, Ww, D)`` with windows ordered ``(B, C1, H1, W1)``."""
    B, C, H, W, D = x.shape
    assert C % ws[0] == 0 and H % ws[1] == 0 and W % ws[2] == 0
    x = x.view(B, C // ws[0], ws[0], H // ws[1], ws[1], W // ws[2], ws[2], D)
    return x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(-1, ws[0], ws[1], ws[2], D)


def window_reverse_3d(windows: torch.Tensor, ws: Res3, C: int, H: int, W: int) -> torch.Tensor:
    """Inverse of :func:`window_partition_3d`."""
    C1, H1, W1 = C // ws[0], H // ws[1], W // ws[2]
    B = windows.shape[0] // (C1 * H1 * W1)
    x = windows.reshape(B, C1, H1, W1, ws[0], ws[1], ws[2], -1)
    return x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(B, C, H, W, -1)


def _merge_groups() -> list:
    """Region ids that are joined so that the left and right edges of the globe connect."""
    base = [(1, 2), (4, 5), (7, 8)]
    return [(a + 9 * i, b + 9 * i) for i in range(3) for a, b in base]


@lru_cache(maxsize=32)
def shifted_window_mask(
    C: int, H: int, W: int, ws: Res3, ss: Res3, device: torch.device, dtype: torch.dtype, warped: bool = True
) -> torch.Tensor:
    """Additive attention mask ``(nW, N, N)`` (0 = may attend, -100 = masked) for SW-MSA.

    Regions of the shifted grid get ids; if ``warped`` the east/west regions are identified
    (longitude is periodic); padded positions get a region of their own.
    """
    img = torch.zeros((1, C, H, W, 1), device=device, dtype=dtype)
    sl = lambda w, s: (slice(0, -w), slice(-w, -s), slice(-s, None))  # noqa: E731
    cnt = 0
    for c, h, w in itertools.product(sl(ws[0], ss[0]), sl(ws[1], ss[1]), sl(ws[2], ss[2])):
        img[:, c, h, w, :] = cnt
        cnt += 1
    if warped:
        for g1, g2 in _merge_groups():
            img = img.masked_fill(img == g1, g2)
    pad = tuple((-n) % w for n, w in zip((C, H, W), ws))
    img = pad_3d(img, pad, value=cnt)
    mw = window_partition_3d(img, ws).view(-1, ws[0] * ws[1] * ws[2])
    diff = mw.unsqueeze(1) - mw.unsqueeze(2)
    return diff.masked_fill(diff != 0, -100.0).masked_fill(diff == 0, 0.0)


# ----------------------------------------------------------------------------------------
# layers
# ----------------------------------------------------------------------------------------
class AdaptiveLayerNorm(nn.Module):
    """LayerNorm (no affine) whose scale and shift are predicted from the lead-time embedding."""

    def __init__(self, dim: int, context_dim: int, scale_bias: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.ln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(context_dim, dim * 2))
        self.scale_bias = scale_bias
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.zeros_(self.ln_modulation[-1].weight)
        nn.init.zeros_(self.ln_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        c = self.ln_modulation(c)  # (B, 2D)
        if c.ndim == 2:
            c = c.unsqueeze(1)
        shift, scale = c.chunk(2, dim=-1)
        return self.ln(x) * (self.scale_bias + scale) + shift


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class WindowAttention(nn.Module):
    """Window multi-head self-attention with an optional additive (shifted-window) mask."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim, self.num_heads, self.head_dim = dim, num_heads, dim // num_heads
        self.attn_drop = attn_drop
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``x``: ``(nW*B, N, D)``; ``mask``: ``(nW, N, N)`` or None."""
        Bw, N, D = x.shape
        qkv = self.qkv(x).reshape(Bw, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (nW*B, h, N, d)
        if mask is not None:
            nW = mask.shape[0]
            mask = mask[None, :, None].expand(Bw // nW, -1, -1, -1, -1).reshape(Bw, 1, N, N)
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.attn_drop if self.training else 0.0
        )
        x = x.transpose(1, 2).reshape(Bw, N, D)
        return self.proj_drop(self.proj(x))


class Swin3DBlock(nn.Module):
    """(Shifted-)window attention + MLP, both with adaptive post-norm (conditioned on lead time)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        time_dim: int,
        window_size: Res3,
        shift_size: Res3 = (0, 0, 0),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        scale_bias: float = 0.0,
    ):
        super().__init__()
        self.window_size, self.shift_size = tuple(window_size), tuple(shift_size)
        self.norm1 = AdaptiveLayerNorm(dim, time_dim, scale_bias)
        self.attn = WindowAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = AdaptiveLayerNorm(dim, time_dim, scale_bias)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)

    def forward(self, x: torch.Tensor, c: torch.Tensor, res: Res3, warped: bool = True) -> torch.Tensor:
        C, H, W = res
        B, L, D = x.shape
        assert L == C * H * W, f"Wrong feature size: {L} vs {C}x{H}x{W}"
        ws, ss = maybe_adjust_windows(self.window_size, self.shift_size, res)

        shortcut = x
        x = x.view(B, C, H, W, D)
        shifted = bool(any(ss))
        if shifted:  # cyclic shift
            x = torch.roll(x, shifts=(-ss[0], -ss[1], -ss[2]), dims=(1, 2, 3))
            mask = shifted_window_mask(C, H, W, ws, ss, x.device, x.dtype, warped)
        else:
            mask = None

        pad = ((-C) % ws[0], (-H) % ws[1], (-W) % ws[2])
        x = pad_3d(x, pad)
        Cp, Hp, Wp = x.shape[1:4]
        win = window_partition_3d(x, ws).view(-1, ws[0] * ws[1] * ws[2], D)
        win = self.attn(win, mask)
        x = window_reverse_3d(win.view(-1, *ws, D), ws, Cp, Hp, Wp)
        x = crop_3d(x, pad)
        if shifted:  # reverse cyclic shift
            x = torch.roll(x, shifts=tuple(ss), dims=(1, 2, 3))
        x = x.reshape(B, C * H * W, D)

        x = shortcut + self.drop_path(self.norm1(x, c))  # post-norm residual (Swin v2)
        x = x + self.drop_path(self.norm2(self.mlp(x), c))
        return x


class PatchMerging3D(nn.Module):
    """Down-sample H, W by 2: concat each 2x2 neighbourhood (4D) -> LayerNorm -> Linear(4D, 2D)."""

    def __init__(self, dim: int):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, res: Res3) -> torch.Tensor:
        C, H, W = res
        B, L, D = x.shape
        assert L == C * H * W and H > 1 and W > 1
        x = x.view(B, C, H, W, D)
        x = pad_3d(x, (0, H % 2, W % 2))  # pad to even size
        Hn, Wn = x.shape[2], x.shape[3]
        x = x.reshape(B, C, Hn // 2, 2, Wn // 2, 2, D).permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, C * (Hn // 2) * (Wn // 2), 4 * D)
        return self.reduction(self.norm(x))


class PatchSplitting3D(nn.Module):
    """Up-sample H, W by 2 (Pangu-style ``UpSample``).

    ``lin1`` (D -> 2D) -> reshape/permute to (2H, 2W, D/4) -> crop the padding added by the
    matching :class:`PatchMerging3D` -> LayerNorm -> ``lin2`` (D/2 -> D/2).
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        out = dim // 2
        self.lin1 = nn.Linear(dim, out * 4, bias=False)
        self.lin2 = nn.Linear(out, out, bias=False)
        self.norm = nn.LayerNorm(out)

    def forward(self, x: torch.Tensor, res: Res3, crop: Res3 = (0, 0, 0)) -> torch.Tensor:
        C, H, W = res
        B, L, D = x.shape
        assert L == C * H * W
        x = self.lin1(x)  # (B, CHW, 2D)
        D4 = x.shape[-1] // 4
        x = x.view(B, C, H, W, 2, 2, D4).permute(0, 1, 2, 4, 3, 5, 6).reshape(B, C, 2 * H, 2 * W, D4)
        x = crop_3d(x, crop).reshape(B, -1, D4)
        return self.lin2(self.norm(x))


class SwinStage(nn.Module):
    """``depth`` blocks (alternating plain / shifted windows) + optional down/up-sampling."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Res3,
        time_dim: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Sequence[float] | float = 0.0,
        downsample: bool = False,
        upsample: bool = False,
    ):
        super().__init__()
        assert not (downsample and upsample)
        ws = window_size
        self.blocks = nn.ModuleList(
            [
                Swin3DBlock(
                    dim, num_heads, time_dim, ws,
                    (0, 0, 0) if i % 2 == 0 else (ws[0] // 2, ws[1] // 2, ws[2] // 2),
                    mlp_ratio, qkv_bias, drop, attn_drop,
                    drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                )
                for i in range(depth)
            ]
        )
        self.downsample = PatchMerging3D(dim) if downsample else None
        self.upsample = PatchSplitting3D(dim) if upsample else None

    def forward(self, x, c, res: Res3, crop: Res3 = (0, 0, 0)):
        """Returns ``(x_out, x_before_resampling)``; the latter is the U-Net skip tensor."""
        for blk in self.blocks:
            x = blk(x, c, res)
        if self.downsample is not None:
            return self.downsample(x, res), x
        if self.upsample is not None:
            return self.upsample(x, res, crop), x
        return x, None

    def init_respostnorm(self) -> None:
        for blk in self.blocks:
            blk.norm1.init_weights()
            blk.norm2.init_weights()


class Swin3DBackbone(nn.Module):
    """Encoder stages (patch merging) -> decoder stages (patch splitting) with skips.

    Skips: additive for the intermediate stages, concatenation (-> 2*embed_dim) at the last one.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        encoder_depths: Sequence[int] = (6, 10, 8),
        encoder_num_heads: Sequence[int] = (8, 16, 32),
        decoder_depths: Sequence[int] = (8, 10, 6),
        decoder_num_heads: Sequence[int] = (32, 16, 8),
        window_size: Res3 = (2, 6, 12),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        use_updated_lead_time_embedding: bool = False,
    ):
        super().__init__()
        assert sum(encoder_depths) == sum(decoder_depths)
        self.window_size = tuple(window_size)
        self.n_enc, self.n_dec = len(encoder_depths), len(decoder_depths)
        self.embed_dim = embed_dim
        self.use_updated_lead_time_embedding = use_updated_lead_time_embedding
        self.time_mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(encoder_depths))]
        self.encoder_layers = nn.ModuleList()
        for i in range(self.n_enc):
            self.encoder_layers.append(
                SwinStage(
                    embed_dim * 2**i, encoder_depths[i], encoder_num_heads[i], self.window_size, embed_dim,
                    mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                    dpr[sum(encoder_depths[:i]) : sum(encoder_depths[: i + 1])],
                    downsample=i < self.n_enc - 1,
                )
            )
        self.decoder_layers = nn.ModuleList()
        for i in range(self.n_dec):
            self.decoder_layers.append(
                SwinStage(
                    embed_dim * 2 ** (self.n_dec - i - 1), decoder_depths[i], decoder_num_heads[i], self.window_size,
                    embed_dim, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                    dpr[sum(decoder_depths[:i]) : sum(decoder_depths[: i + 1])],
                    upsample=i < self.n_dec - 1,
                )
            )
        # truncated-normal init of every Linear (as the encoder/decoder), then re-zero the
        # AdaLN modulation layers (which must start as identity).
        self.apply(_init_weights)
        for s in list(self.encoder_layers) + list(self.decoder_layers):
            s.init_respostnorm()

    def encoder_specs(self, patch_res: Res3):
        """Input resolution and output padding of each encoder stage."""
        all_res, padded = [patch_res], []
        for _ in range(1, self.n_enc):
            C, H, W = all_res[-1]
            pH, pW = H % 2, W % 2
            padded.append((0, pH, pW))
            all_res.append((C, (H + pH) // 2, (W + pW) // 2))
        padded.append((0, 0, 0))
        return all_res, padded

    def forward(self, x: torch.Tensor, lead_times: torch.Tensor, patch_res: Res3) -> torch.Tensor:
        """``x``: ``(B, C*H*W, D)`` -> ``(B, C*H*W, 2D)`` (last skip concatenated)."""
        assert x.shape[1] == patch_res[0] * patch_res[1] * patch_res[2]
        assert patch_res[0] % self.window_size[0] == 0
        all_res, padded = self.encoder_specs(patch_res)
        expand = lead_time_expansion_v3 if self.use_updated_lead_time_embedding else lead_time_expansion
        c = self.time_mlp(expand(lead_times, self.embed_dim).to(dtype=x.dtype))

        skips = []
        for i, layer in enumerate(self.encoder_layers):
            x, x_unscaled = layer(x, c, all_res[i])
            skips.append(x_unscaled)
        for i, layer in enumerate(self.decoder_layers):
            index = self.n_dec - i - 1
            x, _ = layer(x, c, all_res[index], padded[index - 1])
            if 0 < i < self.n_dec - 1:
                x = x + skips[index - 1]  # additive skip
            elif i == self.n_dec - 1:
                x = torch.cat([x, skips[0]], dim=-1)  # concatenation skip (like Pangu)
        return x


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.0)
