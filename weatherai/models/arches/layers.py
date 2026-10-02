"""Building blocks of ArchesWeather / ArchesWeatherGen: Pangu-style 3-D earth-specific window attention with adaLN conditioning.

Written natively from the published architecture; the geoarches package (INRIA, BSD-3) is used only as the numerical oracle in
the parity tests. Parameter / buffer names match the official checkpoints so ``load_state_dict(strict=True)`` works.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def earth_position_index(window_size):
    """Index into the (symmetric) earth-position bias table for every pair of tokens inside one window (Pangu-Weather)."""
    wp, wh, ww = window_size
    zi, zj = torch.arange(wp), -torch.arange(wp) * wp
    hi, hj = torch.arange(wh), -torch.arange(wh) * wh
    w = torch.arange(ww)
    c1 = torch.stack(torch.meshgrid([zi, hi, w], indexing="ij")).flatten(1)
    c2 = torch.stack(torch.meshgrid([zj, hj, w], indexing="ij")).flatten(1)
    c = (c1[:, :, None] - c2[:, None, :]).permute(1, 2, 0).contiguous()
    c[:, :, 2] += ww - 1
    c[:, :, 1] *= 2 * ww - 1
    c[:, :, 0] *= (2 * ww - 1) * wh * wh
    return c.sum(-1)


def pad3d_amounts(res, win):
    """(left, right, top, bottom, front, back) zero padding that makes (Pl, Lat, Lon) divisible by the window."""
    out = []
    for r, w in zip(res[::-1], win[::-1]):  # lon, lat, pl  -> (left,right),(top,bottom),(front,back)
        rem = r % w
        p = (w - rem) if rem else 0
        out += [p // 2, p - p // 2]
    return tuple(out)


def window_partition(x, win):
    B, Pl, Lat, Lon, C = x.shape
    wp, wh, ww = win
    x = x.view(B, Pl // wp, wp, Lat // wh, wh, Lon // ww, ww, C)
    return x.permute(0, 5, 1, 3, 2, 4, 6, 7).contiguous().view(-1, (Pl // wp) * (Lat // wh), wp, wh, ww, C)


def window_reverse(w, win, Pl, Lat, Lon):
    wp, wh, ww = win
    B = int(w.shape[0] / (Lon / ww))
    x = w.view(B, Lon // ww, Pl // wp, Lat // wh, wp, wh, ww, -1)
    return x.permute(0, 2, 4, 3, 5, 1, 6, 7).contiguous().view(B, Pl, Lat, Lon, -1)


class EarthAttention3D(nn.Module):
    """Window attention with a learned earth-position bias (one bias table per window *row*, i.e. per latitude/level band)."""

    def __init__(self, dim, input_resolution, window_size, num_heads):
        super().__init__()
        self.dim, self.window_size, self.num_heads = dim, window_size, num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.type_of_windows = (input_resolution[0] // window_size[0]) * (input_resolution[1] // window_size[1])
        wp, wh, ww = window_size
        self.earth_position_bias_table = nn.Parameter(torch.zeros((wp ** 2) * (wh ** 2) * (ww * 2 - 1), self.type_of_windows, num_heads))
        self.register_buffer("earth_position_index", earth_position_index(window_size))
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.earth_position_bias_table, std=0.02)

    def forward(self, x):  # x: (B*num_lon, num_pl*num_lat, N, C)
        B_, nW, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B_, nW, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.earth_position_bias_table[self.earth_position_index.view(-1)].view(N, N, self.type_of_windows, -1)
        attn = (attn + bias.permute(3, 2, 0, 1).unsqueeze(0)).softmax(-1)
        return self.proj((attn @ v).permute(0, 2, 3, 1, 4).reshape(B_, nW, N, C))


class SwiGLU(nn.Module):
    """timm-compatible gated MLP (GELU gate) (``fc1_g``/``fc1_x``/``fc2``)."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1_g, self.fc1_x, self.fc2 = nn.Linear(dim, hidden), nn.Linear(dim, hidden), nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1_g(x)) * self.fc1_x(x))  # the official config passes act_layer=GELU to timm SwiGLU


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _AxialSelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.dh = heads, dim // heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):  # (N, T, D)
        q, (k, v) = self.to_q(x), self.to_kv(x).chunk(2, dim=-1)
        n, t, d = q.shape
        q, k, v = (a.reshape(n, t, self.heads, self.dh).transpose(1, 2) for a in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, scale=self.dh ** -0.5)
        return self.to_out(o.transpose(1, 2).reshape(n, t, d))


class _AxialAxis(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.fn = _AxialSelfAttention(dim, heads)

    def forward(self, x):
        return self.fn(x)


class VerticalAxialAttention(nn.Module):
    """Self-attention along the vertical (pressure-level) axis only, with a learned level embedding (``axis_pos.param_0``).

    Names mirror lucidrains' ``axial_attention`` (used by the official code) so the checkpoints load strictly."""

    def __init__(self, dim, n_levels, heads=8):
        super().__init__()
        self.axis_pos = nn.Module()
        self.axis_pos.param_0 = nn.Parameter(torch.randn(1, n_levels, dim))
        self.axis_attn = nn.Module()
        self.axis_attn.axial_attentions = nn.ModuleList([_AxialAxis(dim, heads)])

    def forward(self, x):  # x: (N, Pl, C)
        return self.axis_attn.axial_attentions[0](x + self.axis_pos.param_0)


def _drop_path(x, p, training):
    if p == 0.0 or not training:
        return x
    keep = 1 - p
    m = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
    return x * m / keep


class EarthSpecificBlock(nn.Module):
    """adaLN-modulated (shift/scale/gate) window-attention block + vertical axial attention + (Swi)GLU MLP."""

    def __init__(self, dim, input_resolution, num_heads, window_size, shift_size=(1, 3, 5), mlp_ratio=4.0, drop_path=0.0,
                 roll=False, axis_attn=True, swiglu=True):
        super().__init__()
        self.dim, self.input_resolution, self.window_size, self.shift_size = dim, tuple(input_resolution), tuple(window_size), tuple(shift_size)
        self.roll, self.drop_path = roll, drop_path
        self.norm1 = nn.LayerNorm(dim)
        self.padding = pad3d_amounts(self.input_resolution, self.window_size)
        pr = list(self.input_resolution)
        pr[0] += self.padding[4] + self.padding[5]
        pr[1] += self.padding[2] + self.padding[3]
        pr[2] += self.padding[0] + self.padding[1]
        self.attn = EarthAttention3D(dim, pr, self.window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwiGLU(dim, round(dim * mlp_ratio * 2 / 3)) if swiglu else Mlp(dim, int(dim * mlp_ratio))
        if axis_attn:
            vert = VerticalAxialAttention(dim, self.input_resolution[0])
            self.axis_pos, self.axis_attn = vert.axis_pos, vert.axis_attn  # flat names: blocks.N.axis_pos / axis_attn

    def _window_attention(self, x):
        B, L, C = x.shape
        Pl, Lat, Lon = self.input_resolution
        x = x.view(B, Pl, Lat, Lon, C)
        x = F.pad(x.permute(0, 4, 1, 2, 3), self.padding).permute(0, 2, 3, 4, 1)  # zero-pad (lon, lat, pl)
        _, Pp, Hp, Wp, _ = x.shape
        sp, sh, sw = self.shift_size
        if self.roll:
            x = torch.roll(x, shifts=(-sp, -sh, -sw), dims=(1, 2, 3))
        w = window_partition(x, self.window_size)
        w = w.view(w.shape[0], w.shape[1], -1, C)
        w = self.attn(w).view(w.shape[0], w.shape[1], *self.window_size, C)
        x = window_reverse(w, self.window_size, Pp, Hp, Wp)
        if self.roll:
            x = torch.roll(x, shifts=(sp, sh, sw), dims=(1, 2, 3))
        l, r, t, b, f, bk = self.padding  # crop the padding back
        x = x[:, f:Pp - bk, t:Hp - b, l:Wp - r, :]
        return x.reshape(B, Pl * Lat * Lon, C)

    def forward(self, x, c):
        Pl, Lat, Lon = self.input_resolution
        B, L, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = c.chunk(6, dim=1)
        shortcut = x
        h = self.norm1(x) * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        h = self._window_attention(h)
        if hasattr(self, "axis_attn"):  # attention across the Pl vertical positions of every column
            x2 = h.reshape(B, Pl, Lat * Lon, C).movedim(2, 1).flatten(0, 1)  # (B*Lat*Lon, Pl, C)  (uses the attn output, as official)
            x2 = self.axis_attn.axial_attentions[0](x2 + self.axis_pos.param_0)
            h = h + x2.reshape(B, Lat * Lon, Pl, C).movedim(1, 2).flatten(1, 2)
        x = shortcut + gate_msa[:, None, :] * _drop_path(h, self.drop_path, self.training)
        m = self.mlp(self.norm2(x) * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :])
        return x + _drop_path(gate_mlp[:, None, :] * m, self.drop_path, self.training)


class CondBasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, cond_dim, drop_path, mlp_ratio=4.0, axis_attn=True, swiglu=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            EarthSpecificBlock(dim, input_resolution, num_heads, window_size, mlp_ratio=mlp_ratio, drop_path=drop_path[i], roll=bool(i % 2),
                               axis_attn=axis_attn, swiglu=swiglu) for i in range(depth)])
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim))

    def forward(self, x, cond_emb):
        c = self.adaLN_modulation(cond_emb)
        for blk in self.blocks:
            x = blk(x, c)
        return x


class LinVert(nn.Module):
    """Residual linear layer mixing the whole vertical column (levels x channels) at every horizontal position."""

    def __init__(self, in_features, zdim):
        super().__init__()
        self.zdim = zdim
        self.fc1 = nn.Linear(zdim * in_features, zdim * in_features)

    def forward(self, x):
        B, N, C = x.shape
        x2 = x.reshape(B, self.zdim, -1, C).movedim(1, -2).flatten(-2, -1)
        x2 = self.fc1(x2).reshape(B, -1, self.zdim, C).movedim(-2, 1).flatten(1, 2)
        return x + x2


class DownSample(nn.Module):
    def __init__(self, in_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear = nn.Linear(in_dim * 4, in_dim * 2, bias=False)
        self.norm = nn.LayerNorm(4 * in_dim)
        self.input_resolution, self.output_resolution = input_resolution, output_resolution
        hp, wp = output_resolution[1] * 2 - input_resolution[1], output_resolution[2] * 2 - input_resolution[2]
        self.pad = (wp // 2, wp - wp // 2, hp // 2, hp - hp // 2, 0, 0)

    def forward(self, x):
        B, N, C = x.shape
        pl, lat, lon = self.input_resolution
        _, ol, ow = self.output_resolution[0], self.output_resolution[1], self.output_resolution[2]
        x = x.reshape(B, pl, lat, lon, C)
        x = F.pad(x.permute(0, 4, 1, 2, 3), self.pad).permute(0, 2, 3, 4, 1)
        x = x.reshape(B, pl, ol, 2, ow, 2, C).permute(0, 1, 2, 4, 3, 5, 6).reshape(B, pl * ol * ow, 4 * C)
        return self.linear(self.norm(x))


class UpSample(nn.Module):
    def __init__(self, in_dim, out_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.input_resolution, self.output_resolution = input_resolution, output_resolution

    def forward(self, x):
        B, N, C = x.shape
        pl, lat, lon = self.input_resolution
        _, ol, ow = self.output_resolution
        x = self.linear1(x).reshape(B, pl, lat, lon, 2, 2, C // 2).permute(0, 1, 2, 4, 3, 5, 6).reshape(B, pl, lat * 2, lon * 2, -1)
        ph, pw = lat * 2 - ol, lon * 2 - ow
        t, l = ph // 2, pw // 2
        x = x[:, :, t:2 * lat - (ph - t), l:2 * lon - (pw - l), :]
        return self.linear2(self.norm(x.reshape(B, -1, x.shape[-1])))
