"""Native PyTorch UniCM (Yuan et al., *Nature Machine Intelligence* 2026), a re-implementation of
``src/models.py`` + ``src/my_tools.py`` of tsinghua-fib-lab/UniCM-Global-Climate-Modes (MIT).

Two coupled spatio-temporal Transformers: a *field* model on 5 monthly ocean fields (SST, wind stress
x/y, HTC300, T20D) on a 12x72 5-degree grid, and a *mode* model on the climate-mode index series.
The mode model's encoder/decoder outputs are scattered back onto the regions of the field grid and added
to the field model's embeddings (subtracted over the second box of the dipole modes IOD and SIOD).

Module and parameter names match the official ``state_dict`` so an officially trained checkpoint
(``model_save/*.pkl`` from ``app_train.py``) loads with ``strict=True``; the official code shares
``time_emb`` between the field and mode embeddings and uses ``sublayer[1]`` twice in the decoder
(``sublayer[2]`` is created but never used); both are reproduced.  No pretrained weights are public.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["UniCMConfig", "UniCM", "oras5_regions", "patchify", "unpatchify"]

RESO = 5   # degrees


def oras5_regions(reso: int = RESO, lat_start: int = 45):
    """Region boxes (lat0, lat1, lon0, lon1) in grid cells, as set in the official ``make_*_ORAS5`` loaders.

    Order: Nino (ONI box), NPMM, SPMM, IOB, IOD (two boxes), SIOD (two boxes), TNA, Nino1+2, Nino3, Nino4.
    """
    L = lat_start // reso
    r = lambda a, b, c, d: (a // reso - L, b // reso - L, c // reso, d // reso)
    return [
        r(70, 80, 190, 240), r(85, 100, 200, 240), r(50, 60, 250, 270), r(55, 75, 40, 100),
        (r(65, 85, 50, 70), r(65, 75, 90, 110)), (r(50, 65, 65, 80), r(45, 65, 90, 120)),
        r(80, 100, 305, 345), r(65, 75, 270, 280), r(70, 80, 210, 270), r(70, 80, 200, 210),
    ]


@dataclass
class UniCMConfig:
    his_len: int = 12
    pred_len: int = 24
    in_channels: int = 5
    grid: Tuple[int, int] = (12, 72)
    patch_size: Tuple[int, int] = (2, 2)
    d_size: int = 256
    nheads: int = 4
    dim_feedforward: int = 256
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dropout: float = 0.2
    t20d_mode: int = 1
    mode_interaction: str = "1"           # '0': no field coupling in the mode model (time attention only)
    autoregressive: int = 0
    val_relative: List = field(default_factory=oras5_regions)

    @classmethod
    def official(cls):
        return cls()

    @classmethod
    def lite(cls, **kw):
        base = dict(d_size=32, nheads=2, dim_feedforward=64, num_encoder_layers=1, num_decoder_layers=1,
                    his_len=6, pred_len=4, dropout=0.0)
        base.update(kw)
        return cls(**base)

    @property
    def n_patches(self):
        return (self.grid[0] // self.patch_size[0]) * (self.grid[1] // self.patch_size[1])

    @property
    def n_mode_series(self):
        return len(self.val_relative) + self.t20d_mode

    @property
    def cube_dim(self):
        return self.in_channels * self.patch_size[0] * self.patch_size[1]

    @property
    def special_index(self):
        return [4, 5] if len(self.val_relative) > 2 else [1]


def patchify(x: torch.Tensor, p: Sequence[int]) -> torch.Tensor:
    """(B, T, C, H, W) -> (B, T, C*p0*p1, H/p0, W/p1), channel order (c, i, j) = official ``unfold_func``."""
    B, T, C, H, W = x.shape
    x = x.reshape(B, T, C, H // p[0], p[0], W // p[1], p[1])
    return x.permute(0, 1, 2, 4, 6, 3, 5).reshape(B, T, C * p[0] * p[1], H // p[0], W // p[1])


def unpatchify(x: torch.Tensor, p: Sequence[int]) -> torch.Tensor:
    """Inverse of :func:`patchify` (== ``F.fold`` with stride = kernel)."""
    B, T, K, h, w = x.shape
    C = K // (p[0] * p[1])
    x = x.reshape(B, T, C, p[0], p[1], h, w).permute(0, 1, 2, 5, 3, 6, 4)
    return x.reshape(B, T, C, h * p[0], w * p[1])


class Embedding(nn.Module):
    """official ``make_embedding``: linear + month embedding + learned patch-position embedding, LayerNorm."""

    def __init__(self, cube_dim: int, d_size: int, n_space: int):
        super().__init__()
        self.time_emb = nn.Embedding(12, d_size)
        self.emb_space = nn.Embedding(n_space, d_size)
        self.linear = nn.Linear(cube_dim, d_size)
        self.norm = nn.LayerNorm(d_size)
        self.register_buffer("spatial_pos", torch.arange(n_space)[None, :, None], persistent=False)

    def forward(self, x, months):                      # x (B,S,T,cube) months (B,T) long
        t = self.time_emb(months).unsqueeze(1)          # (B,1,T,d)
        s = self.emb_space(self.spatial_pos)            # (1,S,1,d)
        return self.norm(self.linear(x) + t + s)


class Attention(nn.Module):
    """official ``make_attention``: ``kind='T'`` attends over time (causal mask optional), ``'S'`` over space."""

    def __init__(self, d, heads, kind, dropout):
        super().__init__()
        self.h, self.dk, self.kind = heads, d // heads, kind
        self.linears = nn.ModuleList([nn.Linear(d, d) for _ in range(4)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):             # (B,S,T,d)
        B, S, T, _ = q.shape
        q, k, v = [l(x).view(x.shape[0], x.shape[1], x.shape[2], self.h, self.dk).permute(0, 3, 1, 2, 4)
                   for l, x in zip(self.linears, (q, k, v))]                     # (B,h,S,T,dk)
        if self.kind == "T":
            sc = q @ k.transpose(-2, -1) / math.sqrt(self.dk)
            if mask is not None:
                sc = sc.masked_fill(mask[None, None, None], float("-inf"))
            out = self.dropout(sc.softmax(-1)) @ v
        else:                                           # spatial attention; the official code ignores the mask
            qs, ks, vs = q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3)   # (B,h,T,S,dk)
            sc = qs @ ks.transpose(-2, -1) / math.sqrt(self.dk)
            out = (self.dropout(sc.softmax(-1)) @ vs).transpose(2, 3)
        out = out.permute(0, 2, 3, 1, 4).reshape(B, S, T, self.h * self.dk)
        return self.linears[-1](out)


class Sublayer(nn.Module):
    """post-norm residual: LayerNorm(x + dropout(f(x)))."""

    def __init__(self, d, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, f):
        return self.norm(x + self.dropout(f(x)))


def _ffn(d, dff):
    return nn.Sequential(nn.Linear(d, dff), nn.ReLU(), nn.Linear(dff, d))


class EncoderLayer(nn.Module):
    def __init__(self, d, heads, dff, dropout, mode="ST"):
        super().__init__()
        self.sublayer = nn.ModuleList([Sublayer(d, dropout) for _ in range(2)])
        self.time_attn = Attention(d, heads, "T", dropout)
        self.space_attn = Attention(d, heads, "S", dropout)
        self.FC = _ffn(d, dff)
        self.mode = mode

    def attn(self, x, mask):
        t = self.time_attn(x, x, x, mask)
        return self.space_attn(t, t, t, mask) if self.mode == "ST" else t

    def forward(self, x, mask=None):
        x = self.sublayer[0](x, lambda y: self.attn(y, mask))
        return self.sublayer[1](x, self.FC)


class DecoderLayer(nn.Module):
    def __init__(self, d, heads, dff, dropout, mode="ST"):
        super().__init__()
        self.sublayer = nn.ModuleList([Sublayer(d, dropout) for _ in range(3)])   # [2] unused (as in the official code)
        self.encoder_attn = Attention(d, heads, "T", dropout)
        self.time_attn = Attention(d, heads, "T", dropout)
        self.space_attn = Attention(d, heads, "S", dropout)
        self.FC = _ffn(d, dff)
        self.mode = mode

    def self_attn(self, x, mask):
        m = self.time_attn(x, x, x, mask)
        return self.space_attn(m, m, m) if self.mode == "ST" else m

    def forward(self, x, en_out, tgt_mask, memory_mask=None):
        x = self.sublayer[0](x, lambda y: self.self_attn(y, tgt_mask))
        x = self.sublayer[1](x, lambda y: self.encoder_attn(y, en_out, en_out, memory_mask))
        return self.sublayer[1](x, self.FC)             # official code reuses sublayer[1] here


class Stack(nn.Module):
    def __init__(self, layer, n):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n)])

    def forward(self, x, *a):
        for l in self.layers:
            x = l(x, *a)
        return x


class UniCM(nn.Module):
    def __init__(self, cfg: UniCMConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_size
        self.predictor_emb = Embedding(cfg.cube_dim, d, cfg.n_patches)
        self.encoder = Stack(EncoderLayer(d, cfg.nheads, cfg.dim_feedforward, cfg.dropout), cfg.num_encoder_layers)
        self.decoder = Stack(DecoderLayer(d, cfg.nheads, cfg.dim_feedforward, cfg.dropout), cfg.num_decoder_layers)
        self.linear_output = nn.Linear(d, cfg.cube_dim)
        self.predictor_emb_mode = Embedding(1, d, cfg.n_mode_series)
        self.predictor_emb_mode.time_emb = self.predictor_emb.time_emb       # shared, as in the official code
        self.predictand_emb = self.predictor_emb                             # aliases (official state_dict keys)
        self.predictand_emb_mode = self.predictor_emb_mode
        mm = "ST" if cfg.mode_interaction != "0" else "T"
        self.encoder_mode = Stack(EncoderLayer(d, cfg.nheads, cfg.dim_feedforward, cfg.dropout, mm), cfg.num_encoder_layers)
        self.decoder_mode = Stack(DecoderLayer(d, cfg.nheads, cfg.dim_feedforward, cfg.dropout, mm), cfg.num_decoder_layers)
        self.linear_output_mode = nn.Linear(d, 1)

    # official: predictand_emb is predictor_emb (same module registered once)
    @staticmethod
    def causal_mask(n, device):
        return (torch.triu(torch.ones(n, n, device=device)) == 0).T

    def _encode(self, enc, emb, x, months, p, cube, bias=None):
        B, T = x.shape[:2]
        x = patchify(x, p).reshape(B, T, cube, -1).permute(0, 3, 1, 2)       # (B,S,T,cube)
        x = emb(x, months)
        if bias is not None:
            assert bias.shape == x.shape
            x = x + bias
        return enc(x, None)

    def _decode(self, dec, lin, emb, x, months, en_out, mask, p, cube, bias=None):
        B, T, _, H, W = x.shape
        z = patchify(x, p).reshape(B, T, cube, -1).permute(0, 3, 1, 2)
        z = emb(z, months)
        if bias is not None:
            z = z + bias[:, :, :T]
        out = dec(z, en_out, mask, None)
        emb_out = out
        y = lin(out).permute(0, 2, 3, 1).reshape(B, T, cube, H // p[0], W // p[1])
        return unpatchify(y, p), emb_out

    def _branch(self, x, months, enc, dec, lin, emb, cube, p, train, sv_ratio, bias_enc=None, bias_dec=None,
                pred_len=None):
        c = self.cfg
        his = c.his_len
        pre, post = x[:, :his], x[:, his:]
        tpre, tpost = months[:, :his], months[:, his:]
        en = self._encode(enc, emb, pre, tpre, p, cube, bias_enc)
        if train:
            conn = torch.cat([pre[:, -1:], post[:, :-1]], 1)
            t = torch.cat([tpre[:, -1:], tpost[:, :-1]], 1)
            mask = self.causal_mask(conn.size(1), x.device)
            if c.autoregressive == 0:
                with torch.no_grad():
                    out, _ = self._decode(dec, lin, emb, conn, t, en, mask, p, cube, bias_dec)
            else:
                out, _ = self._decode(dec, lin, emb, conn, t, en, mask, p, cube, bias_dec)
            sup = (torch.bernoulli(sv_ratio * torch.ones(post.size(0), post.size(1) - 1, 1, 1, 1, device=x.device))
                   if sv_ratio > 1e-7 else 0)
            tgt = sup * post[:, :-1] + (1 - sup) * out[:, :-1]
            tgt = torch.cat([pre[:, -1:], tgt], 1)
            out, emb_out = self._decode(dec, lin, emb, tgt, t, en, mask, p, cube, bias_dec)
        else:
            tgt, t = pre[:, -1:], tpre[:, -1:]
            n = pred_len if pred_len is not None else c.pred_len
            for i in range(n):
                mask = self.causal_mask(tgt.size(1), x.device)
                out, emb_out = self._decode(dec, lin, emb, tgt, t, en, mask, p, cube, bias_dec)
                tgt = torch.cat([tgt, out[:, -1:]], 1)
                t = torch.cat([t, tpost[:, i:i + 1]], 1)
        return out, en, emb_out

    def forward(self, predictor, predictor_mode, months, train: bool = False, sv_ratio: float = 0.0):
        """
        predictor      (B, T, C, H, W)  normalised monthly anomalies, T = his_len (+ pred_len for training)
        predictor_mode (B, M, T)        mode series (M = n_modes + t20d_mode)
        months         (B, T) long      calendar month 0..11 of every step (future steps are needed even at
                                        inference: only their month index is used)
        returns (field prediction (B, pred_len, C, H, W), mode prediction (B, M, pred_len))
        """
        c = self.cfg
        B, T, C, H, W = predictor.shape
        pm = predictor_mode.permute(0, 2, 1).unsqueeze(-1).unsqueeze(2)         # (B,T,1,M,1)
        pred_mode, en_mode, emb_mode = self._branch(
            pm, months, self.encoder_mode, self.decoder_mode, self.linear_output_mode, self.predictor_emb_mode,
            1, (1, 1), train, sv_ratio)
        p = c.patch_size
        h, w = H // p[0], W // p[1]
        enc_out = predictor.new_zeros(B, h, w, c.his_len, en_mode.shape[-1])
        dec_emb = predictor.new_zeros(B, h, w, emb_mode.shape[-2], emb_mode.shape[-1])
        sp = c.special_index
        for i, reg in enumerate(c.val_relative):
            boxes = [reg] if i not in sp else list(reg)
            signs = [1.0, -1.0]
            for bx, sg in zip(boxes, signs):
                a1, b1, a2, b2 = bx[0] // p[0], bx[1] // p[0], bx[2] // p[1], bx[3] // p[1]
                if a1 == b1: b1 += 1
                if a2 == b2: b2 += 1
                enc_out[:, a1:b1, a2:b2] = enc_out[:, a1:b1, a2:b2] + sg * en_mode[:, i:i + 1].unsqueeze(1)
                dec_emb[:, a1:b1, a2:b2] = dec_emb[:, a1:b1, a2:b2] + sg * emb_mode[:, i:i + 1].unsqueeze(1)
        enc_out = enc_out.reshape(B, h * w, c.his_len, -1)
        dec_emb = dec_emb.reshape(B, h * w, dec_emb.shape[-2], -1)
        pred, _, _ = self._branch(predictor, months, self.encoder, self.decoder, self.linear_output,
                                  self.predictor_emb, c.cube_dim, p, train, sv_ratio, enc_out, dec_emb)
        return pred, pred_mode.squeeze(-1).squeeze(2).permute(0, 2, 1)
