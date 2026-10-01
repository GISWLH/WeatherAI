"""Perceiver blocks (cross-attention resampler) used to aggregate / de-aggregate pressure levels.

Structure and parameter names follow microsoft/aurora (MIT) ``aurora/model/perceiver.py``
(itself adapted from lucidrains/perceiver-pytorch and open_flamingo, both MIT).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Linear -> GELU -> Linear -> Dropout (``net.0`` / ``net.2`` in checkpoints)."""

    def __init__(self, dim: int, hidden_features: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_features), nn.GELU(), nn.Linear(hidden_features, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class PerceiverAttention(nn.Module):
    """Multi-head cross attention: queries from ``latents``, keys/values from ``x``."""

    def __init__(self, latent_dim: int, context_dim: int, head_dim: int = 64, num_heads: int = 8, ln_k_q: bool = False):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        inner = head_dim * num_heads
        self.to_q = nn.Linear(latent_dim, inner, bias=False)
        self.to_kv = nn.Linear(context_dim, inner * 2, bias=False)
        self.to_out = nn.Linear(inner, latent_dim, bias=False)
        if ln_k_q:  # optional extra LayerNorm on keys / queries (before splitting heads)
            self.ln_k, self.ln_q = nn.LayerNorm(inner), nn.LayerNorm(inner)
        else:
            self.ln_k = self.ln_q = nn.Identity()

    def forward(self, latents: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """``latents`` (B, L1, Dl), ``x`` (B, L2, Dc) -> (B, L1, Dl)."""
        B, L1, _ = latents.shape
        h = self.num_heads
        q = self.ln_q(self.to_q(latents))
        k, v = self.to_kv(x).chunk(2, dim=-1)
        k = self.ln_k(k)
        q, k, v = (t.reshape(B, -1, h, self.head_dim).transpose(1, 2) for t in (q, k, v))  # (B, h, L, d)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, L1, h * self.head_dim)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    """``depth`` x [post-norm cross-attention (+residual), post-norm MLP (+residual)]."""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        depth: int = 1,
        head_dim: int = 64,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        residual_latent: bool = True,
        ln_eps: float = 1e-5,
        ln_k_q: bool = False,
    ):
        super().__init__()
        self.residual_latent = residual_latent
        hidden = int(latent_dim * mlp_ratio)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PerceiverAttention(latent_dim, context_dim, head_dim, num_heads, ln_k_q if i == 0 else False),
                        MLP(latent_dim, hidden, drop),
                        nn.LayerNorm(latent_dim, eps=ln_eps),
                        nn.LayerNorm(latent_dim, eps=ln_eps),
                    ]
                )
                for i in range(depth)
            ]
        )

    def forward(self, latents: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        for attn, ff, ln1, ln2 in self.layers:
            attn_out = ln1(attn(latents, x))  # post-norm (Swin-v2 style)
            latents = attn_out + latents if self.residual_latent else attn_out
            latents = ln2(ff(latents)) + latents
        return latents
