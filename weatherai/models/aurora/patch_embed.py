"""Per-variable 3-D patch embedding (history x patch x patch) used by the Aurora encoder.

Structure and parameter names follow microsoft/aurora (MIT) ``aurora/model/patchembed.py``.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LevelPatchEmbed(nn.Module):
    """Embed all variables of one level (surface, or one pressure level) into one token per patch.

    Every variable has its own ``(embed_dim, 1, T, P, P)`` kernel (``weights[name]``); the
    kernels of the variables present are concatenated along the input-channel axis and applied
    with a single stride-``(T, P, P)`` 3-D convolution. Using fewer than ``max_history_size``
    history steps simply uses the first ``T`` slices of every kernel.
    """

    def __init__(self, var_names: Sequence[str], patch_size: int, embed_dim: int, history_size: int = 2):
        super().__init__()
        self.var_names = tuple(var_names)
        self.kernel_size = (history_size, patch_size, patch_size)
        self.embed_dim = embed_dim
        self.weights = nn.ParameterDict(
            {n: nn.Parameter(torch.empty(embed_dim, 1, *self.kernel_size)) for n in self.var_names}
        )
        self.bias = nn.Parameter(torch.empty(embed_dim))
        self.init_weights()

    def init_weights(self) -> None:
        for w in self.weights.values():
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(next(iter(self.weights.values())))
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, var_names: Sequence[str]) -> torch.Tensor:
        """``x``: ``(B, V, T, H, W)`` -> tokens ``(B, (H/P)*(W/P), D)``."""
        B, V, T, H, W = x.shape
        assert len(var_names) == V and len(set(var_names)) == V
        assert self.kernel_size[0] >= T, f"history {T} > max_history_size {self.kernel_size[0]}"
        assert H % self.kernel_size[1] == 0 and W % self.kernel_size[2] == 0
        weight = torch.cat([self.weights[n][:, :, :T] for n in var_names], dim=1)  # (D, V, T, P, P)
        proj = F.conv3d(x, weight, self.bias, stride=(T,) + self.kernel_size[1:])  # (B, D, 1, H/P, W/P)
        return proj.reshape(B, self.embed_dim, -1).transpose(1, 2)
