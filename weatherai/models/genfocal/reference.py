"""Torch twin of the tiny Flax flow network used to generate ``tests/models/genfocal/data/ref_flow.npz`` from the official
GenFocal code (see ``scripts/genfocal_make_ref.py``).  Only used for parity tests."""
import numpy as np
import torch
import torch.nn as nn


class TinyFlow(nn.Module):
    def __init__(self, c: int = 4, h: int = 16):
        super().__init__()
        self.a, self.b1, self.b2, self.d = nn.Linear(c, h), nn.Linear(c, h), nn.Linear(c, h), nn.Linear(h, c)
        self.e = nn.Parameter(torch.zeros(h))

    def forward(self, x, sigma, cond=None):
        if sigma.ndim < 1:
            sigma = sigma.expand(x.shape[0])
        h = self.a(x) + self.b1(cond["channel:mean"]) + self.b2(cond["channel:std"])
        h = h + sigma[:, None, None, None, None] * self.e * 3.0
        return self.d(torch.tanh(h))

    @classmethod
    def from_ref(cls, ref) -> "TinyFlow":
        m = cls()
        sd = {}
        for n in ("a", "b1", "b2", "d"):
            sd[f"{n}.weight"] = torch.from_numpy(np.ascontiguousarray(ref[f"p/{n}/kernel"].T))
            sd[f"{n}.bias"] = torch.from_numpy(ref[f"p/{n}/bias"])
        sd["e"] = torch.from_numpy(ref["p/e"])
        m.load_state_dict(sd, strict=True)
        return m
