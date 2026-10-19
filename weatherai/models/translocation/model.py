from __future__ import annotations

import torch
from torch import nn


class Translocation(nn.Module):
    """Wrap a differentiable forecaster to compute the paper's signed ERF.

    Inputs/output: (B,C,H,W). Supply a physical-unit output wrapper and
    properly normalized inputs when interpreting derivatives scientifically.
    A large ERF alone does not establish causal translocation.
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    @torch.enable_grad()
    def effective_receptive_field(self, x, *, channel=0, location=None, region_mask=None):
        """Derivative of one selected output per batch wrt all input variables.

        Give either (row,col) or a region mask: the latter chooses the largest
        predicted value within that region, as in the Dubai rainfall analysis.
        Returns signed gradients and selected positions; leaves parameter grads
        untouched. Batch samples must be independent (use eval mode).
        """
        if self.training:
            raise ValueError("call eval() before ERF analysis")
        if (location is None) == (region_mask is None):
            raise ValueError("provide exactly one of location or region_mask")
        inputs = x.detach().clone().requires_grad_(True)
        y = self(inputs)
        if y.ndim != 4 or not 0 <= channel < y.shape[1]:
            raise ValueError("backbone must output (B,C,H,W) with valid channel")
        field = y[:, channel]
        if region_mask is not None:
            mask = torch.as_tensor(region_mask, device=y.device, dtype=torch.bool)
            if mask.shape != field.shape[-2:] or not mask.any():
                raise ValueError("region_mask must be a nonempty H,W mask")
            index = field.masked_fill(~mask, -torch.inf).flatten(1).argmax(-1)
        else:
            row, col = location
            if not 0 <= row < field.shape[-2] or not 0 <= col < field.shape[-1]:
                raise ValueError("location is outside the output grid")
            index = torch.full((len(x),), row * field.shape[-1] + col, device=y.device, dtype=torch.long)
        selected = field.flatten(1).gather(1, index[:, None]).sum()
        gradient, = torch.autograd.grad(selected, inputs)
        positions = torch.stack((index // field.shape[-1], index % field.shape[-1]), -1)
        return {"gradient": gradient.detach(), "location": positions, "prediction": y.detach()}


def Translocation_lite(img_size=(8, 16), in_channels=4):
    """Random-weight, small GraphCast diagnostic harness; no Dubai skill claim."""
    from ..graphcast import GraphCast
    return Translocation(GraphCast(img_size=img_size, in_channels=in_channels,
                                  mesh_level=1, processor_layers=2, hidden_dim=16)).eval()
