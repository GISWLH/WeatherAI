import json
import os

import numpy as np
import torch

from weatherai.models.genfocal import GenFocalNet, GenFocalNetConfig, interpolate, reflow_loss, sample_flow
from weatherai.models.genfocal.reference import TinyFlow

REF = os.path.join(os.path.dirname(__file__), "data", "ref_flow.npz")


def test_flow_core_matches_official_jax():
    """loss and RK4 sampler (forward / reversed) vs the unchanged swirl_dynamics code on a tiny Flax network (stored reference)."""
    from scripts.genfocal_parity import run
    r = run(REF)
    for k, v in r.items():
        if k.startswith("loss"):
            assert v["rel"] < 1e-5, (k, v)
        else:
            assert v["rel_l2"] < 1e-5, (k, v)


class _Const(torch.nn.Module):
    def forward(self, x, sigma, cond=None):
        return torch.ones_like(x)


def test_sampler_schedule_stops_at_one_minus_dt():
    x0 = torch.zeros(1, 1, 2, 2, 1)
    assert torch.allclose(sample_flow(_Const(), x0, num_steps=4), torch.full_like(x0, 0.75))          # official schedule
    assert torch.allclose(sample_flow(_Const(), x0, num_steps=4, to_end=True), torch.ones_like(x0))
    assert torch.allclose(sample_flow(_Const(), x0, num_steps=4, solver="euler", to_end=True), torch.ones_like(x0))


def test_interpolate_endpoints():
    a, b = torch.randn(2, 3), torch.randn(2, 3)
    assert torch.allclose(interpolate(a, b, torch.zeros(2)), a) and torch.allclose(interpolate(a, b, torch.ones(2)), b)


def _net():
    cfg = GenFocalNetConfig.lite(4)
    torch.manual_seed(0)
    m = GenFocalNet(cfg).eval()
    torch.nn.init.normal_(m.out.conv.weight, std=0.1)                  # un-zero the output so the tests are informative
    return cfg, m


def test_lite_net_shapes_zero_init_and_grads():
    cfg = GenFocalNetConfig.lite(4)
    m = GenFocalNet(cfg)
    x = torch.randn(2, 3, 16, 8, 4)
    c = {k: torch.randn_like(x) for k in cfg.cond_keys}
    assert m(x, torch.rand(2), c).abs().max() == 0                      # official-style zero-init output kernel
    cfg, m = _net()
    m.train()
    loss = reflow_loss(m, x, torch.randn_like(x), c)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert m.out.conv.weight.grad.abs().max() > 0


def test_lite_net_periodic_in_longitude():
    """circular longitude padding => rolling the inputs by a multiple of the total downsampling (4) rolls the output."""
    cfg, m = _net()
    x = torch.randn(1, 2, 16, 8, 4)
    c = {k: torch.randn_like(x) for k in cfg.cond_keys}
    s = torch.rand(1)
    y = m(x, s, c)
    r = lambda a: a.roll(4, 2)
    assert (m(r(x), s, {k: r(v) for k, v in c.items()}) - r(y)).abs().max() < 1e-4


def test_lite_training_reduces_loss():
    cfg, m = _net()
    m.train()
    torch.manual_seed(1)
    x0 = torch.randn(4, 2, 16, 8, 4)
    x1 = x0 * 0.5 + 1.0
    c = {k: x0.clone() for k in cfg.cond_keys}
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    first = None
    for _ in range(40):
        opt.zero_grad()
        l = reflow_loss(m, x0, x1, c)
        l.backward()
        opt.step()
        first = first if first is not None else float(l)
    assert float(l) < 0.6 * first
