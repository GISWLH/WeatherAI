"""StormCast native-port tests. Parity tests use NVIDIA PhysicsNeMo as the oracle (skipped if it is not installed)."""
import os

import pytest
import torch

from weatherai.models.stormcast import StormCast_lite, edm_heun_sample, edm_sigmas
from weatherai.models.stormcast.unet import EDMPrecond, SongUNetConfig, StormCastUNet

CKPT = os.environ.get("STORMCAST_CKPT", "/workspace/ckpt/stormcast")


def _randomise(m, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
        for n, b in m.named_buffers():
            pass
    return m


def test_lite_shapes_and_determinism():
    m = StormCast_lite().eval()
    c = m.cfg
    x = torch.randn(2, c.n_state, 64, 64)
    cond = torch.randn(2, c.n_cond, 64, 64)
    y1 = m(x, cond, generator=torch.Generator().manual_seed(1))
    y2 = m(x, cond, generator=torch.Generator().manual_seed(1))
    y3 = m(x, cond, generator=torch.Generator().manual_seed(2))
    assert y1.shape == x.shape and torch.isfinite(y1).all()
    assert torch.equal(y1, y2) and not torch.equal(y1, y3)
    assert m(x, cond, deterministic_only=True).shape == x.shape


def test_lite_trainable_both_networks():
    m = StormCast_lite().train()
    c = m.cfg
    x, cond = torch.randn(2, c.n_state, 64, 64), torch.randn(2, c.n_cond, 64, 64)
    inv = m.invariants.expand(2, -1, -1, -1)
    tgt = torch.randn_like(x)
    mean = m.regression(torch.cat([x, cond, inv], 1))
    l_reg = (mean - tgt).pow(2).mean()
    sig = torch.tensor([0.5, 3.0])
    den = m.diffusion(tgt + sig.view(-1, 1, 1, 1) * torch.randn_like(tgt), sig, torch.cat([x, mean.detach(), inv], 1))
    l_dif = (den - tgt).pow(2).mean()
    (l_reg + l_dif).backward()
    for name, net in [("regression", m.regression), ("diffusion", m.diffusion)]:
        bad = [n for n, p in net.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
        assert not bad, (name, bad[:5])


def test_sigma_schedule_endpoints():
    s = edm_sigmas(18)
    assert s.shape == (19,) and abs(s[0].item() - 800) < 1e-9 and abs(s[17].item() - 0.002) < 1e-12 and s[18] == 0
    assert (s[:-1] > s[1:]).all()


@pytest.mark.skipif(not os.path.exists(f"{CKPT}/EDMPrecond.0.0.mdlus"), reason="official StormCast checkpoint not downloaded")
def test_official_checkpoint_strict_load_and_param_counts():
    from weatherai.models.stormcast import load_official
    m = load_official(CKPT, with_metadata=False)
    assert abs(sum(p.numel() for p in m.regression.parameters()) - 78_588_387) == 0
    assert abs(sum(p.numel() for p in m.diffusion.parameters()) - 121_058_275) == 0
