import os

import pytest
import torch

from weatherai.models.fuxi_s2s import FuXiS2S_lite, FuXiS2SConfig, make_relative_tables, make_shift_mask

ROOT = os.environ.get("FUXI_S2S_ROOT", "/workspace/ckpt/fuxi_s2s")
ONNX = os.path.join(ROOT, "model-1.0", "fuxi_s2s.onnx")


def _x(m, B=1):
    c = m.cfg
    torch.manual_seed(0)
    x = torch.randn(B, 2, c.n_channels, *c.img_size)
    x[..., c.precip_channel, :, :] = x[..., c.precip_channel, :, :].abs()
    x[:, :, 2, :4] = float("nan")             # NaN (land) is mapped to 0
    return x


def test_shapes_and_state_passthrough():
    m = FuXiS2S_lite().eval()
    x = _x(m)
    with torch.no_grad():
        y = m(x, torch.zeros(1))
    assert y.shape == x.shape
    torch.testing.assert_close(y[:, 0], x[:, -1], rtol=0, atol=0, equal_nan=True)   # slot 0 = raw last input frame
    assert torch.isfinite(y[:, 1]).all() and (y[:, 1, -1] >= -1).all()


def test_noise_controls_ensemble():
    m = FuXiS2S_lite().eval()
    x = _x(m)
    n1, n2 = m.sample_noise(1), m.sample_noise(1)
    with torch.no_grad():
        a, b, c = m(x, torch.zeros(1), noise=n1), m(x, torch.zeros(1), noise=n1), m(x, torch.zeros(1), noise=n2)
    torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
    assert not torch.allclose(a[:, 1].nan_to_num(), c[:, 1].nan_to_num())


def test_shift_mask_is_latitude_only():
    mk = make_shift_mask((60, 120), 10, 5)
    assert mk.shape == (72, 100, 100)
    assert (mk[:60] == 0).all() and (mk[60:] != 0).any()      # only the 12 windows of the last window row are masked
    t, idx = make_relative_tables(10)
    assert t.shape == (1, 19, 19, 2) and idx.shape == (100, 100) and int(idx.max()) == 19 * 19 - 1


def test_lite_trains():
    m = FuXiS2S_lite()
    x = _x(m, 2)
    tgt = torch.randn(2, m.cfg.n_channels, *m.cfg.img_size)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    nz = m.sample_noise(2)
    losses = []
    for _ in range(8):
        opt.zero_grad()
        l = ((m(x, torch.zeros(2), noise=nz)[:, 1] - tgt).nan_to_num() ** 2).mean()
        l.backward(); opt.step(); losses.append(float(l))
    assert losses[-1] < losses[0]


@pytest.mark.skipif(not os.path.exists(ONNX), reason="official FuXi-S2S weights not downloaded (CC-BY-NC-ND, fetch yourself)")
def test_official_strict_load():
    from weatherai.models.fuxi_s2s.convert import load_official
    m = load_official(ONNX, dtype=torch.float16)
    assert 1.03e9 < sum(p.numel() for p in m.parameters()) < 1.04e9
    assert float(m.dist_p.cov.weight.abs().max()) == 0.0
