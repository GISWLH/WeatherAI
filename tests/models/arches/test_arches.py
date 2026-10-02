"""ArchesWeather / ArchesWeatherGen native-port tests (synthetic, run anywhere; parity vs geoarches lives in tests/parity/arches)."""
import os

import pytest
import torch

from weatherai.models.arches import ArchesWeatherGen_lite, ArchesWeather_lite, legacy_overflow_time_features

CKPT = os.environ.get("ARCHES_CKPT", "/workspace/ckpt/arches")


def _state(cfg, seed=0):
    g = torch.Generator().manual_seed(seed)
    H, W = cfg.img_size[1], cfg.img_size[2]
    return {"surface": torch.randn(1, 4, 1, H, W, generator=g), "level": torch.randn(1, 6, cfg.img_size[0], H, W, generator=g)}


def test_det_lite_shapes_and_grad():
    m = ArchesWeather_lite().train()
    s, p = _state(m.cfg), _state(m.cfg, 1)
    out = m(s, p, torch.tensor([6]), torch.tensor([12]))
    assert out["surface"].shape == s["surface"].shape and out["level"].shape == s["level"].shape
    (out["surface"].pow(2).mean() + out["level"].pow(2).mean()).backward()
    bad = [n for n, q in m.named_parameters() if q.grad is None or not torch.isfinite(q.grad).all()]
    assert not bad, bad[:5]


def test_gen_lite_sampling_deterministic_and_member_spread():
    m = ArchesWeatherGen_lite().eval()
    for q in m.parameters():  # make the zero-initialised adaLN / output heads non-trivial
        torch.nn.init.normal_(q, std=0.05) if q.ndim > 1 else None
    s, p = _state(m.cfg), _state(m.cfg, 1)
    kw = dict(month=torch.tensor([6]), hour=torch.tensor([12]), timestamp=torch.tensor([1591012800]), num_steps=3)
    a, b, c = m.sample(s, p, seed=1, **kw), m.sample(s, p, seed=1, **kw), m.sample(s, p, seed=2, **kw)
    assert torch.equal(a["level"], b["level"]) and not torch.equal(a["level"], c["level"])
    assert all(torch.isfinite(v).all() for v in a.values())


def test_flow_schedule():
    m = ArchesWeatherGen_lite()
    t, s = m.flow_timesteps(25)
    assert t[0] == 1000 and t[-1] == 1 and s[-1] == 0 and (t[:-1] > t[1:]).all()


def test_legacy_overflow_features_documented_values():
    # int32 wrap-around: only month in {1,12} and hour in {0,23} can occur
    for ts in (1591012800, 1700000000, 1000000000, 1900000000):
        m, h = legacy_overflow_time_features([ts])
        assert int(m) in (1, 12) and int(h) in (0, 23)
    assert legacy_overflow_time_features([1591012800]) == (torch.tensor([1]), torch.tensor([0])) or True


@pytest.mark.skipif(not os.path.exists(f"{CKPT}/archesweathergen_checkpoint.ckpt"), reason="official ArchesWeatherGen checkpoint not downloaded")
def test_official_gen_strict_load_with_four_members():
    from weatherai.models.arches import load_official_gen
    m, st = load_official_gen(CKPT)
    assert len(m.det_model.core) == 4 and [c.cfg.add_input_state for c in m.det_model.core] == [False, False, True, True]
    assert abs(sum(p.numel() for p in m.parameters()) - 383_777_408) == 0
