import os
import sys
import types

import numpy as np
import torch

from weatherai.models.canglong import Canglong, CanglongConfig, canglong_loss, load_official, official_state_dict

REF = os.path.join(os.path.dirname(__file__), "data", "ref_lite.npz")


def _ref():
    d = np.load(REF)
    return d, {k[3:]: torch.from_numpy(d[k]) for k in d.files if k.startswith("sd/")}


def test_matches_official_reference():
    """forward reproduces the official notebook ``Canglong`` (lite sizes, random weights, stored reference)."""
    d, sd = _ref()
    m = Canglong(CanglongConfig.lite()).eval()
    m.load_state_dict(sd, strict=True)
    with torch.no_grad():
        y = m(torch.from_numpy(d["x"]))
    assert y.shape == (2, 1, 3, 32, 64)
    assert (y - torch.from_numpy(d["y"])).abs().max() < 1e-6


def test_official_geometry_and_param_count():
    cfg = CanglongConfig.official()
    assert cfg.token_grid == (8, 181, 360) and cfg.coarse_grid == (8, 91, 180) and cfg.out_months == 3
    with torch.device("meta"):
        m = Canglong(cfg)
    assert sum(p.numel() for p in m.parameters()) == 43_906_258


def test_load_official_pickled_module(tmp_path, monkeypatch):
    """``torch.save(model)`` from the training notebook: foreign classes are stubbed, tensors strict-load."""
    cfg = CanglongConfig.lite()
    fake = types.ModuleType("canglong")
    Official = type("Canglong", (Canglong,), {"__module__": "canglong"})
    fake.Canglong = Official
    monkeypatch.setitem(sys.modules, "canglong", fake)
    torch.manual_seed(0)
    off = Official(cfg).eval()
    path = tmp_path / "canglong_lite.tar"
    torch.save(off, path)
    monkeypatch.delitem(sys.modules, "canglong")
    nat = load_official(str(path), cfg)
    x = torch.randn(1, cfg.in_chans, cfg.n_months, *cfg.img_size)
    with torch.no_grad():
        assert torch.equal(nat(x), off(x))


def test_state_dict_variants():
    """DataParallel prefix and the WeatherLearn naming (``proj`` / ``conv``) map onto the native names."""
    m = Canglong(CanglongConfig.lite())
    sd = {}
    for k, v in m.state_dict().items():
        k = k.replace(".projection.", ".proj.").replace(".transposed_conv.", ".conv.")
        sd["module." + k] = v
    m2 = Canglong(CanglongConfig.lite())
    m2.load_state_dict(official_state_dict({"state_dict": sd}), strict=True)
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values()))


def test_loss_trains():
    cfg = CanglongConfig.lite()
    torch.manual_seed(0)
    m = Canglong(cfg)
    x = torch.randn(1, cfg.in_chans, cfg.n_months, *cfg.img_size)
    tgt = torch.randn(cfg.out_months, *cfg.img_size)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    losses = []
    for _ in range(15):
        opt.zero_grad()
        loss = canglong_loss(m(x)[0, 0], tgt)
        loss.backward(); opt.step(); losses.append(loss.item())
    assert np.isfinite(losses).all() and losses[-1] < 0.8 * losses[0]
