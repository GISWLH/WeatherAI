import os
import numpy as np
import torch

from weatherai.models.unicm import UniCM, UniCMConfig, climate_modes, patchify, unpatchify, unicm_loss

REF = os.path.join(os.path.dirname(__file__), "data", "ref_lite.npz")


def _ref():
    d = np.load(REF)
    return d, {k[3:]: torch.from_numpy(d[k]) for k in d.files if k.startswith("sd/")}


def test_matches_official_reference():
    """eval rollout reproduces the official ``models.UniCM`` output (random lite weights, stored reference)."""
    d, sd = _ref()
    m = UniCM(UniCMConfig.lite()).eval()
    m.load_state_dict(sd, strict=True)
    with torch.no_grad():
        f, md = m(torch.from_numpy(d["x"]), torch.from_numpy(d["xm"]), torch.from_numpy(d["months"]).long(), train=False)
    assert (f - torch.from_numpy(d["field"])).abs().max() < 1e-5
    assert (md - torch.from_numpy(d["mode"])).abs().max() < 1e-5


def test_patchify_roundtrip_and_fold():
    x = torch.randn(2, 3, 5, 12, 72)
    p = patchify(x, (2, 2))
    assert p.shape == (2, 3, 20, 6, 36)
    assert torch.equal(unpatchify(p, (2, 2)), x)
    f = torch.nn.functional.fold(p.flatten(0, 1).flatten(-2), (12, 72), 2, stride=2).reshape(x.shape)
    assert torch.equal(f, x)


def test_official_size_param_count():
    assert sum(p.numel() for p in UniCM(UniCMConfig.official()).parameters()) == 12_726_037


def test_climate_modes_and_loss_trains():
    cfg = UniCMConfig.lite()
    torch.manual_seed(0)
    x = torch.randn(4, cfg.his_len + cfg.pred_len, 5, 12, 72)
    months = (torch.arange(x.shape[1])[None] + torch.arange(4)[:, None]) % 12
    assert climate_modes(x, cfg).shape == (4, cfg.n_mode_series, x.shape[1])
    m = UniCM(cfg)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    losses = []
    for _ in range(25):
        opt.zero_grad()
        l, _ = unicm_loss(m, x, months)
        l.backward(); opt.step(); losses.append(float(l))
    assert losses[-1] < 0.8 * losses[0]


def test_dropout_only_in_train():
    cfg = UniCMConfig.lite(dropout=0.3)
    m = UniCM(cfg).eval()
    x = torch.randn(1, cfg.his_len + cfg.pred_len, 5, 12, 72)
    xm = torch.randn(1, cfg.n_mode_series, x.shape[1])
    mo = torch.zeros(1, x.shape[1], dtype=torch.long)
    with torch.no_grad():
        a = m(x, xm, mo)[0]; b = m(x, xm, mo)[0]
    assert torch.equal(a, b)
