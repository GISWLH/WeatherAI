import os
import numpy as np
import pytest
import torch

from weatherai.models.ace2 import ACE2, ACE2Config, SFNO, SFNOConfig, RealSHT, InverseRealSHT

CK = os.environ.get("ACE2_CKPT", "/workspace/ckpt/ace2")


def test_sht_roundtrip_bandlimited():
    h, w = 24, 48
    f, g = RealSHT(h, w, lmax=h, mmax=w // 2 + 1), InverseRealSHT(h, w, lmax=h, mmax=w // 2 + 1)
    c = f(torch.randn(2, 3, h, w))
    c2 = f(g(c))
    assert (c - c2).abs().max() < 1e-4 * c.abs().max()  # forward(inverse(c)) = c on the representable subspace


def test_sht_constant_field():
    h, w = 16, 32
    c = RealSHT(h, w)(torch.ones(1, 1, h, w))
    # Y_00 = 1/sqrt(4 pi): a constant 1 has coefficient sqrt(4 pi)
    assert abs(c[0, 0, 0, 0].real.item() - (4 * np.pi) ** 0.5) < 1e-4
    assert c[0, 0, 1:, :].abs().max() < 1e-4


def _lite_cfg():
    names_in = ["PRESsfc", "TMP2m", "DSWRFtoa"]
    names_out = ["PRESsfc", "TMP2m", "PRATEsfc"]
    return names_in, names_out


def test_sfno_lite_shapes_and_grads():
    cfg = SFNOConfig(in_chans=5, out_chans=4, img_shape=(12, 24), embed_dim=8, num_layers=2)
    net = SFNO(cfg)
    x = torch.randn(2, 5, 12, 24)
    y = net(x)
    assert y.shape == (2, 4, 12, 24)
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_stepper_lite_conservation():
    """Dry-air mass and the global moisture budget are conserved by construction, whatever the network outputs."""
    torch.manual_seed(0)
    h, w, nl = 12, 24, 3
    names = ["PRESsfc", "surface_temperature", "PRATEsfc", "LHTFLsfc", "tendency_of_total_water_path_due_to_advection"] + \
        [f"specific_total_water_{i}" for i in range(nl)]
    ins = ["ocean_fraction", "DSWRFtoa"] + names[:2] + names[5:]
    outs = names
    means = {n: 0.0 for n in ins + outs}
    stds = {n: 1.0 for n in ins + outs}
    means["PRESsfc"], stds["PRESsfc"] = 1e5, 1e3
    for i in range(nl):
        means[f"specific_total_water_{i}"], stds[f"specific_total_water_{i}"] = 5e-3, 1e-3
    cfg = ACE2Config(sfno=SFNOConfig(in_chans=len(ins), out_chans=len(outs), img_shape=(h, w), embed_dim=8, num_layers=2),
                     in_names=tuple(ins), out_names=tuple(outs), n_levels=nl,
                     force_positive=tuple(f"specific_total_water_{i}" for i in range(nl)) + ("PRATEsfc",))
    area = torch.rand(h, w) + 0.1
    ak = torch.tensor([0.0, 2000.0, 1000.0, 0.0])
    bk = torch.tensor([0.0, 0.1, 0.6, 1.0])
    m = ACE2(cfg, means, stds, area, ak, bk).eval()
    st = {n: torch.randn(1, h, w) * 0 + means[n] for n in m.prognostic}
    st["PRESsfc"] = 1e5 + 500 * torch.randn(1, h, w)
    forcing = {"ocean_fraction": (torch.rand(1, h, w) > 0.5).float(), "DSWRFtoa": torch.rand(1, h, w)}
    nxt = {"ocean_fraction": forcing["ocean_fraction"], "surface_temperature": torch.full((1, h, w), 290.0)}
    target = m.global_dry_air(st)
    with torch.no_grad():
        gen, target2 = m.step(st, forcing, nxt)
    assert torch.allclose(target, target2)
    assert (m.global_dry_air(gen) - target).abs().max() < 1e-2           # Pa, fp32 round-off level
    assert all((gen[f"specific_total_water_{i}"] >= 0).all() for i in range(nl))  # (precipitation is rescaled after the clamp, as in fme, so it is not guaranteed >= 0 for a random network)
    oc = forcing["ocean_fraction"] == 1
    assert torch.equal(gen["surface_temperature"][oc], nxt["surface_temperature"][oc])   # SST prescribed over ocean
    # moisture budget: global-mean dTWP/dt = global-mean (E - P)
    tend = (m.total_water_path(gen) - m.total_water_path({**st, **forcing})) / cfg.timestep_seconds
    lhs = m.area_mean(tend).item()
    rhs = m.area_mean(gen["LHTFLsfc"] / 2.5e6 - gen["PRATEsfc"]).item()
    assert abs(lhs - rhs) < 1e-6 * max(1.0, abs(rhs)) + 1e-9


@pytest.mark.skipif(not os.path.exists(os.path.join(CK, "ace2_era5_ckpt.tar")), reason="official ACE2-ERA5 checkpoint not downloaded")
def test_official_strict_load():
    from weatherai.models.ace2 import load_official
    m = load_official(CK)
    assert sum(p.numel() for p in m.net.parameters()) == 455_831_040
    assert len(m.cfg.in_names) == 44 and len(m.cfg.out_names) == 50
