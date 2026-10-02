"""StormCast parity vs NVIDIA PhysicsNeMo (oracle). Skipped when physicsnemo is not installed (``/workspace/venv/pn``).

Small random-weight tests run anywhere with physicsnemo. The full-size official-checkpoint test (512x640, ~3 min CPU) runs
with ``STORMCAST_FULL=1`` and the checkpoint in $STORMCAST_CKPT."""
import os

import pytest
import torch

pytest.importorskip("physicsnemo", reason="PhysicsNeMo oracle not installed")
from weatherai.models.stormcast import edm_heun_sample
from weatherai.models.stormcast.unet import EDMPrecond, SongUNetConfig, StormCastUNet

CKPT = os.environ.get("STORMCAST_CKPT", "/workspace/ckpt/stormcast")




def _oracle_pair(res=(64, 64), n_in=14, n_out=8, ch=32, mult=(1, 2, 2), nb=1):
    from physicsnemo.diffusion.preconditioners import EDMPrecond as PEDM
    from physicsnemo.models.diffusion_unets import StormCastUNet as PReg
    kw = dict(model_channels=ch, channel_mult=list(mult), num_blocks=nb, attn_resolutions=[])
    pr = PReg(img_resolution=list(res), img_in_channels=n_in, img_out_channels=n_out, embedding_type="zero", additive_pos_embed=False, **kw).eval()
    pd = PEDM(img_resolution=list(res), img_channels=3 * n_out + 2, img_out_channels=n_out, model_type="SongUNet", additive_pos_embed=True, **kw).eval()
    nr = StormCastUNet(SongUNetConfig(res, n_in, n_out, ch, mult, num_blocks=nb, embedding_type="zero")).eval()
    nd = EDMPrecond(SongUNetConfig(res, 3 * n_out + 2, n_out, ch, mult, num_blocks=nb, embedding_type="positional", additive_pos_embed=True)).eval()
    for ref in (pr, pd):  # random (non-zero) weights incl. the zero-initialised output convs
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for p in ref.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.08)
    for ref, nat in ((pr, nr), (pd, nd)):
        sd = {k.replace(".attn.norm.", ".norm2.").replace(".attn.", "."): v for k, v in ref.state_dict().items()
              if not k.endswith(("device_buffer", "map_noise.freqs")) and k != "sigma_data"}  # newer PhysicsNeMo nests the block attention in .attn
        nat.load_state_dict(sd, strict=True)  # identical names / shapes -> strict
    return pr, pd, nr, nd


def test_networks_bitwise_vs_physicsnemo_random_weights():
    pr, pd, nr, nd = _oracle_pair()
    x = torch.randn(2, 14, 64, 64)
    with torch.no_grad():
        assert torch.allclose(pr(x), nr(x), atol=1e-6, rtol=1e-5)
        xn, cond = torch.randn(2, 8, 64, 64), torch.randn(2, 18, 64, 64)
        for sig in (0.002, 0.7, 40.0, 800.0):
            s = torch.full((2,), sig)
            assert torch.allclose(pd(xn, s, condition=cond), nd(xn, s, cond), atol=1e-5, rtol=1e-5), sig


def test_heun_sampler_vs_physicsnemo_sample():
    from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    from physicsnemo.diffusion.samplers import sample
    pr, pd, nr, nd = _oracle_pair()
    cond = torch.randn(1, 18, 64, 64)
    lat = 800.0 * torch.randn(1, 8, 64, 64, generator=torch.Generator().manual_seed(3))
    sch = EDMNoiseScheduler(sigma_min=0.002, sigma_max=800, rho=7)
    den = sch.get_denoiser(x0_predictor=lambda z, t: pd(z, t, condition=cond))
    with torch.no_grad():
        ref = sample(den, lat.clone(), noise_scheduler=sch, num_steps=6, solver="edm_stochastic_heun",
                     solver_options=dict(S_churn=0.0, S_min=0.0, S_max=float("inf"), S_noise=1))
        ours = edm_heun_sample(lambda z, s: nd(z, s, cond), lat.clone(), num_steps=6)
    assert (ref - ours).abs().max() < 1e-3 * max(1.0, ref.abs().max().item()), (ref - ours).abs().max()




@pytest.mark.skipif(not (os.environ.get("STORMCAST_FULL") and os.path.exists(f"{CKPT}/EDMPrecond.0.0.mdlus")),
                    reason="set STORMCAST_FULL=1 with the official checkpoint in $STORMCAST_CKPT")
def test_official_checkpoint_full_size_exact():
    from physicsnemo.diffusion.preconditioners import EDMPrecond as PEDM
    from physicsnemo.models.diffusion_unets import StormCastUNet as PReg
    from weatherai.models.stormcast import load_official, read_mdlus
    a_r, sd_r = read_mdlus(f"{CKPT}/StormCastUNet.0.0.mdlus")
    a_d, sd_d = read_mdlus(f"{CKPT}/EDMPrecond.0.0.mdlus")
    ref_r, ref_d = PReg(**a_r["__args__"]).eval(), PEDM(**a_d["__args__"]).eval()
    ref_r.load_state_dict(sd_r, strict=True)
    ref_d.load_state_dict(sd_d, strict=False)  # only 'sigma_data' (a constant) missing
    m = load_official(CKPT, with_metadata=False)
    torch.manual_seed(0)
    with torch.no_grad():
        x = torch.randn(1, 127, 512, 640)
        assert (ref_r(x) - m.regression(x)).abs().max() < 1e-5
        xn, cond = 3 * torch.randn(1, 99, 512, 640), torch.randn(1, 200, 512, 640)
        for sig in (0.002, 5.0, 800.0):
            s = torch.tensor([sig])
            assert (ref_d(xn, s, condition=cond) - m.diffusion(xn, s, cond)).abs().max() < 1e-5
