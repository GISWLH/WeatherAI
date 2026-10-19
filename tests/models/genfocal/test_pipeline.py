import pytest
import torch
from weatherai import GenFocal_lite
from weatherai.models.genfocal import resize_field


def test_daily_to_hourly_interpolation_replicates_time():
    x = torch.tensor([0., 10.]).reshape(1, 2, 1, 1, 1)
    out = resize_field(x, (4, 2, 2))
    torch.testing.assert_close(out[0, :, 0, 0, 0], torch.tensor([0., 0., 10., 10.]))


def test_edm_zero_network_analytical_and_backward():
    m = GenFocal_lite(2)
    x = torch.randn(1, 2, 8, 8, 2)
    c = torch.randn_like(x)
    sigma = torch.tensor([2.])
    # Final convolution is zero initialized; EDM skip is exactly 1/(1+sigma^2).
    torch.testing.assert_close(m.super_resolution(x, sigma, c), x / 5)
    loss = m.super_resolution.loss(x, c, sigma)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.super_resolution.parameters())


def test_two_stage_sampling_and_physical_units():
    m = GenFocal_lite(2).eval()
    x = torch.randn(1, 2, 8, 8, 2)
    cond = {k: torch.zeros_like(x) for k in m.debiasing.cfg.cond_keys}
    kw = dict(output_size=(4, 16, 8), flow_steps=2, diffusion_steps=3)
    out = m.downscale(x, cond, generator=torch.Generator().manual_seed(0), **kw)
    assert out['output'].shape == (1, 4, 16, 8, 2)
    assert torch.isfinite(out['output']).all()
    torch.testing.assert_close(out['debiased'], x)
    physical = m.downscale(x, cond, baseline=torch.full_like(out['output'], 280), residual_mean=2.,
                          residual_std=3., generator=torch.Generator().manual_seed(0), **kw)
    torch.testing.assert_close(physical['output'], 282 + 3 * out['residual'])
    with pytest.raises(ValueError):
        m.super_resolution.sample(x, steps=1)
