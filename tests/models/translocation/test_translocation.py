import pytest
import torch
from weatherai.models.translocation import Translocation, Translocation_lite


def test_signed_erf_and_region_selection_analytical():
    model = Translocation(torch.nn.Conv2d(1, 1, 1, bias=False)).eval()
    with torch.no_grad():
        model.backbone.weight.fill_(-2)
    x = torch.tensor([[[[1., -3.], [2., 4.]]]])
    mask = torch.tensor([[True, True], [False, False]])
    out = model.effective_receptive_field(x, region_mask=mask)
    assert out['location'].tolist() == [[0, 1]]
    torch.testing.assert_close(out['gradient'], torch.tensor([[[[0., -2.], [0., 0.]]]]))
    assert model.backbone.weight.grad is None and x.grad is None


def test_graphcast_erf_finite_difference():
    torch.manual_seed(5)
    model = Translocation_lite()
    x = torch.randn(1, 4, 8, 16)
    result = model.effective_receptive_field(x, location=(3, 4))
    delta = torch.randn_like(x)
    eps = 1e-3
    with torch.no_grad():
        difference = (model(x + eps * delta) - model(x - eps * delta))[0, 0, 3, 4] / (2 * eps)
    expected = (result['gradient'] * delta).sum()
    torch.testing.assert_close(difference, expected, atol=2e-3, rtol=2e-2)
    with pytest.raises(ValueError):
        model.effective_receptive_field(x)
