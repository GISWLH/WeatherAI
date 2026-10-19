import numpy as np
import pytest
import torch
from scipy.optimize import isotonic_regression

from weatherai.models.pbc import PBCLite as PBC, PBC_lite, project_cdf, ensemble_cdf, seasonal_mask


def test_projection_matches_independent_scipy_oracle():
    x = torch.randn(64, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    expected = np.stack([isotonic_regression(row.clip(0, 1)).x for row in x.numpy()])
    np.testing.assert_allclose(project_cdf(x).numpy(), expected, atol=1e-14)


def test_fit_predict_against_numpy_least_squares_and_checkpoint():
    torch.manual_seed(7)
    args = [torch.rand(50, 3, 4, dtype=torch.float64) for _ in range(4)]
    raw, clim, lag1, lag2 = args
    obs = (torch.rand_like(raw) > .5).double()
    model = PBC().fit(raw, obs, clim, lag1, lag2)
    coefficients = np.empty((3, 4, 5))
    for g in range(3):
        for q in range(4):
            design = np.stack([np.ones(50), clim[:, g, q], lag1[:, g, q], lag2[:, g, q], raw[:, g, q]], -1)
            coefficients[g, q] = np.linalg.lstsq(design, obs[:, g, q], rcond=None)[0]
    np.testing.assert_allclose(model.coefficients.numpy(), coefficients, atol=1e-12)
    predicted = model.predict_proba(*args)
    assert predicted.shape == (50, 3, 5)
    assert predicted.min() >= -1e-12
    torch.testing.assert_close(predicted.sum(-1), torch.ones(50, 3, dtype=torch.float64))
    restored = PBC_lite().double()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(*args), model(*args))


def test_rank_deficiency_and_validation():
    x = torch.full((10, 2, 4), .4)
    m = PBC().fit(x, x, x, x, x)
    assert torch.isfinite(m(x, x, x, x)).all()
    with pytest.raises(RuntimeError, match="fitted"):
        PBC()(x, x, x, x)
    with pytest.raises(ValueError):
        project_cdf(torch.tensor([float('nan')]))


def test_empirical_cdf_and_past_only_window():
    torch.testing.assert_close(ensemble_cdf(torch.tensor([0., 0., 1., 2.]), torch.tensor([0., 1., 2., 3.])),
                               torch.tensor([0., .5, .75, 1.]))
    assert seasonal_mask(['2019-10-19', '2020-10-19', '2021-10-19'], '2020-10-19').tolist() == [True, False, False]
