import os
import unittest

import numpy as np
import torch

from weatherai.models.gencast import (
    GenCast,
    GenCast_lite,
    SamplerConfig,
    dpm_solver_pp_2s_sample,
    noise_schedule,
    stochastic_churn_rate_schedule,
)

REF = os.path.join(os.path.dirname(__file__), "data", "sampler_ref.npz")


def _toy_denoise(x, s):  # same closed form used to generate the official-JAX reference
    s = s.view(-1, 1, 1)
    return x * (1 / (1 + s**2)) + 0.1 * x


class TestSamplerParityWithOfficialJAX(unittest.TestCase):
    """Reference arrays were produced by the *official* google-deepmind/weathernext sampler
    (scripts/gencast_sampler_reference.py) with a fixed noise tensor and a closed-form denoiser."""

    @classmethod
    def setUpClass(cls):
        cls.ref = np.load(REF)

    def test_noise_schedule(self):
        np.testing.assert_allclose(noise_schedule(80.0, 0.03, 20, 7.0), self.ref["schedule_80_0.03_20_7"], rtol=1e-6)

    def test_churn_schedule(self):
        lv = noise_schedule(80.0, 0.03, 20, 7.0)
        np.testing.assert_allclose(
            stochastic_churn_rate_schedule(lv, 2.5, 0.75, float("inf")), self.ref["churn_rates"], rtol=1e-6
        )

    def _run(self, churn):
        noise = torch.from_numpy(self.ref["noise"])
        cfg = SamplerConfig(num_noise_levels=6, stochastic_churn_rate=churn)
        return dpm_solver_pp_2s_sample(_toy_denoise, torch.zeros_like(noise), cfg, noise_fn=lambda t: noise)

    def test_sampler_with_churn_matches_official(self):
        np.testing.assert_allclose(self._run(2.5).numpy(), self.ref["out_churn"], rtol=2e-4, atol=1e-5)

    def test_sampler_deterministic_matches_official(self):
        np.testing.assert_allclose(self._run(0.0).numpy(), self.ref["out_det"], rtol=2e-4, atol=1e-5)


class TestGenCastLite(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.m = GenCast_lite()
        self.cond = torch.randn(2, 8, 16, 32)
        self.tgt = torch.randn(2, 4, 16, 32)

    def test_preconditioning_identities(self):
        s = torch.tensor([0.1, 1.0, 10.0])
        # sigma_data = 1: c_skip + c_out^2 = 1, c_skip = c_in^2, c_out = sigma * c_in
        np.testing.assert_allclose((GenCast.c_skip(s) + GenCast.c_out(s) ** 2).numpy(), 1.0, rtol=1e-6)
        np.testing.assert_allclose(GenCast.c_skip(s).numpy(), (GenCast.c_in(s) ** 2).numpy(), rtol=1e-6)
        np.testing.assert_allclose(GenCast.c_out(s).numpy(), (s * GenCast.c_in(s)).numpy(), rtol=1e-6)

    def test_denoise_shape_finite(self):
        sigma = torch.tensor([0.03, 80.0])
        d = self.m.denoise(torch.randn(2, 4, 16, 32), self.cond, sigma)
        self.assertEqual(tuple(d.shape), (2, 4, 16, 32))
        self.assertTrue(torch.isfinite(d).all())

    def test_loss_backward(self):
        loss = self.m.loss(self.tgt, self.cond)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grads = [p.grad for p in self.m.parameters() if p.grad is not None]
        self.assertGreater(len(grads), 0)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_sample_ensemble_shape_finite_and_spread(self):
        g = torch.Generator().manual_seed(1)
        s = self.m.eval().sample(self.cond, num_members=3, generator=g)
        self.assertEqual(tuple(s.shape), (3, 2, 4, 16, 32))
        self.assertTrue(torch.isfinite(s).all())
        self.assertGreater(float(s.std(0).mean()), 0.0)  # members differ

    def test_sample_reproducible_with_seed(self):
        a = self.m.eval().sample(self.cond, 2, torch.Generator().manual_seed(7))
        b = self.m.eval().sample(self.cond, 2, torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(a, b))

    def test_few_train_steps_reduce_loss_on_fixed_batch(self):
        m = GenCast_lite()
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        losses = []
        for _ in range(80):
            g = torch.Generator().manual_seed(0)  # fixed noise/sigma -> loss is a deterministic function of weights
            opt.zero_grad()
            loss = m.loss(self.tgt, self.cond, generator=g)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        self.assertLess(losses[-1], 0.8 * losses[0])


if __name__ == "__main__":
    unittest.main()
