"""NeuralGCM precipitation / evaporation checkpoints (weatherai.models.neuralgcm.precip): official load, rollout diagnostics,
precipitation-rate helpers, differentiable precipitation loss and a short fine-tune. Checkpoints: $NGCM_PRECIP_DIR or
/workspace/ckpt/ngcm_precip (tests that need them skip otherwise). The first dycore compile takes ~1 min on CPU."""
import os
import unittest

import numpy as np

from weatherai.models.neuralgcm import neuralgcm_available
from weatherai.models.neuralgcm import precip as P

D = os.environ.get("NGCM_PRECIP_DIR", "/workspace/ckpt/ngcm_precip")


class TestRateHelpers(unittest.TestCase):
    def test_rate_from_cumulative(self):
        cum = np.cumsum(np.full((4, 1, 3, 2), 1e-4, np.float32), axis=0)      # 0.1 mm per hour
        r = P.precip_rate_mm_per_hour({P.PRECIP_KEY: cum})
        self.assertEqual(r.shape, (4, 3, 2))
        np.testing.assert_allclose(r, 0.1, rtol=1e-5)
        cum2 = np.cumsum(np.full((3, 1, 3, 2), 2e-4, np.float32), axis=0)
        r2 = P.precip_rate_mm_per_hour({P.PRECIP_KEY: cum2}, hours=2)
        np.testing.assert_allclose(r2, 0.1, rtol=1e-5)


@unittest.skipUnless(neuralgcm_available() and all(os.path.exists(os.path.join(D, f)) for f in P.FILES.values()),
                     "needs jax neuralgcm dinosaur and the Zenodo precip/evap pickles")
class TestPrecipCheckpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import optax  # noqa: F401
        except Exception as e:
            raise unittest.SkipTest(f"needs optax: {e}")
        import neuralgcm

        cls.ck = {k: P.load_checkpoint(os.path.join(D, f)) for k, f in P.FILES.items()}
        cls.model = {k: P.build(v) for k, v in cls.ck.items()}
        m = cls.model["precip"]
        cls.ds = neuralgcm.demo.load_data(m.data_coords)

    def test_load_and_diagnostics(self):
        for k, m in self.model.items():
            self.assertEqual(tuple(m.data_coords.horizontal.nodal_shape), (128, 64))
            pred = P.rollout(m, self.ds, steps=3, hours=1)
            self.assertIn(P.PRECIP_KEY, pred)
            self.assertIn(P.EVAP_KEY, pred)
            self.assertEqual(pred[P.PRECIP_KEY].shape, (3, 1, 128, 64))
            self.assertTrue(np.isfinite(np.asarray(pred[P.PRECIP_KEY])).all())
            cum = np.asarray(pred[P.PRECIP_KEY])
            if k == "precip":    # the evaporation-predicting model *diagnoses* precipitation from the budget and can go negative
                self.assertTrue((np.diff(cum, axis=0) > -1e-6).all(), "predicted precipitation should not be negative")

    def test_param_counts_differ_from_deterministic(self):
        n = {k: sum(int(np.prod(v.shape)) for d in c["params"].values() for v in d.values()) for k, c in self.ck.items()}
        self.assertEqual(n["precip"], 11_145_090)
        self.assertEqual(n["evap"], 11_080_210)

    def test_seeds_give_different_members(self):
        m = self.model["precip"]
        a = P.rollout(m, self.ds, steps=2, seed=0)[P.PRECIP_KEY]
        b = P.rollout(m, self.ds, steps=2, seed=1)[P.PRECIP_KEY]
        self.assertGreater(float(np.abs(np.asarray(a) - np.asarray(b)).max()), 0.0)

    def test_loss_gradient_flows_to_all_module_groups(self):
        import jax

        m = self.model["precip"]
        first = self.ds.isel(time=0)
        inp, frc = m.inputs_from_xarray(first), m.forcings_from_xarray(first)
        tgt = np.full((1, 128, 64), 0.1, np.float32)
        loss, g = jax.value_and_grad(P.precip_loss)(m.params, m, inp, frc, tgt, 1)
        self.assertTrue(np.isfinite(float(loss)))
        leaves = jax.tree.leaves(g)
        self.assertTrue(all(bool(np.isfinite(np.asarray(x)).all()) for x in leaves))
        nz = sum(bool(np.abs(np.asarray(x)).max() > 0) for x in leaves)
        self.assertGreater(nz / len(leaves), 0.5)

    def test_short_finetune_reduces_loss_on_constant_target(self):
        import jax

        m = self.model["precip"]
        first = self.ds.isel(time=0)
        inp, frc = m.inputs_from_xarray(first), m.forcings_from_xarray(first)
        tgt = np.full((1, 128, 64), 0.15, np.float32)
        params, hist = P.fit_precip(m, inp, frc, tgt, n_updates=6, steps=1, lr=3e-4, log=lambda *_: None)
        self.assertLess(hist[-1][0], hist[0][0])
        changed = any(float(np.abs(np.asarray(a) - np.asarray(b)).max()) > 0 for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(m.params)))
        self.assertTrue(changed)


if __name__ == "__main__":
    unittest.main()
