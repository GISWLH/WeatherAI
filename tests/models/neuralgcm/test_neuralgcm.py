import os
import unittest

import numpy as np

from weatherai.models.neuralgcm import NeuralGCMWrapper, neuralgcm_available
from weatherai.models.neuralgcm.neuralgcm import CHECKPOINTS, download_checkpoint

CKPT = os.environ.get("NGCM_CKPT")  # optional local path to deterministic_2_8_deg.pkl


@unittest.skipUnless(neuralgcm_available(), "needs `pip install neuralgcm dinosaur jax`")
class TestNeuralGCMWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            path = CKPT or download_checkpoint("deterministic_2_8_deg")
        except Exception as e:  # offline
            raise unittest.SkipTest(f"cannot fetch official checkpoint: {e}")
        cls.w = NeuralGCMWrapper.from_checkpoint_file(path)
        cls.ds = cls.w.demo_data()

    def test_catalogue(self):
        self.assertIn("deterministic_2_8_deg", CHECKPOINTS)
        with self.assertRaises(KeyError):
            download_checkpoint("nope")

    def test_metadata(self):
        self.assertEqual(self.w.grid_shape, (128, 64))
        self.assertEqual(len(self.w.levels), 37)
        self.assertIn("temperature", self.w.input_variables)
        self.assertEqual(sorted(self.w.forcing_variables), ["sea_ice_cover", "sea_surface_temperature"])

    def test_forecast_finite_and_evolves(self):
        out = self.w.forecast(self.ds, steps=2, step_hours=6)
        self.assertEqual(out.temperature.shape, (2, 37, 128, 64))
        for k, v in out.data_vars.items():
            self.assertTrue(np.isfinite(v.values).all(), k)
        t = out.temperature.sel(level=500)
        self.assertGreater(float(abs(t.isel(time=1) - t.isel(time=0)).mean()), 0.0)
        self.assertLess(float(abs(t.isel(time=1) - t.isel(time=0)).max()), 60.0)  # K, sanity bound

    def test_deterministic_model_is_repeatable(self):
        a = self.w.forecast(self.ds, steps=2, step_hours=6, seed=0)
        b = self.w.forecast(self.ds, steps=2, step_hours=6, seed=123)
        np.testing.assert_array_equal(a.temperature.values, b.temperature.values)

    def test_wrapper_equals_direct_upstream_call(self):
        import jax

        m = self.w.model
        first = self.ds.isel(time=0)
        state = m.encode(m.inputs_from_xarray(first), m.forcings_from_xarray(first), jax.random.key(0))
        _, preds = m.unroll(state, m.forcings_from_xarray(self.ds.head(time=1)), steps=2,
                            timedelta=np.timedelta64(6, "h"), start_with_input=True)
        ref = m.data_to_xarray(preds, times=np.arange(2) * 6)
        out = self.w.forecast(self.ds, steps=2, step_hours=6)
        np.testing.assert_array_equal(out.temperature.values, ref.temperature.values)

    def test_to_torch(self):
        import torch

        out = self.w.forecast(self.ds, steps=2, step_hours=6)
        t = NeuralGCMWrapper.to_torch(out, ["temperature"])
        self.assertIsInstance(t["temperature"], torch.Tensor)
        self.assertEqual(tuple(t["temperature"].shape), (2, 37, 128, 64))


if __name__ == "__main__":
    unittest.main()
