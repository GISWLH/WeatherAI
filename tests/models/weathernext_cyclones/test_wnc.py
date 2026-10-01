import os
import unittest

import numpy as np

from weatherai.models.weathernext_cyclones import (
    CHECKPOINTS,
    WeatherNextCyclonesWrapper,
    download_checkpoint,
    download_sample_data,
    weathernext_available,
)

CKPT = os.environ.get("WNC_MINI_CKPT")  # optional local WeatherNextCyclones_Mini_<2024.npz
DATA = os.environ.get("WNC_SAMPLE_NC")  # optional local 1.0deg steps-04 sample .nc


@unittest.skipUnless(weathernext_available(), "needs python>=3.12 + official `weathernext` package")
class TestWNCWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import xarray

        try:
            ck = CKPT or download_checkpoint()
            nc = DATA or download_sample_data()
        except Exception as e:
            raise unittest.SkipTest(f"cannot fetch official files: {e}")
        cls.w = WeatherNextCyclonesWrapper(ck)  # official checkpoint, official config
        cls.ds = xarray.load_dataset(nc).compute()
        cls.out = cls.w.forecast(cls.ds, steps=2, num_members=2)

    def test_catalogue(self):
        self.assertIn("WeatherNextCyclones_Mini_<2024", CHECKPOINTS)
        with self.assertRaises(KeyError):
            download_checkpoint("nope")

    def test_task_is_13_levels_1deg_12h_input(self):
        self.assertEqual(tuple(self.w.task.pressure_levels), (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000))
        self.assertEqual(self.w.task.input_duration, "12h")

    def test_output_shapes_and_finite(self):
        o = self.out
        self.assertEqual(o.sizes["sample"], 2)
        self.assertEqual(o.sizes["time"], 2)
        self.assertEqual(tuple(o.temperature.shape), (2, 2, 1, 13, 181, 360))
        for k in ("temperature", "geopotential", "u_component_of_wind", "v_component_of_wind",
                  "specific_humidity", "mean_sea_level_pressure", "2m_temperature"):
            self.assertTrue(np.isfinite(o[k].values).all(), k)

    def test_members_differ_but_agree_roughly(self):
        t = self.out.temperature.isel(batch=0, level=7, time=1)  # 500 hPa, +12 h
        d = float(np.sqrt(((t.isel(sample=0) - t.isel(sample=1)) ** 2).mean()))
        self.assertGreater(d, 0.0)
        self.assertLess(d, 3.0)  # K; members are noise-perturbed samples of one forecast

    def test_short_range_error_vs_hres_analysis_is_small(self):
        """Sanity (not a skill score): 500 hPa T at +12 h beats persistence against the HRES frame in the sample file."""
        _, tgt, _ = self.w.split(self.ds, 2)
        tg = tgt.temperature.isel(batch=0, level=7, time=1).values
        err = self.out.temperature.isel(sample=0, batch=0, level=7, time=1).values - tg
        pers = self.ds.temperature.isel(batch=0, level=7, time=1).values - tg  # state at init time
        self.assertLess(float(np.sqrt((err**2).mean())), 1.5)
        self.assertLess(float(np.sqrt((err**2).mean())), float(np.sqrt((pers**2).mean())))

    def test_to_torch(self):
        import torch

        t = WeatherNextCyclonesWrapper.to_torch(self.out, ["temperature"])["temperature"]
        self.assertIsInstance(t, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
