import os
import unittest

import numpy as np
import torch

from weatherai.models.aardvark import AardvarkProcessor, AardvarkProcessor_lite, load_official_processor

REF = os.path.join(os.path.dirname(__file__), "data", "processor_ref.npz")
CKPT = os.environ.get("AARDVARK_PROC_CKPT")  # official trained_model/processor/forecast_1/epoch_0


class TestAardvarkProcessorLite(unittest.TestCase):
    def test_shapes_finite(self):
        m = AardvarkProcessor_lite().eval()
        y = m(torch.randn(2, 35, 60, 31))
        self.assertEqual(tuple(y.shape), (2, 31, 60, 24))
        self.assertTrue(torch.isfinite(y).all())

    def test_backward(self):
        m = AardvarkProcessor_lite()
        m(torch.randn(1, 35, 60, 31)).pow(2).mean().backward()
        g = [p.grad for p in m.parameters() if p.grad is not None]
        self.assertGreater(len(g), 0)
        self.assertTrue(all(torch.isfinite(x).all() for x in g))

    def test_wrong_grid_raises(self):
        with self.assertRaises(ValueError):
            AardvarkProcessor_lite()(torch.randn(1, 35, 61, 31))

    def test_param_names_match_official_layout(self):
        m = AardvarkProcessor()
        n = m.num_parameters()
        self.assertEqual(n, 53_923_416)  # official processor ViT (decoder_lr.*) size
        keys = set(m.state_dict())
        for k in ("decoder_lr.var_embed", "decoder_lr.pos_embed", "decoder_lr.token_embeds.34.proj.weight",
                  "decoder_lr.blocks.15.mlp.fc2.bias", "decoder_lr.head.8.weight", "decoder_lr.var_agg.in_proj_weight"):
            self.assertIn(k, keys)

    def test_forecast_step_roundtrip(self):
        m = AardvarkProcessor_lite().eval()
        x = torch.randn(1, 35, 60, 31)
        mean, std = torch.zeros(24), torch.ones(24)
        fc, nxt = m.forecast_step(x, mean, std, torch.zeros(24), torch.ones(24))
        self.assertEqual(tuple(fc.shape), (1, 31, 60, 24))
        self.assertEqual(tuple(nxt.shape), (1, 35, 60, 31))
        self.assertTrue(torch.equal(nxt[:, 24:], x[:, 24:]))  # aux channels carried over


@unittest.skipUnless(CKPT and os.path.exists(CKPT), "set AARDVARK_PROC_CKPT to the official forecast_1/epoch_0 file")
class TestAardvarkOfficialCheckpoint(unittest.TestCase):
    """Reference crops come from the official ConvCNPWeather(forecast, vit) + same checkpoint
    (scripts/aardvark_processor_reference.py)."""

    @classmethod
    def setUpClass(cls):
        cls.m = load_official_processor(CKPT, strict=True).eval()  # strict load of all decoder_lr.* keys
        cls.ref = np.load(REF)

    def _check(self, lt):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(1, 35, 240, 121, generator=g)
        with torch.no_grad():
            y = self.m(x, torch.full((1, 1), float(lt)))
        np.testing.assert_allclose(y[:, ::6, ::6].numpy(), self.ref[f"y_lt{lt}"], rtol=1e-3, atol=2e-4)

    def test_matches_official_lead_time_0(self):
        self._check(0)

    def test_matches_official_lead_time_1(self):
        self._check(1)


if __name__ == "__main__":
    unittest.main()
