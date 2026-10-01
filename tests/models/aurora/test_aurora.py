import os
import unittest
from datetime import datetime

import torch

from weatherai.models import Aurora_lite, Aurora_small
from weatherai.models.aurora import aurora_available

SMALL_CKPT = os.environ.get("AURORA_SMALL_CKPT")  # optional local path to the official small ckpt


def _inputs(H=16, W=32, B=1, T=2, L=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(B, T, 4, H, W, generator=g),
        torch.randn(3, H, W, generator=g),
        torch.randn(B, T, 5, L, H, W, generator=g),
    )


@unittest.skipUnless(aurora_available(), "needs `pip install microsoft-aurora`")
class TestAuroraWrapper(unittest.TestCase):
    def test_lite_shapes_finite(self):
        m = Aurora_lite().eval()
        surf, static, atmos = _inputs()
        with torch.no_grad():
            s, a = m(surf, static, atmos)
        self.assertEqual(tuple(s.shape), (1, 4, 16, 32))
        self.assertEqual(tuple(a.shape), (1, 5, 4, 16, 32))
        self.assertTrue(torch.isfinite(s).all() and torch.isfinite(a).all())

    def test_lite_matches_upstream_batch_api(self):
        """Wrapper must return exactly what upstream Aurora returns for the same Batch."""
        m = Aurora_lite().eval()
        surf, static, atmos = _inputs(seed=1)
        batch = m.make_batch(surf, static, atmos, time=(datetime(2020, 6, 1, 12),))
        with torch.no_grad():
            ref = m.core(batch)
            s, a = m(surf, static, atmos, time=(datetime(2020, 6, 1, 12),))
        self.assertTrue(torch.equal(s[:, 0], ref.surf_vars["2t"][:, 0]))
        self.assertTrue(torch.equal(a[:, 3], ref.atmos_vars["t"][:, 0]))

    def test_lite_backward(self):
        m = Aurora_lite()
        surf, static, atmos = _inputs(seed=2)
        s, a = m(surf, static, atmos)
        (s.pow(2).mean() + a.pow(2).mean()).backward()
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        self.assertGreater(len(grads), 0)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_batch_size_2(self):
        m = Aurora_lite().eval()
        surf, static, atmos = _inputs(B=2)
        with torch.no_grad():
            s, a = m(surf, static, atmos, time=(datetime(2020, 6, 1, 12),) * 2)
        self.assertEqual(s.shape[0], 2)

    def test_wrong_levels_raises(self):
        m = Aurora_lite()
        surf, static, atmos = _inputs(L=3)
        with self.assertRaises(ValueError):
            m(surf, static, atmos)

    @unittest.skipUnless(SMALL_CKPT and os.path.exists(SMALL_CKPT), "set AURORA_SMALL_CKPT to run")
    def test_small_official_checkpoint_strict_load_and_forward(self):
        m = Aurora_small(checkpoint_path=SMALL_CKPT).eval()  # strict=True load
        surf, static, atmos = _inputs(H=17, W=32)
        with torch.no_grad():
            s, a = m(surf, static, atmos)
        self.assertTrue(torch.isfinite(s).all() and torch.isfinite(a).all())
        self.assertEqual(tuple(s.shape[-2:]), (16, 32))


if __name__ == "__main__":
    unittest.main()
