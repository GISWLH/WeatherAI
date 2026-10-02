import os
import unittest

import numpy as np
import torch

from weatherai.models.tropicyclonenet import TCNM, TCNMConfig, relative_to_abs, to_physical, track_error_km

CK = os.environ.get("TCN_CKPT", "/workspace/ckpt/tcn/checkpoint_with_model_16000.pt")
REF = os.path.join(os.path.dirname(__file__), "data", "ref_ni_fani.npz")


def inputs(B=3, T=8):
    from weatherai.models.tropicyclonenet.model import ENV_DIMS, ENV_KEYS
    g = torch.Generator().manual_seed(0)
    return (torch.randn(T, B, 4, generator=g) * 0.1, torch.rand(B, 1, T, 64, 64, generator=g),
            {k: torch.rand(B, T, d, generator=g) for k, d in zip(ENV_KEYS, ENV_DIMS)})


class TestRandomWeights(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.m = TCNM().eval()

    def test_shapes(self):
        rel, img, env = inputs()
        with torch.no_grad():
            gens, logits = self.m.all_generators(rel, img, env, torch.randn(3, 16))
            samp, cls = self.m.sample(rel, img, env, num_samples=5)
        self.assertEqual(tuple(gens.shape), (4, 6, 3, 4))
        self.assertEqual(tuple(logits.shape), (3, 6))
        self.assertEqual(tuple(samp.shape), (4, 5, 3, 4))
        self.assertEqual(tuple(cls.shape), (3, 5))

    def test_sample_fills_every_member_even_for_one_storm(self):
        rel, img, env = inputs(B=1)
        with torch.no_grad():
            for seed in range(6):
                torch.manual_seed(seed)
                samp, _ = self.m.sample(rel, img, {k: v for k, v in env.items()}, num_samples=6)
                self.assertTrue((samp != 0).any(-1).all())      # official code leaves such members at 1.0 when the drawn class id >= n_unique

    def test_gradients(self):
        self.m.train()
        rel, img, env = inputs(B=4)
        gens, logits = self.m.all_generators(rel, img, env, torch.randn(4, 16))
        (gens.square().mean() + logits.square().mean()).backward()
        missing = [n for n, p in self.m.named_parameters() if p.grad is None and not n.startswith(("env_net.", "encoder.time", "encoder_env.time")) and "time_embedding" not in n]
        self.assertEqual(missing, [])

    def test_units_and_error(self):
        ll, me = to_physical(torch.tensor([[1.0, 2.0]]), torch.tensor([[0.0, 0.0]]))
        self.assertAlmostEqual(float(ll[0, 0]), 1850.0)
        self.assertAlmostEqual(float(ll[0, 1]), 100.0)
        self.assertAlmostEqual(float(me[0, 0]), 960.0)
        a = torch.tensor([[1800.0, 0.0]])
        b = torch.tensor([[1800.0, 10.0]])                       # 1 degree latitude = 111 km
        self.assertAlmostEqual(float(track_error_km(b, a)), 111.0, places=3)
        inc = torch.ones(4, 2, 3, 4)
        self.assertEqual(float(relative_to_abs(inc, torch.zeros(3, 4))[-1, 0, 0, 0]), 4.0)


@unittest.skipUnless(os.path.exists(CK), "official checkpoint not downloaded")
class TestOfficialCheckpoint(unittest.TestCase):
    def test_strict_load_and_reference(self):
        from weatherai.models.tropicyclonenet import load_official
        m = load_official(CK)
        self.assertEqual(sum(p.numel() for p in m.parameters()), 4_767_195)
        r = np.load(REF)
        env = {k[4:]: torch.tensor(r[k]) for k in r.files if k.startswith("env_")}
        with torch.no_grad():
            gens, logits = m.all_generators(torch.tensor(r["obs_traj_rel"]), torch.tensor(r["image_obs"]), env, torch.tensor(r["noise"]))
        self.assertLess(float((gens - torch.tensor(r["gens"])).abs().max()), 1e-4)
        self.assertLess(float((logits - torch.tensor(r["logits"])).abs().max()), 1e-4)


if __name__ == "__main__":
    unittest.main()
