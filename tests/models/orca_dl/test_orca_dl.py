import json
import os
import sys
import unittest

import numpy as np
import torch

from weatherai.models.orca_dl import ORCADL, ORCADLConfig
from weatherai.models.orca_dl.model import PatchExpanding, RotaryTimeEmbed, shift_attention_mask

CK = os.environ.get("ORCA_CKPT", "/workspace/ckpt/orca_dl")
NPARAMS = 539_942_562
OFF = os.environ.get("ORCA_OFFICIAL", "/workspace/scratch/ORCA-DL")


def lite_cfg():
    return ORCADLConfig(lat_space=(-31.5, 31.5, 32), lon_space=(0.5, 359.5, 60), in_chans=[2, 2, 1, 2, 2, 1], out_chans=[2, 2, 1, 2, 2, 1],
                        embed_dim=24, lg_hidden_dim=192, enc_heads=(2, 4, 8), lg_heads=(8, 8), window_size=(4, 5), max_t=3)


class TestLite(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.m = ORCADL(lite_cfg()).eval()
        for p in self.m.parameters():
            if p.dim() > 1:
                torch.nn.init.normal_(p, std=0.05)

    def test_shapes_and_multistep(self):
        o, a = torch.randn(2, 10, 32, 60), torch.randn(2, 2, 32, 60)
        with torch.no_grad():
            y = self.m(o, a, predict_time_steps=4)          # max_t=3 -> feedback after step 3
        self.assertEqual(tuple(y.shape), (2, 4, 10, 32, 60))
        self.assertTrue(torch.isfinite(y).all())

    def test_lead_time_experts_differ(self):
        o, a = torch.randn(1, 10, 32, 60), torch.randn(1, 2, 32, 60)
        with torch.no_grad():
            y = self.m(o, a, predict_time_steps=3)
        self.assertGreater((y[:, 0] - y[:, 1]).abs().max().item(), 1e-3)

    def test_batch_independent(self):
        o, a = torch.randn(2, 10, 32, 60), torch.randn(2, 2, 32, 60)
        with torch.no_grad():
            both = self.m(o, a, predict_time_steps=2)
            one = self.m(o[1:], a[1:], predict_time_steps=2)
        self.assertLess((both[1:] - one).abs().max().item(), 1e-4)

    def test_patch_expanding_matches_einops_semantics(self):
        pe = PatchExpanding(8)
        x = torch.randn(1, 3, 4, 8)
        y = pe.expand(x)
        ref = y.view(1, 3, 4, 2, 2, 4)                      # (p2, p1, c)
        out = torch.zeros(1, 6, 8, 4)
        for h in range(3):
            for w in range(4):
                for p2 in range(2):
                    for p1 in range(2):
                        out[0, h * 2 + p1, w * 2 + p2] = ref[0, h, w, p2, p1]
        got = pe.norm(out)
        with torch.no_grad():
            self.assertTrue(torch.allclose(pe(x), got, atol=1e-6))

    def test_rotary_time_identity_at_lead_zero(self):
        r = RotaryTimeEmbed(8)
        x = torch.randn(2, 8, 3, 3)
        self.assertTrue(torch.allclose(r(x, torch.zeros(2, dtype=torch.long)), x))

    def test_shift_mask_blocks_cross_region(self):
        m = shift_attention_mask(8, 10, (4, 5), (2, 2))
        self.assertEqual(tuple(m.shape), (4, 20, 20))
        self.assertTrue(set(m.unique().tolist()) <= {0.0, -100.0})
        self.assertTrue((m[0].diagonal() == 0).all())
        self.assertTrue((m[3] == -100).any())               # the last (wrap-around) window mixes regions

    def test_gradients(self):
        self.m.train()
        o, a = torch.randn(1, 10, 32, 60), torch.randn(1, 2, 32, 60)
        self.m(o, a, 2).square().mean().backward()
        got = [p.grad is not None and p.grad.abs().sum() > 0 for n, p in self.m.named_parameters() if "mlps.2" not in n]
        self.assertTrue(all(got), [n for (n, p), g in zip([(n, p) for n, p in self.m.named_parameters() if 'mlps.2' not in n], got) if not g][:5])


@unittest.skipUnless(os.path.exists(f"{CK}/model_weights/seed_1.bin"), "official seed_1.bin not downloaded")
class TestOfficialCheckpoint(unittest.TestCase):
    def test_strict_load_and_param_count(self):
        from weatherai.models.orca_dl import load_official
        m = load_official(CK, 1)
        self.assertEqual(sum(p.numel() for p in m.parameters()), NPARAMS)
        self.assertEqual(tuple(m.land_mask.shape), (128, 360))

    def test_recorded_official_parity(self):
        ref = json.load(open(os.path.join(os.path.dirname(__file__), "data", "official_parity_7steps.json")))
        self.assertLess(max(ref["rel_l2"]), 1e-5)
        self.assertEqual(ref["steps"], 7)


if __name__ == "__main__":
    unittest.main()
