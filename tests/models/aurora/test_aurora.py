"""Native Aurora: structure, training step, and numerical parity vs the official package."""
import os
import unittest
from datetime import datetime

import torch

from weatherai.models import Aurora, Aurora_lite, Aurora_small
from weatherai.models.aurora import AuroraWrapper, AuroraOfficial_lite, aurora_available
from weatherai.models.aurora.swin3d import (
    PatchMerging3D, PatchSplitting3D, crop_3d, pad_3d, window_partition_3d, window_reverse_3d,
)

SMALL_CKPT = os.environ.get("AURORA_SMALL_CKPT", "/workspace/ckpt/aurora-0.25-small-pretrained.ckpt")
ATOL, RTOL = 1e-5, 1e-4  # same fp32 tolerances as docs/graphcast_parity.md
T0 = (datetime(2020, 6, 1, 12),)


def _inputs(H=16, W=32, B=1, T=2, L=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    surf = torch.randn(B, T, 4, H, W, generator=g) * 0.5 + torch.tensor([278.0, 0, 0, 1e5]).view(1, 1, 4, 1, 1)
    return surf, torch.randn(3, H, W, generator=g), torch.randn(B, T, 5, L, H, W, generator=g)


class TestNativeAurora(unittest.TestCase):
    def test_lite_shapes_finite(self):
        m = Aurora_lite().eval()
        with torch.no_grad():
            s, a = m(*_inputs())
        self.assertEqual(tuple(s.shape), (1, 4, 16, 32))
        self.assertEqual(tuple(a.shape), (1, 5, 4, 16, 32))
        self.assertTrue(torch.isfinite(s).all() and torch.isfinite(a).all())

    def test_odd_latitude_is_cropped(self):
        m = Aurora_lite().eval()
        with torch.no_grad():
            s, a = m(*_inputs(H=17))
        self.assertEqual(tuple(s.shape[-2:]), (16, 32))
        with self.assertRaises(ValueError):
            m(*_inputs(H=18))

    def test_patch_split_merge_shapes_and_crop(self):
        # odd patch grid: merging pads, splitting crops the padding again
        x = torch.randn(2, 2 * 5 * 7, 8)
        down = PatchMerging3D(8)(x, (2, 5, 7))
        self.assertEqual(down.shape, (2, 2 * 3 * 4, 16))
        up = PatchSplitting3D(16)(down, (2, 3, 4), crop=(0, 1, 1))
        self.assertEqual(up.shape, (2, 2 * 5 * 7, 8))

    def test_window_partition_roundtrip_and_pad_crop(self):
        x = torch.randn(2, 4, 6, 12, 5)
        w = window_partition_3d(x, (2, 3, 4))
        self.assertTrue(torch.equal(window_reverse_3d(w, (2, 3, 4), 4, 6, 12), x))
        y = torch.randn(1, 3, 5, 7, 2)
        self.assertTrue(torch.equal(crop_3d(pad_3d(y, (1, 1, 1)), (1, 1, 1)), y))

    def test_param_names_match_official_layout(self):
        names = set(Aurora_lite().state_dict())
        for n in [
            "encoder.atmos_latents", "encoder.surf_token_embeds.weights.2t", "encoder.level_agg.layers.0.0.to_q.weight",
            "backbone.encoder_layers.0.blocks.0.norm1.ln_modulation.1.weight",
            "backbone.encoder_layers.0.downsample.reduction.weight", "backbone.decoder_layers.0.upsample.lin1.weight",
            "decoder.level_decoder.layers.0.0.to_kv.weight", "decoder.surf_heads.msl.weight",
        ]:
            self.assertIn(n, names)

    def test_training_step_reduces_loss(self):
        """Config-driven & trainable: Adam steps on a fixed (normalised-space) target lower the loss."""
        torch.manual_seed(0)
        m = Aurora(embed_dim=32, encoder_depths=(1, 1, 1), decoder_depths=(1, 1, 1), encoder_num_heads=(2, 2, 2),
                   decoder_num_heads=(2, 2, 2), num_heads=2, window_size=(2, 2, 2))
        g = torch.Generator().manual_seed(3)
        # inputs / target drawn in normalised space, mapped to physical units (as real data would be)
        n_surf, n_atmos = torch.randn(1, 2, 4, 16, 32, generator=g), torch.randn(1, 2, 5, 4, 16, 32, generator=g)
        surf = torch.stack([m.unnormalise(n_surf[:, t], n_atmos[:, t], m.atmos_levels)[0] for t in range(2)], 1)
        atmos = torch.stack([m.unnormalise(n_surf[:, t], n_atmos[:, t], m.atmos_levels)[1] for t in range(2)], 1)
        static = torch.randn(3, 16, 32, generator=g)
        tgt_s, tgt_a = n_surf[:, 1], n_atmos[:, 1]  # "persistence" target
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        losses = []
        for _ in range(25):
            s, a = m(surf, static, atmos)
            ps, _, pa = m.normalise(s[:, None], static, a[:, None], m.atmos_levels)
            loss = (ps[:, 0] - tgt_s).pow(2).mean() + (pa[:, 0] - tgt_a).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        self.assertTrue(all(v == v for v in losses))
        self.assertLess(losses[-1], 0.8 * losses[0])

    def test_backward_all_params_get_grad(self):
        m = Aurora_lite()
        s, a = m(*_inputs(seed=2))
        (s.pow(2).mean() + a.pow(2).mean()).backward()
        no_grad = [n for n, p in m.named_parameters() if p.grad is None]
        # surf_level_encoding etc. all take part; only unused history slices stay zero
        self.assertEqual(no_grad, [])

    def test_wrong_levels_raises(self):
        with self.assertRaises(ValueError):
            Aurora_lite()(*_inputs(L=3))


@unittest.skipUnless(aurora_available(), "needs `pip install microsoft-aurora` (reference oracle)")
class TestParityVsOfficial(unittest.TestCase):
    def _pair(self, **cfg):
        import aurora as aur

        core = aur.Aurora(use_lora=False, **cfg).eval()
        torch.manual_seed(0)
        with torch.no_grad():  # random (non-zero) params so every layer matters
            for p in core.parameters():
                p.add_(0.05 * torch.randn_like(p))
        ours = Aurora(**cfg).eval()
        ours.load_state_dict(core.state_dict(), strict=True)
        return AuroraWrapper(core).eval(), ours

    def _check(self, cfg, H, W, B=2):
        off, ours = self._pair(**cfg)
        surf, static, atmos = _inputs(H, W, B, seed=7)
        t = (datetime(2020, 6, 1, 12), datetime(2021, 1, 3, 6))[:B]
        with torch.no_grad():
            so, ao = off(surf, static, atmos, time=t)
            s, a = ours(surf, static, atmos, time=t)
        torch.testing.assert_close(s, so, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(a, ao, atol=ATOL, rtol=RTOL)

    def test_lite_even_and_odd_grids(self):
        cfg = dict(embed_dim=64, encoder_depths=(2, 2, 2), decoder_depths=(2, 2, 2), encoder_num_heads=(2, 2, 2),
                   decoder_num_heads=(2, 2, 2), num_heads=2, window_size=(2, 2, 2))
        for H, W in [(16, 32), (17, 32), (25, 44)]:  # 25x44 -> 6x11 patches: exercises merge padding / split crop
            self._check(cfg, H, W)

    def test_uneven_windows_shifts_and_v3_lead_time(self):
        cfg = dict(embed_dim=32, encoder_depths=(1, 3, 2), decoder_depths=(2, 3, 1), encoder_num_heads=(2, 2, 2),
                   decoder_num_heads=(2, 2, 2), num_heads=2, window_size=(2, 6, 12), use_updated_lead_time_embedding=True)
        self._check(cfg, 32, 64)
        cfg = dict(embed_dim=64, encoder_depths=(2, 2, 2), decoder_depths=(2, 2, 2), encoder_num_heads=(2, 4, 8),
                   decoder_num_heads=(8, 4, 2), num_heads=2, window_size=(2, 3, 5))
        self._check(cfg, 20, 36)

    def test_wrapper_matches_upstream_batch_api(self):
        off = AuroraOfficial_lite().eval()
        surf, static, atmos = _inputs(seed=1)
        batch = off.make_batch(surf, static, atmos, time=T0)
        with torch.no_grad():
            ref = off.core(batch)
            s, _ = off(surf, static, atmos, time=T0)
        self.assertTrue(torch.equal(s[:, 0], ref.surf_vars["2t"][:, 0]))

    @unittest.skipUnless(os.path.exists(SMALL_CKPT), "official small checkpoint not present")
    def test_official_small_checkpoint_strict_load_and_parity(self):
        import aurora as aur

        ours = Aurora_small(checkpoint_path=SMALL_CKPT).eval()  # strict=True
        core = aur.AuroraSmallPretrained()
        core.load_checkpoint_local(SMALL_CKPT, strict=True)
        off = AuroraWrapper(core).eval()
        surf, static, atmos = _inputs(H=33, W=64, seed=5)
        with torch.no_grad():
            so, ao = off(surf, static, atmos)
            s, a = ours(surf, static, atmos)
        self.assertTrue(torch.isfinite(s).all() and torch.isfinite(a).all())
        torch.testing.assert_close(s, so, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(a, ao, atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    unittest.main()
