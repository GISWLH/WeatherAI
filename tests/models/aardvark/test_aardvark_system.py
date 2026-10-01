"""Aardvark encoder / decoder / E2E system: structure tests (no weights) and official-checkpoint
parity against references produced by the OFFICIAL code (scripts/aardvark_system_reference.py)."""
import os
import unittest

import numpy as np
import torch

from weatherai.models.aardvark import (AardvarkDecoder, AardvarkE2E, AardvarkEncoder, AardvarkProcessor, SetConv,
                                       Unet, load_official_decoder, load_official_encoder, load_official_processor,
                                       load_official_sample)
from weatherai.models.aardvark.unet import CylindricalConvTranspose2D

CK = os.environ.get("AARDVARK_CKPT", "/workspace/ckpt/aardvark/trained_model")
SAMPLE = os.environ.get("AARDVARK_SAMPLE", "/workspace/sources/aardvark-weather-public/data/sample_data_final.pkl")
REF = os.path.join(os.path.dirname(__file__), "data", "system_ref.npz")
HAVE = all(os.path.exists(p) for p in (f"{CK}/encoder/epoch_96", f"{CK}/decoder/tas/lt_1/epoch_18",
                                       f"{CK}/processor/forecast_1/epoch_0", SAMPLE))
ATOL, RTOL = 1e-5, 1e-4  # fp32; same as docs/graphcast_parity.md


class TestSetConv(unittest.TestCase):
    def test_off_to_on_matches_naive(self):
        torch.manual_seed(0)
        sc = SetConv(0.05, "OffToOn", density_channel=True)
        N, X, Y = 7, 5, 4
        xlon, xlat = torch.rand(1, N), torch.rand(1, N)
        y = torch.randn(1, 1, N)
        gx, gy = torch.linspace(0, 1, X)[None], torch.linspace(0, 1, Y)[None]
        out = sc([xlon, xlat], y, [gx, gy])  # (1, 2, X, Y)
        dens = torch.zeros(X, Y)
        num = torch.zeros(X, Y)
        for n in range(N):
            w = torch.exp(-0.5 * (gx[0, :, None] - xlon[0, n]) ** 2 / 0.05**2) * torch.exp(-0.5 * (gy[0, None, :] - xlat[0, n]) ** 2 / 0.05**2)
            dens += w
            num += w * y[0, 0, n]
        torch.testing.assert_close(out[0, 0], dens, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(out[0, 1], num / dens.clamp(min=1e-6), atol=1e-5, rtol=1e-4)

    def test_nan_is_missing(self):
        sc = SetConv(0.05, "OffToOn", True)
        gx, gy = torch.linspace(0, 1, 3)[None], torch.linspace(0, 1, 3)[None]
        a = sc([torch.tensor([[0.2, 0.7]]), torch.tensor([[0.3, 0.6]])], torch.tensor([[[1.0, float("nan")]]]), [gx, gy])
        b = sc([torch.tensor([[0.2]]), torch.tensor([[0.3]])], torch.tensor([[[1.0]]]), [gx, gy])
        torch.testing.assert_close(a, b)

    def test_on_to_off_and_on_to_on_shapes(self):
        grid = [torch.linspace(0, 1, 6)[None], torch.linspace(0, 0.5, 5)[None]]
        wt = torch.randn(2, 3, 6, 5)
        self.assertEqual(SetConv(0.1, "OnToOn", True)(grid, wt, grid).shape, (2, 4, 6, 5))
        pts = [torch.rand(2, 9), torch.rand(2, 9)]
        self.assertEqual(SetConv(0.1, "OnToOff", False)(grid, wt, pts).shape, (2, 3, 9))


class TestUnetDecoder(unittest.TestCase):
    def test_cylindrical_transpose_conv_shape_and_unet(self):
        up = CylindricalConvTranspose2D(4, 3, 3, 2)
        self.assertEqual(up(torch.randn(1, 4, 5, 7)).shape, (1, 3, 10, 14))
        u = Unet(6, 2, div_factor=8).eval()
        y = u(torch.randn(1, 6, 24, 13))
        self.assertEqual(y.shape, (1, 24, 13, 2))

    def test_param_names_match_official(self):
        keys = set(AardvarkDecoder().state_dict())
        for k in ("decoder_lr.down1.conv_1.weight", "decoder_lr.up2.up._bias", "decoder_lr.up4.conv.bn_2.running_var",
                  "decoder_lr.out.weight", "decoder_lr.variances", "mlp.mlp.1.block.0.weight", "sc_out.init_ls"):
            self.assertIn(k, keys)
        ek = set(AardvarkEncoder().state_dict())
        for k in ("decoder_lr.token_embeds.0.proj.weight", "decoder_lr.mlp.mlp.0.weight", "decoder_lr.var_agg.in_proj_weight",
                  "ascat_setconvs.init_ls", "sc_out.init_ls", "mlp.mlp.2.0.weight"):
            self.assertIn(k, ek)
        self.assertEqual(AardvarkEncoder().decoder_lr.token_embeds[0].proj.weight.shape, (512, 256, 3, 3))


def _tiny_task(B=1, n_lon=24, n_lat=13, N=6):
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    coords = lambda n: torch.linspace(0, 1, n)[None].repeat(B, 1)  # noqa: E731
    asm = {
        "x_context_hadisd": [torch.rand(B, 2, N, generator=g) for _ in range(5)], "y_context_hadisd": [r(B, N) for _ in range(5)],
        "sat_x": [coords(10), coords(8)], "sat": r(B, 2, 10, 8),
        "icoads_x": [torch.rand(B, N, generator=g), torch.rand(B, N, generator=g)], "icoads": r(B, 5, N),
        "igra_x": [torch.rand(B, N, generator=g), torch.rand(B, N, generator=g)], "igra": r(B, 24, N),
        "amsua_x": [coords(9), coords(7)], "amsua": r(B, 7, 9, 13),
        "amsub_x": [coords(9), coords(7)], "amsub": r(B, 9, 7, 12),
        "hirs_x": [coords(9), coords(7)], "hirs": r(B, 9, 7, 26),
        "iasi": r(B, 12, 6, 52), "ascat": r(B, 12, 6, 17),
        "era5_elev": r(B, 7, n_lat, n_lon), "climatology": r(B, 24, n_lon, n_lat), "aux_time": r(B, 5),
    }
    return asm


class TestEncoderLite(unittest.TestCase):
    def test_tiny_encoder_forward_backward(self):
        enc = AardvarkEncoder(n_lon=24, n_lat=13, vit_size=(24, 12), embed_dim=32, depth=1, patch_size=3, num_heads=2, decoder_depth=1,
                              learnable_setconvs=True)
        x = enc(_tiny_task())
        self.assertEqual(x.shape, (1, 13, 24, 24))
        self.assertTrue(torch.isfinite(x).all())
        x.pow(2).mean().backward()
        self.assertTrue(torch.isfinite(enc.hadisd_setconvs[0].init_ls.grad).all())  # length scales trainable when asked

    def test_train_step_whole_system_lite(self):
        """Config-driven tiny end-to-end: encoder + processor + decoder, one optimiser step lowers the loss."""
        torch.manual_seed(0)
        n_lon, n_lat = 30, 13
        enc = AardvarkEncoder(n_lon, n_lat, (30, 12), 32, 1, 3, 2, 1)
        proc = AardvarkProcessor(35, 24, (n_lon, n_lat), 32, 1, 5, 2, 1)
        dec = AardvarkDecoder(36, 8, div_factor=16, h_channels=16)
        dec.mlp = type(dec.mlp)(8 + 9, 1, 16, 1)
        e2e = AardvarkE2E(enc, [proc], dec)
        t = _tiny_task(n_lon=n_lon, n_lat=n_lat)
        N = 6
        task = {
            "assimilation": t,
            "forecast": {"y_context": torch.randn(1, 35, n_lon, n_lat), "lt": torch.ones(1, 1)},
            "downscaling": {"y_context": torch.randn(1, 36, n_lon, n_lat), "x_context": [torch.linspace(0, 1, n_lon)[None], torch.linspace(-.25, .25, n_lat)[None]],
                            "x_target": torch.rand(1, 2, N), "alt_target": torch.randn(1, 2, N), "aux_time": torch.randn(1, 5, 1, 1)},
        }
        target = torch.randn(1, N)
        opt = torch.optim.Adam(e2e.parameters(), lr=1e-3)
        l0 = None
        for _ in range(8):
            loss = (e2e(task) - target).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            l0 = l0 if l0 is not None else loss.item()
        self.assertLess(loss.item(), l0)


@unittest.skipUnless(HAVE, "needs official checkpoints (AARDVARK_CKPT) and sample data (AARDVARK_SAMPLE)")
class TestOfficialParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = np.load(REF)
        cls.task = load_official_sample(SAMPLE)
        cls.enc = load_official_encoder(f"{CK}/encoder/epoch_96").eval()  # strict
        cls.dec = load_official_decoder(f"{CK}/decoder/tas/lt_1/epoch_18").eval()  # strict
        cls.proc = load_official_processor(f"{CK}/processor/forecast_1/epoch_0").eval()  # strict

    def test_encoder_matches_official(self):
        with torch.no_grad():
            x0 = self.enc(self.task["assimilation"])
        np.testing.assert_allclose(x0.numpy()[:, ::4, ::4], self.ref["encoder_state"], atol=ATOL, rtol=RTOL)

    def test_station_decoder_matches_official(self):
        with torch.no_grad():
            st = self.dec(self.task["downscaling"])
        np.testing.assert_allclose(st.numpy(), self.ref["decoder_station"], atol=ATOL, rtol=RTOL)

    def test_end_to_end_matches_official(self):
        e2e = AardvarkE2E(self.enc, [self.proc], self.dec).eval()
        with torch.no_grad():
            st, fc, init = e2e(self.task, return_gridded=True)
        np.testing.assert_allclose(st.numpy(), self.ref["e2e_station"], atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(fc.numpy()[:, ::4, ::4], self.ref["e2e_forecast"], atol=1e-2, rtol=RTOL)  # |fc| up to 1.2e5 (z)
        np.testing.assert_allclose(init.numpy()[:, ::4, ::4], self.ref["e2e_init"], atol=1e-2, rtol=RTOL)


if __name__ == "__main__":
    unittest.main()
