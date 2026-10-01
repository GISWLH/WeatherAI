"""Native PyTorch NeuralGCM learned components + spherical-harmonic layer vs the official JAX/Haiku model.

Reference data (tests/models/neuralgcm/data/*.npz) were produced by scripts/neuralgcm_*_reference.py from the
official ``neuralgcm`` 1.2.3 / ``dinosaur`` 1.5.0 on the official deterministic 2.8 deg checkpoint (real packed tower
inputs of one encode->advance->decode step on the demo ERA5 snapshot, evaluated on 24 random columns).
Tests needing the checkpoint look at $NGCM_CKPT or download it (skipped offline).
"""
import os
import pickle
import unittest

import numpy as np
import torch

from weatherai.models.neuralgcm.network import (
    EpdTower,
    LearnedOrography,
    NeuralGCMLearnedComponents,
    SurfaceEmbedding,
    VerticalConvTower,
    convert_official_params,
    count_parameters,
)
from weatherai.models.neuralgcm.spectral import SphericalHarmonicsGrid, SpectralGridConfig

DATA = os.path.join(os.path.dirname(__file__), "data")
CKPT = os.environ.get("NGCM_CKPT")


def _load_params():
    path = CKPT
    if path is None:
        try:
            from weatherai.models.neuralgcm.neuralgcm import download_checkpoint

            path = download_checkpoint("deterministic_2_8_deg")
        except Exception as e:  # offline
            raise unittest.SkipTest(f"no checkpoint: {e}")
    try:
        with open(path, "rb") as f:
            return pickle.load(f)["params"]
    except Exception as e:  # unpickling needs jax
        raise unittest.SkipTest(f"cannot unpickle checkpoint (needs jax): {e}")


class TestSpectral(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = np.load(os.path.join(DATA, "sht_ref.npz"))
        cls.g = SphericalHarmonicsGrid(SpectralGridConfig.TL63())
        cls.T = lambda self, k: torch.tensor(self.r[k])

    def close(self, a, k, atol):
        ref = self.r[k]
        err = float(np.abs(a.numpy() - ref).max())
        self.assertLess(err, atol, f"{k}: {err}")

    def test_transforms(self):
        self.assertEqual(self.g.modal_shape, (127, 65))
        self.close(self.g.to_nodal(self.T("modal")), "to_nodal", 2e-4)
        self.close(self.g.to_modal(self.T("nodal")), "to_modal", 1e-6)

    def test_operators(self):
        m = self.T("modal")
        self.close(self.g.laplacian(m), "lap", 1e-3)
        self.close(self.g.inverse_laplacian(m), "inv_lap", 1e-6)
        self.close(self.g.d_dlon(m), "d_dlon", 1e-4)
        self.close(self.g.cos_lat_d_dlat(m), "cos_lat_d_dlat", 1e-3)
        self.close(self.g.sec_lat_d_dlat_cos2(m), "sec_lat_d_dlat_cos2", 1e-3)
        g0, g1 = self.g.cos_lat_grad(m)
        self.close(g0, "grad0", 1e-3)
        self.close(g1, "grad1", 1e-3)
        self.close(self.g.div_cos_lat((m, m.flip(0))), "div", 1e-3)
        self.close(self.g.curl_cos_lat((m, m.flip(0))), "curl", 1e-3)

    def test_uv_vor_div_roundtrip_matches_official(self):
        vo, di = self.g.uv_nodal_to_vor_div_modal(self.T("u"), self.T("v"))
        self.close(vo, "vor", 5e-4)
        self.close(di, "div_uv", 5e-4)
        u, v = self.g.vor_div_to_uv_nodal(vo, di)
        self.close(u, "u_rt", 5e-4)
        self.close(v, "v_rt", 5e-4)

    def test_float64_orthonormality(self):
        """to_modal(to_nodal(x)) == x on the triangular mask (exact basis, f64)."""
        torch.manual_seed(0)
        x = torch.randn(127, 65, dtype=torch.float64) * self.g.mask
        y = self.g.to_modal(self.g.to_nodal(x))
        # the top total wavenumber is not exactly representable on this Gaussian grid for all m; check lower part
        self.assertLess(float((y - x)[:, :-1].abs().max()), 1e-10)


class TestLearnedComponents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = np.load(os.path.join(DATA, "tower_ref.npz"))
        cls.params = _load_params()
        cls.m = convert_official_params(cls.params).eval()

    def _tower_data(self, i):
        return torch.tensor(self.r[f"t{i}_x"]), self.r[f"t{i}_y"], str(self.r[f"t{i}_name"])

    def test_param_count_matches_checkpoint(self):
        n_official = sum(int(np.size(v)) for d in self.params.values() for v in d.values())
        self.assertEqual(count_parameters(self.m), n_official)
        self.assertEqual(n_official, 14_518_180)

    def test_epd_towers_vs_official(self):
        towers = {0: self.m.decoder, 1: self.m.encoder, 2: self.m.encoder_1, 3: self.m.physics}
        for i, tower in towers.items():
            x, y, name = self._tower_data(i)
            with torch.no_grad():
                out = tower(x[:, :, None])[:, :, 0].numpy()
            err = float(np.abs(out - y).max())
            self.assertLess(err, 2e-4 * max(1.0, float(np.abs(y).max())), f"{name}: {err}")

    def test_surface_towers_and_blend_vs_official(self):
        outs = []
        for i, tower in ((4, self.m.surface.land), (5, self.m.surface.sea), (6, self.m.surface.sea_ice)):
            x, y, name = self._tower_data(i)
            with torch.no_grad():
                o = tower(x[:, :, None])[:, :, 0]
            self.assertLess(float(np.abs(o.numpy() - y).max()), 1e-5, name)
            outs.append(x)
        with torch.no_grad():
            blend = self.m.surface(outs[0][:, :, None], outs[1][:, :, None], outs[2][:, :, None],
                                   torch.tensor(self.r["surf_mask"])[:, None], torch.tensor(self.r["surf_ice"])[:, None])[:, :, 0]
        self.assertLess(float(np.abs(blend.numpy() - self.r["surf_out"]).max()), 1e-5)

    def test_vertical_cnn_vs_official(self):
        x, y, name = self._tower_data(7)
        with torch.no_grad():
            out = self.m.volume_cnn(x[..., None])[..., 0].numpy()
        self.assertLess(float(np.abs(out - y).max()), 1e-4 * max(1.0, float(np.abs(y).max())), name)

    def test_learned_orography_vs_official(self):
        mods = {0: self.m.orog_encoder, 1: self.m.orog_encoder_1}
        # sorted order of names: encoder, encoder_1, then custom_coords_corrector -> check by mask shape instead
        names = [str(self.r[f"o{i}_name"]) for i in range(int(self.r["n_orog"]))]
        for i, n in enumerate(names):
            mod = self.m.orog_corrector if "custom_coords_corrector" in n else (self.m.orog_encoder_1 if "encoder_1" in n else self.m.orog_encoder)
            base = torch.tensor(self.r[f"o{i}_base"])
            with torch.no_grad():
                out = mod(base).numpy()
            self.assertLess(float(np.abs(out - self.r[f"o{i}_val"]).max()), 1e-6, n)
            self.assertTrue(np.array_equal(mod.mask.numpy(), self.r[f"o{i}_mask"]))
            self.assertAlmostEqual(mod.scale, float(self.r[f"o{i}_scale"]), places=10)

    def test_positional_features_loaded(self):
        for p in (self.m.pos_decoder, self.m.pos_encoder, self.m.pos_encoder_1, self.m.pos_physics):
            self.assertEqual(tuple(p().shape), (8, 128, 64))
            self.assertGreater(float(p().detach().abs().max()), 0.0)


class TestTrainableNative(unittest.TestCase):
    def test_small_tower_trains(self):
        torch.manual_seed(0)
        tw = EpdTower(6, 3, latent=16, num_blocks=2, process_hidden_layers=1)
        x = torch.randn(2, 6, 8, 4)
        y = x[:, :3].tanh()
        opt = torch.optim.Adam(tw.parameters(), 1e-2)
        l0 = None
        for _ in range(60):
            opt.zero_grad()
            loss = ((tw(x) - y) ** 2).mean()
            loss.backward()
            opt.step()
            l0 = l0 if l0 is not None else float(loss)
        self.assertLess(float(loss), 0.3 * l0)

    def test_gradients_flow_through_sht(self):
        g = SphericalHarmonicsGrid(SpectralGridConfig(8, 9, 24, 12))
        x = torch.randn(3, *g.nodal_shape, dtype=torch.float64, requires_grad=True)
        vo, di = g.uv_nodal_to_vor_div_modal(x, x.flip(0))
        (vo.square().sum() + di.square().sum()).backward()
        self.assertTrue(torch.isfinite(x.grad).all() and float(x.grad.abs().sum()) > 0)

    def test_full_model_construct_and_forward_shapes(self):
        m = NeuralGCMLearnedComponents()
        self.assertEqual(count_parameters(m), 14_518_180)
        x = torch.randn(1052, 4, 2)
        self.assertEqual(tuple(m.encoder(x).shape), (193, 4, 2))


if __name__ == "__main__":
    unittest.main()
