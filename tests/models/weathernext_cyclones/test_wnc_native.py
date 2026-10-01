"""Native WeatherNext Cyclones network: official-weight parity (vs JAX reference arrays), structure, and training."""
import os
import unittest

import numpy as np
import torch

from weatherai.models.weathernext_cyclones import (
    WeatherNextCyclonesNative_lite,
    WeatherNextCyclonesNet_from_official,
    build_wnc_graphs,
    convert_official_params,
    fair_crps,
    stack_grid_inputs,
    stack_mesh_inputs,
    unstack_outputs,
)

DATA = os.path.join(os.path.dirname(__file__), "data")
CKPT = os.environ.get("WNC_MINI_CKPT", "/workspace/ckpt/wn/WNC_Mini_2024.npz")
FULL_REF = os.environ.get("WNC_FULL_REF", "/workspace/ckpt/wn/forward_ref_full.npz")
have_ckpt = os.path.exists(CKPT)


def _group(ref, tag):
    return {k.split("/", 1)[1]: ref[k] for k in ref.files if k.startswith(tag + "/")}


def _run(ref, model, dtype=torch.float32):
    shape = (len(ref["lat"]), len(ref["lon"]))
    gx = stack_grid_inputs(_group(ref, "inputs"), _group(ref, "forcings"), shape).to(dtype)
    mx = stack_mesh_inputs(_group(ref, "inputs"), _group(ref, "forcings")).to(dtype)
    with torch.no_grad():
        y = model(gx, mx, torch.from_numpy(ref["noise"]).to(dtype))
    return unstack_outputs(y, model.cfg, shape)


class TestGraphsMatchOfficial(unittest.TestCase):
    def test_edge_sets_match_official(self):
        """Ball-query grid→mesh and closest-triangle mesh→grid edge multisets (and receiver-sortedness) == official (mesh_splits=2).

        Compared as sorted (receiver, sender) pairs: the official unstable ``np.argsort(receivers)`` orders edges inside one
        receiver in a numpy-version-dependent way, which cannot change a sum aggregation.
        """
        r = np.load(os.path.join(DATA, "forward_ref_small.npz"))
        g = build_wnc_graphs(2, r["lat"], r["lon"])
        try:
            import rtree, trimesh  # noqa: F401, E401
        except Exception:
            self.skipTest("trimesh/rtree missing: mesh→grid tie-breaking differs from official")
        for name, s, rc in (("points_to_mesh_nodes", g.g2m_senders, g.g2m_receivers), ("mesh_to_points_nodes", g.m2g_senders, g.m2g_receivers)):
            rs, rr = r[f"edges/{name}/senders"], r[f"edges/{name}/receivers"]
            self.assertTrue((np.diff(rc) >= 0).all() and (np.diff(rr) >= 0).all())
            np.testing.assert_array_equal(np.stack([rc[np.lexsort((s, rc))], s[np.lexsort((s, rc))]]),
                                          np.stack([rr[np.lexsort((rs, rr))], rs[np.lexsort((rs, rr))]]))


@unittest.skipUnless(have_ckpt, "official WeatherNextCyclones_Mini params not available (gs://dm_graphcast/weathernext2/params/)")
class TestOfficialWeightParity(unittest.TestCase):
    def test_strict_load_all_params_consumed(self):
        npz = np.load(CKPT)
        sd, layout = convert_official_params(npz)  # raises if any params: entry is left over
        n_official = sum(npz[k].size for k in npz.files if k.startswith("params:"))
        self.assertEqual(sum(v.numel() for v in sd.values()), n_official)
        self.assertEqual(len(layout), 29)
        self.assertEqual(sum(n for _, n in layout), 101)

    def test_float64_reference_isolates_structure(self):
        """JAX float64 reference (x64, f32-upcasts disabled in the *reference*): native float64 agrees to 1e-9 for both attention kernels."""
        r = np.load(os.path.join(DATA, "forward_ref_small_f64.npz"))
        for att in ("dense", "triblockdiag"):
            m = WeatherNextCyclonesNet_from_official(CKPT, r["lat"], r["lon"], mesh_splits=int(r["splits"]), attention_k_hop=int(r["k_hop"]), attention_type=att).double()
            out = _run(r, m, torch.float64)
            for k, v in out.items():
                np.testing.assert_allclose(v.numpy(), r["out/" + k], atol=1e-9, rtol=1e-9, err_msg=f"{att}:{k}")

    def test_float32_small_matches_official_jax(self):
        """Official float32 JAX run (mesh_splits=2, k_hop=3, 13x24 grid, batch 2): atol 5e-3, rtol 1e-4 (float32 noise ≈1e-3, see docs)."""
        r = np.load(os.path.join(DATA, "forward_ref_small.npz"))
        m = WeatherNextCyclonesNet_from_official(CKPT, r["lat"], r["lon"], mesh_splits=int(r["splits"]), attention_k_hop=int(r["k_hop"]))
        out = _run(r, m)
        for k, v in out.items():
            np.testing.assert_allclose(v.numpy(), r["out/" + k], atol=5e-3, rtol=1e-4, err_msg=k)

    @unittest.skipUnless(os.path.exists(FULL_REF), "full-resolution reference (181x360, 5 splits, k=16) not generated")
    def test_full_resolution_matches_official_jax(self):
        """Trained config: 5 splits (10 242 mesh nodes), 181×360 grid, k_hop 16, banded attention. Tolerance atol 5e-3 (float32)."""
        r = np.load(FULL_REF)
        m = WeatherNextCyclonesNet_from_official(CKPT, r["lat"], r["lon"], mesh_splits=int(r["splits"]), attention_k_hop=int(r["k_hop"]), attention_type="triblockdiag")
        out = _run(r, m)
        for k, v in out.items():
            np.testing.assert_allclose(v.numpy(), r["out/" + k], atol=5e-3, rtol=1e-4, err_msg=k)


class TestStructure(unittest.TestCase):
    def test_banded_equals_dense_random_weights(self):
        torch.manual_seed(0)
        a = WeatherNextCyclonesNative_lite(attention_type="dense").net
        b = WeatherNextCyclonesNative_lite(attention_type="triblockdiag").net
        b.load_state_dict(a.state_dict())
        gx, mx, nz = torch.randn(2, 16 * 32, 12), torch.randn(2, 2), torch.randn(2, 16)
        with torch.no_grad():
            torch.testing.assert_close(a(gx, mx, nz), b(gx, mx, nz), atol=1e-5, rtol=1e-4)

    def test_noise_changes_output_and_sigmoid_range(self):
        torch.manual_seed(0)
        m = WeatherNextCyclonesNative_lite().eval()
        gx, mx = torch.randn(1, 16 * 32, 12), torch.randn(1, 2)
        with torch.no_grad():
            y1, y2 = m.net(gx, mx, torch.randn(1, 16)), m.net(gx, mx, torch.randn(1, 16))
        self.assertGreater(float((y1 - y2).abs().max()), 0.0)
        p = y1[..., 0]  # cyclone_exists_gaussian_unit_mode is first in alphabetical order
        self.assertTrue(bool(((p > 0) & (p < 1)).all()))

    def test_fair_crps_properties(self):
        t = torch.zeros(1, 3, 1)
        same = torch.ones(4, 1, 3, 1)
        self.assertAlmostEqual(float(fair_crps(same, t).mean()), 1.0, places=6)  # no spread: CRPS = MAE
        spread = torch.stack([torch.full((1, 3, 1), v) for v in (-1.0, 1.0, -1.0, 1.0)])
        self.assertLess(float(fair_crps(spread, t).mean()), 1.0)  # same MAE, spread rewarded

    def test_train_step_native_architecture_and_gradients(self):
        """Config-driven random-init native WN-C: fair-CRPS loss decreases on a fixed batch; all params (except the unused mesh-node MLP of mesh→grid) get grads."""
        torch.manual_seed(0)
        m = WeatherNextCyclonesNative_lite()
        gx, mx = torch.randn(2, 16 * 32, 12), torch.randn(2, 2)
        tgt = torch.randn(2, 16 * 32, m.net.cfg.out_channels)
        tgt[..., 0] = (tgt[..., 0] > 1.5).float()  # "cyclone exists" in {0,1}
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        losses = []
        for _ in range(60):
            opt.zero_grad()
            loss = m.loss(gx, mx, tgt, num_samples=2, generator=torch.Generator().manual_seed(0))
            loss.backward()
            opt.step()
            losses.append(float(loss))
        self.assertLess(losses[-1], 0.85 * losses[0])
        no_grad = [n for n, p in m.named_parameters() if p.grad is None]
        self.assertTrue(all("mesh_to_grid_gnn.mesh_node_mlp" in n for n in no_grad), no_grad)


if __name__ == "__main__":
    unittest.main()
