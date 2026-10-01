"""Native GenCast denoiser: official-weight parity (vs JAX reference arrays), structure, and training."""
import os
import unittest

import numpy as np
import torch

from weatherai.models.gencast import (
    DenoiserConfig,
    GenCast,
    GenCastDenoiser,
    GenCastDenoiser_from_official,
    SparseTransformerConfig,
    build_gencast_graphs,
    convert_official_params,
    denoiser_inputs,
    khop_attention_mask,
    unstack_variables,
)

DATA = os.path.join(os.path.dirname(__file__), "data")
CKPT = os.environ.get("GENCAST_CKPT", "/workspace/ckpt/gencast/mini.npz")
FULL_REF = os.environ.get("GENCAST_FULL_REF", "/workspace/ckpt/gencast/denoiser_ref_full.npz")
have_ckpt = os.path.exists(CKPT)


def _group(ref, tag):
    return {k.split("/", 1)[1]: ref[k] for k in ref.files if k.startswith(tag + "/")}


def _run(ref, model):
    shape = (len(ref["lat"]), len(ref["lon"]))
    x = denoiser_inputs(_group(ref, "inputs"), _group(ref, "forcings"), _group(ref, "noisy_targets"), shape)
    with torch.no_grad():
        y = model(x, torch.from_numpy(ref["noise_levels"]))
    tmpl = {k: ((1, 13) if ref["out/" + k].ndim == 5 else (1,)) for k in ref["target_variables"]}
    return unstack_variables(y, tmpl, shape), tmpl


class TestGraphsMatchOfficial(unittest.TestCase):
    """Graph construction vs the official ``_DenoiserArchitecture`` graphs (mesh_size=2, 13x24 grid)."""

    def test_graphs(self):
        o = np.load(os.path.join(DATA, "graphs_ref_small.npz"))
        r = np.load(os.path.join(DATA, "denoiser_ref_small.npz"))
        g = build_gencast_graphs(2, r["lat"], r["lon"])
        np.testing.assert_array_equal(g.mesh_vertices, o["verts"])  # incl. banded (RCM) permutation
        np.testing.assert_array_equal(g.mesh_faces, o["faces"])
        np.testing.assert_array_equal(g.g2m_senders, o["g2m_s"])
        np.testing.assert_array_equal(g.g2m_receivers, o["g2m_r"])
        np.testing.assert_allclose(g.g2m_edge_attr, o["g2m_f"], atol=1e-6)
        np.testing.assert_allclose(g.grid_node_features, o["gnf"], atol=1e-6)
        np.testing.assert_allclose(g.mesh_node_features, o["mnf"], atol=1e-6)
        if (g.m2g_senders == o["m2g_s"]).all():  # exact only with the trimesh backend (official tie-breaking)
            np.testing.assert_allclose(g.m2g_edge_attr, o["m2g_f"], atol=1e-6)
        else:
            self.skipTest("trimesh/rtree not installed: m2g tie-breaking differs from official")


@unittest.skipUnless(have_ckpt, "official GenCast Mini params not available (gs://dm_graphcast/gencast/params/)")
class TestOfficialWeightParity(unittest.TestCase):
    def test_strict_load_and_param_count(self):
        r = np.load(os.path.join(DATA, "denoiser_ref_small.npz"))
        m = GenCastDenoiser_from_official(CKPT, r["lat"], r["lon"], mesh_size=2, attention_k_hop=3)
        npz = np.load(CKPT)
        n_official = sum(npz[k].size for k in npz.files if k.startswith("params:"))
        self.assertEqual(sum(p.numel() for p in m.parameters()), n_official)
        self.assertEqual(len(convert_official_params(npz)), len([k for k in npz.files if k.startswith("params:")]))

    def _check(self, attention_type, atol, rtol):
        r = np.load(os.path.join(DATA, "denoiser_ref_small.npz"))
        m = GenCastDenoiser_from_official(CKPT, r["lat"], r["lon"], mesh_size=2, attention_k_hop=3, attention_type=attention_type)
        out, _ = _run(r, m)
        for k, v in out.items():
            np.testing.assert_allclose(v.numpy(), r["out/" + k], atol=atol, rtol=rtol, err_msg=k)

    def test_small_mesh_matches_official_jax(self):
        """mesh_size=2, k_hop=3, 13x24 grid, official Mini weights, official plain-MHA reference: atol 2e-4 / rtol 1e-4."""
        self._check("dense", 2e-4, 1e-4)

    def test_small_mesh_banded_attention_matches_official_jax(self):
        self._check("triblockdiag", 2e-4, 1e-4)

    @unittest.skipUnless(os.path.exists(FULL_REF), "full-resolution reference (181x360, mesh 4, k=16) not generated")
    def test_full_resolution_matches_official_jax(self):
        """Trained config: mesh_size=4 (2562 nodes), 181x360 grid, attention_k_hop=16: atol 5e-4."""
        r = np.load(FULL_REF)
        m = GenCastDenoiser_from_official(CKPT, r["lat"], r["lon"], mesh_size=int(r["mesh_size"]), attention_k_hop=int(r["k_hop"]))
        out, _ = _run(r, m)
        for k, v in out.items():
            np.testing.assert_allclose(v.numpy(), r["out/" + k], atol=5e-4, rtol=1e-4, err_msg=k)


class TestStructure(unittest.TestCase):
    def test_khop_mask_matches_matrix_power(self):
        g = build_gencast_graphs(1, np.linspace(-90, 90, 7), np.arange(0, 360, 30))
        n = g.num_mesh_nodes
        a = np.eye(n)
        a[g.mesh_senders, g.mesh_receivers] = 1
        ref = np.linalg.matrix_power(a, 3) > 0
        np.testing.assert_array_equal(khop_attention_mask(g.mesh_senders, g.mesh_receivers, n, 3), ref)

    def test_banded_equals_dense_attention_random_weights(self):
        torch.manual_seed(0)
        lat, lon = np.linspace(-80, 80, 9), np.arange(0, 360, 20)
        cfgs = [DenoiserConfig(transformer=SparseTransformerConfig(attention_k_hop=3, d_model=32, num_layers=2, num_heads=4, ffw_hidden=64, attention_type=t),
                               mesh_size=2, latent_size=32, node_output_size=5, in_channels=7) for t in ("dense", "triblockdiag")]
        a, b = (GenCastDenoiser(c, lat, lon) for c in cfgs)
        b.load_state_dict(a.state_dict())
        x, s = torch.randn(2, 9 * 18, 7), torch.tensor([0.3, 30.0])
        with torch.no_grad():
            torch.testing.assert_close(a(x, s), b(x, s), atol=1e-5, rtol=1e-4)

    def test_train_step_native_architecture_and_gradients(self):
        """Config-driven random-init denoiser trains: loss decreases on a fixed batch, all params get grads."""
        from weatherai.models.gencast import GenCast_lite

        torch.manual_seed(0)
        m = GenCast_lite(img_size=(8, 16), out_channels=3, cond_channels=4, latent=24, transformer_layers=2, mesh_level=1)
        tgt, cond = torch.randn(2, 3, 8, 16), torch.randn(2, 4, 8, 16)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        losses = []
        for _ in range(60):
            opt.zero_grad()
            loss = m.loss(tgt, cond, generator=torch.Generator().manual_seed(0))
            loss.backward()
            opt.step()
            losses.append(float(loss))
        self.assertLess(losses[-1], 0.85 * losses[0])
        # the unused-by-forward mesh-node MLP of the mesh2grid GNN is the only parameter without gradient
        no_grad = [n for n, p in m.named_parameters() if p.grad is None]
        self.assertTrue(all("mesh2grid_gnn.processor_nodes_0_mesh_nodes" in n for n in no_grad), no_grad)


if __name__ == "__main__":
    unittest.main()
