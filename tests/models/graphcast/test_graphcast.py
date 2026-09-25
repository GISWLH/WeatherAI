import unittest
import warnings

import numpy as np
import torch

from weatherai.models import GraphCast, GraphCast_lite
from weatherai.models.graphcast import mesh_num_nodes
from weatherai.models.graphcast.mesh import build_graphs, closest_triangle_sq_distances, max_edge_length, build_mesh_hierarchy


class TestGraphCast(unittest.TestCase):
    def test_mesh_num_nodes(self):
        self.assertEqual(mesh_num_nodes(0), 12)
        self.assertEqual(mesh_num_nodes(1), 42)
        self.assertEqual(mesh_num_nodes(2), 162)
        self.assertEqual(mesh_num_nodes(6), 40962)

    def test_lite_defaults_shape(self):
        model = GraphCast_lite()
        self.assertEqual(model.img_size, (32, 64))
        self.assertEqual(model.mesh_level, 1)
        self.assertEqual(model.processor_layers, 2)
        self.assertEqual(model.hidden_dim, 32)
        self.assertEqual(model.in_channels, 4)
        self.assertEqual(model.num_mesh_nodes, 42)
        x = torch.randn(2, 4, 32, 64)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 4, 32, 64))
        self.assertTrue(torch.isfinite(y).all())

    def test_lite_finite_backward(self):
        model = GraphCast_lite(in_channels=4, hidden_dim=16, processor_layers=1)
        x = torch.randn(1, 4, 32, 64, requires_grad=True)
        y = model(x)
        loss = (y - x.detach()).pow(2).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        # At least one parameter received a gradient
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertGreater(len(grads), 0)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_graphcast_small_multi_mesh(self):
        model = GraphCast(
            img_size=(16, 32),
            in_channels=8,
            mesh_level=1,
            processor_layers=2,
            hidden_dim=16,
            use_multi_mesh=True,
        )
        x = torch.randn(1, 8, 16, 32)
        y = model(x)
        self.assertEqual(tuple(y.shape), (1, 8, 16, 32))
        self.assertTrue(torch.isfinite(y).all())

    def test_out_channels_mismatch(self):
        model = GraphCast_lite(in_channels=4, out_channels=2, hidden_dim=16, processor_layers=1)
        x = torch.randn(1, 4, 32, 64)
        y = model(x)
        self.assertEqual(tuple(y.shape), (1, 2, 32, 64))

    def test_param_count_positive(self):
        model = GraphCast_lite()
        n = sum(p.numel() for p in model.parameters())
        self.assertGreater(n, 1000)

    def test_wrong_spatial_raises(self):
        model = GraphCast_lite()
        with self.assertRaises(ValueError):
            model(torch.randn(1, 4, 16, 32))

    def test_graph_construction_rules(self):
        """Grid2Mesh = radius query (0.6 x longest finest edge); Mesh2Grid = 3 edges/grid point
        to the vertices of the closest (containing) finest triangle."""
        g = build_graphs(img_size=(8, 16), mesh_level=1, use_multi_mesh=True)
        mesh_xyz, faces = build_mesh_hierarchy(1)[-1]
        r = float(max_edge_length(mesh_xyz, faces)) * 0.6
        self.assertAlmostEqual(g.query_radius, r, places=6)
        d = np.linalg.norm(g.grid_nodes[g.g2m_senders] - g.mesh_nodes[g.g2m_receivers], axis=-1)
        self.assertTrue((d <= r + 1e-6).all())
        dense = np.linalg.norm(g.grid_nodes[:, None] - g.mesh_nodes[None], axis=-1)
        self.assertEqual(len(g.g2m_senders), int((dense <= r).sum()))
        self.assertTrue(np.array_equal(np.bincount(g.m2g_receivers, minlength=128), np.full(128, 3)))
        d2 = closest_triangle_sq_distances(g.grid_nodes, g.mesh_nodes, faces)
        tri = g.m2g_senders.reshape(-1, 3)
        chosen = [np.where((np.sort(faces, 1) == np.sort(t_)).all(1))[0][0] for t_ in tri]
        self.assertTrue(np.allclose(d2[np.arange(128), chosen], d2.min(1), atol=1e-12))
        # Multi-mesh contains the finest-level edges plus coarser ones; features are 4-D.
        self.assertEqual(g.mesh_edge_attr.shape[1], 4)
        self.assertEqual(g.grid_node_features.shape, (128, 3))

    def test_deprecated_knn_args_warn(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = GraphCast_lite(hidden_dim=8, processor_layers=1, g2m_k=3, m2g_k=3)
        self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))
        self.assertEqual(tuple(m(torch.randn(1, 4, 32, 64)).shape), (1, 4, 32, 64))

    def test_explicit_grid_and_stages(self):
        lat = np.linspace(-90, 90, 5)
        lon = np.arange(8) * 45.0
        model = GraphCast(in_channels=3, mesh_level=1, processor_layers=2, hidden_dim=8,
                          grid_lat=lat, grid_lon=lon)
        self.assertEqual(model.img_size, (5, 8))
        x = torch.randn(2, 3, 5, 8)
        gi = x.reshape(2, 3, 40).permute(0, 2, 1)
        grid, mesh, e = model.run_grid2mesh(gi)
        self.assertEqual(tuple(mesh.shape), (2, 42, 8))
        self.assertEqual(tuple(e.shape), (2, model.g2m_senders.numel(), 8))
        mesh, me = model.run_processor(mesh)
        out, _, _ = model.run_mesh2grid(mesh, grid)
        self.assertTrue(torch.allclose(model(x), x + out.permute(0, 2, 1).reshape(2, 3, 5, 8), atol=1e-6))

    def test_trimesh_backend_if_available(self):
        try:
            import trimesh  # noqa: F401
            import rtree  # noqa: F401
        except ImportError:
            self.skipTest("trimesh/rtree not installed")
        a = build_graphs(img_size=(6, 12), mesh_level=1, m2g_backend="trimesh")
        self.assertTrue(np.array_equal(np.bincount(a.m2g_receivers), np.full(72, 3)))


if __name__ == "__main__":
    unittest.main()
