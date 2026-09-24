import unittest

import torch

from weatherai.models import GraphCast, GraphCast_lite
from weatherai.models.graphcast import mesh_num_nodes


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


if __name__ == "__main__":
    unittest.main()
