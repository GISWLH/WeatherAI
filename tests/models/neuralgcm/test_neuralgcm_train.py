"""JAX training workflow for NeuralGCM (weatherai.models.neuralgcm.train): configurable construction, from-scratch /
transfer / official-checkpoint fine-tuning, frozen-parameter training, gradient flow through the dynamical core.

All tests use the bundled ERA5 demo snapshot (no network). The first compile of the dycore takes ~30-60 s on CPU.
Checkpoint: $NGCM_CKPT or the official public one (tests skip if neither is available)."""
import os
import pickle
import unittest

import numpy as np

from weatherai.models.neuralgcm import neuralgcm_available

SMALL = {"LATENT_SIZE": 64, "LAYER_SIZE": 64, "NUM_BLOCKS": 2}
N_OFFICIAL = 14_518_180


def _ckpt():
    path = os.environ.get("NGCM_CKPT")
    if path is None:
        try:
            from weatherai.models.neuralgcm.neuralgcm import download_checkpoint

            path = download_checkpoint("deterministic_2_8_deg")
        except Exception as e:
            raise unittest.SkipTest(f"no checkpoint: {e}")
    with open(path, "rb") as f:
        return pickle.load(f)


@unittest.skipUnless(neuralgcm_available(), "needs jax neuralgcm dinosaur")
class TestNeuralGCMTrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import optax  # noqa: F401
        except Exception as e:
            raise unittest.SkipTest(f"needs optax: {e}")
        from weatherai.models.neuralgcm import train as T

        cls.T = T
        cls.ck = _ckpt()
        cls.small, cls.small_report = T.build_model(cls.ck, SMALL, params="init", seed=0)
        cls.inputs, cls.forc = T.demo_snapshot(cls.small)
        cls.rec_targets = {k: v[None] for k, v in cls.inputs.items() if k != "sim_time"}  # decode(encode(x)) ~ x

    # ---------------------------------------------------------------- configuration
    def test_config_overrides_change_architecture(self):
        self.assertEqual(self.small_report["n_params"], 874_020)  # vs 14,518,180 official
        self.assertEqual(self.small_report["mode"], "init")
        off, rep = self.T.build_model(self.ck, params="official")
        self.assertEqual(rep["n_params"], N_OFFICIAL)
        # tower sizes really changed in the haiku params
        shapes = {k.split("~/")[-1]: v for k, v in self.small.params.items()}
        enc = [v["w"].shape for k, v in self.small.params.items() if "learned_weatherbench_to_primitive_encoder/nodal_mapping" in k and "encode_tower" in k]
        self.assertEqual(enc, [(1052, 64)])
        self.assertTrue(any("process_tower_1" in k for k in self.small.params))  # NUM_BLOCKS=2 -> process_tower, process_tower_1
        self.assertFalse(any("process_tower_2" in k for k in self.small.params))

    def test_make_config_str(self):
        s = self.T.make_config_str("A = 1", {"LATENT_SIZE": 8, "X": "'a'"}, ["Y = 2"])
        self.assertIn("LATENT_SIZE = 8", s)
        self.assertTrue(s.rstrip().endswith("Y = 2"))

    def test_transfer_from_official_keeps_matching_weights(self):
        T = self.T
        m, rep = T.build_model(self.ck, {"NUM_BLOCKS": 6}, params="transfer", seed=1, sample=(self.inputs, self.forc))
        self.assertEqual(rep["transferred_weights"], N_OFFICIAL)
        self.assertGreater(rep["reinitialised_weights"], 0)
        self.assertEqual(rep["n_params"], N_OFFICIAL + rep["reinitialised_weights"])
        k = next(k for k in self.ck["params"] if "encode_tower" in k and "learned_weatherbench_to_primitive_encoder/" in k)
        np.testing.assert_array_equal(np.asarray(m.params[k]["w"]), np.asarray(self.ck["params"][k]["w"]))

    # ---------------------------------------------------------------- from scratch
    def test_from_scratch_reconstruction_training_decreases_loss(self):
        T = self.T
        m2, h = T.fit(self.small, self.inputs, self.forc, self.rec_targets, steps=0, n_iters=8, lr=3e-3)
        self.assertTrue(np.isfinite(h["loss"]).all() and np.isfinite(h["grad_norm"]).all())
        self.assertGreater(min(h["grad_norm"]), 0.0)
        self.assertLess(min(h["loss"][1:]), 0.9 * h["loss"][0])
        self.assertLess(h["loss"][-1], h["loss"][0])
        # parameters actually changed
        k = next(iter(m2.params))
        self.assertGreater(float(np.abs(np.asarray(m2.params[k]["w"]) - np.asarray(self.small.params[k]["w"])).max()) if "w" in m2.params[k] else 1.0, 0.0)

    def test_gradients_flow_through_dycore_rollout(self):
        """steps=1 loss: gradients reach the encoder, the physics tower (inside the dycore step) and the decoder."""
        import jax

        T = self.T
        tgt = {k: v * 1.01 for k, v in self.rec_targets.items()}
        g = jax.grad(lambda p: T.rollout_loss(p, self.small, self.inputs, self.forc, tgt, T.variable_scales(self.inputs), steps=1))(self.small.params)
        flat = {k: float(sum(np.abs(np.asarray(x)).sum() for x in jax.tree_util.tree_leaves(v))) for k, v in g.items()}
        self.assertTrue(all(np.isfinite(list(flat.values()))))
        for key in ("learned_weatherbench_to_primitive_encoder/", "div_curl_neural_parameterization/nodal_mapping", "dimensional_learned_primitive_to_weatherbench_decoder/nodal_mapping"):
            tot = sum(v for k, v in flat.items() if key in k)
            self.assertGreater(tot, 0.0, key)

    def test_freeze_physics_only_training(self):
        T = self.T
        m2, h = T.fit(self.small, self.inputs, self.forc, self.rec_targets, steps=0, n_iters=2, lr=1e-2, freeze=["stochastic_physics_parameterization_step"])
        for mod, ps in self.small.params.items():
            same = all(np.array_equal(np.asarray(m2.params[mod][n]), np.asarray(v)) for n, v in ps.items())
            if "stochastic_physics_parameterization_step" in mod:
                self.assertTrue(same, mod)
        changed = [mod for mod, ps in self.small.params.items() if "stochastic_physics" not in mod and not all(np.array_equal(np.asarray(m2.params[mod][n]), np.asarray(v)) for n, v in ps.items())]
        self.assertGreater(len(changed), 0)

    # ---------------------------------------------------------------- official fine-tune
    def test_finetune_from_official_checkpoint_decreases_loss(self):
        T = self.T
        off, rep = T.build_model(self.ck, params="official")
        inputs, forc = T.demo_snapshot(off)
        tg = {k: v[None] for k, v in inputs.items() if k != "sim_time"}
        l0 = T.evaluate(off, inputs, forc, tg, 0)
        m2, h = T.fit(off, inputs, forc, tg, steps=0, n_iters=4, lr=2e-5)
        self.assertAlmostEqual(h["loss"][0], l0, places=5)
        self.assertLess(h["loss"][-1], h["loss"][0])
        self.assertLess(T.evaluate(m2, inputs, forc, tg, 0), l0)
        self.assertEqual(rep["n_params"], N_OFFICIAL)


if __name__ == "__main__":
    unittest.main()
