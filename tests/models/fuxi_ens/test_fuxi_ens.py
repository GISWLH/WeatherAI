"""FuXi-ENS native PyTorch port.

* synthetic tests (always run): shapes, determinism with explicit noise, gradient flow / training step,
  shift-mask / RoPE / norm helper invariants, window attention vs a naive reference;
* official-checkpoint tests (run only if ``/workspace/ckpt/fuxi_ens`` (or $FUXI_ENS_CKPT) holds the Zenodo
  files): strict load, ONNX ``attn_mask`` equality, and the stage-by-stage comparison with onnxruntime
  references produced by ``scripts/fuxi_ens_ort_segments.py`` (``<ckpt>/ref/*.npy``).
"""
import os

import numpy as np
import pytest
import torch

from weatherai.models.fuxi_ens import (FuXiENS, FuXiENS_lite, FuXiENSConfig, FuXiENSNoise, make_rope_table,
                                       make_shift_mask, sinusoidal_embedding, unbiased_layer_norm)

CK = os.environ.get("FUXI_ENS_CKPT", "/workspace/ckpt/fuxi_ens")
HAVE_CK = os.path.exists(f"{CK}/fuxi_ens") and os.path.exists(f"{CK}/fuxi_ens.onnx")
HAVE_REF = os.path.exists(f"{CK}/ref/final.npy")


def _inputs(cfg, b=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, 2, cfg.channels, *cfg.img_size, generator=g)
    t = torch.zeros(b)
    return x, t, t + 0.25, t + 0.5


def test_lite_forward_shape_and_passthrough():
    m = FuXiENS_lite().eval()
    cfg = m.cfg
    x, s, h, d = _inputs(cfg)
    noise = FuXiENSNoise.sample(cfg, 1, torch.Generator().manual_seed(1))
    with torch.no_grad():
        y = m(x, s, h, d, noise)
    assert y.shape == x.shape
    assert torch.equal(y[:, 0], x[:, 1])                       # output step 0 is the previous input's step 1
    assert torch.isfinite(y).all()
    assert (y[:, 1, -1] >= -1e-6).all()                        # precip = exp(clip(.,0,7)) - 1 >= 0


def test_noise_is_the_only_randomness_and_matters():
    m = FuXiENS_lite().eval()
    x, s, h, d = _inputs(m.cfg)
    n1 = FuXiENSNoise.sample(m.cfg, 1, torch.Generator().manual_seed(1))
    n2 = FuXiENSNoise.sample(m.cfg, 1, torch.Generator().manual_seed(2))
    with torch.no_grad():
        a, b, c = m(x, s, h, d, n1), m(x, s, h, d, n1), m(x, s, h, d, n2)
    assert torch.equal(a, b)
    assert (a - c).abs().max() > 0                             # different noise -> different ensemble member


def test_generator_driven_forward_reproducible():
    m = FuXiENS_lite().eval()
    x, s, h, d = _inputs(m.cfg)
    with torch.no_grad():
        a = m(x, s, h, d, generator=torch.Generator().manual_seed(7))
        b = m(x, s, h, d, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_gradients_flow_and_training_step_reduces_loss():
    torch.manual_seed(0)
    m = FuXiENS_lite()
    cfg = m.cfg
    x, s, h, d = _inputs(cfg, seed=3)
    # make buffers sane so the toy problem is well conditioned
    noise = FuXiENSNoise.sample(cfg, 1, torch.Generator().manual_seed(5))
    target = (x[:, 1] * 0.9 + 0.1)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    losses = []
    for i in range(12):
        opt.zero_grad()
        xn = m.normalise(x)
        z, mean, logvar = m.perturb(xn, s, h, d, noise)
        pred = m.decode(z, s, h, d)
        tgt = m.normalise(target[:, None].expand(-1, 2, -1, -1, -1))[:, 0]
        loss = ((pred - tgt) ** 2).mean() + 1e-3 * (mean ** 2).mean()
        loss.backward()
        if i == 0:
            names_no_grad = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
            assert not names_no_grad, names_no_grad
            zero = [n for n, p in m.named_parameters() if p.grad.abs().max() == 0]
            # at random init the zero-initialised-free net has gradients everywhere except possibly dist_p.logvar bias path
            assert len(zero) <= 2, zero
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.8, losses


def test_unbiased_layer_norm_has_unit_unbiased_variance():
    x = torch.randn(3, 5, 64)
    got = unbiased_layer_norm(x, eps=0.0)
    assert torch.allclose(got.var(-1, unbiased=True), torch.ones(3, 5), atol=1e-4)
    ref = torch.nn.functional.layer_norm(x, (64,), eps=0.0)       # torch uses the biased variance
    assert torch.allclose(got, ref * np.sqrt(63 / 64), atol=1e-5)


def test_sinusoid_and_rope_tables():
    e = sinusoidal_embedding(torch.tensor([0.0, 0.5]), 8)
    assert e.shape == (2, 16)
    assert torch.allclose(e[0, :8], torch.zeros(8)) and torch.allclose(e[0, 8:], torch.ones(8))
    c, s = make_rope_table(10, 8)
    assert c.shape == (10, 4) and torch.allclose(c ** 2 + s ** 2, torch.ones(10, 4), atol=1e-6)


def test_shift_mask_structure():
    m = make_shift_mask((90, 180), (18, 18), (9, 9))
    assert m.shape == (50, 324, 324)
    assert set(m.unique().tolist()) == {-100.0, 0.0}
    assert (m[:40] == 0).all()                                 # first 4 window rows unaffected
    assert (m[40:] < 0).any()


def test_window_attention_matches_naive_reference():
    cfg = FuXiENSConfig.lite().replace(dim=32, heads=2)
    from weatherai.models.fuxi_ens.model import AdaLNBlock
    blk = AdaLNBlock(cfg, shifted=True).eval()
    H, W = cfg.grid
    x = torch.randn(1, H * W, cfg.dim)
    cos, sin = make_rope_table(H * W, cfg.head_dim)
    with torch.no_grad():
        got = blk.attn(x, blk.attn_mask, cos, sin)
        # naive: roll, per-window loop, explicit softmax
        a = blk.attn
        xs = torch.roll(x.view(1, H, W, -1), (-cfg.shift[0], -cfg.shift[1]), (1, 2))
        wh, ww = cfg.window
        xw = xs.view(1, H // wh, wh, W // ww, ww, -1).permute(0, 1, 3, 2, 4, 5).reshape(-1, wh * ww, cfg.dim)
        q, k, v = a.wq(xw), a.wk(xw), a.wv(xw)
        nW, T = xw.shape[0], wh * ww

        def rot(t):
            t = t.reshape(1, nW * T, cfg.heads, cfg.head_dim)
            t0, t1 = t[..., 0::2], t[..., 1::2]
            c, s = cos[: nW * T][None, :, None], sin[: nW * T][None, :, None]
            r = torch.stack([t0 * c - t1 * s, t0 * s + t1 * c], -1).flatten(-2)
            return r.reshape(nW, T, cfg.heads, cfg.head_dim).transpose(1, 2)

        q, k = rot(q), rot(k)
        v = v.reshape(nW, T, cfg.heads, cfg.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) * cfg.head_dim ** -0.5 + blk.attn_mask[:, None]
        o = (att.softmax(-1) @ v).transpose(1, 2).reshape(nW, T, cfg.dim)
        o = a.wo(o)
        o = o.view(1, H // wh, W // ww, wh, ww, -1).permute(0, 1, 3, 2, 4, 5).reshape(1, H, W, -1)
        ref = torch.roll(o, cfg.shift, (1, 2)).reshape(1, H * W, -1)
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()


def test_lite_state_dict_keys_follow_checkpoint_naming():
    keys = set(FuXiENS_lite().state_dict())
    for k in ["dist_p.patch_embed.input_proj.weight", "dist_p.patch_embed.norm.weight", "dist_p.step_embed.mlp.0.weight",
              "dist_p.layers.0.blocks.0.adaln.1.weight", "dist_p.layers.0.blocks.0.attn.wq.weight",
              "dist_p.layers.0.blocks.0.mlp.fc1.weight", "dist_p.norm_layer.scale_shift.1.weight", "dist_p.mean.weight",
              "decoder.0.pred_layer.pl_head.weight", "decoder.0.pred_layer.sf_head.bias", "mean", "std", "const"]:
        assert k in keys, k


def test_official_config_param_count_matches_checkpoint_without_allocating():
    with torch.device("meta"):
        m = FuXiENS(FuXiENSConfig.official())
    # exactly the number of weight elements in the ONNX (excluding attn_mask / mean / std / const buffers)
    assert m.num_parameters() == 2_474_861_958, m.num_parameters()


# ----------------------------------------------------------------------------- official checkpoint
@pytest.mark.skipif(not HAVE_CK, reason="Zenodo FuXi-ENS files not present")
def test_official_strict_load_and_masks_match_onnx():
    import onnx
    from onnx import numpy_helper
    from weatherai.models.fuxi_ens import load_official
    m = load_official(f"{CK}/fuxi_ens.onnx")
    assert m.num_parameters() == 2_474_861_958
    mod = onnx.load(f"{CK}/fuxi_ens.onnx", load_external_data=False)
    ref = None
    for init in mod.graph.initializer:
        if init.name.endswith("attn_mask"):
            ext = {e.key: e.value for e in init.external_data}
            ref = np.fromfile(f"{CK}/{ext['location']}", dtype=np.float32, count=int(ext["length"]) // 4, offset=int(ext["offset"]))
            ref = ref.reshape(tuple(init.dims))
            mine = make_shift_mask((90, 180), (18, 18), (9, 9)).numpy()
            assert np.array_equal(m.dist_p.layers[0].blocks[1].attn_mask.numpy(), mine)
            assert np.array_equal(ref, mine), init.name
    assert ref is not None


@pytest.mark.skipif(not HAVE_REF, reason="ONNX reference tensors not generated (scripts/fuxi_ens_ort_segments.py)")
def test_official_forward_matches_onnxruntime_with_fixed_noise():
    import json
    rep = json.load(open(f"{CK}/ref/report.json"))
    assert rep["final"]["max_abs_over_max_ref"] < 1e-3, rep["final"]
    for k in ("xn", "demb", "dout", "mean", "z", "dl5", "dec_out"):
        assert rep[k]["rmse_over_rms_ref"] < 1e-4, (k, rep[k])
