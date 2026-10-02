"""ArchesWeather(Gen) parity vs the official ``geoarches`` package (INRIA, BSD-3) used as the oracle. Skipped if geoarches / diffusers
are missing (oracle env: /workspace/venv/ga). The full-size official-checkpoint test needs ARCHES_FULL=1 and the HF checkpoints."""
import os

import pytest
import torch

pytest.importorskip("geoarches", reason="geoarches oracle not installed")
pytest.importorskip("diffusers")
from geoarches.backbones.archesweather import ArchesWeatherCondBackbone, WeatherEncodeDecodeLayer

from weatherai.models.arches import ArchesConfig, ArchesWeatherGen_lite
from weatherai.models.arches.model import ArchesBackbone, ArchesEmbedder

CKPT = os.environ.get("ARCHES_CKPT", "/workspace/ckpt/arches")


def _lite_pair(n_cat=3):
    cfg = ArchesConfig(img_size=(5, 25, 48), emb_dim=48, cond_dim=32, num_heads=(2, 4, 4, 2), window_size=(1, 3, 4), depth_multiplier=1,
                       n_concatenated_states=n_cat)
    ob = ArchesWeatherCondBackbone(tensor_size=cfg.tensor_size, emb_dim=48, cond_dim=32, num_heads=(2, 4, 4, 2), window_size=(1, 3, 4),
                                   droppath_coeff=0.2, depth_multiplier=1, dropout=0, use_skip=True, first_interaction_layer="linear",
                                   axis_attn=True, mlp_layer="swiglu", mlp_ratio=4.0).eval()
    oe = WeatherEncodeDecodeLayer(img_size=(5, 25, 48), emb_dim=48, out_emb_dim=96, patch_size=(2, 2, 2), surface_ch=4, level_ch=6,
                                  n_concatenated_states=n_cat).eval()
    g = torch.Generator().manual_seed(0)
    for mod in (ob, oe):
        for p in mod.parameters():
            p.data.copy_(torch.randn(p.shape, generator=g) * 0.1)
    masks = torch.randn(3, 1, 24, 48)
    oe.constant_masks = masks
    nb, ne = ArchesBackbone(cfg).eval(), ArchesEmbedder(cfg, masks).eval()
    nb.load_state_dict(ob.state_dict(), strict=True)
    ne.load_state_dict(oe.state_dict(), strict=True)
    return cfg, ob, oe, nb, ne


def test_embedder_backbone_lite_exact():
    from tensordict.tensordict import TensorDict
    cfg, ob, oe, nb, ne = _lite_pair(n_cat=3)
    g = torch.Generator().manual_seed(1)
    mk = lambda: {"surface": torch.randn(2, 4, 1, 25, 48, generator=g), "level": torch.randn(2, 6, 5, 25, 48, generator=g)}
    s, c = mk(), {k: torch.cat([v, v.flip(-1), v * 0.5], 1) for k, v in mk().items()}
    td = lambda d: TensorDict(surface=d["surface"], level=d["level"], batch_size=2)
    cond = torch.randn(2, 32, generator=g)
    with torch.no_grad():
        eo, en = oe.encode(td(s), td(c)), ne.encode(s, c)
        assert torch.equal(eo, en)
        bo, bn = ob(eo, cond), nb(en, cond)
        assert torch.allclose(bo, bn, atol=1e-5, rtol=1e-5), (bo - bn).abs().max()
        do, dn = oe.decode(bo), ne.decode(bn)
        assert torch.allclose(do["surface"], dn["surface"], atol=1e-5) and torch.allclose(do["level"], dn["level"], atol=1e-5)


def test_flow_scheduler_matches_diffusers():
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    sch = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    sch.set_timesteps(25)
    t, s = ArchesWeatherGen_lite().flow_timesteps(25)
    assert torch.allclose(sch.timesteps.float(), t, atol=1e-3) and torch.allclose(sch.sigmas.float(), s, atol=1e-6)
    x, v = torch.randn(2, 3), torch.randn(2, 3)
    out = sch.step(v, sch.timesteps[3], x, s_churn=0.0).prev_sample
    assert torch.allclose(out, x + (s[4] - s[3]) * v, atol=1e-6)


@pytest.mark.skipif(not (os.environ.get("ARCHES_FULL") and os.path.exists(f"{CKPT}/archesweathergen_checkpoint.ckpt")), reason="set ARCHES_FULL=1 with the HF checkpoints")
def test_official_checkpoint_det_member_full_size():
    from geoarches.backbones.archesweather import ArchesWeatherCondBackbone as B  # noqa: F401
    from weatherai.models.arches import load_official_det
    m = load_official_det(CKPT)
    assert m.cfg.depth_multiplier == 2  # full oracle comparison is scripts/arches_parity_full.py (needs the geoarches modelstore layout)
