"""Parity against the official ai2cm/ace (`fme`) implementation. Needs `pip install fme torch-harmonics==0.8.0`."""
import os
import types
import pytest
import torch

pytest.importorskip("fme")
th = pytest.importorskip("torch_harmonics")

from weatherai.models.ace2 import SFNO, SFNOConfig, RealSHT, InverseRealSHT, load_official

CK = os.environ.get("ACE2_CKPT", "/workspace/ckpt/ace2")


@pytest.mark.parametrize("grid,shape", [("legendre-gauss", (24, 48)), ("equiangular", (33, 64)), ("legendre-gauss", (180, 360))])
def test_sht_vs_torch_harmonics(grid, shape):
    h, w = shape
    x = torch.randn(1, 2, h, w)
    a, b = RealSHT(h, w, grid=grid), th.RealSHT(h, w, grid=grid).float()
    ca, cb = a(x), b(x)
    assert (ca - cb).abs().max() <= 1e-6 * cb.abs().max()
    ya = InverseRealSHT(h, w, grid=grid)(cb)
    yb = th.InverseRealSHT(h, w, grid=grid).float()(cb)
    assert (ya - yb).abs().max() <= 1e-6 * yb.abs().max()


def test_lite_sfno_vs_fme_random_weights():
    from fme.ace.models.modulus.sfnonet import SphericalFourierNeuralOperatorNet as Ref
    torch.manual_seed(0)
    # fme reads the grid from `params.data_grid`
    ref = Ref(types.SimpleNamespace(data_grid="legendre-gauss"), spectral_transform="sht", filter_type="linear",
              operator_type="dhconv", img_shape=(24, 48), in_chans=7, out_chans=5, embed_dim=16, num_layers=3,
              normalization_layer="instance_norm").eval()
    with torch.no_grad():                                        # non-trivial pos_embed / biases so every path matters
        for p in ref.parameters():
            p.add_(0.05 * torch.randn_like(p))
    mine = SFNO(SFNOConfig(in_chans=7, out_chans=5, img_shape=(24, 48), embed_dim=16, num_layers=3)).eval()
    mine.load_state_dict(ref.state_dict(), strict=True)
    x = torch.randn(2, 7, 24, 48)
    with torch.no_grad():
        a, b = ref(x), mine(x)
    assert (a - b).abs().max() <= 1e-5 * a.abs().max()


@pytest.mark.skipif(os.environ.get("ACE2_FULL") != "1" or not os.path.exists(os.path.join(CK, "ace2_era5_ckpt.tar")),
                    reason="set ACE2_FULL=1 and download the checkpoint (several minutes on CPU)")
def test_official_two_step_rollout_vs_fme():
    import subprocess, sys, json
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, "..", "..", "..", "scripts", "ace2_parity_fme.py")
    subprocess.run([sys.executable, script, "2"], check=True)
    rep = json.load(open("/tmp/ace2_parity.json"))
    assert rep[0]["max_rel_err"] < 1e-5 and rep[1]["max_rel_err"] < 1e-4
