import os

import numpy as np
import pytest
import torch

from weatherai.inference import orca_dl as O
from weatherai.inference.errors import MissingInputError, UnitError
from weatherai.inference.manifest import RunManifest
from weatherai.models.orca_dl import ORCADL, ORCADLConfig

H, W, NZ = 32, 60, 2
DEPTHS = (10.0, 50.0)
EXAMPLE = os.environ.get("ORCA_EXAMPLE", "/home/user/data/orca_example")


def lite_cfg():
    return ORCADLConfig(lat_space=(-31.5, 31.5, H), lon_space=(0.5, 359.5, W), in_chans=[NZ, NZ, 1, NZ, NZ, 1],
                        out_chans=[NZ, NZ, 1, NZ, NZ, 1], embed_dim=24, lg_hidden_dim=192, enc_heads=(2, 4, 8), lg_heads=(8, 8),
                        window_size=(4, 5), max_t=3)


def stats(month_offset=True):
    """mean[m] = 10*m (per month), std = 1: a zero normalised prediction de-normalises to 10 * stats-month."""
    mean, std = {}, {}
    for v in O.OCEAN_VARS + O.ATMO_VARS:
        nz = 1 if v in ("tos", "zos", "tauu", "tauv") else NZ
        mean[v] = np.stack([np.full((nz, H, W), 0.1 * m if month_offset else 0.0) for m in range(12)])
        std[v] = np.ones((12, nz, H, W))
    return {"mean": mean, "std": std}


def inputs(month="2026-12", land=True):
    rng = np.random.default_rng(0)
    f = {}
    for v in O.OCEAN_VARS + O.ATMO_VARS:
        nz = 1 if v in ("tos", "zos", "tauu", "tauv") else NZ
        a = rng.uniform(0.0, 1.0, (nz, H, W)) + (20.0 if v in ("so", "thetao", "tos") else 0.0)
        if land:
            a[:, :8, :10] = np.nan                                     # a "continent"
        f[v] = a.squeeze() if nz == 1 else a
    return O.ORCADLInputs(f, month, source="synthetic")


class Zero(torch.nn.Module):
    """Stand-in network returning normalised zeros: isolates the de-normalisation month rule."""
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, ocean, atmo, predict_time_steps=1):
        return torch.zeros(ocean.shape[0], predict_time_steps, ocean.shape[1], *ocean.shape[2:]) + 0 * self.p


def lite(seed):
    torch.manual_seed(seed)
    m = ORCADL(lite_cfg()).eval()
    for p in m.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.05)
    return m


def test_month_rule_and_calendar_across_year_end():
    f = O.ORCADLForecaster.from_modules({1: Zero()}, stats(), lite_cfg(), DEPTHS)
    r = f.run(inputs("2026-12"), lead_months=3)
    # init December (index 11): lead L uses stats month (11 + L) % 12 -> Jan, Feb, Mar = 0, 1, 2 -> values 0.0, 0.1, 0.2
    tos = r.data["tos"][0]
    np.testing.assert_allclose([np.nanmean(tos[i]) for i in range(3)], [0.0, 0.1, 0.2], atol=1e-6)
    assert [str(t)[:10] for t in r.coords["valid_start"][1]] == ["2027-01-01", "2027-02-01", "2027-03-01"]
    assert str(r.coords["valid_end"][1][-1])[:10] == "2027-04-01"


def test_masks_are_nan_not_zero_and_members_are_seeds():
    f = O.ORCADLForecaster.from_modules({1: lite(1), 2: lite(2)}, stats(), lite_cfg(), DEPTHS)
    r = f.run(inputs(), lead_months=4)                       # max_t=3 -> feedback after lead 3
    so = r.data["so"]
    assert so.shape == (2, 4, NZ, H, W) and r.dims["so"] == ("member", "lead", "depth", "lat", "lon")
    assert np.isnan(so[:, :, :, :8, :10]).all()               # land stays missing
    assert np.isfinite(so[:, :, :, 8:, :]).all()
    assert not r.masks["tos"][:8, :10].any() and r.masks["tos"][8:].all()
    assert np.nanmax(np.abs(so[0] - so[1])) > 1e-4            # seeds differ
    assert r.attrs["member_kind"] == "checkpoint_seed" and "ONE model" in r.attrs["ensemble_note"]
    assert "proxy" in r.var_attrs["tos"]["sst_definition"]


def test_input_errors():
    f = O.ORCADLForecaster.from_modules({1: Zero()}, stats(), lite_cfg(), DEPTHS)
    bad = inputs(); bad.fields.pop("vo")
    with pytest.raises(MissingInputError, match="vo"):
        f.run(bad, 1)
    k = inputs(); k.fields["thetao"] = k.fields["thetao"] + 273.15
    with pytest.raises(UnitError, match="thetao"):
        f.run(k, 1)
    s = inputs(); s.fields["so"] = s.fields["so"] / 1000.0
    with pytest.raises(UnitError, match="so"):
        f.run(s, 1)
    with pytest.raises(MissingInputError, match="YYYY-MM"):
        f.run(O.ORCADLInputs(inputs().fields, "Dec 2026"), 1)


def test_unit_conversion_and_depth_interpolation():
    np.testing.assert_allclose(O.convert_units("thetao", [273.15, 300.0], "K"), [0.0, 26.85])
    np.testing.assert_allclose(O.convert_units("so", [0.035], "kg/kg"), [35.0])
    np.testing.assert_allclose(O.convert_units("uo", [10.0], "cm/s"), [0.1])
    with pytest.raises(UnitError):
        O.convert_units("zos", [1.0], "furlong")
    src = np.array([5.0, 15.0, 25.0, 35.0])
    col = np.array([20.0, 18.0, 16.0, np.nan])[:, None]
    out = O.interp_depth(col, src, (10.0, 15.0, 30.0))
    np.testing.assert_allclose(out[:2, 0], [19.0, 18.0])     # linear, exact level
    assert np.isnan(out[2, 0])                                 # bracketing NaN propagates (like cdo intlevel)
    with pytest.raises(MissingInputError, match="extrapolation"):
        O.interp_depth(col, src, (2.0,))


def test_manifest_save_and_hashes(tmp_path):
    pytest.importorskip("xarray")
    f = O.ORCADLForecaster.from_modules({1: Zero()}, stats(), lite_cfg(), DEPTHS)
    p = str(tmp_path / "orca.nc")
    r = f.run(inputs(), 2, out_path=p)
    m = RunManifest.read(p + ".manifest.json")
    assert set(m.inputs) == set(O.OCEAN_VARS + O.ATMO_VARS) and set(m.outputs) == set(O.OCEAN_VARS)
    assert m.config["init_month"] == "2026-12" and m.config_hash() == r.manifest.config_hash()
    assert "git_revision" in m.code or "version" in m.code
    import xarray as xr
    ds = xr.open_dataset(p)
    assert ds.tos.attrs["sst_definition"] == "proxy" and "mask_tos" in ds and ds.attrs["member_kind"] == "checkpoint_seed"
    r2 = f.run(inputs(), 2)
    assert r2.manifest.outputs == r.manifest.outputs          # deterministic


@pytest.mark.skipif(not os.path.exists(os.path.join(EXAMPLE, "salt.nc")), reason="official ORCA-DL example_data not downloaded")
def test_official_example_inputs_pass_checks():
    inp = O.ORCADLInputs.from_official_example(EXAMPLE, "1980-01")
    rep = O.check_inputs(inp)
    assert 0.6 < rep["tos"]["finite_fraction_top"] < 0.75
    lat = np.linspace(-63.5, 63.5, 128); lon = np.linspace(0.5, 359.5, 360)
    la, lo = (lat >= -2) & (lat <= 2), (lon >= 180) & (lon <= 260)
    assert np.nanmean(inp.fields["tauu"][np.ix_(la, lo)]) < 0       # eastward-positive stress: January trades negative
