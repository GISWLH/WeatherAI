"""FuXi-S2S forecaster plumbing on a stand-in ONNX graph with the official interface (input [1,2,76,121,240], step [1],
two RandomNormalLike nodes, external-data initialiser). The real weights are CC-BY-NC-ND and not available in CI; these
tests cover the ONNX runner, seeded-noise surgery, member/seed handling, masks, units checks, the 42-day limit and resume.
They say nothing about forecast quality."""
import os

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from weatherai.inference import fuxi_s2s as F  # noqa: E402
from weatherai.inference.errors import HorizonError, MissingInputError, UnitError  # noqa: E402

SAMPLE = os.environ.get("FUXI_SAMPLE", "/home/user/data/fuxi_s2s_sample")


def stub_graph(root):
    d = os.path.join(root, "model-1.0")
    os.makedirs(d, exist_ok=True)
    like1 = numpy_helper.from_array(np.zeros((128, 12), np.float32), "like1")
    like2 = numpy_helper.from_array(np.zeros((128, 7200), np.float32), "like2")
    scale = numpy_helper.from_array(np.full((1,), 1.0, np.float32), "scale")
    big = numpy_helper.from_array(np.zeros((4096,), np.float32), "unused_big")         # forces an external-data file
    nodes = [
        helper.make_node("RandomNormalLike", ["like1"], ["/rn1"], dtype=TensorProto.DOUBLE),
        helper.make_node("RandomNormalLike", ["like2"], ["/rn2"], dtype=TensorProto.DOUBLE),
        helper.make_node("ReduceMean", ["/rn1"], ["m1"], keepdims=0),
        helper.make_node("ReduceMean", ["/rn2"], ["m2"], keepdims=0),
        helper.make_node("Add", ["m1", "m2"], ["m12"]),
        helper.make_node("Cast", ["m12"], ["noise"], to=TensorProto.FLOAT),
        helper.make_node("Constant", [], ["s1"], value=numpy_helper.from_array(np.array([1], np.int64))),
        helper.make_node("Constant", [], ["s2"], value=numpy_helper.from_array(np.array([2], np.int64))),
        helper.make_node("Constant", [], ["ax"], value=numpy_helper.from_array(np.array([1], np.int64))),
        helper.make_node("Slice", ["input", "s1", "s2", "ax"], ["last"]),
        helper.make_node("IsNaN", ["last"], ["isnan"]),
        helper.make_node("Constant", [], ["zero"], value=numpy_helper.from_array(np.array(0.0, np.float32))),
        helper.make_node("Where", ["isnan", "zero", "last"], ["lastfix"]),          # finite everywhere, like the real network
        helper.make_node("Constant", [], ["c01"], value=numpy_helper.from_array(np.array(0.01, np.float32))),
        helper.make_node("Mul", ["step", "c01"], ["sterm"]),
        helper.make_node("Add", ["lastfix", "sterm"], ["p1"]),
        helper.make_node("Add", ["p1", "noise"], ["p2"]),
        helper.make_node("Mul", ["p2", "scale"], ["pred"]),
        helper.make_node("Concat", ["last", "pred"], ["output"], axis=1),
    ]
    g = helper.make_graph(nodes, "stub", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 76, 121, 240]),
                                          helper.make_tensor_value_info("step", TensorProto.FLOAT, [1])],
                          [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 76, 121, 240])],
                          initializer=[like1, like2, scale, big])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    p = os.path.join(d, "fuxi_s2s.onnx")
    onnx.save(m, p, save_as_external_data=True, all_tensors_to_one_file=True, location="fuxi_s2s", size_threshold=1024)
    assert os.path.exists(os.path.join(d, "fuxi_s2s"))
    return root


def synthetic_inputs(init="2026-12-01"):
    rng = np.random.default_rng(1)
    x = np.zeros((2, 76, 121, 240), np.float32)
    std_h = dict(zip(F.LEVELS, [110, 760, 1450, 3000, 4200, 5600, 7200, 9200, 10400, 11800, 13600, 16200, 20600]))
    for c, name in enumerate(F.CHANNELS):
        v, lev = name[0], name[1:]
        if name in ("t2m",): base = 288.0
        elif name == "d2m": base = 280.0
        elif name == "sst": base = 290.0
        elif name == "ttr": base = -240.0
        elif name == "msl": base = 101325.0
        elif name == "tcwv": base = 25.0
        elif name == "tp": base = 0.11
        elif name in ("10u", "10v", "100u", "100v"): base = 0.0
        elif v == "z": base = 9.80665 * std_h[int(lev)]
        elif v == "t": base = 250.0
        elif v == "q": base = 1e-3
        else: base = 0.0
        x[:, c] = base + (0.001 * abs(base) + 1e-6) * rng.standard_normal((2, 121, 240))
    x[:, F.CHANNELS.index("tp")] = np.abs(x[:, F.CHANNELS.index("tp")])
    land = np.zeros((121, 240), bool); land[0:60, 0:150] = True
    x[:, F.CHANNELS.index("sst"), land] = np.nan
    d = np.datetime64(init, "D")
    return F.FuXiS2SInputs(x, [d - np.timedelta64(1, "D"), d], "synthetic")


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return stub_graph(str(tmp_path_factory.mktemp("fuxi")))


@pytest.fixture(scope="module")
def fc(root, tmp_path_factory):
    return F.FuXiS2SForecaster.from_onnx(root, cache_dir=str(tmp_path_factory.mktemp("graph")), threads=2)


def test_42_day_limit_is_enforced():
    with pytest.raises(HorizonError, match="42"):
        F.FuXiS2SForecaster._check_lead(43)
    assert F.FuXiS2SForecaster._check_lead(42) == 42


def test_seeded_noise_reproducible_and_members_differ(fc):
    inp = synthetic_inputs()
    a = fc.run(inp, 3, members=2, seed=0, keep=["t2m", "sst", "tp"])
    b = fc.run(inp, 3, members=2, seed=0, keep=["t2m", "sst", "tp"])
    c = fc.run(inp, 3, members=[0], seed=1, keep=["t2m", "sst", "tp"])
    fa, fb, fcx = a.data["fields"], b.data["fields"], c.data["fields"]
    assert fa.shape == (2, 3, 3, 121, 240)
    np.testing.assert_array_equal(fa, fb)
    assert np.nanmax(np.abs(fa[0] - fa[1])) > 1e-4
    assert np.nanmax(np.abs(fa[0] - fcx[0])) > 1e-4
    # stand-in: pred_k = last + 0.01 * k + mean(eps1_k) + mean(eps2_k), k = 0-based lead index fed as `step`
    e1, e2 = F.noise_for(0, 0, 1, [(128, 12), (128, 7200)])
    expect = 0.01 + float(np.mean(e1.astype(np.float64)) + np.mean(e2.astype(np.float64)))
    diff = fa[0, 1, 0] - fa[0, 0, 0]                                       # t2m channel, member 0
    np.testing.assert_allclose(diff, expect, atol=2e-4)
    assert a.manifest.config["members"] == [0, 1] and a.manifest.weights and a.manifest.inputs["x"]


def test_masks_units_and_time_axis(fc):
    inp = synthetic_inputs("2026-12-01")
    r = fc.run(inp, 2, members=1, keep=["sst", "tp", "t2m"])
    sst = r.data["fields"][0, :, 0]
    land = np.isnan(inp.x[-1, F.CHANNELS.index("sst")])
    assert np.isnan(sst[:, land]).all() and np.isfinite(sst[:, ~land]).all()
    assert np.isfinite(r.data["fields"][0, :, 1:]).all()                 # other channels are never masked
    assert list(r.coords["units"][1]) == ["K", "mm h-1", "K"] and "x24" in r.attrs["tp_units"]
    assert [str(t)[:10] for t in r.coords["valid_start"][1]] == ["2026-12-02", "2026-12-03"]
    assert r.attrs["member_kind"] == "stochastic_latent_draw"


def test_resume_skips_identical_members(fc, tmp_path):
    pytest.importorskip("xarray")
    inp = synthetic_inputs()
    idx = fc.run(inp, 2, members=2, keep=["t2m"], out_dir=str(tmp_path))
    t0 = [os.path.getmtime(f) for f in idx.files]
    idx2 = fc.run(inp, 2, members=2, keep=["t2m"], out_dir=str(tmp_path))
    assert [os.path.getmtime(f) for f in idx2.files] == t0                 # nothing recomputed
    fc.run(inp, 2, members=2, seed=5, keep=["t2m"], out_dir=str(tmp_path))
    assert [os.path.getmtime(f) for f in idx.files] != t0                  # different seed -> rewritten
    ds = idx.open()
    assert ds.fields.shape == (2, 2, 1, 121, 240) and os.path.exists(idx.files[0] + ".manifest.json")


def test_internal_noise_mode_runs(root, tmp_path):
    f = F.FuXiS2SForecaster.from_onnx(root, noise="internal", threads=2)
    r = f.run(synthetic_inputs(), 1, members=1, keep=["t2m"])
    assert np.isfinite(r.data["fields"]).all() and r.manifest.notes


def test_input_unit_errors():
    base = synthetic_inputs()
    def mod(ch, fn):
        x = base.x.copy(); i = F.CHANNELS.index(ch); x[:, i] = fn(x[:, i])
        return F.FuXiS2SInputs(x, base.times)
    with pytest.raises(UnitError, match="tp"):
        F.check_inputs(mod("tp", lambda a: a * 24))                       # mm/day given instead of mm/h
    with pytest.raises(UnitError, match="tp"):
        F.check_inputs(mod("tp", lambda a: a / 1000))                     # metres given
    with pytest.raises(UnitError, match="msl"):
        F.check_inputs(mod("msl", lambda a: a / 100))
    with pytest.raises(UnitError, match="ttr"):
        F.check_inputs(mod("ttr", lambda a: a * 3600))
    with pytest.raises(UnitError, match="z500"):
        F.check_inputs(mod("z500", lambda a: a / 9.80665))
    with pytest.raises(MissingInputError, match="land"):
        F.check_inputs(mod("sst", lambda a: np.nan_to_num(a, nan=290.0)))
    with pytest.raises(MissingInputError, match="consecutive"):
        F.check_inputs(F.FuXiS2SInputs(base.x, [np.datetime64("2026-11-28"), np.datetime64("2026-12-01")]))


def test_noise_is_backend_independent():
    a = F.noise_for(3, 7, 11, [(128, 12), (128, 7200)])
    b = F.noise_for(3, 7, 11, [(128, 12), (128, 7200)])
    c = F.noise_for(3, 8, 11, [(128, 12), (128, 7200)])
    np.testing.assert_array_equal(a[1], b[1]) and None
    assert not np.array_equal(a[1], c[1]) and a[0].dtype == np.float32


def test_channel_list_matches_model_module():
    from weatherai.models.fuxi_s2s.model import CHANNELS
    assert CHANNELS == F.CHANNELS


@pytest.mark.skipif(not os.path.exists(os.path.join(SAMPLE, "sample", "total_precipitation.nc")), reason="official sample not downloaded")
def test_official_sample_passes_checks():
    inp = F.FuXiS2SInputs.from_official_files(os.path.join(SAMPLE, "sample"), os.path.join(SAMPLE, "mask.nc"))
    rep = F.check_inputs(inp)
    assert 0.10 < rep["tp_global_mean_mm_per_h"] < 0.13            # 2.76 mm/day / 24 (docs/results/fuxi_s2s_input_check.json)
    assert str(inp.init) == "2020-06-02"
