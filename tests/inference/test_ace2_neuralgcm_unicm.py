import os

import numpy as np
import pytest

from weatherai.inference.errors import (BackendUnavailableError, MissingForcingError, MissingInputError,
                                        WeightsUnavailableError)


def lite_ace2():
    torch = pytest.importorskip("torch")
    from weatherai.models.ace2 import ACE2, ACE2Config, SFNOConfig
    torch.manual_seed(0)
    h, w, nl = 12, 24, 3
    names = ["PRESsfc", "surface_temperature", "PRATEsfc", "LHTFLsfc", "tendency_of_total_water_path_due_to_advection"] + \
        [f"specific_total_water_{i}" for i in range(nl)]
    ins = ["ocean_fraction", "DSWRFtoa"] + names[:2] + names[5:]
    means = {n: 0.0 for n in ins + names}; stds = {n: 1.0 for n in ins + names}
    means["PRESsfc"], stds["PRESsfc"] = 1e5, 1e3
    means["surface_temperature"], stds["surface_temperature"] = 288.0, 10.0
    for i in range(nl):
        means[f"specific_total_water_{i}"], stds[f"specific_total_water_{i}"] = 5e-3, 1e-3
    cfg = ACE2Config(sfno=SFNOConfig(in_chans=len(ins), out_chans=len(names), img_shape=(h, w), embed_dim=8, num_layers=2),
                     in_names=tuple(ins), out_names=tuple(names), n_levels=nl,
                     force_positive=tuple(f"specific_total_water_{i}" for i in range(nl)) + ("PRATEsfc",))
    m = ACE2(cfg, means, stds, torch.rand(h, w) + 0.1, torch.tensor([0.0, 2000.0, 1000.0, 0.0]), torch.tensor([0.0, 0.1, 0.6, 1.0])).eval()
    from weatherai.inference.ace2 import ACE2Forecaster
    f = ACE2Forecaster(m)
    ic = {n: np.full((h, w), means[n], np.float32) for n in f.prognostic}
    ic["PRESsfc"] = (1e5 + 500 * np.random.default_rng(0).standard_normal((h, w))).astype(np.float32)
    return f, ic, (h, w)


def forcing(T, hw, ocean):
    sst = 290.0 + np.arange(T, dtype=np.float32)[:, None, None] * 0.5 + np.zeros((T, *hw), np.float32)
    return {"ocean_fraction": np.broadcast_to(ocean.astype(np.float32), (T, *hw)).copy(),
            "DSWRFtoa": np.full((T, *hw), 340.0, np.float32), "surface_temperature": sst}


def test_ace2_forcing_must_cover_every_step_and_provenance_is_recorded():
    f, ic, hw = lite_ace2()
    ocean = np.zeros(hw, bool); ocean[:, : hw[1] // 2] = True
    with pytest.raises(MissingForcingError, match="surface_temperature"):
        f.run(ic, {k: v for k, v in forcing(5, hw, ocean).items() if k != "surface_temperature"}, 4, "forecast")
    with pytest.raises(MissingForcingError, match="3 of 5"):
        f.run(ic, forcing(3, hw, ocean), 4, "forecast")
    init = np.datetime64("2027-01-01T00")
    times = init + np.arange(4) * np.timedelta64(6, "h")                     # one 6-h time short
    with pytest.raises(MissingForcingError, match="time axis"):
        f.run(ic, forcing(5, hw, ocean), 4, "scenario", init=init, forcing_times=times)
    with pytest.raises(ValueError):
        f.run(ic, forcing(5, hw, ocean), 4, "obs")                             # provenance is mandatory and checked
    with pytest.raises(MissingInputError):
        f.run({}, forcing(5, hw, ocean), 4, "forecast")
    r = f.run(ic, forcing(5, hw, ocean), 4, "observed", init=init)
    assert r.attrs["experiment_type"] == "conditional_hindcast" and r.attrs["usable_as_forecast"] == "no"
    sst = r.data["surface_temperature"][0]
    np.testing.assert_allclose(sst[:, ocean], (290.0 + 0.5 * np.arange(1, 5))[:, None] * np.ones((1, ocean.sum())))   # prescribed
    assert str(r.coords["valid_time"][1][-1])[:13] == "2027-01-02T00" and list(r.coords["lead"]) == [6, 12, 18, 24]
    assert set(k.split(":")[0] for k in r.manifest.inputs) == {"ic", "forcing"}


def test_unicm_is_blocked_without_weights():
    from weatherai.inference.unicm import UniCMForecaster
    with pytest.raises(WeightsUnavailableError, match="no published checkpoint"):
        UniCMForecaster.from_checkpoint()
    with pytest.raises(BackendUnavailableError):
        UniCMForecaster(None, "x").run()


def test_neuralgcm_torch_backend_is_not_a_forecaster():
    from weatherai.inference.neuralgcm import NeuralGCMForecaster
    with pytest.raises(BackendUnavailableError, match="partial"):
        NeuralGCMForecaster.from_pretrained(backend="torch")


@pytest.mark.skipif(os.environ.get("WEATHERAI_NETWORK") != "1", reason="downloads the official 2.8 deg checkpoint (58 MB); set WEATHERAI_NETWORK=1")
def test_neuralgcm_official_jax_demo_rollout():
    pytest.importorskip("neuralgcm")
    from weatherai.inference.neuralgcm import NeuralGCMForecaster
    f = NeuralGCMForecaster.from_pretrained("deterministic_2_8_deg")
    ds = f.wrapper.demo_data()
    with pytest.raises(MissingForcingError):
        f.run(ds, 4, forcing_policy="provided")                                # demo has one time only
    r = f.run(ds, 4, step_hours=6, forcing_policy="persist_initial", variables=["temperature", "geopotential"])
    t = r.data["temperature"]
    assert t.shape[:2] == (1, 4) and np.isfinite(t).all()
    # lead 0 is the encoded/decoded initial state (start_with_input=True), not the analysis: on the demo snapshot it has 157 K at
    # 10 hPa where the input minimum is 183 K, so physical-range checks start at lead 6 h
    assert 175 < float(t[:, 1:].min()) and float(t[:, 1:].max()) < 330
    assert r.attrs["forcing_policy"] == "persist_initial" and r.manifest.weights
    with pytest.raises(ValueError, match="deterministic"):
        f.run(ds, 2, members=2)
