"""Per-model smoke tests that print one JSON line (used locally and by the HF Space).

    python scripts/gpu_smoke.py aurora [--device cuda] [--pretrained]

Records: pass/fail, wall-clock, device name, torch version, peak GPU memory, and *what was
actually checked* (``checks``). A passing smoke test means a finite forward pass of the
stated shapes — it is not a numerical-parity or skill claim.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback

import torch


def _env(device: str) -> dict:
    d = {"torch": torch.__version__, "device": device}
    if device.startswith("cuda") and torch.cuda.is_available():
        d["gpu"] = torch.cuda.get_device_name(0)
        d["cuda"] = torch.version.cuda
    return d


def smoke_aurora(device: str, pretrained: bool = False) -> dict:
    """Native Aurora: finite fwd/bwd of the lite model; with ``pretrained`` the official small
    checkpoint is strict-loaded into the native model and compared with the official package
    (reference oracle) on the same inputs and device."""
    from weatherai.models import Aurora_lite, Aurora_small

    checks = {}
    g = torch.Generator().manual_seed(0)

    def inputs(H, W):
        return (
            torch.randn(1, 2, 4, H, W, generator=g).to(device),
            torch.randn(3, H, W, generator=g).to(device),
            torch.randn(1, 2, 5, 4, H, W, generator=g).to(device),
        )

    m = Aurora_lite().to(device).eval()
    with torch.no_grad():
        s, a = m(*inputs(16, 32))
    checks["native_lite_forward_finite"] = bool(torch.isfinite(s).all() and torch.isfinite(a).all())
    checks["native_lite_params_M"] = round(sum(p.numel() for p in m.parameters()) / 1e6, 2)
    m.train()
    s, a = m(*inputs(16, 32))
    (s.pow(2).mean() + a.pow(2).mean()).backward()
    checks["native_lite_backward_finite"] = all(
        torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None
    )
    try:
        import aurora  # noqa: F401
        have_official = True
    except Exception:
        have_official = False
    checks["official_package_available"] = have_official

    if have_official:  # random-weight parity, lite config, same device
        from weatherai.models.aurora import AuroraOfficial_lite

        off = AuroraOfficial_lite().to(device).eval()
        ours = Aurora_lite().to(device).eval()
        ours.load_state_dict(off.core.state_dict(), strict=True)
        x = inputs(17, 32)
        with torch.no_grad():
            so, ao = off(*x)
            s, a = ours(*x)
        checks["lite_parity_vs_official_max_abs"] = [float((s - so).abs().max()), float((a - ao).abs().max())]
        checks["lite_parity_allclose_atol1e-5_rtol1e-4"] = bool(
            torch.allclose(s, so, atol=1e-5, rtol=1e-4) and torch.allclose(a, ao, atol=1e-5, rtol=1e-4)
        )
    if pretrained:
        m = Aurora_small(pretrained=True).to(device).eval()  # strict official ckpt load (native model)
        checks["native_small_official_ckpt_strict_load"] = True
        x = inputs(33, 64)
        with torch.no_grad():
            s, a = m(*x)
        checks["native_small_forward_finite"] = bool(torch.isfinite(s).all() and torch.isfinite(a).all())
        checks["native_small_params_M"] = round(sum(p.numel() for p in m.parameters()) / 1e6, 1)
        if have_official:
            import aurora as aur
            from huggingface_hub import hf_hub_download
            from weatherai.models.aurora import AuroraWrapper

            core = aur.AuroraSmallPretrained()
            core.load_checkpoint_local(
                hf_hub_download("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt",
                                revision="0be7e57c685dac86b78c4a19a3ab149d13c6a3dd"), strict=True)
            off = AuroraWrapper(core).to(device).eval()
            with torch.no_grad():
                so, ao = off(*x)
            checks["small_parity_vs_official_max_abs"] = [float((s - so).abs().max()), float((a - ao).abs().max())]
            checks["small_parity_allclose_atol1e-5_rtol1e-4"] = bool(
                torch.allclose(s, so, atol=1e-5, rtol=1e-4) and torch.allclose(a, ao, atol=1e-5, rtol=1e-4)
            )
    return checks


def smoke_graphcast(device: str) -> dict:
    from weatherai.models import GraphCast_lite

    m = GraphCast_lite(in_channels=4, hidden_dim=16, processor_layers=1).to(device)
    x = torch.randn(1, 4, 32, 64, device=device)
    y = m(x)
    (y - x).pow(2).mean().backward()
    return {
        "forward_shape_ok": tuple(y.shape) == (1, 4, 32, 64),
        "forward_finite": bool(torch.isfinite(y).all()),
        "backward_finite": all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None),
    }


def smoke_neuralgcm(device: str, pretrained: bool = False) -> dict:
    """Native NeuralGCM learned components + torch SHT layer (the official JAX model is the oracle).

    Always: native SHT on the 2.8 deg grid vs official dinosaur outputs (``tests/models/neuralgcm/data/sht_ref.npz``)
    and a trainable tower fwd/bwd on ``device``. With ``pretrained``: strict-load the official checkpoint (HF/GCS)
    into the native towers and compare with the real official tower outputs (``tower_ref.npz``). Finally the official
    JAX wrapper forecast is run as before (JAX device reported separately)."""
    import os
    import pickle

    import jax
    import numpy as np

    from weatherai.models import NeuralGCM_lite
    from weatherai.models.neuralgcm.network import EpdTower, convert_official_params
    from weatherai.models.neuralgcm.neuralgcm import download_checkpoint
    from weatherai.models.neuralgcm.spectral import SphericalHarmonicsGrid, SpectralGridConfig

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(here, "tests", "models", "neuralgcm", "data")
    res = {}
    r = np.load(os.path.join(d, "sht_ref.npz"))
    g = SphericalHarmonicsGrid(SpectralGridConfig.TL63()).to(device)
    T = lambda k: torch.tensor(r[k]).to(device)
    vo, di = g.uv_nodal_to_vor_div_modal(T("u"), T("v"))
    u2, v2 = g.vor_div_to_uv_nodal(vo, di)
    res["sht_to_nodal_maxerr"] = float((g.to_nodal(T("modal")).cpu().numpy() - r["to_nodal"]).__abs__().max())
    res["sht_vor_maxerr"] = float((vo.cpu().numpy() - r["vor"]).__abs__().max())
    res["sht_uv_roundtrip_maxerr"] = float(max(abs(u2.cpu().numpy() - r["u_rt"]).max(), abs(v2.cpu().numpy() - r["v_rt"]).max()))
    torch.manual_seed(0)
    tw = EpdTower(6, 3, latent=16, num_blocks=2, process_hidden_layers=1).to(device)
    loss = tw(torch.randn(2, 6, 8, 4, device=device)).square().mean()
    loss.backward()
    res["tower_train_step_finite"] = bool(torch.isfinite(loss).item())
    if pretrained:
        path = download_checkpoint("deterministic_2_8_deg")
        params = pickle.load(open(path, "rb"))["params"]
        m = convert_official_params(params).to(device).eval()
        ref = np.load(os.path.join(d, "tower_ref.npz"))
        errs = {}
        with torch.no_grad():
            for i, tower in enumerate((m.decoder, m.encoder, m.encoder_1, m.physics, m.surface.land, m.surface.sea, m.surface.sea_ice)):
                x = torch.tensor(ref[f"t{i}_x"]).to(device)
                y = ref[f"t{i}_y"]
                o = tower(x[:, :, None])[:, :, 0].cpu().numpy()
                errs[f"tower{i}_maxabs"] = float(abs(o - y).max())
                errs[f"tower{i}_relmax"] = float(abs(o - y).max() / max(1.0, abs(y).max()))
            x = torch.tensor(ref["t7_x"]).to(device)
            errs["vertical_cnn_maxabs"] = float(abs(m.volume_cnn(x[..., None])[..., 0].cpu().numpy() - ref["t7_y"]).max())
        res.update(errs)
        res["official_params_strict_loaded"] = True
    w = NeuralGCM_lite()
    ds = w.demo_data()
    out = w.forecast(ds, steps=2, step_hours=6)
    t = out.temperature
    res.update({
        "jax_backend": jax.default_backend(),
        "jax_devices": str(jax.devices()),
        "official_ckpt_loaded": True,
        "output_shape_ok": tuple(t.shape) == (2, 37, 128, 64),
        "all_vars_finite": bool(all(np.isfinite(v.values).all() for v in out.data_vars.values())),
        "state_evolves": bool(float(abs(t.isel(time=1) - t.isel(time=0)).mean()) > 0),
    })
    return res


def smoke_neuralgcm_train(device: str, pretrained: bool = False) -> dict:
    """JAX training workflow of ``weatherai.models.neuralgcm.train`` on the Space: reports which JAX backend is used,
    builds a modified (small) from-scratch NeuralGCM from the official gin config, runs 4 reconstruction training steps
    on the bundled ERA5 snapshot and one dycore-rollout gradient (gradients finite / non-zero)."""
    import pickle
    import time

    import jax
    import numpy as np

    from weatherai.models.neuralgcm import train as T
    from weatherai.models.neuralgcm.neuralgcm import download_checkpoint

    res = {"jax_version": jax.__version__, "jax_backend": jax.default_backend(), "jax_devices": str(jax.devices())}
    ck = pickle.load(open(download_checkpoint("deterministic_2_8_deg"), "rb"))
    t = time.time()
    m, rep = T.build_model(ck, {"LATENT_SIZE": 64, "LAYER_SIZE": 64, "NUM_BLOCKS": 2}, params="init")
    res["small_model_params"] = rep["n_params"]
    res["build_seconds"] = round(time.time() - t, 1)
    inputs, forc = T.demo_snapshot(m)
    tg = {k: v[None] for k, v in inputs.items() if k != "sim_time"}
    m2, h = T.fit(m, inputs, forc, tg, steps=0, n_iters=4, lr=3e-3)
    res["loss_history"] = [round(x, 4) for x in h["loss"]]
    res["grad_norm_min"] = round(min(h["grad_norm"]), 4)
    res["loss_decreased"] = bool(h["loss"][-1] < h["loss"][0])
    res["train_seconds"] = round(h["seconds"], 1)
    sc = T.variable_scales(inputs)
    g = jax.grad(lambda p: T.rollout_loss(p, m, inputs, forc, {k: v * 1.01 for k, v in tg.items()}, sc, steps=1))(m.params)
    leaves = jax.tree_util.tree_leaves(g)
    res["rollout_grad_all_finite"] = bool(all(np.isfinite(np.asarray(x)).all() for x in leaves))
    res["rollout_grad_nonzero_leaves"] = int(sum(bool((np.asarray(x) != 0).any()) for x in leaves))
    res["rollout_grad_leaves"] = len(leaves)
    res["params_on_device"] = str(jax.tree_util.tree_leaves(m2.params)[0].devices())
    return res


def smoke_gencast(device: str, pretrained: bool = False) -> dict:
    """Native GenCast denoiser: lite fwd/bwd/sample; with ``pretrained`` also the official GenCast-1p0deg-Mini
    weights (gs://dm_graphcast, public) strict-loaded and compared with the official JAX denoiser outputs
    stored in ``tests/models/gencast/data/denoiser_ref_small.npz`` (mesh_size=2, k_hop=3, 13x24 grid)."""
    import os
    import urllib.request

    import numpy as np

    from weatherai.models import GenCast_lite
    from weatherai.models.gencast import GenCastDenoiser_from_official, denoiser_inputs, unstack_variables

    torch.manual_seed(0)
    m = GenCast_lite().to(device)
    cond = torch.randn(2, 8, 16, 32, device=device)
    tgt = torch.randn(2, 4, 16, 32, device=device)
    loss = m.loss(tgt, cond)
    loss.backward()
    m.eval()
    s = m.sample(cond, num_members=4, generator=torch.Generator(device=device).manual_seed(0))
    checks = {
        "params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 3),
        "loss_finite": bool(torch.isfinite(loss)),
        "backward_finite": all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None),
        "sample_shape_ok": tuple(s.shape) == (4, 2, 4, 16, 32),
        "sample_finite": bool(torch.isfinite(s).all()),
        "members_differ": bool(s.std(0).mean() > 0),
    }
    if not pretrained:
        return checks
    ck = "/tmp/gencast_mini.npz"
    if not os.path.exists(ck):
        urllib.request.urlretrieve("https://storage.googleapis.com/dm_graphcast/gencast/params/GenCast%201p0deg%20Mini%20%3C2019.npz", ck)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = np.load(os.path.join(here, "tests", "models", "gencast", "data", "denoiser_ref_small.npz"))
    grp = lambda tag: {k.split("/", 1)[1]: r[k] for k in r.files if k.startswith(tag + "/")}  # noqa: E731
    shape = (len(r["lat"]), len(r["lon"]))
    tmpl = {k: ((1, 13) if r["out/" + k].ndim == 5 else (1,)) for k in r["target_variables"]}

    def errs(attention_type):
        net = GenCastDenoiser_from_official(ck, r["lat"], r["lon"], mesh_size=int(r["mesh_size"]),
                                            attention_k_hop=int(r["k_hop"]), attention_type=attention_type).to(device)
        x = denoiser_inputs(grp("inputs"), grp("forcings"), grp("noisy_targets"), shape).to(device)
        with torch.no_grad():
            y = net(x, torch.from_numpy(r["noise_levels"]).to(device))
        out = unstack_variables(y, tmpl, shape)
        return max(float(np.abs(out[k].cpu().numpy() - r["out/" + k]).max()) for k in tmpl)

    checks["official_weights_strict_load"] = True
    for att in ("dense", "triblockdiag"):
        checks[f"max_abs_err_vs_official_jax_{att}_default_tf32"] = errs(att)
    old = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    e = {att: errs(att) for att in ("dense", "triblockdiag")}
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = old
    for att, v in e.items():
        checks[f"max_abs_err_vs_official_jax_{att}_tf32_off"] = v
        checks[f"{att}_matches_official_atol_2e-4_tf32_off"] = bool(v < 2e-4)
    return checks


def smoke_aardvark(device: str, pretrained: bool = False) -> dict:
    """Lite processor fwd/bwd; with ``pretrained``: official processor / encoder / station-decoder checkpoints
    strict-loaded and the full observation->station E2E run on the official sample, compared with outputs
    stored from the official code (CPU reference; ``tests/models/aardvark/data``)."""
    import os
    import urllib.request

    import numpy as np

    from weatherai.models import AardvarkProcessor_lite

    checks = {}
    m = AardvarkProcessor_lite().to(device)
    y = m(torch.randn(2, 35, 60, 31, device=device))
    y.pow(2).mean().backward()
    checks["lite_shape_ok"] = tuple(y.shape) == (2, 31, 60, 24)
    checks["lite_finite"] = bool(torch.isfinite(y).all())
    checks["lite_backward_finite"] = all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    if pretrained:
        from huggingface_hub import hf_hub_download

        from weatherai.models.aardvark import (AardvarkE2E, load_official_decoder, load_official_encoder,
                                               load_official_processor, load_official_sample)

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dl = lambda f: hf_hub_download("av555/aardvark-weather", f, repo_type="dataset")  # noqa: E731
        ref = np.load(os.path.join(root, "tests/models/aardvark/data/processor_ref.npz"))
        full = load_official_processor(dl("trained_model/processor/forecast_1/epoch_0"), strict=True, device=device).eval()
        checks["official_processor_strict_load"] = True
        g = torch.Generator().manual_seed(0)
        x = torch.randn(1, 35, 240, 121, generator=g).to(device)
        with torch.no_grad():
            out = {lt: full(x, torch.full((1, 1), float(lt), device=device))[:, ::6, ::6].float().cpu().numpy() for lt in (0, 1)}
        for lt in (0, 1):
            err = float(np.abs(out[lt] - ref[f"y_lt{lt}"]).max())
            checks[f"processor_max_abs_err_vs_official_lt{lt}"] = err
            checks[f"processor_matches_official_lt{lt}"] = bool(err < 1e-3)
        # encoder + decoder + E2E vs the official full system (reference made with scripts/aardvark_system_reference.py)
        sref = np.load(os.path.join(root, "tests/models/aardvark/data/system_ref.npz"))
        sp = "/tmp/aardvark_sample_data_final.pkl"
        if not os.path.exists(sp):
            urllib.request.urlretrieve("https://raw.githubusercontent.com/anna-allen/aardvark-weather-public/main/data/sample_data_final.pkl", sp)
        task = load_official_sample(sp)
        mv = lambda o: {k: (mv(v) if isinstance(v, dict) else [t.to(device) for t in v] if isinstance(v, list) else v.to(device)) for k, v in o.items()}  # noqa: E731
        task = mv(task)
        enc = load_official_encoder(dl("trained_model/encoder/epoch_96"), device=device).eval()
        dec = load_official_decoder(dl("trained_model/decoder/tas/lt_1/epoch_18"), device=device).eval()
        checks["official_encoder_decoder_strict_load"] = True
        e2e = AardvarkE2E(enc, [full], dec).to(device).eval()

        def sys_errs():
            with torch.no_grad():
                st, fc, init = e2e(task, return_gridded=True)
                e_state = enc(task["assimilation"])
            return {
                "encoder_state": float(np.abs(e_state.cpu().numpy()[:, ::4, ::4] - sref["encoder_state"]).max()),
                "e2e_station": float(np.abs(st.cpu().numpy() - sref["e2e_station"]).max()),
                "e2e_forecast_rel": float((np.abs(fc.cpu().numpy()[:, ::4, ::4] - sref["e2e_forecast"]) / (np.abs(sref["e2e_forecast"]) + 1.0)).max()),
            }

        # reference is a CPU fp32 run; GPU conv/matmul default to TF32 on recent GPUs -> report both settings
        checks["system_errors_vs_cpu_official_default_tf32"] = sys_errs()
        old = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
        errs = sys_errs()
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = old
        checks["system_errors_vs_cpu_official_tf32_off"] = errs
        checks["system_matches_official_tol_1e-3"] = bool(errs["encoder_state"] < 1e-3 and errs["e2e_station"] < 1e-3 and errs["e2e_forecast_rel"] < 1e-3)
    return checks


def smoke_weathernext_cyclones(device: str, pretrained: bool = False) -> dict:
    """Native WN-C network: lite fwd/bwd/sample + CRPS train step; with ``pretrained`` also the official
    WeatherNextCyclones_Mini_<2024 weights (gs://dm_graphcast, public) strict-loaded and compared with the official JAX
    ``ForwardPass`` outputs stored in ``tests/models/weathernext_cyclones/data`` (mesh_splits=2, k_hop=3, 13x24 grid):
    float64 reference (structure, atol 1e-9) and float32 reference (atol 5e-3)."""
    import os
    import urllib.request

    import numpy as np

    from weatherai.models.weathernext_cyclones import (
        WeatherNextCyclonesNative_lite,
        WeatherNextCyclonesNet_from_official,
        stack_grid_inputs,
        stack_mesh_inputs,
        unstack_outputs,
    )

    torch.manual_seed(0)
    m = WeatherNextCyclonesNative_lite().to(device)
    gx, mx = torch.randn(2, 16 * 32, 12, device=device), torch.randn(2, 2, device=device)
    tgt = torch.randn(2, 16 * 32, m.net.cfg.out_channels, device=device)
    loss = m.loss(gx, mx, tgt, num_samples=3)
    loss.backward()
    with torch.no_grad():
        s = m.sample(gx, mx, num_samples=4, generator=torch.Generator(device=device).manual_seed(0))
    checks = {
        "params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 3),
        "loss_finite": bool(torch.isfinite(loss)),
        "backward_finite": all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None),
        "sample_shape_ok": tuple(s.shape) == (4, 2, 16 * 32, m.net.cfg.out_channels),
        "sample_finite": bool(torch.isfinite(s).all()),
        "members_differ": bool(s.std(0).mean() > 0),
    }
    if not pretrained:
        return checks
    ck = "/tmp/wnc_mini.npz"
    if not os.path.exists(ck):
        urllib.request.urlretrieve("https://storage.googleapis.com/dm_graphcast/weathernext2/params/WeatherNextCyclones_Mini_%3C2024.npz", ck)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(here, "tests", "models", "weathernext_cyclones", "data")
    checks["official_weights_strict_load"] = True

    def errs(ref_name, dtype, attention_type):
        r = np.load(os.path.join(d, ref_name))
        net = WeatherNextCyclonesNet_from_official(ck, r["lat"], r["lon"], mesh_splits=int(r["splits"]), attention_k_hop=int(r["k_hop"]),
                                                   attention_type=attention_type).to(device=device, dtype=dtype)
        grp = lambda tag: {k.split("/", 1)[1]: r[k] for k in r.files if k.startswith(tag + "/")}  # noqa: E731
        shape = (len(r["lat"]), len(r["lon"]))
        g = stack_grid_inputs(grp("inputs"), grp("forcings"), shape).to(device=device, dtype=dtype)
        mm = stack_mesh_inputs(grp("inputs"), grp("forcings")).to(device=device, dtype=dtype)
        with torch.no_grad():
            y = net(g, mm, torch.from_numpy(r["noise"]).to(device=device, dtype=dtype))
        out = unstack_outputs(y, net.cfg, shape)
        return max(float(np.abs(out[k].cpu().numpy() - r["out/" + k]).max()) for k in r["target_variables"])

    for att in ("dense", "triblockdiag"):
        checks[f"max_abs_err_vs_official_jax_float64_{att}"] = errs("forward_ref_small_f64.npz", torch.float64, att)
        checks[f"{att}_float64_matches_official_atol_1e-9"] = bool(checks[f"max_abs_err_vs_official_jax_float64_{att}"] < 1e-9)
    for att in ("dense", "triblockdiag"):
        checks[f"max_abs_err_vs_official_jax_float32_{att}_default_tf32"] = errs("forward_ref_small.npz", torch.float32, att)
    old = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    e = {att: errs("forward_ref_small.npz", torch.float32, att) for att in ("dense", "triblockdiag")}
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = old
    for att, v in e.items():
        checks[f"max_abs_err_vs_official_jax_float32_{att}_tf32_off"] = v
        checks[f"{att}_float32_matches_official_atol_5e-3_tf32_off"] = bool(v < 5e-3)
    return checks


_FUXI_CPU = {}


def preload_fuxi_ens(ck: str) -> dict:
    """Strict-load the official checkpoint into CPU RAM (outside the 120 s ZeroGPU window) and keep it for ``smoke_fuxi_ens``."""
    import time

    from weatherai.models.fuxi_ens import load_official

    t0 = time.time()
    _FUXI_CPU["model"] = load_official(f"{ck}/fuxi_ens.onnx", data_dir=ck, device="cpu", materialize=True)
    return {"cpu_load_seconds": round(time.time() - t0, 1)}


def smoke_fuxi_ens(device: str, pretrained: bool = False) -> dict:
    """FuXi-ENS native PyTorch port.

    Always: lite config forward + one training step on ``device``.
    With ``pretrained``: the official 2.49 B-param checkpoint (downloaded beforehand by the Space's ``fetch_fuxi_ens``
    endpoint to $FUXI_ENS_CKPT, default /tmp/fuxi_ens) is strict-loaded to ``device``, run on the official input.nc with the
    same fixed noise as the onnxruntime reference (torch CPU generator, seed 1234) and compared with the committed
    stride-12 slice of the onnxruntime output (``tests/models/fuxi_ens/data/onnx_final_t1_stride12.npz``)."""
    import os

    import numpy as np

    from weatherai.models.fuxi_ens import FuXiENS_lite, FuXiENSNoise

    res = {}
    m = FuXiENS_lite().to(device)
    cfg = m.cfg
    x = torch.randn(1, 2, cfg.channels, *cfg.img_size, device=device)
    t = torch.zeros(1, device=device)
    noise = FuXiENSNoise.sample(cfg, 1, torch.Generator().manual_seed(0), device=device)
    xn = m.normalise(x)
    z, mean, _ = m.perturb(xn, t, t, t, noise)
    loss = (m.decode(z, t, t, t) - xn[:, 1]).pow(2).mean() + 1e-3 * mean.pow(2).mean()
    loss.backward()
    res["lite_params"] = m.num_parameters()
    res["lite_loss_finite"] = bool(torch.isfinite(loss))
    res["lite_grads_finite"] = all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    if not pretrained:
        return res

    import time

    ck = os.environ.get("FUXI_ENS_CKPT", "/tmp/fuxi_ens")
    if not (os.path.exists(f"{ck}/fuxi_ens") and os.path.exists(f"{ck}/input.nc")):
        res["weights_present"] = False
        res["error"] = f"weights not found in {ck}; call the Space's fetch_fuxi_ens endpoint first and wait for it to finish"
        return res
    import xarray as xr
    import pandas as pd

    from weatherai.models.fuxi_ens import load_official

    t0 = time.time()
    model = _FUXI_CPU.get("model")
    res["preloaded_in_cpu_ram"] = model is not None
    if model is None:
        model = load_official(f"{ck}/fuxi_ens.onnx", data_dir=ck, device="cpu", materialize=True)
    model = model.to(device)          # fp32 (2.47 B params = 9.9 GB); no fp16/bf16 needed on the 48 GB MIG slice
    res["move_to_device_seconds"] = round(time.time() - t0, 1)
    res["official_params"] = model.num_parameters()
    res["official_params_match_onnx"] = model.num_parameters() == 2_474_861_958
    ds = xr.open_dataset(f"{ck}/input.nc")
    xin = torch.from_numpy(ds["fuxi_ens"].values[None].astype(np.float32)).to(device)
    tt = pd.to_datetime(ds.time.values[-1])
    step, hour, doy = (torch.tensor([v], dtype=torch.float32, device=device) for v in (0.0, tt.hour / 24, min(365, tt.day_of_year) / 365))
    ref_npz = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "models", "fuxi_ens", "data", "onnx_final_t1_stride12.npz"))
    g = torch.Generator().manual_seed(int(ref_npz["seed"]))
    d0 = torch.rand(1, 16200, 1536, generator=g)
    res["noise_rng_matches_reference_box"] = abs(float(d0.double().sum()) - float(ref_npz["rng_checksum"])) < 1e-6 * abs(float(ref_npz["rng_checksum"]))
    d1 = torch.rand(1, 16200, 1536, generator=g)
    eps = torch.randn(1, 156, 721, 1440, generator=g)
    noise = FuXiENSNoise(d0.to(device), d1.to(device), eps.to(device))
    ref = ref_npz["final_t1"]

    def run(label):
        torch.cuda.synchronize() if device.startswith("cuda") else None
        t1 = time.time()
        with torch.no_grad():
            y = model(xin, step, hour, doy, noise)
        torch.cuda.synchronize() if device.startswith("cuda") else None
        sl = y[0, 1, :, ::12, ::12].float().cpu().numpy()
        d = np.abs(sl - ref)
        std = model.std.view(-1).cpu().numpy()[:, None, None]
        res[f"{label}_seconds"] = round(time.time() - t1, 2)
        res[f"{label}_max_abs_err_vs_onnxruntime_slice"] = float(d.max())
        res[f"{label}_max_err_over_channel_std"] = float((d / std).max())
        res[f"{label}_finite"] = bool(np.isfinite(sl).all())
        res[f"{label}_t0_is_input_step1"] = bool(torch.equal(torch.nan_to_num(y[:, 0], nan=-7.0), torch.nan_to_num(xin[:, 1], nan=-7.0)))  # input has NaNs; torch.equal(nan, nan) is False
        return y

    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        res["peak_gpu_mem_after_load_MB"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    run("fp32")
    res["fp32_matches_onnxruntime_atol_1e-2_of_std"] = res["fp32_max_err_over_channel_std"] < 1e-2
    # a second member with different noise: ensemble spread is non-zero
    g2 = torch.Generator().manual_seed(1)
    noise2 = FuXiENSNoise.sample(model.cfg, 1, g2, device=device)
    with torch.no_grad():
        y2 = model(xin, step, hour, doy, noise2)
    res["second_member_differs"] = bool((y2[0, 1, :, ::12, ::12].float().cpu().numpy() != ref).any())
    del y2
    return res


def smoke_stormcast(device: str, pretrained: bool = False) -> dict:
    """Lite: forward + one training step + 4-step EDM sampling. Pretrained: official nvidia/stormcast-v1-era5-hrrr (Apache-2.0,
    downloaded to $STORMCAST_CKPT by the Space's ``fetch_weights`` endpoint) strict-loaded; regression net + one EDM denoise
    evaluated at 512x640 on seeded noise inputs and compared with the CPU reference slices
    (``tests/models/stormcast/data/ref_cpu_fp32.npz``, CPU outputs of the native port that are bit-identical to the PhysicsNeMo
    modules); then a full 18-step Heun diffusion sample is timed. No real HRRR/GFS data are used."""
    import os

    import numpy as np
    from weatherai.models.stormcast import StormCast_lite

    res: dict = {}
    m = StormCast_lite().to(device)
    c = m.cfg
    x = torch.randn(2, c.n_state, 64, 64, device=device)
    cond = torch.randn(2, c.n_cond, 64, 64, device=device)
    with torch.no_grad():
        y = m.eval()(x, cond, generator=torch.Generator(device).manual_seed(0))
    res["lite_forward_finite"] = bool(torch.isfinite(y).all()) and tuple(y.shape) == tuple(x.shape)
    m.train()
    inv = m.invariants.expand(2, -1, -1, -1)
    mean = m.regression(torch.cat([x, cond, inv], 1))
    sig = torch.tensor([0.5, 3.0], device=device)
    den = m.diffusion(x + sig.view(-1, 1, 1, 1) * torch.randn_like(x), sig, torch.cat([x, mean.detach(), inv], 1))
    ((mean - x).pow(2).mean() + (den - x).pow(2).mean()).backward()
    res["lite_train_grads_finite"] = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in m.parameters())
    if not pretrained:
        return res

    import time

    ck = os.environ.get("STORMCAST_CKPT", "/tmp/stormcast")
    if not os.path.exists(f"{ck}/EDMPrecond.0.0.mdlus"):
        res["error"] = f"weights not found in {ck}; call the Space's fetch_weights endpoint (model=stormcast) first"
        res["checkpoint_present"] = False
        return res
    from weatherai.models.stormcast import load_official

    t = time.time()
    try:
        model = load_official(ck, with_metadata=True)
        res["metadata_loaded"] = True
    except Exception as e:  # zarr/xarray API differences on the Space's python 3.10 stack
        model = load_official(ck, with_metadata=False)
        res["metadata_loaded"] = False
        res["metadata_error"] = repr(e)[:200]
    res["load_strict_seconds"] = round(time.time() - t, 1)
    res["params_regression_M"] = round(sum(p.numel() for p in model.regression.parameters()) / 1e6, 2)
    res["params_diffusion_M"] = round(sum(p.numel() for p in model.diffusion.parameters()) / 1e6, 2)
    model = model.to(device).eval()
    ref = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "models", "stormcast", "data", "ref_cpu_fp32.npz"))
    g = torch.Generator().manual_seed(0)
    xin = torch.randn(1, 127, 512, 640, generator=g)
    xn = 3 * torch.randn(1, 99, 512, 640, generator=g)
    cd = torch.randn(1, 200, 512, 640, generator=g)
    sl = (slice(None), slice(None, None, 4), slice(None, None, 31), slice(None, None, 37))

    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    def rel(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    old = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    for tag, tf32 in (("tf32_off", False), ("tf32_default", old[0])):
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = tf32
        with torch.no_grad():
            sync(); t = time.time()
            yr = model.regression(xin.to(device)); sync()
            res[f"regression_512x640_seconds_{tag}"] = round(time.time() - t, 2)
            yd = model.diffusion(xn.to(device), torch.tensor([5.0], device=device), cd.to(device)); sync()
        res[f"rel_err_regression_vs_cpu_{tag}"] = rel(yr[sl].float().cpu().numpy(), ref["reg"])
        res[f"rel_err_edm_sigma5_vs_cpu_{tag}"] = rel(yd[sl].float().cpu().numpy(), ref["edm_sigma5"])
        res[f"regression_matches_cpu_{tag}"] = res[f"rel_err_regression_vs_cpu_{tag}"] < (1e-4 if not tf32 else 2e-2)
        res[f"edm_matches_cpu_{tag}"] = res[f"rel_err_edm_sigma5_vs_cpu_{tag}"] < (1e-4 if not tf32 else 2e-2)
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = old
    del yr, yd

    # full one-hour step on synthetic physical-scale inputs: regression + 18-step Heun diffusion (35 denoiser evals)
    xs = (model.means + model.stds * torch.randn(1, 99, 512, 640, device=device)).contiguous()
    cs = model.cond_means + model.cond_stds * torch.randn(1, 26, 512, 640, device=device)
    sync(); t = time.time()
    out = model(xs, cs, generator=torch.Generator(device).manual_seed(0))
    sync()
    res["full_step_18_heun_steps_seconds"] = round(time.time() - t, 1)
    res["full_step_finite"] = bool(torch.isfinite(out).all()) and tuple(out.shape) == (1, 99, 512, 640)
    out2 = model(xs, cs, generator=torch.Generator(device).manual_seed(1))
    res["different_seeds_differ"] = bool((out != out2).any())
    return res


def smoke_arches(device: str, pretrained: bool = False) -> dict:
    """Lite: det + gen forward/train/sample. Pretrained: official ArchesWeatherGen checkpoint (4 embedded deterministic members + generative
    network; HF gcouairon/ArchesWeather, BSD) strict-loaded; checked against CPU reference slices (random inputs; the CPU model is
    parity-checked vs geoarches at full size) and run on a real ERA5 case (WeatherBench2, 2020-06-01 12Z) -> +24 h vs persistence."""
    import os
    import time

    import numpy as np
    from weatherai.models.arches import ArchesWeatherGen_lite, ArchesWeather_lite

    res: dict = {}
    det = ArchesWeather_lite().to(device).train()
    H, W = det.cfg.img_size[1], det.cfg.img_size[2]
    mk = lambda: {"surface": torch.randn(1, 4, 1, H, W, device=device), "level": torch.randn(1, 6, det.cfg.img_size[0], H, W, device=device)}
    s, p = mk(), mk()
    out = det(s, p, torch.tensor([6], device=device), torch.tensor([12], device=device))
    (out["surface"].pow(2).mean() + out["level"].pow(2).mean()).backward()
    res["lite_det_train_grads_finite"] = all(q.grad is not None and bool(torch.isfinite(q.grad).all()) for q in det.parameters())
    gen = ArchesWeatherGen_lite().to(device).eval()
    o = gen.sample(s, p, torch.tensor([6], device=device), torch.tensor([12], device=device), timestamp=torch.tensor([1591012800]), num_steps=3, seed=0)
    res["lite_gen_sample_finite"] = all(bool(torch.isfinite(v).all()) for v in o.values())
    if not pretrained:
        return res

    ck = os.environ.get("ARCHES_CKPT", "/tmp/arches")
    if not os.path.exists(f"{ck}/archesweathergen_checkpoint.ckpt"):
        res["error"] = f"weights not found in {ck}; call the Space's fetch_weights endpoint (model=arches) first"
        return res
    from weatherai.models.arches import denormalize, load_official_gen, normalize, wb2_to_state

    t = time.time()
    model, st = load_official_gen(ck)
    res["load_strict_seconds"] = round(time.time() - t, 1)
    res["members"] = len(model.det_model.core)
    t = time.time()
    model = model.to(device)
    res["to_device_seconds"] = round(time.time() - t, 1)

    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    ref = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "models", "arches", "data", "ref_cpu_fp32.npz"))
    g = torch.Generator().manual_seed(0)
    rs = lambda: {"surface": torch.randn(1, 4, 1, 121, 240, generator=g) * 0.7, "level": torch.randn(1, 6, 13, 121, 240, generator=g) * 0.7}
    s0, p0, noise = rs(), rs(), rs()
    dv = lambda d: {k: v.to(device) for k, v in d.items()}
    month, hour, ts = torch.tensor([6], device=device), torch.tensor([12], device=device), torch.tensor([1591012800])
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    old = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    with torch.no_grad():
        sync(); t = time.time()
        d = model.det_model.core[2](dv(s0), dv(p0), month, hour); sync()
        res["det_member_forward_seconds"] = round(time.time() - t, 2)
        res["rel_err_det_skip_member_vs_cpu"] = max(rel(d["surface"][..., ::7, ::11].cpu().numpy(), ref["det_skip_surface"]),
                                                    rel(d["level"][..., ::7, ::11].cpu().numpy(), ref["det_skip_level"]))
        avg = model.det_model(dv(s0), dv(p0), month, hour)
        o5 = model.sample(dv(s0), dv(p0), month, hour, timestamp=ts, num_steps=5, noise=dv(noise), pred_state=avg)
        res["rel_err_5step_sample_vs_cpu"] = max(rel(o5["surface"][..., ::7, ::11].cpu().numpy(), ref["sample5_surface"]),
                                                 rel(o5["level"][..., ::7, ::11].cpu().numpy(), ref["sample5_level"]))
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = old
    res["matches_cpu_tf32_off"] = res["rel_err_det_skip_member_vs_cpu"] < 1e-3 and res["rel_err_5step_sample_vs_cpu"] < 1e-3

    # real ERA5 case (WeatherBench2 1.5 deg) -> +24 h, latitude-weighted RMSE vs persistence
    f = os.path.join(ck, "wb2_sample_2020.nc")
    if os.path.exists(f):
        import xarray as xr

        ds = xr.open_dataset(f)
        tt = list(ds.time.values)
        raw = {x: dv(wb2_to_state(ds, x)) for x in tt}
        w = torch.cos(torch.linspace(torch.pi / 2, -torch.pi / 2, 121, device=device))[None, :, None]
        w = w / w.mean()
        z500 = lambda st_: st_["level"][:, 0, 7]
        t850 = lambda st_: st_["level"][:, 3, 10]
        rm = lambda a, b: float(((a - b).pow(2) * w).mean().sqrt())
        init, truth = raw[tt[1]], raw[tt[2]]
        m_, h_ = torch.tensor([6], device=device), torch.tensor([12], device=device)
        with torch.no_grad():
            s_, p_ = normalize(init, st), normalize(raw[tt[0]], st)
            sync(); t = time.time()
            dd = denormalize(model.det_model(dv(s_), dv(p_), m_, h_), st); sync()
            res["real_det_mean_seconds"] = round(time.time() - t, 2)
            sync(); t = time.time()
            gg = denormalize(model.sample(dv(s_), dv(p_), m_, h_, timestamp=torch.tensor([1591012800]), seed=0, num_steps=25), st); sync()
            res["real_gen_member_25step_seconds"] = round(time.time() - t, 2)
        res["real_case"] = str(tt[1])
        res["rmse_z500_persistence"] = rm(z500(init), z500(truth)); res["rmse_z500_det_mean"] = rm(z500(dd), z500(truth)); res["rmse_z500_gen_member"] = rm(z500(gg), z500(truth))
        res["rmse_t850_persistence"] = rm(t850(init), t850(truth)); res["rmse_t850_det_mean"] = rm(t850(dd), t850(truth)); res["rmse_t850_gen_member"] = rm(t850(gg), t850(truth))
        res["det_beats_persistence_z500_t850"] = res["rmse_z500_det_mean"] < res["rmse_z500_persistence"] and res["rmse_t850_det_mean"] < res["rmse_t850_persistence"]
        res["gen_member_finite"] = all(bool(torch.isfinite(v).all()) for v in gg.values())
    else:
        res["real_case"] = "wb2_sample_2020.nc not present (fetch_weights arches downloads it)"
    return res


SMOKES = {"arches": smoke_arches, "stormcast": smoke_stormcast, "fuxi_ens": smoke_fuxi_ens, "neuralgcm_train": smoke_neuralgcm_train, "weathernext_cyclones": smoke_weathernext_cyclones, "aardvark": smoke_aardvark, "gencast": smoke_gencast, "neuralgcm": smoke_neuralgcm, "aurora": smoke_aurora, "graphcast": smoke_graphcast}


def run(name: str, device: str, **kw) -> dict:
    out = {"model": name, **_env(device)}
    t0 = time.time()
    try:
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        checks = SMOKES[name](device, **kw)
        out["checks"] = checks
        out["status"] = "PASS" if all(v for k, v in checks.items() if isinstance(v, bool)) else "FAIL"
    except Exception:
        out["status"] = "FAIL"
        out["error"] = traceback.format_exc()[-1500:]
    out["seconds"] = round(time.time() - t0, 2)
    if device.startswith("cuda") and torch.cuda.is_available():
        out["peak_gpu_mem_MB"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(SMOKES))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pretrained", action="store_true")
    a = ap.parse_args()
    kw = {"pretrained": a.pretrained} if a.model in ("aurora", "aardvark", "gencast", "weathernext_cyclones", "neuralgcm", "fuxi_ens", "stormcast", "arches") else {}
    print(json.dumps(run(a.model, a.device, **kw)))
