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


def smoke_neuralgcm(device: str) -> dict:
    """Official JAX NeuralGCM 2.8 deg checkpoint. JAX device is reported separately: torch's
    ``device`` is irrelevant here (jax[cpu] unless jax[cuda] is installed)."""
    import jax
    import numpy as np

    from weatherai.models import NeuralGCM_lite

    w = NeuralGCM_lite()
    ds = w.demo_data()
    out = w.forecast(ds, steps=2, step_hours=6)
    t = out.temperature
    return {
        "jax_backend": jax.default_backend(),
        "jax_devices": str(jax.devices()),
        "official_ckpt_loaded": True,
        "output_shape_ok": tuple(t.shape) == (2, 37, 128, 64),
        "all_vars_finite": bool(all(np.isfinite(v.values).all() for v in out.data_vars.values())),
        "state_evolves": bool(float(abs(t.isel(time=1) - t.isel(time=0)).mean()) > 0),
    }


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


SMOKES = {"aardvark": smoke_aardvark, "gencast": smoke_gencast, "neuralgcm": smoke_neuralgcm, "aurora": smoke_aurora, "graphcast": smoke_graphcast}


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
    kw = {"pretrained": a.pretrained} if a.model in ("aurora", "aardvark", "gencast") else {}
    print(json.dumps(run(a.model, a.device, **kw)))
