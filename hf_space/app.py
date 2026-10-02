"""ZeroGPU smoke runner for WeatherAI models (see README)."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
import traceback

# JAX/XLA on the ZeroGPU Blackwell slice: the GEMM autotuner aborts the process on the dycore's HIGHEST-precision
# (dot_bf16_bf16_f32_x6) matmuls (fatal "reference (cublas)" compile failure); disable autotuning (must be set before jax is imported).
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
import gradio as gr
import spaces
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(ROOT, "WeatherAI")
sys.path.insert(0, PKG)

MODELS = ["graphcast", "aurora", "neuralgcm", "gencast", "aardvark", "weathernext_cyclones", "neuralgcm_train", "fuxi_ens", "stormcast"]  # extended as models are added


def _smoke(model: str, pretrained: bool) -> str:
    from scripts.gpu_smoke import run

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kw = {"pretrained": pretrained} if model in ("aurora", "aardvark", "gencast", "weathernext_cyclones", "neuralgcm", "fuxi_ens", "stormcast") else {}
    return json.dumps(run(model, dev, **kw), indent=2)


@spaces.GPU(duration=120)
def smoke_gpu(model: str, pretrained: bool = False) -> str:
    try:
        return _smoke(model, pretrained)
    except Exception:
        return "STATUS=FAIL\n" + traceback.format_exc()


def smoke_cpu(model: str, pretrained: bool = False) -> str:
    try:
        from scripts.gpu_smoke import run

        return json.dumps(run(model, "cpu", **({"pretrained": pretrained} if model in ("aurora", "aardvark", "gencast", "weathernext_cyclones", "neuralgcm", "fuxi_ens", "stormcast") else {})), indent=2)
    except Exception:
        return "STATUS=FAIL\n" + traceback.format_exc()


_FETCH = {"state": "idle", "log": []}


def fetch_fuxi_ens(background: bool = True) -> str:
    """Download FuXi-ENS (Zenodo 10.5281/zenodo.15124541, CC-BY-NC-4.0) to $FUXI_ENS_CKPT (default /tmp/fuxi_ens).

    Runs on the Space CPU in a background thread (10 GB, outside the 120 s ZeroGPU window); call again to poll."""
    import threading

    out = os.environ.get("FUXI_ENS_CKPT", "/tmp/fuxi_ens")
    if _FETCH["state"] == "idle":
        def work():
            try:
                from scripts.fetch_fuxi_ens import fetch

                _FETCH["state"] = "running"
                fetch(out, log=lambda m: _FETCH["log"].append(m))
                _FETCH["state"] = "loading"       # strict-load into CPU RAM now, so the 120 s GPU call only has to move it
                from scripts.gpu_smoke import preload_fuxi_ens

                _FETCH["log"].append(f"preloaded: {preload_fuxi_ens(out)}")
                _FETCH["state"] = "done"
            except Exception:
                _FETCH["state"] = "error"
                _FETCH["log"].append(traceback.format_exc()[-1500:])

        _FETCH["state"] = "starting"
        threading.Thread(target=work, daemon=True).start()
    try:
        files = {f: os.path.getsize(os.path.join(out, f)) for f in sorted(os.listdir(out))}
    except FileNotFoundError:
        files = {}
    import shutil

    free = round(shutil.disk_usage(out if os.path.exists(out) else "/tmp").free / 2**30, 1)
    return json.dumps({"state": _FETCH["state"], "dir": out, "files": files, "free_disk_GiB": free, "log": _FETCH["log"][-6:]}, indent=1)


_WFETCH: dict = {}


def fetch_weights(model: str) -> str:
    """Download a model's official weights on the Space CPU in a background thread (to $<MODEL>_CKPT, default /tmp/<model>).
    Call again to poll. See scripts/fetch_weights.py for the registry; weights never enter git."""
    import threading

    from scripts.fetch_weights import FETCHERS

    out = os.environ.get(f"{model.upper()}_CKPT", f"/tmp/{model}")
    st = _WFETCH.setdefault(model, {"state": "idle", "log": []})
    if st["state"] in ("idle", "error"):
        def work():
            try:
                st["state"] = "running"
                FETCHERS[model](out, log=lambda m: st["log"].append(str(m)))
                st["state"] = "done"
            except Exception:
                st["state"] = "error"
                st["log"].append(traceback.format_exc()[-1500:])

        st["state"] = "starting"
        threading.Thread(target=work, daemon=True).start()
    return json.dumps({"model": model, "state": st["state"], "dir": out, "log": st["log"][-6:]}, indent=1)


def unit_tests(model: str) -> str:
    r = subprocess.run(
        [sys.executable, "-m", "pytest", f"tests/models/{model}", "-q", "-x", "--no-header", "-p", "no:cacheprovider"],
        cwd=PKG, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": PKG},
    )
    return (r.stdout + r.stderr)[-4000:] + f"\nEXIT={r.returncode}"


with gr.Blocks(title="WeatherAI models smoke") as demo:
    gr.Markdown("# WeatherAI model smoke tests (ZeroGPU)\nRepo: https://github.com/GISWLH/WeatherAI")
    model = gr.Dropdown(MODELS, value=MODELS[0], label="model")
    pre = gr.Checkbox(False, label="load official pretrained checkpoint (where available)")
    with gr.Row():
        b_gpu = gr.Button("Smoke (ZeroGPU)", variant="primary")
        b_cpu = gr.Button("Smoke (CPU)")
        b_ut = gr.Button("Unit tests (CPU)")
    out = gr.Textbox(label="Result", lines=24)
    b_gpu.click(smoke_gpu, [model, pre], out, api_name="smoke_gpu")
    b_cpu.click(smoke_cpu, [model, pre], out, api_name="smoke_cpu")
    b_fetch = gr.Button("FuXi-ENS: download / poll weights (CPU, Zenodo)")
    b_fetch.click(fetch_fuxi_ens, [], out, api_name="fetch_fuxi_ens")
    b_wf = gr.Button("Download official weights for the selected model (CPU, background; click again to poll)")
    b_wf.click(fetch_weights, [model], out, api_name="fetch_weights")
    b_ut.click(unit_tests, [model], out, api_name="unit_tests")

if __name__ == "__main__":
    demo.launch()
