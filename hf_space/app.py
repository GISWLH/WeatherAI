"""ZeroGPU smoke runner for WeatherAI models (see README)."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
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

MODELS = ["graphcast", "aurora", "neuralgcm", "gencast", "aardvark", "weathernext_cyclones", "neuralgcm_train"]  # extended as models are added


def _smoke(model: str, pretrained: bool) -> str:
    from scripts.gpu_smoke import run

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kw = {"pretrained": pretrained} if model in ("aurora", "aardvark", "gencast", "weathernext_cyclones", "neuralgcm") else {}
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

        return json.dumps(run(model, "cpu", **({"pretrained": pretrained} if model in ("aurora", "aardvark", "gencast", "weathernext_cyclones", "neuralgcm") else {})), indent=2)
    except Exception:
        return "STATUS=FAIL\n" + traceback.format_exc()


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
    b_ut.click(unit_tests, [model], out, api_name="unit_tests")

if __name__ == "__main__":
    demo.launch()
