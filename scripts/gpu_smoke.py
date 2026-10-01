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
    checks["lite_forward_finite"] = bool(torch.isfinite(s).all() and torch.isfinite(a).all())
    checks["lite_params_M"] = round(sum(p.numel() for p in m.parameters()) / 1e6, 2)
    m.train()
    s, a = m(*inputs(16, 32))
    (s.pow(2).mean() + a.pow(2).mean()).backward()
    checks["lite_backward_finite"] = all(
        torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None
    )
    if pretrained:
        m = Aurora_small(pretrained=True).to(device).eval()  # strict official ckpt load
        with torch.no_grad():
            s, a = m(*inputs(32, 64))
        checks["small_official_ckpt_strict_load"] = True
        checks["small_official_forward_finite"] = bool(torch.isfinite(s).all() and torch.isfinite(a).all())
        checks["small_params_M"] = round(sum(p.numel() for p in m.parameters()) / 1e6, 1)
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


SMOKES = {"aurora": smoke_aurora, "graphcast": smoke_graphcast}


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
    kw = {"pretrained": a.pretrained} if a.model == "aurora" else {}
    print(json.dumps(run(a.model, a.device, **kw)))
