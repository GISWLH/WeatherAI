---
title: WeatherAI Models Smoke (GraphCast, Aurora, ...)
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.25.2"
app_file: app.py
pinned: false
license: mit
suggested_hardware: zero-a10g
---

# WeatherAI model smoke tests (ZeroGPU)

Runs `scripts/gpu_smoke.py <model>` and the model unit tests from
[`GISWLH/WeatherAI`](https://github.com/GISWLH/WeatherAI) on a ZeroGPU (A10G/H200 slice)
Pro GPU and prints one JSON line per run: status, wall-clock seconds, GPU name, peak memory,
and exactly which checks passed. A pass = finite forward/backward pass at the stated shapes;
it is **not** a numerical-parity or forecast-skill claim.

Deployed from the repo with `scripts/deploy_hf_space.py` (token read from `HF_TOKEN`).
