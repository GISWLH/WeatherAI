# StormCast — native PyTorch regression U-Net + EDM diffusion U-Net (official weights load; bit-identical to PhysicsNeMo on CPU)

[← Model zoo](../../README.md) · [All model pages](README.md) · **Paper:** *Science Advances* 2026, [doi:10.1126/sciadv.adv0423](https://doi.org/10.1126/sciadv.adv0423)

`weatherai/models/stormcast` re-implements NVIDIA's StormCast v1 (ERA5 → HRRR, 3 km, 512×640 grid, 99 state + 26 conditioning + 2 invariant channels)
without any PhysicsNeMo dependency. Weights: HF [`nvidia/stormcast-v1-era5-hrrr`](https://huggingface.co/nvidia/stormcast-v1-era5-hrrr) (Apache-2.0, ungated; not committed here).

```python
from weatherai.models.stormcast import StormCast_lite, load_official
from weatherai.models.stormcast.convert import download

m = StormCast_lite()                                      # tiny trainable config (64x64, 8 state channels), same code path
y = m(x_state, x_cond)                                    # one 1-hour step: regression mean + 18/4-step Heun diffusion residual

download("/workspace/ckpt/stormcast")                     # ~800 MB from the Hub (HF_TOKEN only needed if you are rate limited)
model = load_official("/workspace/ckpt/stormcast")        # strict load of both networks + means/stds/invariants
nxt = model(x_hrrr, x_cond_on_hrrr_grid)                  # (B,99,512,640) physical units, (B,26,512,640) coarse conditioning interpolated to HRRR
mean_only = model(x_hrrr, x_cond_on_hrrr_grid, deterministic_only=True)   # regression mean without the diffusion residual
```

* **What is verified (CPU, fp32, real official checkpoint, 512×640):** the native `StormCastUNet` and `EDMPrecond` match the PhysicsNeMo 2.2.2 modules loaded from the *same* `.mdlus` state dicts with **max|Δ| = 0.0** (regression net, and the EDM denoiser at σ = 0.002 / 0.5 / 5 / 800; random-normal inputs). Strict loading works (only the two empty `device_buffer` tensors are dropped); parameter counts 78,588,387 and 121,058,275.
  Small random-weight parity (64×64) and the 6-step Heun sampler against `physicsnemo.diffusion.samplers.sample` + `EDMNoiseScheduler` also pass (`tests/parity/stormcast`).
  Quirk reproduced on purpose: PhysicsNeMo's eval-mode GroupNorm uses the **unbiased** variance.
* **HF ZeroGPU (RTX PRO 6000 Blackwell MIG 2g.48gb, torch 2.13+cu130, 2026-10-02): PASS** — official weights downloaded on the Space, strict-loaded in 2.7 s; regression 512×640 forward 0.3–0.5 s; relative error vs the stored CPU reference slices 1.7e-6 (regression) / 7.7e-6 (EDM σ = 5) with TF32 off (8e-4 / 6.5e-3 with default TF32); a full step (regression + 18 Heun steps = 35 denoiser evaluations) took 10.3 s; peak GPU memory 5.2 GB; different seeds give different samples.
* **NOT verified:** forecast skill on real data — no real HRRR/GFS/ERA5 case was run (inputs were random tensors at physical mean/std), no multi-hour rollout, no comparison with the Earth2Studio pipeline end-to-end, no reproduction of paper scores, no bf16/fp16, no training of the official-size networks (only the lite config was trained for gradient flow). The HRRR/ERA5 data pipeline (hybrid-level fields, interpolation of the global conditioning to the HRRR grid) is not included.
