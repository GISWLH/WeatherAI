# ORCA-DL — native PyTorch Swin ocean model (official weights load; matches official code)

[← Model zoo](../../README.md) · [All model pages](README.md) · **Paper:** *Science Advances* 2025, [doi:10.1126/sciadv.adu2488](https://doi.org/10.1126/sciadv.adu2488)

`weatherai/models/orca_dl` re-implements ORCA-DL (*Science Advances* 2025) without `transformers`/`timm`/`global_land_mask`; parameter names equal the official ones. Weights: HF dataset `JayKuo/ORCA-DL-data` (`seed_1..8.bin`, 2.16 GB each; licence not stated, GitHub repo has no LICENSE file; never committed). Input: GODAS-style monthly fields on 128×360 (−63.5…63.5°), 16 levels, normalised by the shipped monthly anomaly statistics.

```python
from weatherai.models.orca_dl import download, download_example, load_official, load_example
download("/workspace/ckpt/orca_dl", seeds=(1,)); download_example("/tmp/orca_example")   # example input is fetched from GitHub, not vendored
m = load_official("/workspace/ckpt/orca_dl", seed=1)           # strict load, 539,942,562 parameters
ocean, atmo = load_example("/tmp/orca_example", "/workspace/ckpt/orca_dl", month=0)    # (1,66,128,360), (1,2,128,360)
y = m(torch.from_numpy(ocean), torch.from_numpy(atmo), predict_time_steps=11)           # (1,11,66,128,360), normalised; channel 32 = tos
```

* **Structure:** six ocean encoders (so, thetao, tos, uo, vo, zos) and one wind-stress encoder (patch 2×3, three Swin stages, 2-D rotary positions in windows), a fusion block (wind-stress latent gates the ocean latent after a lead-time rotation, then two global-window Swin blocks), six decoders. Every Swin MLP except the wind-stress encoder is a mixture of six lead-month experts. Lead months 1–6 are predicted *directly* from the same input; the lead-6 output becomes the next input; wind stress stays at its initial value.
* **Finding:** the official land-mask attention mask is a no-op (its mask is built with an all-zero convolution kernel, so it is uniform), so the native port uses standard Swin masks; numerics agree. The land mask is used only for the loss.
* **Verified (CPU):** strict load; 7-month rollout from the official demo input vs the official modules (seed_1): rel. L2 4.4e-07–7.3e-07, max abs 1.1e-04 (normalised units; `docs/results/orca_dl_parity.json`); random-weight small config vs official: rel. L2 7.3e-07 (`orca_dl_parity_lite.json`); 9 unit tests.
* **Real case (one initial month):** GODAS Jan 1980 (the official demo input), seed_1, tos vs GODAS 5 m temperature, lat-weighted RMSE (°C), ORCA-DL vs persistence vs shipped climatology: +1 mo: 0.43 vs 0.70 vs 0.99; +3 mo: 0.64 vs 1.04 vs 0.77; +6 mo: 0.72 vs 3.55 vs 0.68; +9 mo: 0.84 vs 3.38 vs 0.68; +11 mo: 0.96 vs 1.14 vs 0.90. It beats persistence at every lead and the climatology only at leads 1–2 (the climatology is the shipped training-period mean, not GODAS's own); Niño3.4 anomaly at +1 mo 0.49 vs truth 0.42 °C (`docs/results/orca_dl_real_case.json`).
* **HF ZeroGPU (RTX PRO 6000 Blackwell MIG, torch 2.13+cu130, 2026-10-02): PASS** — strict load 5.2 s, 7-month rollout 0.82 s, 2.4 GB peak; GPU rel. error vs CPU ≤8e-7 (tos +1/+7, zos +1).
* **NOT verified / not done:** multi-seed ensemble (only seed_1 downloaded), more than one initial month, decadal rollouts, the paper's Niño/IOD/AMV skill, other reanalysis inputs, training, the licence of the weights.
