# ACE2-ERA5 — native PyTorch spherical Fourier neural operator (official weights load; matches `fme` rollouts)

[← Model zoo](../../README.md) · [All model pages](README.md) · **Paper:** *npj Climate and Atmospheric Science* 2025, [doi:10.1038/s41612-025-01090-0](https://doi.org/10.1038/s41612-025-01090-0)

`weatherai/models/ace2` re-implements Ai2's ACE2-ERA5 (180×360 Gauss–Legendre grid, 44 inputs → 50 outputs, 6 h step) without depending on `fme` or `torch-harmonics`:
`sht.py` (real SHT/iSHT on Gauss–Legendre and equiangular grids), `sfno.py` (SFNO: 8 blocks, embed 384, instance norm, per-degree complex channel mixing = Driscoll–Healy convolution, inner linear skip + outer identity skip, big skip), `model.py` (the stepper around the network) and `convert.py`.
Weights: HF [`allenai/ACE2-ERA5`](https://huggingface.co/allenai/ACE2-ERA5) (Apache-2.0, not committed).

```python
from weatherai.models.ace2 import download, load_official
from weatherai.models.ace2.data import load_case

download("/workspace/ckpt/ace2")                       # 1.8 GB checkpoint + 2020 initial conditions + forcing (0.55 GB)
m = load_official("/workspace/ckpt/ace2")              # strict-load; 455,831,040 network parameters
ic, forcing, times = load_case("/workspace/ckpt/ace2", ic_index=5, n_steps=32)   # 2020-06-01T00Z, 8 days
outs = m.rollout(ic, forcing, 32)                      # list of dicts name -> [1, 180, 360], physical units (all 50 outputs)
```

* **Stepper (taken from `fme`'s `step_with_adjustments` + the checkpoint's config, not guessed):** normalise → SFNO → denormalise → clamp the 16 `force_positive` fields → pin the global-mean dry-air surface pressure to its initial value (float64) → close the global moisture budget by rescaling precipitation and recomputing the advective tendency → overwrite surface temperature where `round(ocean_fraction)==1` with the next-step forcing. `DSWRFtoa` is read at t+1.
* **Verified (CPU fp32):** SHT/iSHT equal `torch_harmonics` 0.8.0 (max relative difference ≤3e-8 forward, 0 inverse; Gauss–Legendre 180×360, equiangular and small grids); a random-weight lite SFNO matches `fme`'s `SphericalFourierNeuralOperatorNet` to 1e-5; with the official checkpoint a **12-step (3-day) rollout from the shipped 2020-01-01 initial condition matches the official `fme` Stepper**: max relative error over all 50 output fields 2.6e-7 after step 1, 4.6e-6 after step 2, 7.9e-5 after step 12 (float32 round-off growth; surface pressure differs by ≤0.03 Pa). Unit tests: conservation of dry air and global moisture budget by construction, SST prescription, gradients.
* **Real case (CPU, one 8-day rollout from 2020-06-01T00Z, lat-weighted RMSE vs WeatherBench2 ERA5 interpolated to 1.5°; ACE2 vs persistence):** day 1: h500 7.27 vs 56.9 m, TMP850 0.739 vs 2.86 K, TMP2m 0.687 vs 1.76 K; day 3: h500 17.8 vs 92.3 m, TMP850 1.35 vs 3.96 K, TMP2m 1.09 vs 2.35 K; day 7: h500 53.3 vs 112 m, TMP850 2.62 vs 4.77 K, TMP2m 1.98 vs 3.1 K. Global dry air drifted by 3.2e-05 Pa over 8 days; global-mean precipitation 2.90 mm/day; CPU time 236 s for 32 steps.
* **HF ZeroGPU (RTX PRO 6000 Blackwell MIG 2g.48gb, torch 2.13+cu130, 2026-10-02): PASS** — strict load 3.6 s; relative error vs the stored CPU slices 1.5e-06 (1 step) / 6.5e-06 (4 steps) with TF32 off; 32-step (8-day) rollout 3.53 s; dry-air drift over 8 days 3.1e-05 Pa; peak GPU memory 3.1 GB; GPU RMSEs reproduce the CPU report (day-1 h500 7.266 m, day-7 h500 53.278 m).
* **NOT verified:** multi-year / climate-scale stability and the paper's AMIP-style forced-response results (the repository's 100-year inference was not run), skill statistics over more than one case, comparison with the paper's scores, other ACE2 checkpoints (ACE2-SOM, ACE2-EAMv3), training, bf16/fp16. The forecast case uses the interpolation of 1° output to ERA5's 1.5° grid, which smooths slightly. The native SHT is O(L²) memory per transform table (about 24 MB per direction in fp32).
