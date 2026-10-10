# WeatherNext Cyclones Mini — native PyTorch network (official Mini weights load), JAX wrapper kept as oracle

[← Model zoo](../../README.md) · [All model pages](README.md) · **Paper:** *Nature* 2026, [doi:10.1038/s41586-026-10953-2](https://doi.org/10.1038/s41586-026-10953-2)

WN-C / WeatherNext 2 (FGN, a *Functional Generative Network*) is the same family as GenCast/GraphCast (grid → icosahedral mesh → 16-layer k-hop
transformer → grid) but with its own wiring: instead of a diffusion noise level, a 32-channel N(0,1) vector conditions every normalisation, the
encoders/decoder are one big split-input / split-output linear over all variables, and the GNN edge MLPs use the "pre-gather matmul" form.
`weatherai.models.weathernext_cyclones` now contains an **explicit native PyTorch port of that network** (`network.py`; transformer, attention
and graph code are shared with the GenCast port), with the official `WeatherNextCyclones_Mini_<2024` weights converted 1:1 and
`WeatherNextCyclonesWrapper` (official JAX, Python ≥ 3.12) retained only as a test/oracle path.

```python
import numpy as np, torch
from weatherai.models.weathernext_cyclones import (
    WeatherNextCyclonesNet_from_official, stack_grid_inputs, stack_mesh_inputs, unstack_outputs, download_checkpoint)

lat = np.arange(-90, 90.5, 1.0, dtype=np.float32); lon = np.arange(0, 360, 1.0, dtype=np.float32)   # official 1° grid, 181x360
net = WeatherNextCyclonesNet_from_official(download_checkpoint(), lat, lon, attention_type="triblockdiag")   # 56.7 M params, strict load
# inputs: *normalised* fields in the official layout (see stack_grid_inputs), noise ~ N(0,1) of shape (batch, 32)
gx = stack_grid_inputs(inputs, forcings, (181, 360)); mx = stack_mesh_inputs(inputs, forcings)
y = net(gx, mx, torch.randn(1, 32))                  # (B, H*W, 101) one 6 h step, all 29 target variables
fields = unstack_outputs(y, net.cfg, (181, 360))     # {name: (B,1,[13,]181,360)}
```

Trainable, config-driven variant (same modules, ~0.2 M params, fair-CRPS ensemble loss as in the official `fgn.Predictor`):
`WeatherNextCyclonesNative_lite()` → `m.loss(grid_x, mesh_x, target, num_samples=2)`; example in `tests/models/weathernext_cyclones/test_wnc_native.py`.

**Status — verified:**
* All 376 official parameter arrays (56 684 133 weights) are consumed exactly once by `convert_official_params` and strict-load.
* Network vs the **official JAX `ForwardPass`** (official Mini weights, random inputs + noise, plain `mha`; JAX/Haiku run on CPU):
  * float64 reference, mesh_splits=2, k_hop=3, 13×24 grid, batch 2 (the *reference* is run with x64 and its bf16-oriented fp32 up-casts disabled, so rounding
    cannot hide a structural error): native vs official max |Δ| **2.2e-12** (dense attention) / **2.1e-12** (banded attention) — test tolerance 1e-9.
  * float32 official run, same small config: max |Δ| 1.4e-3 (outputs up to ~26; official-JAX f32 vs official-JAX f64 differ by 9.3e-4) — test tolerance 5e-3.
  * **trained size** (mesh_splits=5 → 10 242 mesh nodes, k_hop 16, 181×360 grid, banded attention, float32): max |Δ| 2.8e-3 on outputs up to ~25
    (per variable ≤ 2.9e-3; probabilities 3.5e-6) — test tolerance 5e-3 (the 65 MB reference is not committed; regenerate with `scripts/wnc_forward_reference.py full`).
  * graph edge sets (ball query grid→mesh 101 892 edges; closest triangle mesh→grid 195 480 edges at full size, 576/936 at the small size) equal the official arrays.
* HF ZeroGPU (RTX PRO 6000 Blackwell MIG, torch 2.13+cu130) `--pretrained` smoke: PASS — strict load, float64 max |Δ| 1.9e-12 / 1.6e-12 (dense / banded), float32 max |Δ|
  7.6e-4 / 1.2e-3 (TF32 off) and 8.3e-4 / 1.3e-3 (TF32 default), peak 533 MB; Space unit tests: 5 passed, 10 skipped (the skips need local official ckpt or the JAX package; parity is covered by the smoke).

**Not verified / not ported:** the `fgn.Predictor` wrapper stack (input/residual normalisation constants, NaN cleaning of SST, autoregressive rollout, the
2-sample training ensemble), data pipeline/ERA5-HRES loading, the **cyclone tracker** (`cyclones/direct_tracker.py`) and IBTrACS pipeline; multi-step forecasts of the
native model; the 0.25° / WN2 / `<2023` checkpoints; skill. The comparison is network-level on random inputs, not forecast-level on real data. The official TPU
`splash_mha` kernel is not used (neither on our side nor in the reference). Stochastic sampling: noise is drawn iid N(0,1) as in the official `gaussian_noise_generator`.
