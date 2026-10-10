# NeuralGCM — native learned components + torch spherical harmonics (JAX wrapper kept as oracle)

[← Model zoo](../../README.md) · [All model pages](README.md) · **Paper:** *Nature* 2024, [doi:10.1038/s41586-024-07744-y](https://doi.org/10.1038/s41586-024-07744-y)

NeuralGCM (*Nature* 2024) = a differentiable spectral dynamical core (`dinosaur`: moist primitive equations on
sigma levels, IMEX-RK stepping) + learned Haiku components. **This is only partly ported:**

| ported natively (`weatherai/models/neuralgcm/network.py`, `spectral.py`) | not ported (exact stuck points: docs/model_status.md) |
|---|---|
| `EpdTower` / `ColumnMLP` (gelu-MLP encode → 5 residual process blocks → decode; the decoder, 2 encoders and the physics tendency tower) | the dinosaur dycore: sigma-coordinate primitive equations, `imex_rk_sil3` stepper, filters, `DycoreWithPhysicsCorrector` |
| `SurfaceEmbedding` (land/sea/sea-ice towers + mask/ice blend), `VerticalConvTower` (5 × Conv1D over levels) | tower *input feature builders* (dimensional velocity/prognostic/pressure/radiation/latitude/memory features, normalisation, input clipping/filters) and output transforms |
| `LearnedPositionalFeatures`, `LearnedOrography` | the base orography dataset (shipped in the checkpoint's aux data), forcing interpolation, `encode/advance/decode/unroll` plumbing |
| `SphericalHarmonicsGrid`: real SH basis, `to_modal/to_nodal`, ∇², ∂λ, cosφ∂φ, div/curl/grad, u,v ↔ vorticity/divergence | stochastic checkpoints (noise/`stochastic.py`), other resolutions |

```python
import pickle, torch
from weatherai.models.neuralgcm import NeuralGCMLearnedComponents, convert_official_params, SphericalHarmonicsGrid, SpectralGridConfig

params = pickle.load(open("deterministic_2_8_deg.pkl", "rb"))["params"]   # official checkpoint (trusted file; needs jax to unpickle)
net = convert_official_params(params)                 # strict: raises if any of the 121 Haiku modules / 14,518,180 weights is unused
y = net.physics(torch.randn(2365, 128, 64))           # (192, 128, 64) physics-tendency tower on packed nodal features (C, lon, lat)
g = SphericalHarmonicsGrid(SpectralGridConfig.TL63()) # 2.8° Gaussian grid, modal shape (127, 65)
vor, div = g.uv_nodal_to_vor_div_modal(u, v)          # (..., lon, lat) -> (..., m, l)
```

Everything is differentiable and trainable (`tests/.../test_neuralgcm_native.py::TestTrainableNative`: tower training step; gradients
through the SH layer). The official JAX wrapper (`NeuralGCM_lite()`, `NeuralGCMWrapper.forecast`) is unchanged and is the only
path that produces forecasts.

**Status — verified (numerically vs the official `neuralgcm` 1.2.3 / `dinosaur` 1.5.0, float32, official 2.8° deterministic checkpoint):**
all 121 Haiku modules / 14,518,180 weights strict-load; real packed tower inputs from one official encode→advance→decode step of the
demo ERA5 snapshot (24 random columns), max |native − official|: decoder 1.5e-4 (|y|≤206), encoder 1.3e-3 (|y|≤699), encoder_1 1.1e-4,
physics 1.3e-5 (|y|≤20), surface towers ≤5.8e-6, surface blend ≤1e-5, vertical CNN 2.2e-7 (|y|≤0.11), learned orography 0.0;
SH layer on random fields: ≤3.1e-5 for transforms, vorticity/divergence and the u,v round trip (|values| up to ~100), exact (0.0) for ∇²,
∂λ; float64 basis orthonormal to 1e-10. **Not verified / not claimed:** any forecast produced by native code (there is none),
whole-model parity, the dycore, feature builders, other checkpoints (1.4°/0.7°/stochastic). (JAX-on-GPU and the JAX training workflow: see the next subsection.) Details and results:
[docs/model_status.md](../model_status.md).

## NeuralGCM — training / modification workflow (JAX, official `neuralgcm` + `dinosaur` + Haiku; `weatherai/models/neuralgcm/train.py`)

JAX is allowed for NeuralGCM, so the model that can actually be **trained and modified** is the *official* one: the dycore is
official code, the learned parts are re-configured through the official gin config and trained with `optax`.
(The torch towers / SH layer above are kept for inspection and porting; they cannot forecast.)

**What you can modify** (all verified to change the Haiku parameter tree and train; `CONFIG_KEYS` lists them):

| knob | where | effect |
|---|---|---|
| `LATENT_SIZE`, `LAYER_SIZE` | `build_model(ck, {...})` | width of the decoder / 2 encoder / physics EPD towers (default 384 / 384) |
| `NUM_BLOCKS`, `process/MlpUniform.num_hidden_layers` | same | depth: residual process blocks (5) and layers per block (3) |
| `N_CNN_FEATURES`, `POSITIONAL_LATENT_SIZE` | same | vertical 1-D CNN width (32), learned positional-feature channels (8) |
| `SURFACE_MODEL_LATENT_SIZE/LAYER_SIZE/OUTPUT_SIZE` | same | land / sea / sea-ice surface towers (8) |
| `N_INNER_DYCORE_STEPS` | same | dycore substeps per physics call (5) |
| any other gin binding | `extra_bindings=["name = value"]` | e.g. other tower options in the shipped config |
| which parameters are trained | `fit(..., freeze=[regex,…])` | e.g. freeze everything except `physics` / the decoder |
| initialisation | `params="official" \| "init" \| "transfer"` | keep the checkpoint; random Haiku init (from scratch); random init + copy every official array whose path *and* shape still match |
| loss | `rollout_loss(..., steps, variables, lat_weights, level_w)` | `steps=0` encode→decode reconstruction, `steps≥1` differentiable `unroll` through the dycore |

**Not modified / not claimed:** the dynamical core equations, the feature builders and the data pipeline are the official code (not
re-implemented). No paper-scale training (multi-year ERA5, 1–4 day rollouts, spectral losses, multi-stage schedule, TPUs) was run, and
nothing here claims forecast skill from the demos below.

```python
import pickle, numpy as np
from weatherai.models.neuralgcm import train as T
from weatherai.models.neuralgcm.neuralgcm import download_checkpoint
ck = pickle.load(open(download_checkpoint("deterministic_2_8_deg"), "rb"))        # trusted official file

# (a) from scratch, smaller towers (874,020 params instead of 14,518,180)
m, rep = T.build_model(ck, {"LATENT_SIZE": 64, "LAYER_SIZE": 64, "NUM_BLOCKS": 2}, params="init", seed=0)
x, forc = T.demo_snapshot(m)                                  # bundled ERA5 snapshot
tg = {k: v[None] for k, v in x.items() if k != "sim_time"}   # reconstruction target
m, hist = T.fit(m, x, forc, tg, steps=0, n_iters=8, lr=3e-3)  # hist["loss"], hist["grad_norm"]

# (b) fine-tune the official checkpoint on real ERA5 (public ARCO-ERA5 on GCS, regridded; needs gcsfs)
off, _ = T.build_model(ck, params="official")
ds = T.fetch_era5_window(off, "2020-01-01T00", steps=1, hours=1)              # cached .nc
x, forc = T.demo_snapshot(off)  # or build inputs from ds[0]; see tests / scripts for the exact call
off2, hist = T.fit(off, x, forc, T.stack_targets(off, ds, 1), steps=1, n_iters=6, lr=3e-5, freeze=[r"^(?!.*physics)"])
```

**Training recipe used for the demos:** `optax.chain(clip_by_global_norm(1.0), adam(lr))`; loss = mean squared error of the decoded
state vs the target, each variable normalised by one std (per-level scales blow up at the model top), cos(lat) weights, levels above
50 hPa masked (the encoder→decoder reconstruction error dominates there); `steps=0` for reconstruction, `steps=1` (1 h) through the
dycore for fine-tuning; learning rate 3e-3 from scratch, 3e-5 for fine-tuning.

**Verified** (`tests/models/neuralgcm/test_neuralgcm_train.py`, 7 tests, JAX 0.10.2 CPU on the box, 176 s):
config overrides change the architecture (874,020 vs 14,518,180 params; `process_tower_1` exists, `_2` does not); `transfer` copies all
14,518,180 official weights when only unrelated knobs change; from-scratch reconstruction training decreases the loss
(1.288 → 0.929 in 8 Adam steps); gradients through a dycore rollout reach the encoder, physics tower and decoder (finite, 129/129
parameter leaves non-zero); training with the physics tower frozen leaves it bit-identical; fine-tuning the *official* model on
real ERA5 (2020-01-01 00Z → +1 h) lowers the loss 0.002582 → 0.002217 after 6 steps (held-out 2020-07-01 12Z: 0.004205 → 0.004002; `scripts/neuralgcm_finetune_era5.py`, 331 s for 6 steps on 8 CPU
cores, peak RSS ~7 GB). **These are mechanics checks, not skill:** at +1 h the official model's error vs ERA5 is dominated by the
encoder→decoder reconstruction error near the model top and is *larger* than persistence at many levels; fine-tuning for 6 steps on one
snapshot is overfitting to that snapshot, and no improvement in forecast skill is claimed.

**HF GPU (ZeroGPU RTX PRO 6000 Blackwell MIG 2g.48gb):** `neuralgcm_train` smoke **PASS with JAX actually on the GPU**
(`jax_backend: gpu`, `CudaDevice`, params live on the device; 4 reconstruction steps 0.0030 → 0.0027, rollout gradient finite, 129/129 leaves non-zero).
What it took: the Space runs Python 3.10, so JAX is capped at 0.6.2, which has only the `jax[cuda12]` extra (not `cuda13`) — with plain `jax[cpu]`
(the earlier setting) or an unmatched extra JAX silently falls back to CPU; and on the Blackwell slice XLA's GEMM autotuner aborts the process on the
dycore's `HIGHEST`-precision matmuls (`Algorithm not supported by the ElementalIrEmitter: ALG_DOT_BF16_BF16_F32`), fixed with
`XLA_FLAGS=--xla_gpu_autotune_level=0` (set in `hf_space/app.py` before `import jax`). ZeroGPU exposes the GPU only inside `@spaces.GPU`
calls (120 s limit), so only the small config was timed there; the full-size fine-tune was run on CPU only.
