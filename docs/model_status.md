# New-model status & verification log

What each added model **is** (re-implementation vs. wrapper) and what was **actually verified**.
"Smoke" = finite forward (and backward where stated) at the listed shapes. Nothing here is a
forecast-skill claim. GPU runs: Hugging Face Space
[`LonghaoWang/weatherai-graphcast-smoke`](https://huggingface.co/spaces/LonghaoWang/weatherai-graphcast-smoke)
(ZeroGPU; app in `hf_space/`, deployed with `scripts/deploy_hf_space.py`; same smoke code is
`scripts/gpu_smoke.py <model>`).

## Aurora — `weatherai.models.aurora` — **native PyTorch implementation** (official package = test oracle)

* Source: <https://github.com/microsoft/aurora> (MIT), commit b628d7c. Re-implemented explicitly in
  `weatherai/models/aurora/` (`patch_embed`, `perceiver`, `encoder`, `swin3d`, `decoder`, `aurora`); parameter names / layouts
  follow upstream so official checkpoints load with `strict=True`. The former wrapper lives on as
  `weatherai.models.aurora.official` (`AuroraWrapper`, `AuroraOfficial_lite/_small`), used **only** as a numerical reference.
* `Aurora_lite()`: 1 block/stage, embed 64 → 2.96 M params, random init. `Aurora_small()`: upstream `AuroraSmallPretrained`
  config, 112.8 M params; `pretrained=True` / `checkpoint_path=` strict-loads the official
  `aurora-0.25-small-pretrained.ckpt`. `Aurora(...)` exposes all depth/heads/window/patch/embed settings for retraining.
* Verified (CPU, fp32, tolerance `atol=1e-5, rtol=1e-4`; observed max |diff| = 0.0): native vs. official package with
  identical weights/inputs: 3 random-weight configs × grids (16×32, 17×32, 20×36, 25×44, 25×44, 32×64, incl. odd-latitude
  crop and odd patch grids), v3 lead-time embedding, window sizes (2,2,2)/(2,3,5)/(2,6,12); **official small checkpoint**
  (strict load) on 33×64 and 65×128 inputs. Training: 25 Adam steps lower the loss ≥20% (test); all parameters receive gradients.
* Not ported: LoRA, stochastic/ensemble mode, level-conditioned patch embeddings, dynamic / atmos-static variables,
  separate-perceiver and modulation heads, activation checkpointing, roll-out helpers → fine-tuned 1.3 B / HRES / air-pollution /
  wave / v1.5 checkpoints are not loadable. Not verified: skill on real data, rollouts, the 1.3 B model, fp16/bf16.
* Results: CPU (box): `tests/models/aurora` 12 passed (7 s); smoke with official small ckpt PASS 5.1 s. **HF ZeroGPU** (NVIDIA RTX PRO 6000
  Blackwell MIG 2g.48gb, torch 2.13+cu130): smoke PASS 8.4 s, peak 958 MB: native lite fwd/bwd finite; lite native-vs-official
  max |diff| 0.0 / 0.0 (allclose 1e-5/1e-4 True); official small ckpt strict-loaded into the native model, native-vs-official on
  33x64 inputs max |diff| 0.0 / 0.0 (allclose True). Space unit tests: 11 passed, 1 skipped (the skipped one needs the local
  small ckpt path; the same parity is covered by the smoke).

## NeuralGCM — `weatherai.models.neuralgcm` — **partial native port (learned components + spherical-harmonic layer, numerically checked vs official JAX); dynamical core and feature builders NOT ported; official-JAX wrapper kept as oracle**

* Source: <https://github.com/neuralgcm/neuralgcm> (Apache-2.0 code; weights CC-BY-SA-4.0) at commit eb0485b (pip `neuralgcm` 1.2.3), plus `dinosaur` 1.5.0 (spectral dycore).
  Checkpoint: official `v1/deterministic_2_8_deg.pkl` (58 MB, public `gs://neuralgcm/models/`), 128×64 grid, 37 pressure levels / 32 sigma levels, 1 h internal step.
* **Checkpoint anatomy** (from the pickle's gin config + 121 Haiku modules, 14,518,180 weights, all consumed by `convert_official_params`):
  decoder EPD tower (784→384, 5×[4-layer 384 gelu-MLP residual], 384→259); two encoder EPD towers (1052→384→193) each with learned orography (4223 modal coefficients) and
  learned positional features (8×128×64); physics `div_curl_neural_parameterization` EPD tower (2365→384→192) + its learned positional features + three small surface EPD towers
  (land/sea/sea-ice, latent 8) + a 5-layer vertical Conv1D tower (k=5, 64 ch → 32 features); learned dycore-corrector orography (4094 coefficients). Activation = tanh-approximate GELU.
* **Ported natively** (`network.py`, `spectral.py`): `ColumnMLP`/`EpdTower`, `VerticalConvTower`, `SurfaceEmbedding`, `LearnedPositionalFeatures`, `LearnedOrography`, `NeuralGCMLearnedComponents`
  (+ `convert_official_params`, strict), and `SphericalHarmonicsGrid` (real Fourier × associated-Legendre basis on a Gauss–Legendre grid; `to_modal/to_nodal`, Laplacian/inverse, ∂λ, cosφ∂φ,
  secφ∂φcos²φ, grad/div/curl with top-wavenumber clip, u,v ↔ vorticity/divergence). Everything is differentiable.
* **Numerical verification (CPU, float32; reference = official JAX run with Haiku method interception, `scripts/neuralgcm_tower_reference.py`, `scripts/neuralgcm_sht_reference.py`; data in
  `tests/models/neuralgcm/data/`)**: real packed tower inputs of one official `encode → advance → decode` step on the demo ERA5 snapshot, 24 random columns. max |native − official|:
  decoder 1.5e-4 (|y|≤206), encoder 1.3e-3 (|y|≤699, rel 1.9e-6), encoder_1 1.1e-4, physics 1.3e-5 (|y|≤20), surface towers 9.5e-7 / 7.2e-7 / 5.8e-6, surface blend ≤1e-5,
  vertical CNN 2.2e-7 (|y|≤0.11), learned orography (base + scale·correction) exactly 0.0, masks identical. SH layer on random fields (grid 128×64, modal 127×65): `to_nodal` 3.1e-5 (|y|≤77),
  `to_modal` 6e-8, ∇²/∇⁻²/∂λ exactly 0.0, cosφ∂φ 1.5e-5, grad/div/curl ≤1.5e-5, vorticity/divergence from u,v ≤3.8e-5 (|y|≤98), u,v round trip ≤3.2e-5 (|y|≤44); float64 `to_modal∘to_nodal`
  is the identity to 1e-10. Test tolerances are looser than these (tests/models/neuralgcm/test_neuralgcm_native.py, 13 tests; the 5 tower/orography tests need the checkpoint).
* **Not ported — exact stuck points**
  1. *Dynamical core (`dinosaur`, ~26k lines incl. tests; `primitive_equations.py` 3.8 k, `time_integration.py` 1.2 k, `sigma_coordinates.py`, `filtering.py`)*: moist primitive equations with cloud moisture on 32 sigma
     levels (`MoistPrimitiveEquationsWithCloudMoisture`), `imex_rk_sil3` implicit-explicit stepper with a semi-implicit linear operator solved in spectral space per total wavenumber, 5 inner dycore substeps per
     physics call, exponential/sequential filters, `DycoreWithPhysicsCorrector`. Only the horizontal SH operators of it are ported; the vertical operators, implicit solve, explicit terms and the stepper are not,
     so there is no torch time-stepper and no native forecast.
  2. *Feature builders around the towers*: the 784/1052/2365-channel tower inputs are assembled from dimensional-unit conversions, velocity/prognostic features in `(u,v)` and `(vor,div)` forms, pressure/radiation/latitude/forcing
     features, memory features, `ShiftAndNormalize` constants (stored in the checkpoint's `aux_ds_dict`), input clipping and exponential filters, and output transforms (`div_curl_tendency_outputs`, encoder combined transform). Tower
     inputs/outputs were verified in isolation on captured *real* tensors; the code that builds those tensors from model state is not ported, so tower-level equality is the strongest claim.
  3. *Orography data* (`FilteredCustomOrography`: filtered ERA5 orography from `aux_ds_dict` regridded in modal space) was used only through its captured output (`o*_base` in the reference data).
  4. Stochastic checkpoints (noise modules, `stochastic.py`), the 1.4°/0.7° checkpoints and `unroll` plumbing (forcing interpolation) — not ported/not run natively.
* Wrapper (unchanged, oracle): `NeuralGCMWrapper`/`NeuralGCM_lite()` runs the official code; 6 tests (checkpoint loads; grid/levels/vars; finite and evolving forecast from the bundled ERA5 snapshot; seed-independence;
  wrapper == direct upstream call bit-for-bit; torch conversion). Not verified: skill vs ERA5 beyond the single demo snapshot, other checkpoints. (JAX now runs on the Space GPU — see the training workflow bullet below.)
* Results: CPU (box): native tests 13 passed; smoke PASS (all checks above, plus the JAX forecast). HF Space (RTX PRO 6000 Blackwell MIG 2g.48gb, torch 2.13+cu130): smoke PASS, official checkpoint strict-loaded on the Space; native-vs-official max |Δ| on GPU: decoder 1.4e-4, encoder 1.4e-3 (rel 2.0e-6), encoder_1 1.1e-4, physics 1.2e-5, surface towers ≤5e-6, SH transforms/vorticity/round trip ≤3.2e-5 (as on CPU); vertical CNN 1.6e-4 on GPU vs 2.2e-7 on CPU (|y|≤0.11 — the GPU conv runs with TF32 enabled by default, not investigated further); peak GPU memory 125 MB. The JAX wrapper forecast in the same smoke ran on JAX's **CPU** backend (`jax[cpu]`), so that part is still not a GPU test of the official model. Space unit tests: 19 passed (13 native + 6 wrapper), 0 skipped.
* **Training / modification workflow (JAX, added 2026-10-01; `train.py`, tests `test_neuralgcm_train.py`)** — user decision: JAX is allowed, goal = be able to train and modify the model.
  Built on the *official* `neuralgcm`/`dinosaur`/Haiku code (nothing of the dycore re-implemented): `build_model(ckpt, overrides, params="official"|"init"|"transfer")` rebuilds the model from the checkpoint's
  gin config with overrides (towers' `LATENT_SIZE`/`LAYER_SIZE`, `NUM_BLOCKS`, `process/MlpUniform.num_hidden_layers`, `N_CNN_FEATURES`, `POSITIONAL_LATENT_SIZE`, `SURFACE_MODEL_*`, `N_INNER_DYCORE_STEPS`, any gin binding),
  Haiku-initialises it (`init_params`, encode→advance→decode) or transfers every matching official array; `rollout_loss` (encode → `unroll` through the dycore → decode vs targets; per-variable std
  normalisation, cos φ weights, levels above 50 hPa masked) ; `make_train_step`/`fit` (optax clip+Adam, jitted, `freeze` regex over Haiku module paths); data helpers (bundled demo snapshot; public ARCO-ERA5 → 2.8° conservative regrid).
  * Tests (7, JAX 0.10.2 CPU, box, 176 s, all pass): override changes architecture (874,020 vs 14,518,180 params); `make_config_str`; transfer keeps 14,518,180 official weights (NUM_BLOCKS 6, exact equality of an encoder weight);
    from-scratch reconstruction training: loss 1.288 → 0.929 in 8 steps, grad norms 0.35–0.9, parameters change; gradients through a 1-step dycore rollout finite and non-zero for encoder / physics / decoder
    (129/129 parameter leaves non-zero); freeze-physics training keeps the frozen modules bit-identical while others change; fine-tune of the official checkpoint (reconstruction loss on the demo snapshot) decreases.
  * Real-data fine-tune of the **official** checkpoint (`scripts/neuralgcm_finetune_era5.py`, 1 h rollout loss, ERA5 2020-01-01 00Z, Adam lr 3e-5, 6 steps, CPU 8 cores, 331 s for 6 steps): train loss 0.002582 → 0.002217 (after 6 steps; recorded in-loop values 0.002582 … 0.002259), held-out
    (2020-07-01 12Z) 0.004205 → 0.004002 (see `docs/results/neuralgcm_finetune_era5.json`). **Interpretation limits:** a mechanics check on one snapshot; at +1 h the official model's RMSE vs ERA5 is larger than persistence at many levels
    because the encode→decode reconstruction error dominates near the top (e.g. level-0 T RMSE ≈ 15 K), which is why those levels are masked in the loss; no forecast-skill improvement is claimed. Not run: multi-day rollouts, multi-year data,
    spectral/CRPS losses, multi-stage schedule, other resolutions/checkpoints, distributed training.
  * **HF GPU (ZeroGPU RTX PRO 6000 Blackwell MIG 2g.48gb)**: `neuralgcm_train` smoke PASS **with JAX on the GPU** (`jax_backend gpu`, `CudaDevice(id=0)`, parameters on the device; 4 steps 0.0030 → 0.0027; rollout-grad finite, 129/129 leaves non-zero;
    JAX 0.6.2). Exact obstacles found and solved: (1) the Space image is Python 3.10 ⇒ JAX ≤ 0.6.2 ⇒ only the `jax[cuda12]` extra exists (`jax[cuda13]`/`jax[cpu]` ⇒ silent CPU fallback, observed: first two attempts reported `jax_backend cpu`);
    (2) XLA's GEMM autotuner crashes the process on the dycore's HIGHEST-precision matmuls on the Blackwell slice (`Algorithm not supported by the ElementalIrEmitter: ALG_DOT_BF16_BF16_F32`, fatal "GPU task aborted") ⇒ `XLA_FLAGS=--xla_gpu_autotune_level=0`.
    Space unit tests (CPU, Python 3.10): **26 passed** (native 13 + wrapper 6 + train 7, 413 s). Only the small config was run on the GPU (ZeroGPU calls are limited to 120 s and include a ~25–55 s model build); the full-size fine-tune ran on CPU only. Not measured: GPU speed-up.

## FuXi-ENS — `weatherai.models.fuxi_ens` — **native PyTorch (reverse-engineered from the official ONNX); one forward step numerically checked vs onnxruntime with fixed noise; full 15-day ensemble NOT run**

* Source/licence: Zenodo record 10.5281/zenodo.15124541 (published 2025-04-02, **CC-BY-NC-4.0**, non-commercial). `FuXi-ENS-main.zip` (9.5 GB) → `model/fuxi_ens.onnx` (109 MB graph, opset 17, 26,018 nodes, 498 initialisers), `model/fuxi_ens`
  (9,945,359,232 B external fp32 weights), `data/input.nc` (648 MB: 2018-09-13 18Z + 2018-09-14 00Z, 78 channels, 0.25°; no `target.nc` in the zip). Downloaded on 2026-10-01 with the user's approval to `/workspace/ckpt/fuxi_ens/`
  (range-extracted from the zip; **never committed**; `scripts/fetch_fuxi_ens.py`). There is no official PyTorch definition and no training code — **the ONNX graph is the only oracle** and the PyTorch architecture below was inferred from it.
* **What the graph is** (read node by node; constants dumped and matched to closed forms): normalise → `dist_p` noise-distribution network → latent sample → `decoder.0` → de-normalise (details in README). Recovered specifics, each checked numerically:
  8×8 patch embed `Conv2d(156→1536)` + `Conv2d(6→1536)` on the static `const [6,721,1440]` field, summed, then LN with scale only; AdaLN blocks (`Linear(1536→9216)` on SiLU(cond) → shift/scale/gate ×2); `cond` = three sin/cos(256)→MLP embeddings of step, hour/24, doy/365;
  window attention 18×18 on a 90×180 token grid (50 windows × 324 tokens), odd blocks cyclic shift 9 with the stored additive mask (reproduced by `make_shift_mask`, equal to all 22 ONNX `attn_mask` tensors; mask = −100 between the wrapped bottom 9 token rows and the rest);
  RoPE: the `[16200,32]` cos/sin tables are the 1-D rotary table of the *window-flattened* index (`freqs = 10000^(-2i/64)`, interleaved pairs; equals the stored constants to 5.96e-8), applied to q and k; q,k scaled by 64^-1/4; gated-GELU MLP (`a*gelu(b)`, `fc1` 1536→8192, `fc2` 4096→1536, no biases); unbiased-variance LayerNorm (eps 1e-6);
  `dist_p` 2 layers × 7 blocks with dropout (p=0.2) after the patch embed and after layer 0, `mean`/`logvar` ConvTranspose heads; `decoder.0` 6 layers × 7 blocks, `pl_head` 65 ch + `sf_head` 13 ch; ConvTranspose output rows 728 → crop 721; channels 73–77 (`ssr ssrd fdir ttr tp`, accumulations) zeroed in the model input and in the decoder input;
  `tp = exp(clip(y,0,7)) − 1`. Parameters: 2,474,861,958 = exactly the ONNX weight elements (excl. masks); buffers `mean/std [78,1,1]`, `const [6,721,1440]`.
* **Random ops in the graph:** `RandomUniformLike` ×2 (dropout masks `dist_p/drop0`, `drop1`; p=0.2, kept ×1.25) and `RandomNormalLike` ×1 (latent noise) — all active in the exported graph (no eval mode). For a deterministic comparison they were replaced by graph inputs
  (`scripts/fuxi_ens_fixed_noise_onnx.py`: 3 nodes removed, nothing else changed) and fed with the same tensors as the PyTorch model (`torch.Generator` seed 1234: `rand`, `rand`, `randn`). Without explicit `FuXiENSNoise`, `forward` draws them from a generator.
* **Verification method / caveat:** onnxruntime 1.30 CPU needs > 15 GB RAM for the whole 2.5 B-parameter graph (OOM-killed on the 15 GB box: the external weights are copied to anonymous memory), so the official graph was run **in 11 consecutive sub-graphs** (`scripts/fuxi_ens_ort_segments.py`: sub-graphs cut at named tensors and chained by their saved outputs; no node changed),
  each tensor saved to disk, and the PyTorch model was compared stage-by-stage (`scripts/fuxi_ens_verify.py`, chunked float64 statistics).
* **Numerical results (CPU, fp32, official `input.nc`, step=0, hour 0, doy 257/365, fixed noise).** max |Δ|, with max |ref| in brackets:
  normalised input 0.0 (14.3); condition 3e-7 (1.2); dist_p patch-embed 1.3e-5 (26.5); +drop0 1.9e-5; dist_p layer 0 6.0e-5 (33.4); after drop1 6.9e-5; dist_p out 3.8e-5 (2.76); mean 1.9e-9 (4.0e-3); logvar 2.4e-8 (2.0e-2); sample 7.2e-7 (5.74); decoder input z 9.5e-7 (15.7);
  decoder patch embed 2.4e-6 (7.5); layers 0..3 1.1e-5 / 1.3e-5 / 2.0e-5 / 5.8e-5 (≤39); layer 4 1.4e-2 (62), layer 5 1.4e-2 (82) (rel-RMSE 1.4e-5 / 9e-6; a handful of tokens carry the largest differences); decoder out 3.8e-3 (47).
  **Final output [1,2,78,721,1440] in physical units: max |Δ| = 1.66 (|values| ≤ 2.06e5 for geopotential, 8.1e-6 relative); per channel max|Δ|/std(channel) ≤ 3.3e-3 (`tp`) and ≤ 1.8e-3 for all others; relative RMSE ≤ 1.1e-5 for every channel; NaN (SST land) positions identical (351,876 NaNs); the passed-through step is bit-identical.**
  This is fp32-accumulation-order level for a 56-block, 2.5 B-parameter network (not bit-exact; the threshold in the test is 1e-3 of max|ref| for the output and 1e-4 rel-RMSE for the stages). Report: `tests/models/fuxi_ens/data/cpu_vs_onnxruntime_report.json`.
* **Tests** (`tests/models/fuxi_ens/test_fuxi_ens.py`): 12 pass on the box (10 pass on the Space) (10 synthetic: shapes/pass-through, noise determinism, generator reproducibility, gradient flow + loss decrease of the lite model (all parameters receive finite gradients), unbiased LN, sin/RoPE tables, shift-mask structure,
  window attention vs an explicit per-window reference, state-dict naming, official config parameter count on the meta device; plus 2 that need the local Zenodo files: strict load + mask equality, and the stored ORT-vs-torch report thresholds).
* **HF GPU (2026-10-01, ZeroGPU Space `LonghaoWang/weatherai-graphcast-smoke`, RTX PRO 6000 Blackwell MIG 2g.48gb, torch 2.13+cu130): PASS** (`scripts/gpu_smoke.py fuxi_ens`). Weights downloaded on the Space from Zenodo in 431–483 s (parallel range + inflate; free disk ≈ 3 TB, not a blocker), strict-loaded on CPU in 0.8 s (memmap),
  moved to the GPU in fp32 in 3 s inside the 120 s call (11.5 GB resident, 16.0 GB peak — fits the 48 GB slice, so no bf16/fp16/chunking needed); one forward 2.5 s. Versus the committed stride-12 slice of the onnxruntime output (same `input.nc`, same fixed noise, TF32 off): max |Δ| = 0.078, max |Δ|/channel-std = 5.9e-4 (< 1e-2·std tolerance); pass-through step equal; a second noise draw differs.
  Space unit tests: 10 passed, 2 skipped (those need the local Zenodo files). A first GPU attempt failed only because of a bug in `FuXiENSNoise.sample` (CPU generator with a CUDA device) and a smoke check that compared NaN-containing tensors with `torch.equal`; both fixed. Not run on GPU: any multi-step rollout, bf16/fp16, training.
* **Lite / trainable:** `FuXiENS_lite()` (2.1 M parameters, 33×64 grid, dim 96, 3 levels) — same classes; random init; trains (test: loss ↓ >20 % in 12 Adam steps). `FuXiENSConfig` is all-dataclass, so dims/heads/window/blocks/levels can be changed.
  Official weights are **inference-only in practice**: dropout masks and latent noise are part of the model, no training recipe/loss is published (the paper's CRPS/KL training was not reproduced), so no fine-tune was attempted.
* **Not verified / not done:** a multi-step (up to 15-day, 60 × 6 h) rollout — only one forward step (step index 0) was compared; step > 0 embeddings use the same code (sin/cos of `t`) but were not compared; other `hour`/`doy` values; ensembles/statistics (spread, CRPS) vs the paper or any `target.nc`;
  the c88 hourly / interpolation variant (`--interp`, `c88_hourly.onnx` is not in the Zenodo zip); bf16/fp16 inference accuracy; the exact fp32 differences were not bit-matched (see above); wall-clock comparisons.

## GenCast — `weatherai.models.gencast` — **native PyTorch denoiser (official architecture; official Mini weights strict-load; network numerically checked vs official JAX) + verified sampler**

* Source: `google-deepmind/weathernext` (`weathernext1_gen`, `utils/legacy/deep_typed_graph_net.py`, `utils/sparse_transformer.py`, `utils/dense.py`; Apache-2.0 code) at commit f2f2c51.
  Official weights downloaded from the public bucket `gs://dm_graphcast/gencast/params/` (`GenCast 1p0deg Mini <2019.npz`, 230 MB; CC BY-NC-SA 4.0 — non-commercial). The other three
  params files (0p25deg <2019, 0p25deg Operational <2022, 1p0deg <2019) exist in the same bucket and use the same code path; they were **not** downloaded/run.
* Implemented natively (`denoiser.py`, `graphs.py`): noise-level `FourierFeaturesMLP` (log σ → cos/sin features → Linear-gelu(tanh)-Linear); `Grid2MeshGNN` (node/edge encoders for grid, mesh (zeros ‖ structural) and edges,
  one InteractionNetwork step, f32 aggregation); `MeshTransformer` (16 × [LN→norm-conditioning→k-hop MHA; LN→norm-conditioning→FFW gelu-tanh] + final conditioned LN; k-hop mask = (A+I)^k on the
  reverse-Cuthill-McKee-permuted finest mesh); `Mesh2GridGNN` (edge encoder, one step, decoder MLP). Norm conditioning = `x·(1+s)+o` with `(s,o)=Linear(noise-encoding)`; MLP layer norms have no own affine.
  Attention has two interchangeable implementations: dense masked SDPA, and the official banded tri-block-diagonal scheme (`attention_type="triblockdiag"`, memory-light); they agree to 2e-5 on random weights.
  The EDM wrapper, schedule and DPM-Solver++ 2S sampler (`gencast.py`) predate this work and are unchanged. `DenoiserNet`/`GenCast_lite` are now config-driven instances of the *same* native architecture (random init, trainable).
* Parameter mapping: official Haiku keys → torch names are bijective (`convert_official_params`, 278 official parameter tensors (the other 85 npz entries are config/description), 57,492,612 weights); linear `w` is transposed. One official quirk: the
  mesh2grid GNN contains a mesh-node MLP whose output is discarded upstream; its weights are loaded (strict) but it is not evaluated.
* **Verified numerically against the official JAX denoiser with the official Mini weights (CPU, fp32)** — reference outputs made by `scripts/gencast_denoiser_reference.py` (official `Denoiser` class, `hk.transform`, random
  inputs/noise levels, official plain-`mha` attention because TPU `splash_mha` is unavailable on CPU; same mask):
  * trained configuration mesh_size=4 (2,562 mesh nodes), `attention_k_hop=16`, 181×360 grid (65,160 grid nodes, 102,486 grid→mesh edges), batch 1, σ=0.7: **max |Δ| = 4.8e-5** over all 12 target variables (84 channels; output magnitudes up to 22.3; relative to output max 3.2e-6). Test tolerance atol 5e-4, rtol 1e-4 (runs when the 81 MB reference exists locally; not committed).
  * small configuration mesh_size=2, k_hop=3, 13×24 grid, batch 2, σ∈{0.7, 23}: max |Δ| = 7.4e-5 (dense) / 6.9e-5 (banded); committed reference (0.8 MB) test tolerance atol 2e-4, rtol 1e-4.
  * graphs: banded mesh vertices/faces and grid2mesh/mesh2grid senders/receivers identical to the official ones; edge/node features ≤ 3e-8 (mesh2grid containing-triangle search needs `trimesh`+`rtree` to reproduce official tie-breaking exactly; numpy fallback otherwise).
* **Not verified / not ported (stated precisely):**
  * Only the *denoiser network* was compared. No comparison of a full official ensemble forecast (`GenCast.__call__` + `InputsAndResiduals` normalisation + `NaNCleaner` + autoregressive rollout) with this port; the preconditioning D(x,σ)=c_skip·x+c_out·F(c_in·x,σ) is
    the official formula but was checked by identities only, and the sampler by the toy-denoiser reference.
  * Sampling noise: iid Gaussian on the grid instead of official spherical-harmonic isotropic white noise (needs `dinosaur`'s SHT) → ensemble statistics differ in spatial correlation.
  * Not ported: ERA5 data/normalisation (`InputsAndResiduals` with stats from `gs://dm_graphcast/gencast/stats/`, downloaded but unused), NaN cleaning, training pipeline (`weighted_mse_per_level`, per-variable weights), bf16/remat/sharding paths, splash attention.
  * The comparison used random inputs, not real ERA5 states → no skill or physical-plausibility claim; 0.25° / Operational checkpoints not run.
* Tests (`tests/models/gencast`, 18 on CPU): sampler/schedule parity (4), EDM identities, lite fwd/bwd/sample/seed, graph parity vs official, strict-load + parameter count, official-weight parity (small/dense, small/banded, full res), k-hop mask vs matrix power,
  banded==dense attention, 60-step training example (loss −15 %+ and gradients on every parameter except the discarded mesh-node MLP).
* Results: CPU: 18 passed (≈15 s; full-resolution test needs the reference file). HF ZeroGPU (RTX PRO 6000 Blackwell MIG 2g.48gb, torch 2.13+cu130) `--pretrained` smoke: PASS 7.2 s, peak 313 MB — official Mini weights
  download from GCS + strict-load on the Space, small-config (mesh 2, k=3) max |Δ| vs official JAX reference 6.5e-5 (dense) / 7.2e-5 (banded) (identical with TF32 on or off at this size); Space unit tests 14 passed, 4 skipped (the skips need the
  230 MB checkpoint / full-res reference locally). The full-resolution (mesh 4, 181×360) comparison was **CPU only**; not run on GPU.

## Aardvark Weather — `weatherai.models.aardvark` — **native PyTorch: observation encoder + processor + station decoder (official checkpoints load, numerically checked vs official code)**

* Source: <https://github.com/anna-allen/aardvark-weather-public> (CC0) at commit 8fb35a0; weights/data in the public HF *dataset*
  repo `av555/aardvark-weather` (`trained_model/`; data licence non-commercial/no-derivatives — see the repo FAQ).
* Implemented (explicit modules, official parameter names): `SetConv` (ConvDeepSet), `ViT` (both variants), `AardvarkProcessor`, cylindrical
  `Unet`, `AardvarkEncoder` (ConvCNPWeather assimilation: instrument set-convs + elevation/climatology/time → 277 ch → MLP → patch-3 ViT),
  `AardvarkDecoder` (ConvCNPWeatherOnToOff), `AardvarkE2E` (official `ConvCNPWeatherE2E.forward`, non-mutating).
* **What upstream does not provide / what is missing here:** the official data-loading + training pipeline (`loader*.py`, `train_*.py`,
  `trainer.py`, `finetune.py`, `e2e_train.py`) needs multi-TB local memmaps and is documented by the authors as non-runnable → not ported.
  The U-Net's FiLM and attention branches and the `film_index` path are not ported (unused by the released checkpoints; config.pkl has `film=None`).
  Official quirks reproduced on purpose: AMSU-B reuses the AMSU-A set-conv modules, only `ascat_setconvs` / `sc_out` length scales are stored, lon/lat
  transposes, cylindrical pad applied along the last tensor axis. Official in-place mutation of the input dict is not reproduced (outputs identical).
* Verified (CPU, fp32; reference arrays from the official classes + official checkpoints on the official `sample_data_final.pkl`, made by
  `scripts/aardvark_system_reference.py` with CPU shims only; tolerance `atol=1e-5, rtol=1e-4` unless stated): encoder `encoder/epoch_96`
  strict-load, initial state max |Δ| = 0.0; station decoder `decoder/tas/lt_1/epoch_18` strict-load, max |Δ| = 0.0; E2E encoder→processor
  `forecast_1`→decoder: station max |Δ| = 2.9e-6 (outputs up to 5.9), gridded forecast within atol 1e-2/rtol 1e-4 (max rel err 8e-5; values up to 1.2e5 for z);
  processor alone on random input max |Δ| ≈ 2e-4 (tolerance 1e-3 / 2e-4 abs as before). Gridded references are stride-4 sub-samples (repo size).
* Tests: `tests/models/aardvark` (set-conv vs naive loop, NaN-as-missing, cylindrical transposed-conv shapes, param names, tiny encoder fwd/bwd,
  tiny whole-system training step, and 3 official-checkpoint parity tests that skip without checkpoints).
* Not verified: forecast skill vs. observations, other lead times / processors `forecast_2..10`, other decoders (`ws`, other `lt_*`), `e2e_finetuned`,
  multi-step rollouts, any other sample than the single official one, fp16.
* HF ZeroGPU (RTX PRO 6000 Blackwell MIG, torch 2.13+cu130) `--pretrained` smoke: PASS. All three official ckpts strict-load; processor vs official max |Δ| 2.0e-4 / 1.8e-4 (lt0/lt1).
  System vs the CPU official reference: with TF32 off encoder-state 1.4e-5, station 5.7e-6, gridded forecast rel 9.4e-5 (PASS at 1e-3). With the GPU default (TF32 convs/matmul) errors are 4.7e-4 / 1.5e-3 / 2.1e-3, i.e. TF32 noise, not a structural difference.
  Space unit tests: 12 passed, 5 skipped (the skips need local official ckpts).

## WeatherNext Cyclones (WN-C) — `weatherai.models.weathernext_cyclones` — **native PyTorch network (official Mini weights load); official-JAX wrapper kept as oracle**

* Source: `google-deepmind/weathernext` (Apache-2.0 code; weights CC-BY-4.0) at commit f2f2c51 (`weathernext2/architecture.py`, `fgn.py`, `utils/{deep_gnn,points_mesh_gnn,xarray_dense,dense,sparse_transformer,mesh_transformer}.py`).
* **Same family as GenCast, not the same layout.** Reused from the GenCast port: conditioned MLP/LayerNorm blocks, the 16-layer k-hop mesh transformer, banded (RCM) mesh + dense/`triblockdiag` attention,
  graph helpers. WN-C specific (new in `network.py`/`graphs.py`/`fgn.py`): 32-channel noise → conditioning (`Linear(32,32,no bias)`), split-input-matmul grid / mesh encoders over all
  (variable, time) channels (+ sin lat, sin lon, cos lon), `pre_gather_matmul` GNN edge MLPs (`W_e e + W_s v_s[s] (+ W_r v_r[r]) + b → Linear → LN → cond`; grid→mesh has no receiver term), edge MLP with
  32-d edge encoder, receiver-sorted edges, ball-query (0.6 × longest mesh edge) grid→mesh, closest-triangle mesh→grid, decoder `Linear→swish→Linear(101)` with `sigmoid(x−2)` on
  `cyclone_exists_gaussian_unit_mode`; node positions re-derived from float32 lat/lon exactly as the official code (this matters for edge tie-breaking).
* Conversion: `convert_official_params` consumes **every** one of the 376 `params:` arrays (56 684 133 weights) and fails on leftovers; the per-(variable,time) split weights are concatenated in the
  official channel order (names sorted: `forcing_*` < `input_*` < `spatial`, then time, then level). `strict=True` load.
* Verification (references from the official `ForwardPass`, Mini weights, random inputs/noise, plain `mha`; `scripts/wnc_forward_reference.py`):
  | case | precision | native vs official max abs diff | test tolerance |
  |---|---|---|---|
  | splits 2, k_hop 3, 13×24, batch 2, dense attention | JAX x64 reference | 2.2e-12 | 1e-9 |
  | same, banded attention | JAX x64 reference | 2.1e-12 | 1e-9 |
  | same | float32 | 1.4e-3 (JAX f32 vs JAX f64: 9.3e-4; outputs ≤ ~26) | 5e-3 |
  | **trained size**: splits 5 (10 242 nodes), k_hop 16, 181×360, banded | float32 | 2.8e-3 (outputs ≤ ~25; cyclone probability 3.5e-6) | 5e-3 |
  The float64 case is the structural proof; the float32 cases are bounded by float32 rounding of the 16-layer network (official f32 vs official f64 differs by the same order). In the f64 reference
  the official bf16-motivated up-cast of the attention softmax and the `f32_aggregation` flag are switched off (reference-side only), otherwise they would add 4e-5 of artificial difference.
  Edge sets equal the official arrays (compared as sorted (receiver, sender) pairs: the official unstable `np.argsort` orders edges inside a receiver numpy-version-dependently; sums are order-free).
* Trainable: `WeatherNextCyclonesNative_lite()` (config-driven; random init) + `WNCEnsemble.loss` (fair CRPS over noise draws, per-channel weights) — test: 60 Adam steps lower the loss by >15 %, all
  parameters except the (official, unused-in-output) mesh-node MLP of the mesh→grid GNN receive gradients.
* Tests (`tests/models/weathernext_cyclones/test_wnc_native.py`, 9, plus the 6 original wrapper tests that skip without the official `weathernext` package): edge sets, strict-load/param accounting, f64 and f32
  parity, full-resolution parity (skips without the 65 MB reference), banded == dense attention with random weights, noise dependence/sigmoid range, CRPS properties, training step.
* HF ZeroGPU (RTX PRO 6000 Blackwell MIG, torch 2.13+cu130) `--pretrained` smoke: **PASS**; strict-load; f64 max |Δ| 1.9e-12 (dense) / 1.6e-12 (banded); f32 TF32-off 7.6e-4 / 1.2e-3,
  TF32-default 8.3e-4 / 1.3e-3; peak 533 MB. Space unit tests: 5 passed, 10 skipped (skips need the local official ckpt / JAX package; parity is covered by the smoke). The full-resolution comparison is CPU only (50 s on the box).
* **Not ported / not verified:** `fgn.Predictor` wrappers (InputsAndResiduals normalisation constants, NaN cleaner for SST, autoregressive rollout, ensemble `WithSampleDim`), data loading, the
  cyclone direct tracker (`cyclones/direct_tracker.py`, 1.9 k lines) and IBTrACS handling, 0.25° / WN2 / `<2023` checkpoints, forecast-level (real-data, multi-step) comparison and skill. Verification is
  of the neural network on random inputs, not of a forecast. `splash_mha` (TPU) not used on either side.
* The JAX wrapper (`WeatherNextCyclonesWrapper`, Python ≥ 3.12, official package) is unchanged and remains the oracle for forecast-level checks; its earlier results (CPU: 6 tests, 500 hPa T RMSE vs HRES 0.40/0.50 K at +6/+12 h
  on one case, not a skill score) still stand.

## Blocked (not started; nothing faked)

| Model | Blocking point (checked 2026-10-01) |
|---|---|
| **NowcastNet** | Only official source is Code Ocean capsule `10.24433/CO.0832447.v1` → `codeocean.com/capsule/3935105/tree/v1`; HTTP 403 for anonymous access (also for `/download`); needs a Code Ocean login (not available to the box). No official GitHub; community re-implementations exist but are not official and were not used. |
| **FuXi-DA** | `xuxiaoze/FuXi-DA` contains only the inference driver (`inference.py`, `read_data.py`, `result_plot.py`). It imports `assimilation_v6.AssimilationNetv6` and loads `final_cast_10_assim_model.pth` + `test_data/`, none of which are in the repo or linked (README only shows the expected directory tree; upstream issue #2 "Where can I get `model` and `test_data` dir?" is open with no reply). Only constructor hyper-parameters are visible (`bg_chans=70, embed_dim=256, obs_chans=15, obs_frames=8, depth=(1,1,1), obs_rect=(40,680,210,850)`), not the network. |


## Batch 2 — Nature / Science-family models (plan)

Selected 2026-10-02 from [docs/candidate_models_nature_science.md](candidate_models_nature_science.md) (54 Crossref-verified candidates). **Status of every model below: PLANNED — nothing implemented or verified yet**; this table is updated per model as work lands (each model = one commit; weights are never committed).
Workflow per model: official source/weights → explicit native PyTorch in `weatherai/models/<name>` (JAX only where the official stack is JAX and trainability is the goal) → lite/trainable config + tests → numerical check vs the official implementation where feasible → smoke test on the HF ZeroGPU Space (120 s per call; reuse the existing Space) → record pass/fail, memory, runtime and exact blockers here.

| # | Model | Paper (Crossref-verified) | Code | Weights / licence | Plan & expected difficulty |
|--|--|--|--|--|--|
| 1 | **StormCast** | Pathak et al., “Kilometer-scale convection-allowing model emulation using generative diffusion modeling”, *Science Advances* 12, eadv0423 (2026), doi:10.1126/sciadv.adv0423 | NVIDIA PhysicsNeMo `examples/weather/stormcast` (Apache-2.0); Zenodo 10.5281/zenodo.15588531 | HF `nvidia/stormcast-v1-era5-hrrr` (Apache-2.0, not gated): `StormCastUNet` 315 MB + `EDMPrecond` 484 MB (`.mdlus`) | Native EDM-preconditioned UNet (regression UNet + diffusion UNet), strict-load the `.mdlus` state dicts, parity vs the PhysicsNeMo modules if installable; HRRR/ERA5 inputs for the real run are large (only synthetic/sample-size check expected). **Medium** |
| 2 | **ArchesWeather / ArchesWeatherGen** | Couairon et al., “ArchesWeatherGen: Skillful and compute-efficient probabilistic weather forecasting with machine learning”, *Science Advances* 12, eadx2372 (2026), doi:10.1126/sciadv.adx2372 | GitHub INRIA/geoarches (BSD-3; Zenodo 17122601), PyTorch + Lightning | HF `gcouairon/ArchesWeather` (BSD): deterministic ckpts 340 MB, generative 1.9 GB | Native 3D-attention deterministic model + generative (residual) model, strict-load, parity vs `geoarches`. **Easy–Medium** |
| 3 | **ACE2** (+ Kent et al. seasonal use) | Watt-Meyer et al., “ACE2: accurately learning subseasonal to decadal atmospheric variability and forced responses”, *npj Clim. Atmos. Sci.* 8, 205 (2025), doi:10.1038/s41612-025-01090-0 | GitHub ai2cm/ace (Apache-2.0) | HF `allenai/ACE2-ERA5` (Apache-2.0; ckpt tar 1.8 GB; 450 M params) | Native spherical-operator climate emulator; strict-load checkpoint; one-step parity vs `fme`; reuse the verified torch SH layer where applicable. **Medium** |
| 4 | **NeuralGCM precipitation** (checkpoints) | Yuval et al., “Neural general circulation models for modeling precipitation”, *Science Advances* 12, eadv6891 (2026), doi:10.1126/sciadv.adv6891 | GitHub neuralgcm/neuralgcm 1.1.2 (Apache-2.0) | Zenodo 10.5281/zenodo.17109230: `stochastic_precip_2_8_deg.pkl`, `stochastic_evap_2_8_deg.pkl` (45 MB each, CC-BY-4.0) | Extend the existing NeuralGCM native/JAX work: load the precipitation/evaporation checkpoints, forecast + train/fine-tune on the JAX path, native-tower parity. **Easy–Medium** |
| 5 | **ORCA-DL** | Guo et al., “Data-driven global ocean modeling for seasonal to decadal prediction”, *Science Advances* 11, eadu2488 (2025), doi:10.1126/sciadv.adu2488 | GitHub OpenEarthLab/ORCA-DL (**no LICENSE file** in the repo → all rights reserved by default; Zenodo 15383228 is CC-BY-4.0) | HF dataset `JayKuo/ORCA-DL-data` (`model_weights/seed_*.bin`, `stat/`) + OneDrive; licence of the HF files not stated | Native PyTorch model + weight load, parity vs the repo; licence ambiguity to be recorded. **Easy–Medium** |
| 6 | **TropiCycloneNet (TCN_M)** | Huang et al., “Benchmark dataset and deep learning method for global tropical cyclone forecasting”, *Nature Communications* 16, 5923 (2025), doi:10.1038/s41467-025-61087-4 | GitHub xiaochengfuhuo/TropiCycloneNet (**no LICENSE file**; Zenodo 15024028 CC-BY-4.0) | Zenodo 15024028 `checkpoint_with_model_16000.pt` (56 MB, CC-BY-4.0); dataset TCN_D Zenodo 15009527 (CC-BY-4.0) | Native PyTorch model, strict-load the checkpoint, parity vs official code on sample inputs. **Easy–Medium** |
| 7 | **FuXi-S2S** | Chen et al., “A machine learning model that outperforms conventional global subseasonal forecast models”, *Nature Communications* 15, 6425 (2024), doi:10.1038/s41467-024-50714-1 | no PyTorch source; ONNX only | Zenodo 10.5281/zenodo.15718402 `model-1.0.tar` 2.08 GB, **CC-BY-NC-ND-4.0** → weights are downloaded by the user only, **never redistributed or committed**; the native port is our own code | Reverse-engineer the ONNX like FuXi-ENS (initialiser→parameter map, strict load, onnxruntime parity). **Medium** |
| 8 | **UniCM** | Yuan et al., “Learning the coupled dynamics of global climate modes”, *Nature Machine Intelligence* 8, 930–941 (2026), doi:10.1038/s42256-026-01245-5 | GitHub tsinghua-fib-lab/UniCM-Global-Climate-Modes (MIT) + Zenodo 19173780 (code, MIT) | no checkpoint file found in the Zenodo record (1 MB code zip); README has a `--pretrained_path` option only — to be checked in the repo | Native two-transformer encoder–decoder (`src/models.py`), trainable lite config; parity vs the repo model with random weights if no checkpoint exists. **Medium** |
| 9 | **GenFocal** | Wan et al., “Regional climate risk assessment from climate models using probabilistic machine learning”, *Nature Machine Intelligence* (online 2026-09-28; volume/pages not yet assigned in Crossref), doi:10.1038/s42256-026-01308-7 | GitHub google-research/swirl-dynamics `swirl_dynamics/projects/genfocal` (Apache-2.0) + Zenodo 21268077 (CC-BY-4.0); **JAX**, Python ≥ 3.12, A100/H100-class GPU recommended | Zenodo 21285802 (CC-BY-4.0): `super_resolution_checkpoints.tar.gz` 8.7 GB, `debiasing_checkpoints.tar.gz` 39 GB | Debiasing (generative flow/diffusion in a latent space) + super-resolution stages; official stack is JAX and the checkpoints are very large → expect a lite re-implementation checked on small/synthetic cases, not a full parity run. **Hard** |

Availability statements: UniCM — “source code … publicly available via GitHub … and Zenodo (10.5281/zenodo.19173780); CMIP6, ORAS5, ERA5, GODAS and SODA data are public”. GenFocal — “source code … Zenodo 10.5281/zenodo.21268077 and GitHub …/genfocal; pretrained model weights via Google Cloud and Zenodo 10.5281/zenodo.21285802; training data and downscaled forecasts via Google Cloud; WUS-D3 via AWS”.
