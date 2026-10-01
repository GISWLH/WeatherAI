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

## NeuralGCM — `weatherai.models.neuralgcm` — **thin wrapper over official JAX code** (no PyTorch port)

* Source: <https://github.com/neuralgcm/neuralgcm> (Apache-2.0 code; weights CC-BY-SA-4.0) at commit eb0485b,
  plus `dinosaur` (spectral dynamical core). Model = differentiable primitive-equation solver +
  learned Haiku/JAX components. **Decision:** a PyTorch re-implementation (spherical-harmonic
  transforms, semi-implicit stepping, trained correctors, checkpoint conversion) was judged out of scope
  and would not be verifiable against the checkpoints; so `NeuralGCMWrapper` runs the official code
  unchanged and only converts I/O (xarray ↔ torch tensors). Inference only: no torch autograd through JAX.
* `NeuralGCM_lite()` = official `v1/deterministic_2_8_deg.pkl` (58 MB, 128×64 grid, 37 levels, 1 h
  internal step), public GCS bucket `gs://neuralgcm/models/` (no credentials).
* Verified (6 tests): checkpoint loads; grid/levels/variables as expected; forecast from the
  official bundled ERA5 demo snapshot (1959-01-02 00Z) is finite for all variables and evolves
  (500 hPa T RMS change vs. t=0: 1.0 K at 6 h, 1.6 K at 12 h, 2.2 K at 18 h in a 4-step run);
  deterministic model is seed-independent; wrapper output is bit-identical to a direct upstream
  call; torch conversion works.
* Not verified: skill against ERA5 (no verification against truth beyond the single demo
  snapshot), the higher-resolution (1.4°/0.7°) or stochastic checkpoints, GPU execution of JAX.
* Results: CPU (box, 8 cores): tests 6 passed (~4-5 min, dominated by JAX compile), smoke PASS 62 s.
  HF Space (ZeroGPU allocation, NVIDIA RTX PRO 6000 Blackwell MIG 2g.48gb): smoke PASS 60 s, **but
  JAX ran on its CPU backend** (`jax[cpu]`; GPU peak memory 0 MB) — so this is *not* a GPU test of
  NeuralGCM, only a check that the official stack installs and runs on the Space.

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

## WeatherNext Cyclones (WN-C) — `weatherai.models.weathernext_cyclones` — **thin wrapper over official JAX code (no PyTorch port)**

* Source: `google-deepmind/weathernext` (Apache-2.0 code; weights CC-BY-4.0) at commit f2f2c51. FGN = GraphCast-style grid↔mesh model with a
  16-layer sparse-transformer processor and learned input noise; JAX/Haiku, TPU-oriented attention. **Decision:** no PyTorch re-implementation
  (nothing but the JAX code to validate a port against, plus Haiku `.npz` conversion); `WeatherNextCyclonesWrapper` runs the official
  predictor (config `weathernext2/configs/WeatherNextCyclones_Mini`) and returns xarray / torch tensors. Inference only; Python ≥ 3.12.
* `WeatherNextCyclones_lite()` = official **WeatherNextCyclones_Mini_<2024** (1°, 13 levels, 12 h input / 6 h steps, 227 MB, public bucket
  `gs://dm_graphcast/weathernext2/params`), run on the official 1° HRES-initialised sample (init 2024-10-07 00Z, 159 MB) with plain `mha`
  attention (the official TPU `splash_mha` is unavailable on CPU; GPU would need `triblockdiag_mha`).
* Verified (6 tests, CPU): checkpoint + config load; 2-member × 2-step ensemble forecast has the expected shapes, all fields finite; members differ
  (RMS 0.51 K at 500 hPa/+12 h); 500 hPa T RMSE vs the HRES frames in the sample file is 0.40/0.50 K at +6/+12 h (persistence at +12 h: 3.05 K) —
  a sanity check on one case, **not** a skill evaluation.
* Not verified / not wrapped: the cyclone **tracker** and IBTrACS pipeline; the larger 0.25° models (need ≥H100-class memory); WN2 (100 m wind);
  any statistics over more than one initial condition; GPU/TPU execution; numerical equality with an independent run of the official code (the
  wrapper *is* the official code).
* Results: CPU (box 8 cores, 15 GB): 6 tests passed in ~2 min (JAX compile dominated), 12 s per jitted forward after compile.
  **HF GPU Space: not run** — the shared Space is Python 3.10 and pinned to a different JAX stack, ZeroGPU (10/10 Spaces already used) would
  only provide JAX's CPU backend anyway unless jax[cuda] is set up, and the 3.12 weathernext install was not validated there.

## Blocked (not started; nothing faked)

| Model | Blocking point (checked 2026-10-01) |
|---|---|
| **NowcastNet** | Only official source is Code Ocean capsule `10.24433/CO.0832447.v1` → `codeocean.com/capsule/3935105/tree/v1`; HTTP 403 for anonymous access (also for `/download`); needs a Code Ocean login (not available to the box). No official GitHub; community re-implementations exist but are not official and were not used. |
| **FuXi-ENS** | `tpys/FuXi-ENS` contains `inference.py`/`eval.py`/`data_util.py` that call an ONNX Runtime session on `model/fuxi_ens.onnx`; the model and sample data are on a restricted Google Drive (README: contact the authors). No PyTorch architecture is published, so there is nothing official to wrap or compare a re-implementation against. |
| **FuXi-DA** | `xuxiaoze/FuXi-DA` contains only the inference driver (`inference.py`, `read_data.py`, `result_plot.py`). It imports `assimilation_v6.AssimilationNetv6` and loads `final_cast_10_assim_model.pth` + `test_data/`, none of which are in the repo or linked (README only shows the expected directory tree; upstream issue #2 "Where can I get `model` and `test_data` dir?" is open with no reply). Only constructor hyper-parameters are visible (`bg_chans=70, embed_dim=256, obs_chans=15, obs_frames=8, depth=(1,1,1), obs_rect=(40,680,210,850)`), not the network. |
