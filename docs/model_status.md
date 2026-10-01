# New-model status & verification log

What each added model **is** (re-implementation vs. wrapper) and what was **actually verified**.
"Smoke" = finite forward (and backward where stated) at the listed shapes. Nothing here is a
forecast-skill claim. GPU runs: Hugging Face Space
[`LonghaoWang/weatherai-graphcast-smoke`](https://huggingface.co/spaces/LonghaoWang/weatherai-graphcast-smoke)
(ZeroGPU; app in `hf_space/`, deployed with `scripts/deploy_hf_space.py`; same smoke code is
`scripts/gpu_smoke.py <model>`).

## Aurora — `weatherai.models.aurora` — **wrapper** (official PyTorch code)

* Source: <https://github.com/microsoft/aurora> (MIT), cloned at commit b628d7c. Aurora is already PyTorch, so **no
  re-implementation**: `Aurora_lite()` / `Aurora_small()` build the upstream `aurora.Aurora`
  (`pip install microsoft-aurora`) and `AuroraWrapper.forward` converts plain tensors to the
  upstream `Batch`.
* `Aurora_lite()`: upstream architecture, 1 block/stage, embed 64 → 2.96 M params, random init.
* `Aurora_small()`: upstream `AuroraSmallPretrained` config (112.8 M params);
  `pretrained=True` loads official `aurora-0.25-small-pretrained.ckpt` (HF `microsoft/aurora`,
  ~450 MB, public) with `strict=True`.
* Verified: wrapper output is bit-identical to upstream `model(Batch)` (unit test); lite
  fwd/bwd finite; official small checkpoint strict-loads and forward is finite (CPU + GPU).
* Not verified: numerical agreement with published Aurora forecasts / real ERA5 or HRES inputs
  (only random-normal inputs were used); only the 6 h step (no rollout); the 1.3 B
  fine-tuned model was not run.
* Results: CPU (box, 8 cores): lite+small smoke PASS 8.1 s. HF ZeroGPU (NVIDIA RTX PRO 6000
  Blackwell MIG 2g.48gb, torch 2.13+cu130): smoke with official small ckpt PASS, 5.9 s,
  peak 536 MB; Space unit tests 5 passed / 1 skipped (the skipped test needs a local ckpt path
  via `AURORA_SMALL_CKPT`).

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

## GenCast — `weatherai.models.gencast` — **PyTorch re-implementation (lite), partially checked against official JAX**

* Source: `google-deepmind/weathernext` (`weathernext1_gen`; Apache-2.0 code) at commit f2f2c51. Official
  weights are Haiku `.npz` files in `gs://dm_graphcast/gencast/params/` (230 MB each, public), but the official
  denoiser needs the full `weathernext` stack + xarray_jax (Python ≥3.12) and TPU-style splash attention.
* **Decision:** re-implement in PyTorch (a) the EDM preconditioning / loss weighting / noise schedules / **DPM-Solver++ 2S
  sampler with stochastic churn** — the diffusion logic, which is small and verifiable — and (b) a *small* denoiser network
  in the same spirit (noise-level Fourier-MLP → conditional LayerNorm; grid2mesh GNN → k-hop-masked mesh transformer →
  mesh2grid GNN, on WeatherAI's GraphCast graphs). **Not** a wrapper, and **no official weights are loaded**; the lite
  denoiser is *not* layout- or numerics-compatible with the official network.
* Verified (10 tests): noise schedule, churn schedule and the full sampler (with and without churn) match
  the official JAX `dpm_solver_plus_plus_2s.Sampler` to rtol 2e-4 (reference arrays generated by
  `scripts/gencast_sampler_reference.py` from the official code with a fixed noise tensor and a closed-form toy denoiser
  — i.e. the *sampler* is verified, not the network); EDM coefficient identities; denoiser/loss/backward finite;
  ensemble members differ; seed-reproducible; 40 Adam steps reduce the loss on a fixed batch (overfit sanity check).
* Not verified / differences: network architecture vs. official; official checkpoint conversion; spherical-harmonic white
  noise (lite uses iid Gaussian); ERA5 normalisation/forcing pipeline; any forecast skill.
* Results: CPU: 10 passed (3.8 s); HF ZeroGPU (RTX PRO 6000 Blackwell MIG 2g.48gb): smoke PASS 2.0 s, peak 74 MB, Space unit
  tests 10 passed. Lite model = 0.059 M parameters.

## Aardvark Weather — `weatherai.models.aardvark` — **partial: processor module only (PyTorch re-implementation, official checkpoint loads, numerically checked)**

* Source: <https://github.com/anna-allen/aardvark-weather-public> (CC0) at commit 8fb35a0; weights/data in the public HF *dataset*
  repo `av555/aardvark-weather` (`trained_model/`; data licence CC-BY-NC-ND-style non-commercial, no derivatives — see the repo FAQ).
  Official model = encoder (set-conv + ViT assimilation of raw satellite/in-situ obs) → processor (ViT, 24 h step on a 1.5° 240×121 grid,
  24 channels) → decoder (set-conv + MLP, station forecasts).
* **Implemented here: the processor ViT only** (`AardvarkProcessor`, parameter names identical to the official `decoder_lr.*`,
  53.9 M params), plus the official un/normalisation step (`forecast_step`). **Not implemented:** observation encoder, station
  decoder, end-to-end finetune, data loaders — they need the multi-terabyte observation pipeline (sample data in the repo is a CUDA-pickled
  dict only). So *this is not the end-to-end Aardvark system* and cannot go from observations to a forecast.
* Verified: official `processor/forecast_1/epoch_0` (648 MB) loads with `strict=True`; output matches the official
  `ConvCNPWeather(mode="forecast", decoder="vit")` + same checkpoint on seeded random input at lead_time 0 and 1
  (max |Δ| ≈ 2e-4 on outputs of mean |y| ≈ 0.17, CPU and GPU; reference arrays from `scripts/aardvark_processor_reference.py`, which needs a
  one-kwarg timm shim for the official `vit.py`). Lite model (0.23 M params, 60×31 grid) fwd/bwd finite.
* Not verified: forecast skill on real ERA5-like/Aardvark-assimilated states, multi-step use, other lead-time checkpoints (`forecast_2..10`).
* Results: CPU: 7 tests passed (4.7 s); HF ZeroGPU (RTX PRO 6000 Blackwell MIG 2g.48gb): smoke with official checkpoint PASS 6.6 s,
  peak 696 MB, Space unit tests 5 passed / 2 skipped (official-checkpoint tests need `AARDVARK_PROC_CKPT`).

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
