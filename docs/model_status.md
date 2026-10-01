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
