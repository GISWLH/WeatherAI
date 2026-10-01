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
