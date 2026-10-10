# UniCM — native PyTorch, bit-exact vs official code (random weights, no public checkpoint)

[← Model zoo](../../README.md) · [All model pages](README.md) · **Paper:** *Nature Machine Intelligence* 2026, [doi:10.1038/s42256-026-01245-5](https://doi.org/10.1038/s42256-026-01245-5)

`weatherai/models/unicm` re-implements UniCM (*Nature Machine Intelligence* 2026): two encoder–decoder transformers (spatio-temporal field branch and climate-mode series branch) on a 12×72 ocean grid with 5 channels (SST, taux, tauy, HTC300, T20D), 2×2 patches, d=256, 4 heads, 4+4 layers, 12 monthly steps in → 24 out; 12,726,037 parameters, state-dict keys identical to the official `UniCM`. `climate_modes` computes the mode series from the fields (region means; dipoles; WWV proxy), `unicm_loss` is the teacher-forced SST + mode loss (the skill term is not ported), `load_official(path)` strict-loads an official `model_best.pkl`. Official-code quirks are reproduced (spatial attention ignores its mask; decoder FFN reuses `sublayer[1]`; `time_emb` shared between branches).

* **Verified (CPU, random weights, official `models.py` unchanged; `scripts/unicm_parity.py`):** 24-step eval rollout bit-identical (max abs 0.0, field and modes; full size and lite), teacher-forced pass rel. L2 3e-7, gradients rel. L2 ≤2e-5 per parameter (`docs/results/unicm_parity*.json`); 5 unit tests with a stored lite reference.
* **Checkpoint:** none public (GitHub and Zenodo 19173780 hold code only; weights come from local training with `app_train.py` on CMIP6/ORAS5, not obtainable here).
* **HF ZeroGPU (2026-10-02):** **PASS** — 24-step rollout 0.36 s, batch-4 train step 0.21 s, peak 3.8 GB, lite vs official reference max abs 7e-7
* **NOT verified / not done:** any pretrained forecast or skill, training on real data, the lambda3 skill loss.
