# CAS-Canglong — native PyTorch, bit-exact vs the official code (S2S global SST)

[← Model zoo](../../README.md) · [All model pages](README.md) · **Paper:** *Advanced Science*, "Improved Skills for Predicting Sub-Seasonal to Seasonal Global Sea Surface Temperature", [doi:10.1002/advs.78271](https://doi.org/10.1002/advs.78271) (preprint: arXiv:2409.05369)

`weatherai/models/canglong` re-implements CAS-Canglong, a 3-D Earth-specific transformer (Pangu-style, U-shaped with a skip
concatenation) that predicts global sea surface temperature 1–3 months ahead from 16 monthly frames of 8 ocean/atmosphere
variables on the 0.25° ERA5 grid. Official code: [GISWLH/CAS-Canglong](https://github.com/GISWLH/CAS-Canglong) (MIT; the model is
defined in `notebook/model.ipynb`, helpers in `utils/`). Weights: figshare [10.6084/m9.figshare.26779969](https://doi.org/10.6084/m9.figshare.26779969)
(not committed here). Parameter names equal the official module, so an official `state_dict` strict-loads.

```python
import torch
from weatherai.models.canglong import Canglong, CanglongConfig, Canglong_lite, canglong_loss, load_official

m = Canglong_lite()                                   # 0.29 M params, 8 months x 8 vars on 32x64 -> 3 months; CPU friendly
y = m(torch.randn(1, 8, 8, 32, 64))                   # (1, 1, 3, 32, 64) normalised SST
loss = canglong_loss(y[0, 0], torch.randn(3, 32, 64)); loss.backward()   # official loss: lead-weighted MSE + correlation penalty

full = Canglong(CanglongConfig.official())            # 43.9 M params; input (B, 8, 16, 721, 1440) -> (B, 1, 3, 721, 1440)
model = load_official("canglong3_0005_600ep_base.tar")   # figshare checkpoint (torch.save of the module; trusted file)
```

* **Structure:** 3-D patch embedding (2 months × 4 × 4 px) → 8 × 181 × 360 tokens, embed 72 → 2 Earth-specific blocks →
  down-sample (8 × 91 × 180, 144 ch) → 6 + 6 blocks → up-sample → 2 blocks → concat with the first-stage skip → transposed-conv
  recovery (1 × 8 × 8) → centre crop to 3 × 721 × 1440. Window (2, 6, 12), heads (6, 12, 12, 6), drop-path 0–0.2.
  Channel 0 is SST (land set to 0, then per-variable standardisation); the notebook appends net solar radiation as channel 7.
* **`load_official`** accepts the released full-module pickle (the notebook's `__main__` / `canglong.*` / `weatherlearn.*`
  classes are unpickled as empty stand-ins and only their tensors are read), plain state dicts, `nn.DataParallel` prefixes and
  the WeatherLearn naming (`proj` / `conv`) used by `notebook/Infer.ipynb`.
* **Verified (CPU, fp32, random weights; `scripts/canglong_parity.py`, `docs/results/canglong_parity*.json`):** the unchanged
  official notebook cell + `utils` is executed and its `state_dict` strict-loaded into the native model.
  Full size (1 × 8 × 16 × 721 × 1440, 43,906,258 params): **max |Δ| = 0.0**, output (1, 1, 3, 721, 1440), forward 52 s native /
  64 s official on 4 CPU cores, peak RSS 9.1 GB for both models together. Lite config: forward max |Δ| = 0.0, gradients through
  the official loss identical (max rel. error 0.0 over 115 parameter tensors). 5 unit tests (stored lite reference, parameter
  count, pickled-module loading with stubbed classes, state-dict variants, training-loss decrease).
* **Official-code behaviour reproduced on purpose:**
  1. The output patch (1, 8, 8) is twice the embedding patch (4, 4); the recovered 1448 × 2880 field is centre-cropped to
     721 × 1440, so output pixels are decoded from the central half of the token grid (token rows 45–135 of 181, columns 90–269
     of 360). The time axis keeps the last 3 of the 8 time tokens (the crop depth is Pangu's 13 levels).
  2. Every second block shifts its windows by a hard-coded (1, 3, 6) (= window // 2 for the official window).
  3. `patchrecovery2d` / `patchrecovery3d` are computed and discarded, `patchembed2d` / `patchembed3d` are never called; they
     are kept (`legacy_heads=True`) only so official checkpoints strict-load. The native forward skips them (same output).
  4. Loss: lead weights (1, 1.5, 2) multiply *inside* the square, and the correlation penalty box (named `nino` in the
     notebook, rows 288:336 × cols 480:640) covers 18°N–6°N, 120°E–160°E on the 90°N→90°S, 0→359.75°E grid.
* **Released checkpoint on released data (GPU, fp32; `scripts/canglong_official_checkpoint.py`,
  [`docs/results/canglong_official_checkpoint.json`](../results/canglong_official_checkpoint.json), 2026-10-18):** the figshare
  `canglong3_0005_600ep_base.tar`, `normalized_data.pt` and `ocean_mask.npy` (MD5s equal to the figshare record) on an
  NVIDIA RTX 4090. `load_official` and the unchanged notebook cell strict-load the same weights; on the last 16-month window
  of the released data the two outputs are **identical (max |Δ| = 0.0)**, finite, 0.8 s per full-size forward. Against the
  following 3 months of that file the predicted SST has area-weighted ocean RMSE 1.20 / 1.01 / 0.87 K at leads 1–3
  (persistence 1.19 / 2.39 / 3.59 K) — one window, a sanity check rather than a skill estimate.
* **GPU smoke:** `scripts/gpu_smoke.py canglong` (lite parity + lite training step + one full-size forward) is wired into the
  HF Space; the same smoke on CPU (2026-10-10): **PASS** — lite vs official reference max |Δ| 0.0, lite gradients finite,
  full-size forward 54 s. Log: [model_status.md](../model_status.md).
* **Live forecasts from the same model family:** [canglong.earth](https://canglong.earth) and the Hugging Face Space
  [LonghaoWang/CAS-Canglong](https://huggingface.co/spaces/LonghaoWang/CAS-Canglong) (16-month SST and ENSO outlook).
  The operational 16-month model is a later Swin-transformer variant described in the
  [CAS-Canglong repository](https://github.com/GISWLH/CAS-Canglong); it is not part of this port.
* **NOT verified / not done:** the paper's skill scores over the full hindcast period, the data pipeline (ERA5 download,
  normalisation statistics), training at full size.
