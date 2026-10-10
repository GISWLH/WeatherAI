# FuXi-S2S — native PyTorch from the ONNX, onnxruntime parity (weights CC-BY-NC-ND, never committed)

[← Model zoo](../../README.md) · [All model pages](README.md) · **Paper:** *Nature Communications* 2024, [doi:10.1038/s41467-024-50714-1](https://doi.org/10.1038/s41467-024-50714-1)

`weatherai/models/fuxi_s2s` re-implements FuXi-S2S (*Nature Communications* 2024), reverse-engineered from the released ONNX graph (no PyTorch source is published): input two 76-channel 121×240 frames + step index, 2×2 patch embedding (1536), step-embedding MLP, Swin-V2 U-Transformer (2 stages × 6 blocks, 10×10 windows, cosine attention, continuous relative-position bias, gated-GELU MLP), a variational latent head (`mean`, log-variance, low-rank `cov`) sampled with two noise tensors, a 2-stage 12-block decoder and a pixel-shuffle head; precipitation is log-transformed, and the output is fed back autoregressively (1.039 B parameters). Weights: Zenodo 15718402 `model-1.0.tar` (2.08 GB, **CC-BY-NC-ND-4.0**): downloaded by the user (`download()` / `scripts/fetch_weights.py fuxi_s2s`), **never committed or redistributed**; only scalar fingerprints (`tests/models/fuxi_s2s/data/cpu_summary.json`) are stored.

```python
from weatherai.models.fuxi_s2s import make_input, download_sample
from weatherai.models.fuxi_s2s.convert import download, load_official
m = load_official(download("/workspace/ckpt/fuxi_s2s"))                  # download() returns the .onnx path; strict load, fp32
download_sample("/workspace/ckpt/fuxi_s2s/data")                         # raw ERA5 sample inputs from the authors' GitHub
x, dates = make_input("/workspace/ckpt/fuxi_s2s/data")                   # (2, 76, 121, 240)
y = m(torch.from_numpy(x)[None], torch.zeros(1))                         # step 0: (1, 2, 76, 121, 240) = [last input, forecast]
```

* **Verified (CPU, fp32 native vs onnxruntime fp16-graph, official sample 2020-06-01/02, identical injected noise; `scripts/fuxi_s2s_compare.py`, `docs/results/fuxi_s2s_parity.json`):** strict load of all 576 initialisers; intermediate taps rel. L2 2e-4 to 1e-3 (embedding 4.8e-4, encoder 9.9e-4, latent `z` 3.3e-5, decoder 6.2e-4); step-0 output rel. L2 6.3e-6 (worst channel 8.2e-4), NaN mask identical; 3-step autoregressive chain rel. L2 6.3e-6 / 1.1e-5 / 1.4e-5 (worst channel ≤1.2e-3). The residual is fp16-vs-fp32 rounding, not an architecture difference; 5 unit tests.
* **Findings:** the released `dist_p.cov` weight is exactly zero, so ensemble spread comes only from the diagonal noise (`eps2`); the shifted-window mask covers latitude only (longitude periodic), which differs from a standard Swin mask in 6 windows; the data.zip `input.nc` shipped with the repo is already normalised and unusable with this ONNX (it normalises internally), so inputs are rebuilt from the raw `data/sample` ERA5 files.
* **HF ZeroGPU (2026-10-02):** **PASS** — strict load 1.2 s, one fp32 step 0.67 s, peak 5.4 GB, probe statistics vs CPU within 5.6e-6 (of std)
* **NOT verified / not done:** forecast skill or S2S metrics, the 42-step / 11-member production runs, training (lite config is trainable, gradients tested), bit-level emulation of the ORT fp16 kernels.
