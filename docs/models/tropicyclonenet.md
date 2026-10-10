# TropiCycloneNet (TCN_M) — native PyTorch, official weights load, bit-exact vs official code

[← Model zoo](../../README.md) · [All model pages](README.md) · **Paper:** *Nature Communications* 2025, [doi:10.1038/s41467-025-61087-4](https://doi.org/10.1038/s41467-025-61087-4)

`weatherai/models/tropicyclonenet` re-implements the TCN_M generator (*Nature Communications* 2025): `Unet3D` (8 observed 64×64 500 hPa geopotential frames → 12), two `EnvNet`s (tabular environment + geopotential, 2-layer Transformer), two LSTM encoders, six LSTM decoders (generators) and a class chooser; 6-hourly steps, 8 observed → 4 forecast steps (6–24 h) of (Δlon, Δlat, Δpressure, Δwind). Checkpoint: Zenodo 15024028 `checkpoint_with_model_16000.pt` (CC-BY-4.0, 4,767,195 parameters; not committed). Data: public TCND release (Zenodo 15009527, CC-BY-4.0); `scripts/tcn_prepare_ni.py` builds the official loader layout (NI test split).

```python
from weatherai.models.tropicyclonenet import download, load_official, make_sample, sequences
m = load_official(download("/workspace/ckpt/tcn"))                      # strict load of g_state
f = f"{root}/BST_data/NI/test/NI2019BSTFANI.txt"
s = make_sample(root, f, sequences(f)[5])
inc, cls = m.sample(s["obs_traj_rel"], s["image_obs"], s["env"], num_samples=6)   # (4, 6, 1, 4) increments; works on CPU (the official code calls .cuda())
# relative_to_abs + to_physical + track_error_km give km errors: see scripts/tcn_eval_ni.py
```

* **Verified (CPU):** official-code parity on 24 real NI test windows built by the *official* dataset class: my data loader (no OpenCV; bilinear resize) matches it to 6e-7, the six generators and the chooser logits are bit-identical (max abs 0) at matched RNG, and the 6-member sampling path is identical (`docs/results/tcn_parity.json`, `scripts/tcn_parity.py`); 5 unit tests.
* **Official-code defects found (not copied):** (1) the sampling loop iterates `range(n_unique)` instead of the drawn class ids, so on a single storm 42 of 48 members stay at the `ones` initialisation (native: 0); (2) `all_g_out=True` ignores `user_noise`. The chooser is almost uniform (mean max probability 0.174 vs 1/6 = 0.167; argmax classes [101, 197, 4, 0, 5, 0] on the NI test set).
* **NI test split (20 storms 2017–2022, 307 windows, 3 sampling seeds), track error at 6/12/18/24 h in km:** best of 6 members (official metric) 30 / 53 / 79 / 113; mean member 39 / 77 / 118 / 164; constant velocity 43 / 89 / 137 / 191; persistence 84 / 166 / 246 / 326. At 24 h the pressure MAE is 9.4 hPa and the wind MAE 7.7 m/s (best-track member) (`docs/results/tcn_eval_ni.json`). The official script quotes 24 / 47 / 72 / 103 km (all six basins, checkpoint 5100); not directly comparable.
* **HF ZeroGPU (2026-10-02):** **PASS** — load 0.1 s, forward+6-member sample 1.3 s, peak 0.35 GB, max abs vs CPU 3.7e-6
* **NOT verified / not done:** the other five basins and the paper's all-basin scores, checkpoints other than `g_state` (`g_best_state`, `g_best_nl_state` load but were not evaluated), the Himawari visualisation, training (the discriminator/GAN loop is not ported), the code licence (GitHub repo has no LICENSE file).
