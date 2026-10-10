# NeuralGCM precipitation — official checkpoints via official code, trainable loss

[← Model zoo](../../README.md) · [All model pages](README.md) · **Paper:** *Science Advances* 2026, [doi:10.1126/sciadv.adv6891](https://doi.org/10.1126/sciadv.adv6891)

`weatherai/models/neuralgcm/precip.py` wraps the two checkpoints of the *Science Advances* 2026 paper: `stochastic_precip_2_8_deg.pkl` (**predicts** precipitation; evaporation is diagnosed from the column water budget) and `stochastic_evap_2_8_deg.pkl` (predicts evaporation; precipitation is **diagnosed** from the budget). Zenodo [17109230](https://zenodo.org/records/17109230), CC-BY-4.0, downloaded to `/workspace/ckpt/ngcm_precip` and never committed. Both are stochastic models at 2.8° (128×64, 32 levels) with 1 h steps.

```python
from weatherai.models.neuralgcm import precip as P
import neuralgcm
paths = P.download("/workspace/ckpt/ngcm_precip")          # two 45 MB pickles
m = P.build(P.load_checkpoint(paths["precip"]))            # pickle: trusted official file
ds = neuralgcm.demo.load_data(m.data_coords)
pred = P.rollout(m, ds, steps=24, hours=1, seed=0)         # unchanged official stochastic rollout
rate = P.precip_rate_mm_per_hour(pred)                     # mm/h from the cumulative diagnostic
params, hist = P.fit_precip(m, inp, forcing, target_mm_per_h, n_updates=100, steps=6, lr=3e-4)   # differentiable loss + optax
```

* **Diagnostics:** `precipitation_cumulative_mean` (m of water since encode) and `evaporation` (kg m⁻² s⁻¹, negative = evaporation); the rate helper differences the cumulative field.
* **Verified (CPU):** both pickles load and run through the unchanged official code; parameter counts 11,145,090 (precip, 293 tensors) and 11,080,210 (evap, 230 tensors); different seeds give different members; precipitation of the precip model is non-negative and monotone; the precipitation loss is differentiable through the dycore (gradient is non-zero on the parameters upstream of the precipitation diagnostic, 38 % of the leaves; the rest correctly receive zero); a short fine-tune on a constant target reduces the loss (`tests/models/neuralgcm/test_neuralgcm_precip.py`, 6 tests).
* **Real case (CPU, one case):** ERA5 from ARCO at 2.8°, init 2020-01-01T00Z, 24 hourly steps, 4 members, vs ERA5 `total_precipitation` 24 h accumulation (ERA5 global mean 2.97 mm). *precip model:* member global means 2.50–2.82 mm, member RMSE 3.82–4.04 mm, pattern correlation 0.82–0.83, ensemble-mean RMSE 3.25 mm / correlation 0.868, spread 2.15 mm, no negative rates. *evap model:* global means 2.64–2.80 mm, member RMSE 4.01–4.07 mm, ensemble-mean RMSE 3.33 mm / correlation 0.870; **its diagnosed precipitation is negative in 38 % of cell-hours (minimum −1.24 mm/h)** — the budget-diagnosed field is not constrained to be positive. A constant-global-mean baseline has RMSE 6.29 mm (`docs/results/neuralgcm_precip_eval.json`, `scripts/neuralgcm_precip_eval.py`).
* **HF ZeroGPU (JAX 0.6.2 CUDA backend, RTX PRO 6000 Blackwell MIG 2g.48gb, 2026-10-02):** `ngcm_precip` rollout **PASS** (6-step global-mean cumulative precipitation 0.5612 mm vs CPU 0.5620 mm; 58 s incl. XLA compile), `ngcm_evap` rollout **PASS** (0.7586 vs 0.7594 mm; 62 s incl. compile), `ngcm_precip_train` **PASS** (3 optax updates through the loss: 0.127 → 0.104 → 0.069, gradients finite; 164 s incl. compile — the training call is longer than the 120 s nominal ZeroGPU allotment but completed). The first, combined rollout+training smoke aborted ("GPU task aborted") and was therefore split in two. JAX peak GPU memory was not measured.
* **NOT done / not verified:** a **native-PyTorch port of these checkpoints** — the existing converter for the deterministic checkpoint does not strict-load them (no memory encoder; decoder encode tower 784→1040 inputs, physics decode 192→193 outputs, physics encode 2365→2118 inputs, extra parameters), so tower-level torch parity was not attempted; the paper's skill claims, IMERG/GPCP comparison, multi-day / multi-case statistics, extremes and long-horizon behaviour, multi-year training of the precipitation loss (only mechanics were tested), the fit loop on real multi-step targets. The sqrt-MSE precipitation loss is our choice, not the paper's training recipe.
