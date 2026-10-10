# ArchesWeatherGen — native PyTorch 3D-attention forecaster + flow-matching generator (official weights load; parity with geoarches)

[← Model zoo](../../README.md) · [All model pages](README.md) · **Paper:** *Science Advances* 2026, [doi:10.1126/sciadv.adx2372](https://doi.org/10.1126/sciadv.adx2372)

`weatherai/models/arches` re-implements INRIA's ArchesWeather (deterministic, 24 h step, 1.5° = 121×240 grid, 6 surface + 6 levels×13 pressure-level variables) and ArchesWeatherGen
(4 deterministic members embedded + a conditional flow-matching network that samples the residual). No `geoarches` dependency at runtime. Weights: HF [`gcouairon/ArchesWeather`](https://huggingface.co/gcouairon/ArchesWeather) (BSD; not committed) and normalisation stats from the geoarches repo (BSD-3, fetched from GitHub raw, not committed).

```python
from weatherai.models.arches import ArchesWeatherGen_lite, convert

m = ArchesWeatherGen_lite()                                   # tiny trainable config, same code path
convert.download("/workspace/ckpt/arches")                    # ~2.2 GB; loading the Lightning ckpt needs `pip install omegaconf`
model, stats = convert.load_official_gen("/workspace/ckpt/arches")   # strict-load of gen net + 4 embedded det members
```
See `scripts/arches_real_data.py` (WeatherBench2 ERA5 1.5° sample -> det ensemble mean and generative members).

* **Verified (CPU fp32, oracle = `geoarches` in `/workspace/venv/ga`, `scripts/arches_parity_full.py`, `tests/parity/arches`):** official checkpoints, full size, random normalised inputs: deterministic member max|Δ| 1.6e-5 (rel 1.8e-6); 4-member average rel 8.5e-7; generative-net call rel 8.3e-7; a full 5-step seeded sample max|Δ| 1.5e-5 (rel 9.9e-7). Lite-config exact embedder/backbone parity and the flow-matching scheduler vs diffusers `FlowMatchEulerDiscreteScheduler` pass (unit tests: 7 passed, 1 full-size test gated by `ARCHES_FULL=1`).
* **Quirks reproduced on purpose:** (1) the released generative net was trained with an int32-overflow bug in its month/hour time features (they come out as month∈{1,12}, hour∈{0,23}); `legacy_overflow_time_features` reproduces it (deterministic members use true month/hour); (2) the SwiGLU block uses a GELU gate; (3) the two "skip" members add the input state.
* **Real-data check (CPU, WeatherBench2 ERA5 1.5°, init 2020-06-01 12Z, lat-weighted RMSE):** z500 +24 h: persistence 561, det ensemble mean 39.0, generative member 54.2, 3-member mean 46.2 m²/s²; t850: 2.84 / 0.589 / 0.829 / 0.672 K; +48 h generative member z500 110 vs persistence 749. This is a **single case**; no ensemble-skill/CRPS statistics, no comparison to the paper's scores.
* **HF ZeroGPU (RTX PRO 6000 Blackwell MIG 2g.48gb, torch 2.13+cu130, 2026-10-02): PASS** — strict load 4.8 s; det member forward 0.29 s; relative error vs CPU 2.5e-6 (det) / 2.0e-6 (5-step sample) with TF32 off; real case reproduces the CPU RMSEs (z500 det 39.0, gen member 54.4); 25-step generative member 4.4 s; peak GPU memory 3.0 GB.
* **NOT verified:** probabilistic skill (CRPS/spread-skill) over many cases, long rollouts, parity with random inputs only (no real-input element-wise comparison vs geoarches), the 4 separate public deterministic seeds beyond seed 0 on GPU, bf16, and training (only gradient flow of the lite configs).
