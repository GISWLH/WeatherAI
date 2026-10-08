# `weatherai.inference` — reproducible forecast runs (ORCA-DL, FuXi-S2S, ACE2, NeuralGCM; UniCM blocked)

Stable entry points used by WeatherPre (`weatherpre.adapters.weatherai_run`). Base commit of this work: `4ce651c`.

```python
from weatherai import inference as wi
wi.card("fuxi-s2s")                    # CheckpointCard: inputs, units, history, grid, levels, normalisation, outputs, step,
                                       # forcings, masks, sources, licences, technical vs validated horizon, six evidence levels
f = wi.load("orca-dl", root="/ckpt/orca_dl", seeds=(1, 2), device="cuda")              # torch-native
r = f.run(inputs, lead_months=7, out_path="orca.nc")                                     # ForecastResult + orca.nc.manifest.json
f = wi.load("fuxi-s2s", path="/ckpt/fuxi_s2s", device="cuda")                           # official ONNX, seeded noise
idx = f.run(inp, lead_days=42, members=11, seed=0, out_dir="run/")                      # one file per member, resumable
f = wi.load("fuxi-s2s", backend="torch", path="/ckpt/fuxi_s2s")                         # native port (secondary)
f = wi.load("ace2", ckpt_dir="/ckpt/ace2"); f.run(ic, forcing, n_steps, forcing_provenance="forecast", init=..., forcing_times=...)
f = wi.load("neuralgcm", name="deterministic_2_8_deg"); f.run(ds, n_steps, forcing_policy="persist_initial")   # official JAX
wi.load("unicm")                       # -> WeightsUnavailableError (no published checkpoint)
```

`python -m weatherai.inference cards [--json]` prints the cards (committed as `docs/model_cards.json`, kept in sync by a
test); `python -m weatherai.inference check-fuxi-input DIR --mask MASK` validates official-layout FuXi-S2S inputs without
weights. Examples: `examples/inference/{orca_dl_example,fuxi_s2s_example,weatherpre_handoff}.py`.

## Evidence levels (independent; none is a skill claim except the last)

| model | implemented | weights obtainable | real-data inference | official parity | multi-init / long rollout | independent hindcast | technical max | validated |
|---|---|---|---|---|---|---|---|---|
| ORCA-DL | yes | yes (HF; licence unstated → **not redistributable**) | partial (1 GODAS init) | yes (7 steps ≤7.3e-7) | no | no | none (official 24 mo) | — |
| FuXi-S2S | yes (ONNX ref + native) | yes (Zenodo; CC-BY-NC-ND → **not redistributable**) | partial (official sample) | yes (native vs ORT) | external (SST_S2S 11×42) | no | **42 d** | — |
| ACE2 | yes | yes (Apache-2.0) | partial (1 × 8 d) | yes (vs fme 12 steps) | no | no | none | — |
| NeuralGCM | yes (official JAX; torch port partial, not a forecaster) | yes (GCS) | partial (demo, precip 24 h) | yes (is the official code) | no | no | none | — |
| UniCM | structure only | **no** | no | random weights only | no | no | 24 mo (config) | — |

## Design

* **Lazy imports**: `import weatherai` / `import weatherai.inference` import no torch / onnxruntime / jax and download
  nothing; `weatherai.models` resolves its exports lazily so the JAX NeuralGCM path runs in a torch-free environment.
* **Explicit failures** (`weatherai.inference.errors`): `WeightsUnavailableError`, `BackendUnavailableError`,
  `MissingInputError`, `UnitError` (values outside the physical range of the documented unit), `MissingForcingError`,
  `HorizonError` (FuXi-S2S > 42 days).
* **Results keep model-native layouts** (ORCA-DL: member × lead[month] × depth × lat × lon on 128×360; FuXi-S2S:
  member × lead[day] × channel × lat × lon), with masks; values outside masks are NaN, never zero.
* **Run manifest** (`<file>.manifest.json`): code revision (git) + version, sha256 of weights and normalisation files
  (cached by path/size/mtime), sha256 of every input and output array, full configuration (checkpoint, seeds, members,
  device, dtype, horizon, noise mode, forcing provenance), package versions, timings, peak RSS / GPU memory; a
  configuration hash drives resumable chunked runs.

## Model notes

**ORCA-DL** (`inference/orca_dl.py`). Channel order so(16) thetao(16) tos(1) uo(16) vo(16) zos(1) + tauu, tauv; GODAS names
salt, pottmp, sst, ucur, vcur, sshg, uflx, vflx; units g/kg, °C, m/s, m, N/m² (`convert_units`: K → °C, kg/kg → g/kg,
cm → m); depths 10…1000 m by linear interpolation without extrapolation (`interp_depth`, as `cdo intlevel`); bilinear to
lat −63.5…63.5 / lon 0.5…359.5 (`to_model_grid`, as `cdo remapbil`); normalisation with the statistics of the initial
calendar month, lead L de-normalised with month (m+L) mod 12 (official demo); NaN → 0 after normalisation (official);
outputs masked with each variable's input-valid mask. `tos` with GODAS initial conditions is the 5 m potential temperature
(official example: equal to 6e-6 °C) — a near-surface **proxy**, labelled as such. The official example's wind stress is
eastward-positive (equatorial Pacific January mean −0.047 N/m²). Members = seeds 1–8, loaded one at a time; one model.

**FuXi-S2S** (`inference/fuxi_s2s.py`). Reference = official ONNX with the official session options; the two
`RandomNormalLike` nodes are replaced by graph inputs fed with `numpy.random.default_rng([seed, member, step])` draws
(the modified 1.6 MB graph is cached; the 2 GB external weights are symlinked, never copied); `noise="internal"` keeps the
unmodified graph and calls `onnxruntime.set_seed`. Input checks: shape, two consecutive days, sst land fraction (official
mask), finite elsewhere, physical ranges (geopotential not height, Pa not hPa, ttr flux not J/m², tp daily-mean rate).
**tp units, verified** (`docs/results/fuxi_s2s_input_check.json`, scripts `fuxi_s2s_input_check.py`,
`fuxi_s2s_tp_units_check.py`): the official ERA5 sample is the daily mean of *hourly* accumulations (global mean
1.152e-4 m = WB2 mean hourly accumulation 1.150e-4 m; daily total would be 2.76e-3 m), so the channel is mm/h and ×24
gives mm/day; ttr/3600 = mean top net thermal flux (−244.80 vs WB2 −244.82 W/m²); instantaneous fields are 00–23 UTC means,
accumulations 01–24 UTC. The earlier docstring calling the channel mm/day was corrected. The fed-back state is the
unmodified network output (as the official driver); only saved sst is masked over land.

**ACE2** (`inference/ace2.py`). Every required forcing (`in_names − out_names`, next-step ocean fraction and SST) must
cover all `n_steps + 1` times, optionally checked against an explicit time axis; `forcing_provenance` is mandatory and
labels the output (`observed` → `conditional_hindcast`, not a forecast; `forecast`, `scenario`, `persistence`). SST over
ocean is prescribed: never an SST forecast.

**NeuralGCM** (`inference/neuralgcm.py`). Only the complete official JAX model is offered; `backend="torch"` raises.
`forcing_policy` is explicit: `persist_initial` (SST / sea ice held, recorded) or `provided` (dataset must cover every
output time, else `MissingForcingError`). Lead 0 is the decoded initial state, not the analysis.

**UniCM** (`inference/unicm.py`). Re-checked 2026-10-08: official GitHub `tsinghua-fib-lab/UniCM-Global-Climate-Modes`
HEAD `67fe4c1` (2026-02-06) has no checkpoint and no link (README expects a locally trained `runs/SaveModel`, 20 seeds);
Zenodo / Hugging Face could not be reached from this environment (2026-10-01 check: code only). Blocked. Retraining is a
separate project (CMIP6 CESM2-FV2 + ORAS5/ERA5/GODAS/SODA, 20 seeds, explicit non-overlapping train / validation / test
years, then an independent hindcast).

## What was run for this change (2026-10-08, cloud CPU box: 4 cores, 15 GB, no GPU; Hugging Face, Zenodo, NOAA PSL blocked)

* `tests/inference` (25 tests): cards, light import, backend errors; ORCA-DL month rule across year end, masks, seeds,
  unit / depth helpers, manifest round trip, official GODAS example passes the input checks; FuXi-S2S ONNX runner on a
  stand-in graph with the official interface (seeded noise reproducibility, member / seed differences, step input, sst
  mask, time axis, resume, internal-noise mode, unit errors, 42-day limit); ACE2 forcing gaps / provenance / prescribed
  SST; UniCM refusal; NeuralGCM torch refusal.
* NeuralGCM official JAX forecaster, real demo data, CPU, separate torch-free venv: `WEATHERAI_NETWORK=1 pytest
  tests/inference/test_ace2_neuralgcm_unicm.py -k neuralgcm` (PASS, 185 s).
* Full suite: 184 passed, 217 skipped (base `4ce651c`: 160 passed, 216 skipped; no regressions).
* **Not run here** (weights unreachable): ORCA-DL and FuXi-S2S with official weights, GPU memory, 42-day × 11-member
  runs. Reproduce with `examples/inference/orca_dl_example.py` and `examples/inference/fuxi_s2s_example.py` (commands in
  their docstrings); expected ORCA-DL seed-1 RMSE vs GODAS: 0.43 °C (+1 mo) … 0.96 °C (+11 mo).
