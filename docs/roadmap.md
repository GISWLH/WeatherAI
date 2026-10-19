# Roadmap & planned models

19 October update: [PBC calibration, two-stage GenFocal lite and gray-swan ERF](research_20261019.md)
are available as imports and in WeatherPre's research catalog. Full official
GenFocal checkpoint inference and real-data paper replication remain open.

[← Model zoo](../README.md) · Status log: [model_status.md](model_status.md) · Survey: [candidate_models_nature_science.md](candidate_models_nature_science.md)

## Roadmap

- [ ] Pretrained weight download helpers (no invented checkpoints in-repo)
- [ ] Training / finetune recipes (kept out of v0.1 for a clean starter zoo)
- [x] GraphCast small-scale JAX parity (mesh / MLP / one GraphNet layer)
- [x] GraphCast stage parity (graphs, Grid2Mesh, processor N≤16, Mesh2Grid) vs JAX + PhysicsNeMo (tiny cases; NV processor aggregation is an inherent, documented difference)
- [ ] GraphCast checkpoint loading / full-scale parity
- [ ] More models and evaluation utilities

## Planned models (batch 1 & 2)

Citations checked against Crossref; "code" links point to the authors' official releases (not vendored here). Batch 2 (rows after FuXi-DA, selected 2026-10-02) is **not blocked**: official code and/or weights are public. Plan: [docs/model_status.md](model_status.md#batch-2--nature--science-family-models-plan); survey: [docs/candidate_models_nature_science.md](candidate_models_nature_science.md).

引用已对照 Crossref 核实；“Code”为作者官方仓库（本仓库不内置）。第二批（FuXi-DA 之后）并未受阻：官方代码和/或权重公开。

| Model | Journal | Official code / weights |
|-------|---------|-------------------------|
| **NowcastNet** | *Nature* 2023, [doi:10.1038/s41586-023-06184-4](https://doi.org/10.1038/s41586-023-06184-4) | [Code Ocean capsule](https://doi.org/10.24433/CO.0832447.v1) (code + weights per paper) — not started, blocked (403 without login) |
| **FuXi-ENS** | *Science Advances* 2025 ⚠️ not Nature family, [doi:10.1126/sciadv.adu2854](https://doi.org/10.1126/sciadv.adu2854) | [tpys/FuXi-ENS](https://github.com/tpys/FuXi-ENS) (inference scripts) · [Zenodo 10.5281/zenodo.15124541](https://zenodo.org/records/15124541) (CC-BY-NC-4.0, ONNX only) — **implemented, see FuXi-ENS section** |
| **FuXi-DA** | *npj Climate and Atmospheric Science* 2025, [doi:10.1038/s41612-025-01039-3](https://doi.org/10.1038/s41612-025-01039-3) | [xuxiaoze/FuXi-DA](https://github.com/xuxiaoze/FuXi-DA) (inference driver only; `model/`, `test_data/` unpublished) — blocked |
| **StormCast** | *Science Advances* 2026, [doi:10.1126/sciadv.adv0423](https://doi.org/10.1126/sciadv.adv0423) | [NVIDIA/physicsnemo](https://github.com/NVIDIA/physicsnemo/tree/main/examples/weather/stormcast) (Apache-2.0) · HF [nvidia/stormcast-v1-era5-hrrr](https://huggingface.co/nvidia/stormcast-v1-era5-hrrr) (Apache-2.0) — **implemented (batch 2 #1)** |
| **ArchesWeatherGen** | *Science Advances* 2026, [doi:10.1126/sciadv.adx2372](https://doi.org/10.1126/sciadv.adx2372) | [INRIA/geoarches](https://github.com/INRIA/geoarches) (BSD) · HF [gcouairon/ArchesWeather](https://huggingface.co/gcouairon/ArchesWeather) (BSD) — **implemented (batch 2 #2)** |
| **ACE2** | *npj Climate and Atmospheric Science* 2025, [doi:10.1038/s41612-025-01090-0](https://doi.org/10.1038/s41612-025-01090-0) | [ai2cm/ace](https://github.com/ai2cm/ace) (Apache-2.0) · HF [allenai/ACE2-ERA5](https://huggingface.co/allenai/ACE2-ERA5) (Apache-2.0) — **implemented (batch 2 #3)** |
| **NeuralGCM precipitation** | *Science Advances* 2026, [doi:10.1126/sciadv.adv6891](https://doi.org/10.1126/sciadv.adv6891) | [neuralgcm/neuralgcm](https://github.com/neuralgcm/neuralgcm) (Apache-2.0) · [Zenodo 17109230](https://zenodo.org/records/17109230) (CC-BY-4.0) — **implemented (batch 2 #4)** |
| **ORCA-DL** | *Science Advances* 2025, [doi:10.1126/sciadv.adu2488](https://doi.org/10.1126/sciadv.adu2488) | [OpenEarthLab/ORCA-DL](https://github.com/OpenEarthLab/ORCA-DL) (no LICENSE file) · HF dataset `JayKuo/ORCA-DL-data` — **implemented (batch 2 #5), see the ORCA-DL section** |
| **TropiCycloneNet** | *Nature Communications* 2025, [doi:10.1038/s41467-025-61087-4](https://doi.org/10.1038/s41467-025-61087-4) | [xiaochengfuhuo/TropiCycloneNet](https://github.com/xiaochengfuhuo/TropiCycloneNet) (no LICENSE file) · [Zenodo 15024028](https://doi.org/10.5281/zenodo.15024028) (CC-BY-4.0) — **implemented (batch 2 #6), see the TropiCycloneNet section** |
| **FuXi-S2S** | *Nature Communications* 2024, [doi:10.1038/s41467-024-50714-1](https://doi.org/10.1038/s41467-024-50714-1) | ONNX only · [Zenodo 15718402](https://zenodo.org/records/15718402) (**CC-BY-NC-ND-4.0**, weights not redistributed) — *native port* |
| **UniCM** | *Nature Machine Intelligence* 2026, [doi:10.1038/s42256-026-01245-5](https://doi.org/10.1038/s42256-026-01245-5) | [tsinghua-fib-lab/UniCM-Global-Climate-Modes](https://github.com/tsinghua-fib-lab/UniCM-Global-Climate-Modes) (MIT) · [Zenodo 19173780](https://doi.org/10.5281/zenodo.19173780) (code only; no checkpoint) — *native port* |
| **GenFocal** | *Nature Machine Intelligence* 2026 (online 2026-09-28), [doi:10.1038/s42256-026-01308-7](https://doi.org/10.1038/s42256-026-01308-7) | [swirl-dynamics/genfocal](https://github.com/google-research/swirl-dynamics/tree/main/swirl_dynamics/projects/genfocal) (Apache-2.0, JAX) · [Zenodo 21285802](https://zenodo.org/records/21285802) (CC-BY-4.0, 8.7 + 39 GB) — *lite flow core; checkpoints not loaded* |

Notes / 备注:
- FuXi-ENS is *Science Advances* (AAAS), not a Nature-family journal; kept as a FuXi-family model. FuXi-ENS 发表于 *Science Advances*，不属于 Nature 系列。
- FuXi-DA is in *npj Climate and Atmospheric Science* (Nature family). FuXi-DA 发表于 *npj Climate and Atmospheric Science*（Nature 系列）。
- GraphCast status unchanged: only Grid2Mesh / processor / Mesh2Grid checked at mesh levels 0–1; no official checkpoint loaded, no whole-model parity claim.

Roadmap checklist:

- [ ] NowcastNet — **blocked**: only source is the Code Ocean capsule (codeocean.com/capsule/3935105), HTTP 403 without login; no official GitHub; not started
- [~] NeuralGCM — **partial**: learned components + SH layer native and checked vs official JAX; dycore/feature builders not ported (JAX wrapper is the only forecast path)
- [x] GenCast — native denoiser (official Mini weights load, network matches JAX to 5e-5) + verified sampler; full-model/SH-noise/normalisation not ported
- [x] Aurora — native PyTorch; official small checkpoint strict-loads, output matches official package
- [x] Aardvark Weather — native encoder + processor + station decoder; official checkpoints strict-load, E2E matches official code on the sample (data loaders/training scripts not ported)
- [x] WeatherNext Cyclones (WN-C) — native network; official Mini weights strict-load, match JAX (2e-12 float64 / 3e-3 float32 at trained size); tracker/rollout/normalisation not ported
- [~] FuXi-ENS — **native port from official ONNX** (weights from Zenodo, CC-BY-NC-4.0, never committed): strict load, one forward step with fixed noise matches onnxruntime (CPU and HF GPU, fp32); multi-step/15-day ensemble, skill and official-weight training **not** verified; lite config trainable
- [ ] FuXi-DA — **blocked**: `model/assimilation_v6.py`, `final_cast_10_assim_model.pth` and test data are not public (upstream issue xuxiaoze/FuXi-DA#2 unanswered); architecture cannot be reproduced
- [~] Batch 2 (one commit per model; status in docs/model_status.md): **done** — StormCast (parity + HF GPU smoke, no real-data run), ArchesWeatherGen (geoarches parity, HF GPU smoke, one real case), ACE2 (fme rollout parity, HF GPU smoke, one 8-day real case), NeuralGCM precipitation (official checkpoints run, trainable loss, one real case, HF GPU smoke; no native-torch port), ORCA-DL (official-code parity, HF GPU smoke, one 11-month real case), TropiCycloneNet (bit-exact vs official code, NI test split, HF GPU smoke), FuXi-S2S (onnxruntime parity, HF GPU smoke), UniCM (bit-exact vs official code with random weights, HF GPU smoke; no checkpoint exists); GenFocal (lite: flow loss + RK4 sampler parity vs official JAX, HF GPU smoke; official network and checkpoints not ported); **remaining** — none

- [x] CAS-Canglong (*Advanced Science*, doi:10.1002/advs.78271) — native port, bit-exact vs the official notebook code at full size (random weights); figshare checkpoint loader tested on synthetic pickles only; not yet run on GPU or real data — [page](models/canglong.md)
