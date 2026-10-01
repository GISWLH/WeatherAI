# WeatherAI

**Personal weather AI model zoo (PyTorch)** — Pangu, FuXi, FengWu, GraphCast, and more.

个人天气 AI 模型库（PyTorch）：Pangu、FuXi、FengWu、GraphCast 等，将逐步改进与扩展。

> **Independent project / 独立项目.** This is **not** a fork of
> [lizhuoq/WeatherLearn](https://github.com/lizhuoq/WeatherLearn). It starts from
> WeatherLearn-style MIT implementations of Pangu / FuXi and paper-inspired
> skeletons (see [NOTICE](NOTICE)), then evolves as Longhao Wang (GISWLH)’s own zoo.
>
> 本仓库是 Longhao Wang（GISWLH）的**个人**模型库，**不是** WeatherLearn 的
> GitHub fork。代码在 MIT 许可下改编自 WeatherLearn 风格实现，并将持续改进。

## Models / 模型

| Model | Import | Notes |
|-------|--------|-------|
| **Pangu** / `Pangu_lite` | `from weatherai.models import Pangu, Pangu_lite` | 3D Earth-attention style; lite for smaller grids |
| **FuXi** (`Fuxi`) | `from weatherai.models import FuXi` | Cube embedding + U-Transformer (Swin V2) |
| **FengWu** / `FengWu_lite` | `from weatherai.models import FengWu, FengWu_lite` | Multi-modal encode–fuse–decode; optional uncertainty |
| **GraphCast** / `GraphCast_lite` | `from weatherai.models import GraphCast, GraphCast_lite` | Grid ↔ icosahedral mesh encode–process–decode |

Architecture notes:
- FengWu: [docs/fengwu_specs.md](docs/fengwu_specs.md)
- GraphCast: [docs/graphcast_specs.md](docs/graphcast_specs.md)
- GraphCast parity vs DeepMind JAX + NVIDIA PhysicsNeMo (stage-wise): [docs/graphcast_parity.md](docs/graphcast_parity.md)

## Install / 安装

```bash
git clone https://github.com/GISWLH/WeatherAI.git
cd WeatherAI
pip install -e .
# optional: pip install -e ".[dev]"
```

Requires Python ≥ 3.9, `torch`, `timm`, `numpy`.

## Quick start / 快速开始

```python
from weatherai.models import Pangu, FuXi, FengWu, FengWu_lite, GraphCast, GraphCast_lite

# FengWu lite — CPU-friendly defaults (64×128, 13 levels → 69 channels)
fw = FengWu_lite()
import torch
x = torch.randn(1, 69, 64, 128)
y = fw(x)  # (1, 69, 64, 128)

# GraphCast lite — tiny mesh (level 1), residual next-step
gc = GraphCast_lite()  # in_channels=4, img_size=(32, 64)
y = gc(torch.randn(1, 4, 32, 64))

# GraphCast with FengWu-width channels on a small grid
gc69 = GraphCast_lite(in_channels=69)
y69 = gc69(torch.randn(1, 69, 32, 64))

# Pangu lite
from weatherai.models import Pangu_lite
pangu = Pangu_lite()
```

```python
# Package-level re-exports
from weatherai import Pangu, FuXi, FengWu, FengWu_lite, GraphCast, GraphCast_lite
```

## Tests / 测试

```bash
pip install -e ".[dev]"
pytest tests/models/fengwu tests/models/graphcast -q
# optional GraphCast parity (JAX needs clone + [parity] extras; PhysicsNeMo tests
# need a PyG-enabled env and skip otherwise — see docs/graphcast_parity.md):
# git clone https://github.com/google-deepmind/graphcast.git /workspace/tmp/graphcast-jax
# pip install -e ".[parity]"
# pytest tests/parity/graphcast -q
# optional fuller suite (may be slower / need more RAM for full Pangu):
# pytest tests -q
```

## Attribution / 致谢与许可

- **License:** [MIT](LICENSE) — Copyright (c) 2026 Longhao Wang (GISWLH), plus
  retained WeatherLearn copyright as required by MIT.
- **NOTICE:** [NOTICE](NOTICE) — WeatherLearn (Copyright Zhuoqun Li / contributors),
  FengWu paper (Chen et al., arXiv:2304.02948), GraphCast paper (Lam et al.,
  arXiv:2212.12794; independent reimplementation).
  Parity tests may reference google-deepmind/graphcast and NVIDIA/physicsnemo (both
  Apache-2.0) from local clones; that code is not vendored into `weatherai/`.
- Upstream: https://github.com/lizhuoq/WeatherLearn  
- FengWu branch/PR context: `GISWLH/WeatherLearn@feat/fengwu`,
  [lizhuoq/WeatherLearn#14](https://github.com/lizhuoq/WeatherLearn/pull/14)

### Papers / 论文

- Pangu-Weather — Bi et al., [arXiv:2211.02556](https://arxiv.org/abs/2211.02556)
- FuXi — Chen et al., [arXiv:2306.12873](https://arxiv.org/abs/2306.12873)
- FengWu — Chen et al., [arXiv:2304.02948](https://arxiv.org/abs/2304.02948)
- GraphCast — Lam et al., [arXiv:2212.12794](https://arxiv.org/abs/2212.12794)

## Roadmap / 后续

- [ ] Pretrained weight download helpers (no invented checkpoints in-repo)
- [ ] Training / finetune recipes (kept out of v0.1 for a clean starter zoo)
- [x] GraphCast small-scale JAX parity (mesh / MLP / one GraphNet layer)
- [x] GraphCast stage parity (graphs, Grid2Mesh, processor N≤16, Mesh2Grid) vs JAX + PhysicsNeMo (tiny cases; NV processor aggregation is an inherent, documented difference)
- [ ] GraphCast checkpoint loading / full-scale parity
- [ ] More models and evaluation utilities

### Planned models / 计划加入的模型

Planned additions (**not implemented yet**; no code, weights, or parity claims in this
repo for any of them). Citations were checked against Crossref / the publisher pages;
"code" links point to the authors' official releases, which this repo does not vendor.

计划加入的模型（**尚未实现**，本仓库目前没有这些模型的代码、权重或一致性验证）。论文信息已对照
Crossref / 出版社页面核实；“Code”为作者官方发布的仓库，本仓库不内置其代码。

| Model | Paper (journal, year) | Official code / weights |
|-------|-----------------------|-------------------------|
| **NowcastNet** | Zhang et al., [Skilful nowcasting of extreme precipitation with NowcastNet](https://doi.org/10.1038/s41586-023-06184-4) — *Nature* 619, 2023 | [Code Ocean capsule](https://doi.org/10.24433/CO.0832447.v1) (code + pretrained weights, per the paper) |
| **NeuralGCM** | Kochkov et al., [Neural general circulation models for weather and climate](https://doi.org/10.1038/s41586-024-07744-y) — *Nature* 632, 2024 | [neuralgcm/neuralgcm](https://github.com/neuralgcm/neuralgcm) (models + checkpoints), [neuralgcm/dinosaur](https://github.com/neuralgcm/dinosaur) (dynamical core) |
| **GenCast** | Price et al., [Probabilistic weather forecasting with machine learning](https://doi.org/10.1038/s41586-024-08252-9) — *Nature* 637, 2024 | Inside [google-deepmind/graphcast](https://github.com/google-deepmind/graphcast) (now redirects to `google-deepmind/weathernext`); code + weights per the paper |
| **Aurora** | Bodnar et al., [A foundation model for the Earth system](https://doi.org/10.1038/s41586-025-09005-y) — *Nature* 641, 2025 | [microsoft/aurora](https://github.com/microsoft/aurora) (code + weights) |
| **Aardvark Weather** | Allen et al., [End-to-end data-driven weather prediction](https://doi.org/10.1038/s41586-025-08897-0) — *Nature* 641, 2025 | [anna-allen/aardvark-weather-public](https://github.com/anna-allen/aardvark-weather-public) (the repo URL given in the paper, `annavaughan/...`, now redirects here) |
| **WeatherNext Cyclones (WN-C)** | Alet et al., [Operational tropical cyclone forecasting with AI](https://doi.org/10.1038/s41586-026-10953-2) — *Nature* 657, 2026 | [google-deepmind/weathernext](https://github.com/google-deepmind/weathernext) (Apache-2.0; code + weights) |
| **FuXi-ENS** | Zhong et al., [FuXi-ENS: A machine learning model for efficient and accurate ensemble weather prediction](https://doi.org/10.1126/sciadv.adu2854) — *Science Advances* 11, 2025 ⚠️ **not a Nature-family journal** | [tpys/FuXi-ENS](https://github.com/tpys/FuXi-ENS) (model files on a Google Drive; access limited, request from the authors) |
| **FuXi-DA** | Xu et al., [FuXi-DA: a generalized deep learning data assimilation framework for assimilating satellite observations](https://doi.org/10.1038/s41612-025-01039-3) — *npj Climate and Atmospheric Science* 8, 2025 (Nature Portfolio) | [xuxiaoze/FuXi-DA](https://github.com/xuxiaoze/FuXi-DA) (inference example + checkpoint) |

Notes / 备注:
- FuXi-ENS is listed once. Its published venue is *Science Advances* (AAAS), not Nature or a
  Nature-family journal; it is kept here as a FuXi-family model of interest.
  FuXi-ENS 仅列一次；正式发表于 *Science Advances*，不属于 Nature 系列期刊。
- FuXi-DA is in *npj Climate and Atmospheric Science* (Nature Portfolio / Nature-family).
  FuXi-DA 发表于 *npj Climate and Atmospheric Science*（Nature 系列）。
- GraphCast status is unchanged: only Grid2Mesh / processor / Mesh2Grid are checked at
  mesh levels 0–1; no official checkpoint is loaded, so there is no whole-model parity claim.

Roadmap checklist:

- [ ] NowcastNet (precipitation nowcasting)
- [ ] NeuralGCM (hybrid dynamical core + ML)
- [ ] GenCast (diffusion-based ensemble forecasting)
- [ ] Aurora (Earth-system foundation model)
- [ ] Aardvark Weather (end-to-end, observations → forecast)
- [ ] WeatherNext Cyclones / WN-C (tropical cyclone ensembles)
- [ ] FuXi-ENS (ensemble forecasting)
- [ ] FuXi-DA (satellite data assimilation)

## Disclaimer

Research / educational code. Forecast skill depends on data, training, and
weights — this repo ships **architectures**, not operational NWP replacements.
