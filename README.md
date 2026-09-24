# WeatherAI

**Personal weather AI model zoo (PyTorch)** — Pangu, FuXi, FengWu, and more.

个人天气 AI 模型库（PyTorch）：Pangu、FuXi、FengWu 等，将逐步改进与扩展。

> **Independent project / 独立项目.** This is **not** a fork of
> [lizhuoq/WeatherLearn](https://github.com/lizhuoq/WeatherLearn). It starts from
> WeatherLearn-style MIT implementations of Pangu / FuXi and a FengWu skeleton
> (see [NOTICE](NOTICE)), then evolves as Longhao Wang (GISWLH)’s own zoo.
>
> 本仓库是 Longhao Wang（GISWLH）的**个人**模型库，**不是** WeatherLearn 的
> GitHub fork。代码在 MIT 许可下改编自 WeatherLearn 风格实现，并将持续改进。

## Models / 模型

| Model | Import | Notes |
|-------|--------|-------|
| **Pangu** / `Pangu_lite` | `from weatherai.models import Pangu, Pangu_lite` | 3D Earth-attention style; lite for smaller grids |
| **FuXi** (`Fuxi`) | `from weatherai.models import FuXi` | Cube embedding + U-Transformer (Swin V2) |
| **FengWu** / `FengWu_lite` | `from weatherai.models import FengWu, FengWu_lite` | Multi-modal encode–fuse–decode; optional uncertainty |

Architecture notes for FengWu: [docs/fengwu_specs.md](docs/fengwu_specs.md).

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
from weatherai.models import Pangu, FuXi, FengWu, FengWu_lite

# FuXi (alias of class Fuxi) — pass smaller dims for experiments
# Default paper-scale ctor is heavy; override embed_dim/depth for smoke use.
fuxi = FuXi

# FengWu lite — CPU-friendly defaults (64×128, 13 levels → 69 channels)
fw = FengWu_lite()
import torch
x = torch.randn(1, 69, 64, 128)
y = fw(x)  # (1, 69, 64, 128)

# Pangu lite
from weatherai.models import Pangu_lite
pangu = Pangu_lite()
```

```python
# Package-level re-exports
from weatherai import Pangu, FuXi, FengWu, FengWu_lite
```

## Tests / 测试

```bash
pip install -e ".[dev]"
pytest tests/models/fengwu -q
# optional fuller suite (may be slower / need more RAM for full Pangu):
# pytest tests -q
```

## Attribution / 致谢与许可

- **License:** [MIT](LICENSE) — Copyright (c) 2026 Longhao Wang (GISWLH), plus
  retained WeatherLearn copyright as required by MIT.
- **NOTICE:** [NOTICE](NOTICE) — WeatherLearn (Copyright Zhuoqun Li / contributors)
  and FengWu paper (Chen et al., arXiv:2304.02948).
- Upstream: https://github.com/lizhuoq/WeatherLearn  
- FengWu branch/PR context: `GISWLH/WeatherLearn@feat/fengwu`,
  [lizhuoq/WeatherLearn#14](https://github.com/lizhuoq/WeatherLearn/pull/14)

### Papers / 论文

- Pangu-Weather — Bi et al., [arXiv:2211.02556](https://arxiv.org/abs/2211.02556)
- FuXi — Chen et al., [arXiv:2306.12873](https://arxiv.org/abs/2306.12873)
- FengWu — Chen et al., [arXiv:2304.02948](https://arxiv.org/abs/2304.02948)

## Roadmap / 后续

- [ ] Pretrained weight download helpers (no invented checkpoints in-repo)
- [ ] Training / finetune recipes (kept out of v0.1 for a clean starter zoo)
- [ ] More models and evaluation utilities

## Disclaimer

Research / educational code. Forecast skill depends on data, training, and
weights — this repo ships **architectures**, not operational NWP replacements.
