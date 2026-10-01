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
| **NeuralGCM** (wrapper, JAX) | `from weatherai.models import NeuralGCM_lite, NeuralGCMWrapper` | Thin inference wrapper over the official JAX `neuralgcm` package (not a PyTorch port) — see [NeuralGCM](#neuralgcm--wrapper-around-the-official-jax-package) |
| **GenCast** (native denoiser + sampler) | `from weatherai.models.gencast import GenCastDenoiser_from_official, GenCast_lite` | explicit PyTorch port of the official denoiser (grid2mesh GNN, k-hop mesh transformer, mesh2grid GNN); official Mini weights strict-load and match official JAX (network only) — see [GenCast](#gencast--native-pytorch-denoiser-official-architecture-official-mini-weights-load--edmdpm-solver-sampler) |
| **Aardvark** (encoder + processor + station decoder) | `from weatherai.models.aardvark import AardvarkEncoder, AardvarkProcessor, AardvarkDecoder, AardvarkE2E` | Explicit PyTorch port of everything the official public code defines (set-conv observation encoder + ViT, forecast ViT, U-Net + set-conv + MLP station decoder); the three official checkpoints strict-load and the E2E output matches the official code on the official sample — see [Aardvark](#aardvark-weather--native-pytorch-encoder--processor--decoder) |
| **WeatherNext Cyclones Mini** (wrapper, JAX) | `from weatherai.models import WeatherNextCyclones_lite` | Thin inference wrapper over the official JAX `weathernext` package (Python ≥3.12; not a PyTorch port; tracker not wrapped) — see [WN-C](#weathernext-cyclones-mini--wrapper-around-the-official-jax-package) |
| **Aurora** (native PyTorch) | `from weatherai.models import Aurora, Aurora_lite, Aurora_small` | Explicit Perceiver encoder → Swin-3D U-Net → Perceiver decoder; official small ckpt strict-loads and matches the official package bit-for-bit on the tested inputs — see [Aurora](#aurora--native-pytorch-implementation) |

Architecture notes:
- FengWu: [docs/fengwu_specs.md](docs/fengwu_specs.md)
- GraphCast: [docs/graphcast_specs.md](docs/graphcast_specs.md)
- GraphCast parity vs DeepMind JAX + NVIDIA PhysicsNeMo (stage-wise): [docs/graphcast_parity.md](docs/graphcast_parity.md)

## Aurora — native PyTorch implementation

`weatherai.models.aurora` re-implements Aurora (Bodnar et al., *Nature* 2025; MIT,
[microsoft/aurora](https://github.com/microsoft/aurora)) as explicit `nn.Module`s with readable `forward()`s,
in the style of the Pangu/FuXi code (no dependency on the `aurora` pip package):

| file | contents |
|---|---|
| `patch_embed.py` | `LevelPatchEmbed` – per-variable 3-D patch conv over (history × P × P) |
| `perceiver.py` | `PerceiverAttention`, `PerceiverResampler` (post-norm cross-attention + MLP) |
| `encoder.py` | `Perceiver3DEncoder` – patch-embed surface(+static) and every level → level aggregation → + position / patch-scale / lead-time / absolute-time Fourier embeddings |
| `swin3d.py` | `Swin3DBackbone`: `SwinStage`s of `Swin3DBlock` (shifted 3-D windows with periodic-longitude mask, AdaLN on lead time), `PatchMerging3D`, `PatchSplitting3D` (linear1 → reshape/permute → crop padding → norm → linear2) |
| `decoder.py` | `Perceiver3DDecoder` – level de-aggregation + per-variable linear unpatchify |
| `aurora.py` | `Aurora` (plain-tensor I/O, official normalisation statistics, `Aurora_lite`, `Aurora_small`) |
| `official.py` | the old thin wrapper over `microsoft-aurora` – **reference oracle only** (tests / parity) |

```python
import torch
from weatherai.models import Aurora, Aurora_lite, Aurora_small

m = Aurora_lite().eval()                      # ~3 M params, random init, CPU friendly
surf   = torch.randn(1, 2, 4, 16, 32) + torch.tensor([278., 0, 0, 1e5]).view(1, 1, 4, 1, 1)  # (B,T=2,[2t,10u,10v,msl],H,W)
static = torch.randn(3, 16, 32)               # [lsm, z, slt]
atmos  = torch.randn(1, 2, 5, 4, 16, 32)      # (B,T,[z,u,v,t,q],L=4,H,W)
with torch.no_grad():
    surf_next, atmos_next = m(surf, static, atmos)   # physical units, +6 h

# fully configurable / trainable: any depths, heads, window, patch size, embed dim
my = Aurora(embed_dim=128, encoder_depths=(2, 4, 2), decoder_depths=(2, 4, 2),
            encoder_num_heads=(4, 8, 16), decoder_num_heads=(16, 8, 4), num_heads=4, window_size=(2, 6, 12))
loss = (my(surf, static, atmos)[0] - surf[:, 1]).pow(2).mean(); loss.backward()   # see tests for a full training step

# Official small pretrained checkpoint (~450 MB from HF microsoft/aurora), strict=True load into the native model
small = Aurora_small(pretrained=True).eval()  # or Aurora_small(checkpoint_path="aurora-0.25-small-pretrained.ckpt")
```

**Verified (numerically, vs. the official `microsoft-aurora` package at b628d7c, same weights, same inputs; fp32,
`atol=1e-5, rtol=1e-4` – observed max abs difference was exactly 0.0 on CPU):** random-weight state-dict transfer
`strict=True` for 3 configs (window (2,2,2)/(2,3,5)/(2,6,12), uneven depths, v3 lead-time embedding), grids
16×32 … 32×64 incl. the odd-latitude crop (H%4==1) and patch grids that need merge-padding / split-cropping
(25×44); the official `aurora-0.25-small-pretrained.ckpt` strict-loads into the native `Aurora_small` and
reproduces the official output on 33×64 inputs. HF ZeroGPU results: see [docs/model_status.md](docs/model_status.md).
Unit tests: `tests/models/aurora` (shapes, odd grids, merge/split + window helpers, param-name layout,
25-step training demo, grads, parity).
**Not verified / not included:** skill on real ERA5/HRES data (random inputs only), multi-step rollout;
upstream features that are *not* ported: LoRA, stochastic/ensemble mode, level-conditioned embeddings,
dynamic/atmos-static variables, separate/modulation heads (so the fine-tuned 1.3 B, HRES-0.1°, air-pollution,
wave and v1.5 checkpoints are **not** loadable; `aurora-0.25-pretrained` and `-small-pretrained` are); the 1.3 B
checkpoint was not run; built-in normalisation stats only for the 13 standard ERA5 levels.

## NeuralGCM — wrapper around the official JAX package

NeuralGCM (Kochkov et al., *Nature* 2024) is JAX (a differentiable spectral dynamical core + learned
physics). There is **no PyTorch re-implementation** here: `weatherai.models.neuralgcm` runs the official
code/checkpoints (`pip install -e ".[neuralgcm]"` → `jax neuralgcm dinosaur`) and hands results back as
`xarray` / `torch` tensors. Inference only (no autograd across JAX↔torch).

```python
from weatherai.models import NeuralGCM_lite, NeuralGCMWrapper

w = NeuralGCM_lite()            # official 2.8° deterministic checkpoint (58 MB, public GCS)
ds = w.demo_data()              # 1 ERA5 snapshot shipped with the official package
out = w.forecast(ds, steps=4, step_hours=6)        # xarray.Dataset, (time, level, lon, lat)
t = NeuralGCMWrapper.to_torch(out, ["temperature"])["temperature"]   # torch.Tensor (4, 37, 128, 64)
```

**Status — verified:** official checkpoint loads; forecast from the official demo snapshot is finite and
evolves; wrapper == direct upstream call bit-for-bit; seed-independent (deterministic model); passes on the
box CPU and on the HF Space. **Not verified:** forecast skill vs. truth, other checkpoints (1.4°/0.7°/stochastic),
JAX on GPU (the HF run used JAX's CPU backend). Details: [docs/model_status.md](docs/model_status.md).

## GenCast — native PyTorch denoiser (official architecture, official Mini weights load) + EDM/DPM-Solver++ sampler

GenCast (Price et al., *Nature* 2025) is a conditional EDM diffusion model whose denoiser is a GraphCast-style
grid→mesh→grid network with a 16-layer k-hop mesh transformer; official code is JAX (`google-deepmind/weathernext`).
`weatherai.models.gencast` contains explicit `nn.Module`s of the **official denoiser** (parameter names mirror the Haiku names)
plus the EDM wrapper / DPM-Solver++ 2S sampler. No wrapper around JAX is involved at runtime.

| file | contents |
|---|---|
| `denoiser.py` | `FourierFeaturesMLP` (noise-level encoder), `LinearNormConditioning`, `ConditionedMLP` (Linear→swish→Linear→LN→cond-affine), `Grid2MeshGNN`, `MeshTransformer` (+ `MeshAttention`: dense masked or banded tri-block-diagonal k-hop attention), `Mesh2GridGNN`, `GenCastDenoiser`, `convert_official_params`, `GenCastDenoiser_from_official`, channel stacking helpers |
| `graphs.py` | grid2mesh radius graph, finest-mesh banded (reverse Cuthill-McKee) permutation, mesh2grid containing-triangle graph, k-hop mask (re-uses `graphcast.mesh`) |
| `gencast.py` | EDM preconditioning/loss, noise schedule, DPM-Solver++ 2S + churn sampler, `DenoiserNet` (config-driven adapter), `GenCast`, `GenCast_lite` |

```python
import numpy as np, torch
from weatherai.models.gencast import (GenCast_lite, GenCastDenoiser_from_official, denoiser_inputs, unstack_variables)

# (a) small config-driven, trainable instance of the SAME native architecture (random init)
m = GenCast_lite(img_size=(16, 32), out_channels=4, cond_channels=8, latent=32, transformer_layers=2, mesh_level=1)
cond, target = torch.randn(2, 8, 16, 32), torch.randn(2, 4, 16, 32)
loss = m.loss(target, cond); loss.backward()                     # EDM-weighted denoising loss (training step)
members = m.eval().sample(cond, num_members=4)                   # (4, 2, 4, 16, 32) DPM-Solver++ 2S + churn

# (b) official GenCast-1p0deg-Mini weights (public GCS bucket, 230 MB) in the native denoiser
#     gs://dm_graphcast/gencast/params/GenCast 1p0deg Mini <2019.npz
lat, lon = np.linspace(-90, 90, 181), np.arange(360)             # official 1° grid
net = GenCastDenoiser_from_official("mini.npz", lat, lon)        # strict load; mesh_size=4, 16 layers, 57.5 M params
x = denoiser_inputs(inputs, forcings, noisy_targets, (181, 360)) # dict-of-arrays -> (B, 181*360, 264), official channel order
raw = net(x, noise_levels)                                       # (B, Ng, 84) raw network output
```

**Status — verified (CPU, fp32):**
* **Denoiser vs official JAX with the official Mini weights** (reference from the official `weathernext1_gen.denoiser.Denoiser`,
  `scripts/gencast_denoiser_reference.py`, random inputs, official plain-`mha` attention because the TPU splash kernel is unavailable):
  - *trained configuration* (mesh_size=4 / 2562 mesh nodes, `attention_k_hop=16`, 181×360 grid, 102 k grid→mesh edges, batch 1): max |Δ| = 4.8e-5 on all 84 outputs (output magnitude up to 22; max rel. to output max 3.2e-6);
  - small configuration (mesh_size=2, k_hop=3, 13×24, batch 2, two noise levels): max |Δ| = 7.4e-5; banded tri-block-diagonal attention gives the same (6.9e-5).
  - graph construction (permuted mesh, grid2mesh/mesh2grid indices and edge features) equals the official graphs (indices exactly, features ≤3e-8).
  - all all official parameter tensors (57,492,612 weights) strict-load; unit tests use atol 2e-4 / rtol 1e-4 (small) and atol 5e-4 (full).
* EDM coefficients, noise/churn schedules and the DPM-Solver++ 2S sampler (± churn) match the official JAX sampler (rtol 2e-4, closed-form toy denoiser, fixed noise).
* Training step example/test: loss decreases on a fixed batch, all parameters receive gradients (`tests/models/gencast`).

**Not verified / not ported:** the *network* only was compared against the official network — **not** the full model (no end-to-end official
ensemble forecast comparison; the official `InputsAndResiduals` normalisation, NaN-cleaning of SST and the autoregressive rollout are not ported); the sampling noise
is iid Gaussian on the grid, official is spherical-harmonic white noise (needs `dinosaur`); the 0.25° and Operational checkpoints were not run
(same code path, larger graphs); the lite config is not a trained model. Details: [docs/model_status.md](docs/model_status.md).

## Aardvark Weather — native PyTorch encoder + processor + decoder

Aardvark Weather (Allen et al., *Nature* 2025; official code CC0) = observation **encoder** → **processor** → station **decoder**.
`weatherai.models.aardvark` now contains explicit ports of all three (parameter names identical to the official ones):

| file | contents |
|---|---|
| `setconv.py` | `SetConv` (Gaussian ConvDeepSet: off-grid→grid, grid→grid, grid→off-grid, density channel) |
| `aardvark.py` | `ViT` (per-variable patch embed + variable aggregation, or MLP + single embed), `AardvarkProcessor` (24 h step) |
| `unet.py` | cylindrical-padding conv / transposed conv, `Down`, `Up`, `Unet` |
| `system.py` | `AardvarkEncoder` (instrument set-convs + 277-channel stack + patch-3 ViT), `AardvarkDecoder` (U-Net → station set-conv → MLP), `AardvarkE2E`, official-checkpoint / sample loaders |

```python
import torch
from weatherai.models.aardvark import (AardvarkProcessor_lite, AardvarkE2E, load_official_encoder, load_official_decoder,
                                       load_official_processor, load_official_sample)

m = AardvarkProcessor_lite().eval()                    # 0.23 M params, random init, 60x31 grid; trainable
y = m(torch.randn(1, 35, 60, 31))                      # (1, 31, 60, 24): normalised 24 h tendency

# Official weights (HF dataset av555/aardvark-weather, trained_model/...) - all strict=True loads:
enc = load_official_encoder("trained_model/encoder/epoch_96")
dec = load_official_decoder("trained_model/decoder/tas/lt_1/epoch_18")
proc = load_official_processor("trained_model/processor/forecast_1/epoch_0")
task = load_official_sample("aardvark-weather-public/data/sample_data_final.pkl")     # official sample (CUDA pickle -> CPU)
station, forecast, init_state = AardvarkE2E(enc, [proc], dec).eval()(task, return_gridded=True)  # 24 h forecast; station = normalised tas
# every piece can be built with other sizes / depths and trained (see tests: tiny encoder+processor+decoder training step)
```

**Verified (CPU; official code run on the official sample as reference, `scripts/aardvark_system_reference.py`):**
encoder initial state max |Δ| = 0.0 (atol 1e-5, rtol 1e-4); station decoder (tas, lt_1) max |Δ| = 0; E2E (encoder → 1 processor → decoder)
station output max |Δ| = 2.9e-6, gridded forecast max relative error 8e-5 (values up to 1.2e5); processor on random input max |Δ| ≈ 2e-4
(unchanged earlier check). Unit tests: set-conv vs naive loop, NaN handling, cylindrical conv shapes, parameter names, a training step on a tiny
whole system. HF ZeroGPU: see [docs/model_status.md](docs/model_status.md).
**Not implemented / not verified:** the official data loaders and training scripts (they depend on multi-TB local memmaps; the repo itself says they
cannot be executed), the FiLM / attention options of the official U-Net (unused by the released weights), `e2e_finetuned` checkpoints (not loaded/tested),
decoder checkpoints other than `tas/lt_1`, processors `forecast_2..10`, multi-step E2E, other station variables (`ws`), skill vs. observations.
The official sample has one timestep only, so the comparison covers that single case.

## WeatherNext Cyclones Mini — wrapper around the official JAX package

WN-C / WeatherNext 2 (FGN) is a JAX/Haiku GraphCast-style model with a sparse-transformer processor. **No PyTorch port**:
`weatherai.models.weathernext_cyclones` runs the official code with the official **WeatherNextCyclones_Mini_<2024** checkpoint
(1°, 227 MB; `pip install -e ".[weathernext]"`, **Python ≥ 3.12**) and returns `xarray` / `torch` tensors. Inference only.

```python
import xarray
from weatherai.models import WeatherNextCyclones_lite
from weatherai.models.weathernext_cyclones import download_sample_data

w = WeatherNextCyclones_lite()                                   # official config + weights (downloads 227 MB)
ds = xarray.load_dataset(download_sample_data()).compute()       # official 1° HRES sample (159 MB), init 2024-10-07 00Z
out = w.forecast(ds, steps=2, num_members=2)                     # 2 members x 2 six-hour steps, dims (sample, time, batch, [level,] lat, lon)
```

**Status — verified (CPU only):** official checkpoint loads; ensemble forecast finite with expected shapes; members differ; one-case sanity check
(500 hPa T RMSE vs the sample's HRES frames 0.5 K at +12 h vs 3.0 K for persistence — not a skill score).
**Not verified:** cyclone tracker (not wrapped), 0.25° models, anything on GPU/TPU (no HF GPU run), multi-case statistics. Details: [docs/model_status.md](docs/model_status.md).

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

Planned additions (**not implemented yet** — each is blocked on unavailable official model code/weights, see the roadmap checklist below; no code, weights, or parity claims in this
repo for any of them). Citations were checked against Crossref / the publisher pages;
"code" links point to the authors' official releases, which this repo does not vendor.

计划加入的模型（**尚未实现**，本仓库目前没有这些模型的代码、权重或一致性验证）。论文信息已对照
Crossref / 出版社页面核实；“Code”为作者官方发布的仓库，本仓库不内置其代码。

| Model | Paper (journal, year) | Official code / weights |
|-------|-----------------------|-------------------------|
| **NowcastNet** | Zhang et al., [Skilful nowcasting of extreme precipitation with NowcastNet](https://doi.org/10.1038/s41586-023-06184-4) — *Nature* 619, 2023 | [Code Ocean capsule](https://doi.org/10.24433/CO.0832447.v1) (code + pretrained weights, per the paper) |
| **FuXi-ENS** | Zhong et al., [FuXi-ENS: A machine learning model for efficient and accurate ensemble weather prediction](https://doi.org/10.1126/sciadv.adu2854) — *Science Advances* 11, 2025 ⚠️ **not a Nature-family journal** | [tpys/FuXi-ENS](https://github.com/tpys/FuXi-ENS) (model files on a Google Drive; access limited, request from the authors) |
| **FuXi-DA** | Xu et al., [FuXi-DA: a generalized deep learning data assimilation framework for assimilating satellite observations](https://doi.org/10.1038/s41612-025-01039-3) — *npj Climate and Atmospheric Science* 8, 2025 (Nature Portfolio) | [xuxiaoze/FuXi-DA](https://github.com/xuxiaoze/FuXi-DA) (inference driver only; the README lists `model/` and `test_data/` but they are not published — see roadmap) |

Notes / 备注:
- FuXi-ENS is listed once. Its published venue is *Science Advances* (AAAS), not Nature or a
  Nature-family journal; it is kept here as a FuXi-family model of interest.
  FuXi-ENS 仅列一次；正式发表于 *Science Advances*，不属于 Nature 系列期刊。
- FuXi-DA is in *npj Climate and Atmospheric Science* (Nature Portfolio / Nature-family).
  FuXi-DA 发表于 *npj Climate and Atmospheric Science*（Nature 系列）。
- GraphCast status is unchanged: only Grid2Mesh / processor / Mesh2Grid are checked at
  mesh levels 0–1; no official checkpoint is loaded, so there is no whole-model parity claim.

Roadmap checklist:

- [ ] NowcastNet (precipitation nowcasting) — **blocked**: the only official source is the Code Ocean capsule (codeocean.com/capsule/3935105), which returns HTTP 403 without a login; no official GitHub; not started
- [x] NeuralGCM (hybrid dynamical core + ML) — wrapper over official JAX package; CPU-tested (see above)
- [x] GenCast (diffusion-based ensemble forecasting) — native PyTorch denoiser (official Mini weights load, network matches official JAX to 5e-5) + verified sampler; full-model/SH-noise/normalisation pipeline not ported (see above)
- [x] Aurora (Earth-system foundation model) — native PyTorch; official small checkpoint strict-loads, output matches official package (see above)
- [x] Aardvark Weather — native PyTorch encoder + processor + station decoder; official checkpoints strict-load and E2E matches official code on the official sample (data loaders / training scripts not ported)
- [x] WeatherNext Cyclones / WN-C (tropical cyclone ensembles) — wrapper over official JAX package (Mini checkpoint, CPU-tested; tracker not wrapped)
- [ ] FuXi-ENS (ensemble forecasting) — **blocked**: official repo has inference scripts only; model (`fuxi_ens.onnx`) and sample data are on a restricted Google Drive (request from the authors); no PyTorch definition published
- [ ] FuXi-DA (satellite data assimilation) — **blocked**: `model/assimilation_v6.py` and `final_cast_10_assim_model.pth` + test data are not in the repo and not linked anywhere (open upstream issue xuxiaoze/FuXi-DA#2 asks for them, unanswered); only the inference driver is public, so the architecture cannot be reproduced or verified

## Disclaimer

Research / educational code. Forecast skill depends on data, training, and
weights — this repo ships **architectures**, not operational NWP replacements.
