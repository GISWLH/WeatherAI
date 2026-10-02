# WeatherAI

**Personal weather AI model zoo (PyTorch)** — Pangu, FuXi, FengWu, GraphCast, and more.

个人天气 AI 模型库（PyTorch）：Pangu、FuXi、FengWu、GraphCast 等，将逐步改进与扩展。

## Models / 模型

| Model | Import | Notes |
|-------|--------|-------|
| **Pangu** / `Pangu_lite` | `from weatherai.models import Pangu, Pangu_lite` | 3D Earth-attention style; lite for smaller grids.<br>**论文 / Paper:** *Nature* 2023, [doi:10.1038/s41586-023-06185-3](https://doi.org/10.1038/s41586-023-06185-3) (preprint [arXiv:2211.02556](https://arxiv.org/abs/2211.02556)) |
| **FuXi** (`Fuxi`) | `from weatherai.models import FuXi` | Cube embedding + U-Transformer (Swin V2).<br>**论文 / Paper:** *npj Climate and Atmospheric Science* 2023, [doi:10.1038/s41612-023-00512-1](https://doi.org/10.1038/s41612-023-00512-1) (preprint [arXiv:2306.12873](https://arxiv.org/abs/2306.12873)) |
| **FengWu** / `FengWu_lite` | `from weatherai.models import FengWu, FengWu_lite` | Multi-modal encode–fuse–decode; optional uncertainty. `FengWu` follows the 2023 arXiv architecture ([specs](docs/fengwu_specs.md)).<br>**论文 / Paper:** arXiv 2023 ([arXiv:2304.02948](https://arxiv.org/abs/2304.02948)); journal version (retitled, revised): *Communications Earth & Environment* 2025, [doi:10.1038/s43247-025-02502-y](https://doi.org/10.1038/s43247-025-02502-y) |
| **GraphCast** / `GraphCast_lite` | `from weatherai.models import GraphCast, GraphCast_lite` | Grid ↔ icosahedral mesh encode–process–decode.<br>**论文 / Paper:** *Science* 2023, [doi:10.1126/science.adi2336](https://doi.org/10.1126/science.adi2336) (preprint [arXiv:2212.12794](https://arxiv.org/abs/2212.12794)) |
| **NeuralGCM** (partial native + JAX training) | `from weatherai.models.neuralgcm import NeuralGCMLearnedComponents, convert_official_params, SphericalHarmonicsGrid` · `from weatherai.models import NeuralGCM_lite, NeuralGCMWrapper` · `from weatherai.models.neuralgcm import train` (configure / from-scratch / fine-tune, JAX, tested on HF GPU) | Native learned components + spherical-harmonic layer (14.5 M official weights strict-load, match JAX); dycore and feature builders not ported, forecasts run via the official-JAX wrapper (oracle); trainable/fine-tunable in JAX, tested on HF GPU — [section](#neuralgcm--native-learned-components--torch-spherical-harmonics-jax-wrapper-kept-as-oracle).<br>**论文 / Paper:** *Nature* 2024, [doi:10.1038/s41586-024-07744-y](https://doi.org/10.1038/s41586-024-07744-y) |
| **FuXi-ENS** (native, from ONNX) | `from weatherai.models.fuxi_ens import FuXiENS, FuXiENS_lite, FuXiENSNoise, load_official` | PyTorch port reverse-engineered from the official ONNX; 2.47 B official weights (CC-BY-NC-4.0, not in git) strict-load; one forward step matches onnxruntime (max\|Δ\|/max\|ref\| 8e-6, CPU); 15-day ensemble not run — [section](#fuxi-ens--native-pytorch-reverse-engineered-from-the-official-onnx-numerically-checked-vs-onnxruntime).<br>**论文 / Paper:** *Science Advances* 2025 (not Nature family), [doi:10.1126/sciadv.adu2854](https://doi.org/10.1126/sciadv.adu2854) |
| **StormCast** (native, official weights load) | `from weatherai.models.stormcast import StormCast, StormCast_lite, load_official` | Regression U-Net + EDM diffusion U-Net (HRRR 512×640); official weights (Apache-2.0, not in git) strict-load, bit-identical to PhysicsNeMo on CPU; HF GPU run on synthetic inputs only, skill not verified — [section](#stormcast--native-pytorch-regression-u-net--edm-diffusion-u-net-official-weights-load-bit-identical-to-physicsnemo-on-cpu).<br>**论文 / Paper:** *Science Advances* 2026, [doi:10.1126/sciadv.adv0423](https://doi.org/10.1126/sciadv.adv0423) |
| **ArchesWeather / ArchesWeatherGen** (native, official weights load) | `from weatherai.models.arches import ArchesWeather_lite, ArchesWeatherGen_lite, convert` | 3D-attention forecaster + flow-matching generator (383.8 M); official checkpoint strict-loads, matches `geoarches` on CPU (rel. err ≈1e-6); one real case only, no skill statistics — [section](#archesweathergen--native-pytorch-3d-attention-forecaster--flow-matching-generator-official-weights-load-parity-with-geoarches).<br>**论文 / Paper:** *Science Advances* 2026, [doi:10.1126/sciadv.adx2372](https://doi.org/10.1126/sciadv.adx2372) |
| **ACE2-ERA5** (native, official weights load) | `from weatherai.models.ace2 import ACE2, load_official, SFNO, RealSHT` | 455.8 M-param spherical Fourier neural operator + inference wrapper (dry-air/moisture conservation, SST forcing); official checkpoint (Apache-2.0, not in git) strict-loads, 12-step rollout matches `fme` (max rel. err 7.9e-5); one 8-day real case, no climate-scale/ensemble statistics — [section](#ace2-era5--native-pytorch-spherical-fourier-neural-operator-official-weights-load-matches-fme-rollouts).<br>**论文 / Paper:** *npj Climate and Atmospheric Science* 2025, [doi:10.1038/s41612-025-01090-0](https://doi.org/10.1038/s41612-025-01090-0) |
| **NeuralGCM precipitation** (official JAX checkpoints; trainable loss) | `from weatherai.models.neuralgcm import precip as P` | Two official stochastic precipitation checkpoints (Zenodo, CC-BY-4.0, not in git) run through unchanged official JAX code, plus differentiable precipitation loss and `fit_precip` fine-tune loop; one real 24 h case, reproduced on HF GPU; not ported to native torch, no skill/multi-case verification — [section](#neuralgcm-precipitation--official-checkpoints-via-official-code-trainable-loss).<br>**论文 / Paper:** *Science Advances* 2026, [doi:10.1126/sciadv.adv6891](https://doi.org/10.1126/sciadv.adv6891) |
| **GenCast** (native denoiser + sampler) | `from weatherai.models.gencast import GenCastDenoiser_from_official, GenCast_lite` | Native denoiser (grid2mesh GNN, k-hop mesh transformer, mesh2grid GNN) + EDM/DPM-Solver++ sampler; official Mini weights strict-load, network matches official JAX (network only) — [section](#gencast--native-pytorch-denoiser-official-architecture-official-mini-weights-load--edmdpm-solver-sampler).<br>**论文 / Paper:** *Nature* 2025, [doi:10.1038/s41586-024-08252-9](https://doi.org/10.1038/s41586-024-08252-9) |
| **Aardvark** (encoder + processor + station decoder) | `from weatherai.models.aardvark import AardvarkEncoder, AardvarkProcessor, AardvarkDecoder, AardvarkE2E` | Native port of everything the official public code defines; three official checkpoints strict-load, E2E output matches official code on the official sample — [section](#aardvark-weather--native-pytorch-encoder--processor--decoder).<br>**论文 / Paper:** *Nature* 2025, [doi:10.1038/s41586-025-08897-0](https://doi.org/10.1038/s41586-025-08897-0) |
| **WeatherNext Cyclones Mini** (native network + JAX oracle) | `from weatherai.models.weathernext_cyclones import WeatherNextCyclonesNet_from_official, WeatherNextCyclonesNative_lite` | Native FGN network (noise-conditioned encoders, 16-layer k-hop mesh transformer); official Mini weights strict-load, match official JAX; tracker/rollout/normalisation not ported — [section](#weathernext-cyclones-mini--native-pytorch-network-official-mini-weights-load-jax-wrapper-kept-as-oracle).<br>**论文 / Paper:** *Nature* 2026, [doi:10.1038/s41586-026-10953-2](https://doi.org/10.1038/s41586-026-10953-2); underlying FGN report [arXiv:2506.10772](https://arxiv.org/abs/2506.10772) (preprint only) |
| **Aurora** (native PyTorch) | `from weatherai.models import Aurora, Aurora_lite, Aurora_small` | Perceiver encoder → Swin-3D U-Net → Perceiver decoder; official small ckpt strict-loads, matches official package exactly on tested inputs — [section](#aurora--native-pytorch-implementation).<br>**论文 / Paper:** *Nature* 2025, [doi:10.1038/s41586-025-09005-y](https://doi.org/10.1038/s41586-025-09005-y) |

Architecture notes:
- FengWu: [docs/fengwu_specs.md](docs/fengwu_specs.md)
- GraphCast: [docs/graphcast_specs.md](docs/graphcast_specs.md)
- GraphCast parity vs DeepMind JAX + NVIDIA PhysicsNeMo (stage-wise): [docs/graphcast_parity.md](docs/graphcast_parity.md)

## Aurora — native PyTorch implementation

`weatherai.models.aurora` re-implements Aurora (*Nature* 2025; MIT,
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

## NeuralGCM — native learned components + torch spherical harmonics (JAX wrapper kept as oracle)

NeuralGCM (*Nature* 2024) = a differentiable spectral dynamical core (`dinosaur`: moist primitive equations on
sigma levels, IMEX-RK stepping) + learned Haiku components. **This is only partly ported:**

| ported natively (`weatherai/models/neuralgcm/network.py`, `spectral.py`) | not ported (exact stuck points: docs/model_status.md) |
|---|---|
| `EpdTower` / `ColumnMLP` (gelu-MLP encode → 5 residual process blocks → decode; the decoder, 2 encoders and the physics tendency tower) | the dinosaur dycore: sigma-coordinate primitive equations, `imex_rk_sil3` stepper, filters, `DycoreWithPhysicsCorrector` |
| `SurfaceEmbedding` (land/sea/sea-ice towers + mask/ice blend), `VerticalConvTower` (5 × Conv1D over levels) | tower *input feature builders* (dimensional velocity/prognostic/pressure/radiation/latitude/memory features, normalisation, input clipping/filters) and output transforms |
| `LearnedPositionalFeatures`, `LearnedOrography` | the base orography dataset (shipped in the checkpoint's aux data), forcing interpolation, `encode/advance/decode/unroll` plumbing |
| `SphericalHarmonicsGrid`: real SH basis, `to_modal/to_nodal`, ∇², ∂λ, cosφ∂φ, div/curl/grad, u,v ↔ vorticity/divergence | stochastic checkpoints (noise/`stochastic.py`), other resolutions |

```python
import pickle, torch
from weatherai.models.neuralgcm import NeuralGCMLearnedComponents, convert_official_params, SphericalHarmonicsGrid, SpectralGridConfig

params = pickle.load(open("deterministic_2_8_deg.pkl", "rb"))["params"]   # official checkpoint (trusted file; needs jax to unpickle)
net = convert_official_params(params)                 # strict: raises if any of the 121 Haiku modules / 14,518,180 weights is unused
y = net.physics(torch.randn(2365, 128, 64))           # (192, 128, 64) physics-tendency tower on packed nodal features (C, lon, lat)
g = SphericalHarmonicsGrid(SpectralGridConfig.TL63()) # 2.8° Gaussian grid, modal shape (127, 65)
vor, div = g.uv_nodal_to_vor_div_modal(u, v)          # (..., lon, lat) -> (..., m, l)
```

Everything is differentiable and trainable (`tests/.../test_neuralgcm_native.py::TestTrainableNative`: tower training step; gradients
through the SH layer). The official JAX wrapper (`NeuralGCM_lite()`, `NeuralGCMWrapper.forecast`) is unchanged and is the only
path that produces forecasts.

**Status — verified (numerically vs the official `neuralgcm` 1.2.3 / `dinosaur` 1.5.0, float32, official 2.8° deterministic checkpoint):**
all 121 Haiku modules / 14,518,180 weights strict-load; real packed tower inputs from one official encode→advance→decode step of the
demo ERA5 snapshot (24 random columns), max |native − official|: decoder 1.5e-4 (|y|≤206), encoder 1.3e-3 (|y|≤699), encoder_1 1.1e-4,
physics 1.3e-5 (|y|≤20), surface towers ≤5.8e-6, surface blend ≤1e-5, vertical CNN 2.2e-7 (|y|≤0.11), learned orography 0.0;
SH layer on random fields: ≤3.1e-5 for transforms, vorticity/divergence and the u,v round trip (|values| up to ~100), exact (0.0) for ∇²,
∂λ; float64 basis orthonormal to 1e-10. **Not verified / not claimed:** any forecast produced by native code (there is none),
whole-model parity, the dycore, feature builders, other checkpoints (1.4°/0.7°/stochastic). (JAX-on-GPU and the JAX training workflow: see the next subsection.) Details and results:
[docs/model_status.md](docs/model_status.md).

### NeuralGCM — training / modification workflow (JAX, official `neuralgcm` + `dinosaur` + Haiku; `weatherai/models/neuralgcm/train.py`)

JAX is allowed for NeuralGCM, so the model that can actually be **trained and modified** is the *official* one: the dycore is
official code, the learned parts are re-configured through the official gin config and trained with `optax`.
(The torch towers / SH layer above are kept for inspection and porting; they cannot forecast.)

**What you can modify** (all verified to change the Haiku parameter tree and train; `CONFIG_KEYS` lists them):

| knob | where | effect |
|---|---|---|
| `LATENT_SIZE`, `LAYER_SIZE` | `build_model(ck, {...})` | width of the decoder / 2 encoder / physics EPD towers (default 384 / 384) |
| `NUM_BLOCKS`, `process/MlpUniform.num_hidden_layers` | same | depth: residual process blocks (5) and layers per block (3) |
| `N_CNN_FEATURES`, `POSITIONAL_LATENT_SIZE` | same | vertical 1-D CNN width (32), learned positional-feature channels (8) |
| `SURFACE_MODEL_LATENT_SIZE/LAYER_SIZE/OUTPUT_SIZE` | same | land / sea / sea-ice surface towers (8) |
| `N_INNER_DYCORE_STEPS` | same | dycore substeps per physics call (5) |
| any other gin binding | `extra_bindings=["name = value"]` | e.g. other tower options in the shipped config |
| which parameters are trained | `fit(..., freeze=[regex,…])` | e.g. freeze everything except `physics` / the decoder |
| initialisation | `params="official" \| "init" \| "transfer"` | keep the checkpoint; random Haiku init (from scratch); random init + copy every official array whose path *and* shape still match |
| loss | `rollout_loss(..., steps, variables, lat_weights, level_w)` | `steps=0` encode→decode reconstruction, `steps≥1` differentiable `unroll` through the dycore |

**Not modified / not claimed:** the dynamical core equations, the feature builders and the data pipeline are the official code (not
re-implemented). No paper-scale training (multi-year ERA5, 1–4 day rollouts, spectral losses, multi-stage schedule, TPUs) was run, and
nothing here claims forecast skill from the demos below.

```python
import pickle, numpy as np
from weatherai.models.neuralgcm import train as T
from weatherai.models.neuralgcm.neuralgcm import download_checkpoint
ck = pickle.load(open(download_checkpoint("deterministic_2_8_deg"), "rb"))        # trusted official file

# (a) from scratch, smaller towers (874,020 params instead of 14,518,180)
m, rep = T.build_model(ck, {"LATENT_SIZE": 64, "LAYER_SIZE": 64, "NUM_BLOCKS": 2}, params="init", seed=0)
x, forc = T.demo_snapshot(m)                                  # bundled ERA5 snapshot
tg = {k: v[None] for k, v in x.items() if k != "sim_time"}   # reconstruction target
m, hist = T.fit(m, x, forc, tg, steps=0, n_iters=8, lr=3e-3)  # hist["loss"], hist["grad_norm"]

# (b) fine-tune the official checkpoint on real ERA5 (public ARCO-ERA5 on GCS, regridded; needs gcsfs)
off, _ = T.build_model(ck, params="official")
ds = T.fetch_era5_window(off, "2020-01-01T00", steps=1, hours=1)              # cached .nc
x, forc = T.demo_snapshot(off)  # or build inputs from ds[0]; see tests / scripts for the exact call
off2, hist = T.fit(off, x, forc, T.stack_targets(off, ds, 1), steps=1, n_iters=6, lr=3e-5, freeze=[r"^(?!.*physics)"])
```

**Training recipe used for the demos:** `optax.chain(clip_by_global_norm(1.0), adam(lr))`; loss = mean squared error of the decoded
state vs the target, each variable normalised by one std (per-level scales blow up at the model top), cos(lat) weights, levels above
50 hPa masked (the encoder→decoder reconstruction error dominates there); `steps=0` for reconstruction, `steps=1` (1 h) through the
dycore for fine-tuning; learning rate 3e-3 from scratch, 3e-5 for fine-tuning.

**Verified** (`tests/models/neuralgcm/test_neuralgcm_train.py`, 7 tests, JAX 0.10.2 CPU on the box, 176 s):
config overrides change the architecture (874,020 vs 14,518,180 params; `process_tower_1` exists, `_2` does not); `transfer` copies all
14,518,180 official weights when only unrelated knobs change; from-scratch reconstruction training decreases the loss
(1.288 → 0.929 in 8 Adam steps); gradients through a dycore rollout reach the encoder, physics tower and decoder (finite, 129/129
parameter leaves non-zero); training with the physics tower frozen leaves it bit-identical; fine-tuning the *official* model on
real ERA5 (2020-01-01 00Z → +1 h) lowers the loss 0.002582 → 0.002217 after 6 steps (held-out 2020-07-01 12Z: 0.004205 → 0.004002; `scripts/neuralgcm_finetune_era5.py`, 331 s for 6 steps on 8 CPU
cores, peak RSS ~7 GB). **These are mechanics checks, not skill:** at +1 h the official model's error vs ERA5 is dominated by the
encoder→decoder reconstruction error near the model top and is *larger* than persistence at many levels; fine-tuning for 6 steps on one
snapshot is overfitting to that snapshot, and no improvement in forecast skill is claimed.

**HF GPU (ZeroGPU RTX PRO 6000 Blackwell MIG 2g.48gb):** `neuralgcm_train` smoke **PASS with JAX actually on the GPU**
(`jax_backend: gpu`, `CudaDevice`, params live on the device; 4 reconstruction steps 0.0030 → 0.0027, rollout gradient finite, 129/129 leaves non-zero).
What it took: the Space runs Python 3.10, so JAX is capped at 0.6.2, which has only the `jax[cuda12]` extra (not `cuda13`) — with plain `jax[cpu]`
(the earlier setting) or an unmatched extra JAX silently falls back to CPU; and on the Blackwell slice XLA's GEMM autotuner aborts the process on the
dycore's `HIGHEST`-precision matmuls (`Algorithm not supported by the ElementalIrEmitter: ALG_DOT_BF16_BF16_F32`), fixed with
`XLA_FLAGS=--xla_gpu_autotune_level=0` (set in `hf_space/app.py` before `import jax`). ZeroGPU exposes the GPU only inside `@spaces.GPU`
calls (120 s limit), so only the small config was timed there; the full-size fine-tune was run on CPU only.

## FuXi-ENS — native PyTorch (reverse-engineered from the official ONNX; numerically checked vs onnxruntime)

FuXi-ENS (*Science Advances* 2025) is a latent-noise ensemble forecaster. The only official artefact is `fuxi_ens.onnx`
(Zenodo [10.5281/zenodo.15124541](https://zenodo.org/records/15124541), **CC-BY-NC-4.0 — non-commercial; weights are never committed**:
download with `python scripts/fetch_fuxi_ens.py --out /workspace/ckpt/fuxi_ens`, 9.9 GB). `weatherai/models/fuxi_ens/model.py` is an
explicit PyTorch re-implementation whose structure, constants and parameter names were read out of the 26,018-node graph; **the official
code has no PyTorch definition, so the ONNX graph itself is the oracle**.

```
input [B,2,78,721,1440] (2 steps x (5 vars x 13 levels + 13 surface))  --(x-mean)/std, NaN->0, accumulated channels 73-77 := 0-->  xn
 dist_p  : PatchEmbed (Conv 8x8 on [state;static const] -> LN) -> dropout "drop0" -> layers.0 (7 AdaLN blocks) -> dropout "drop1"
           -> layers.1 (7 blocks) -> final AdaLN (+patch-embed residual) -> ConvTranspose 8x8 heads: mean, logvar     (14 blocks)
 sample  = mean + exp(logvar/2) * eps                                  <- the ensemble noise ("RandomNormalLike")
 decoder.0: z = xn + sample (channels 73-77 := 0) -> PatchEmbed -> 6 layers x 7 AdaLN blocks -> final AdaLN (+residual)
           -> pl_head (65 ch) || sf_head (13 ch)  -> x*std+mean ; tp = exp(clip(.,0,7)) - 1                          (42 blocks)
output [B,2,78,721,1440] = concat(input[:, -1], new state)           (== the ONNX output; inference.py feeds it back as next input)
```

AdaLN block = `adaln = Linear(SiLU(cond))` -> `(shift, scale, gate)` x 2; `x += g1*Attn(LN(x)(1+s1)+b1)`, `x += g2*MLP(LN(x)(1+s2)+b2)`;
`cond = MLP(sin/cos(step)) + MLP(hour/24) + MLP(doy/365)`. Attention: 24 heads x 64, 18x18 windows over the 90x180 token grid,
odd blocks cyclic-shifted by 9 with the stored −100 mask, **1-D RoPE on the window-flattened index** (not 2-D), q and k scaled by d^-1/4;
MLP is a gated GELU (`a*gelu(b)`, `fc1` 1536→8192 no bias); LayerNorm is the *unbiased-variance* form (eps 1e-6). All of this was recovered
from the graph and then verified numerically (below). 2,474,861,958 parameters + buffers `mean/std [78,1,1]` and `const [6,721,1440]`
(parameter count equals the ONNX weight elements exactly; the 22 `attn_mask` initialisers are derived by `make_shift_mask` and
tested equal to the ONNX tensors).

```python
from weatherai.models.fuxi_ens import load_official, FuXiENSNoise, FuXiENS_lite, FuXiENSConfig
model = load_official("/workspace/ckpt/fuxi_ens/fuxi_ens.onnx")          # strict state-dict load, weights memory-mapped (low RAM)
g = torch.Generator().manual_seed(0)
noise = FuXiENSNoise.sample(model.cfg, 1, g)                             # 2 dropout masks + latent normal noise, all explicit
y = model(x, step, hour, doy, noise)                                     # x: [1,2,78,721,1440] physical units; step=t, hour=h/24, doy=min(365,doy)/365
y2 = model(x, step, hour, doy, generator=torch.Generator().manual_seed(1))  # another ensemble member
lite = FuXiENS_lite()                                                    # ~2.1 M params, same code, trainable (see tests)
```

**Random ops in the exported graph (handled):** the ONNX has no eval switch — `RandomUniformLike` (×2: dropout p=0.2 on the patch embedding
`drop0` and on the output of `dist_p.layers.0` `drop1`, keep-prob 0.8, scale 1/0.8) and `RandomNormalLike` (latent noise) are always active, so
two ONNX runs differ. For a deterministic comparison `scripts/fuxi_ens_fixed_noise_onnx.py` rewrites the 108 MB graph so these three tensors
are graph *inputs* (no op is changed), and both runtimes receive identical tensors (torch generator seed 1234). `FuXiENSNoise` is the
corresponding explicit input of the PyTorch model; `FuXiENS.forward` without `noise` draws all three from a `torch.Generator`.

**Verified (CPU, fp32; `scripts/fuxi_ens_ort_segments.py` + `scripts/fuxi_ens_verify.py`; official `input.nc` 2018-09-13 18Z/2018-09-14 00Z,
step=0, fixed noise):** one full forward (all 14 + 42 blocks), strict load of all 498 initialisers. Stage-by-stage vs onnxruntime 1.30 (max |Δ| / max |ref|):
normalised input 0 (exact), patch embed 5e-7, dist_p layer 0 1.8e-6, dist_p out 1.4e-5, mean/logvar 4.7e-7 / 1.2e-6, sample 1.2e-7, decoder input `z` 6e-8,
decoder layers 0–3 ≤ 1.5e-6, layer 4–5 1.7e-4–2.3e-4 (a few outlier tokens, rel-RMSE 1e-5), decoder out 8e-5. **Final output** (physical units, all 78 channels × 2 steps):
max |Δ| = 1.66 on |values| up to 2.06e5 (geopotential; max |Δ|/max|ref| = 8.1e-6); per-channel max |Δ| / channel-std ≤ 3.3e-3 (precipitation `tp`, whose
`exp(clip)-1` amplifies logit errors) and ≤ 1.8e-3 for every other channel; relative RMSE ≤ 1.1e-5 for every channel; NaN (SST land) positions identical; the pass-through
step is bit-identical. That is fp32-rounding level for a 2.5 B-parameter network, **not** bit-exactness.

**HF GPU (ZeroGPU Space, NVIDIA RTX PRO 6000 Blackwell MIG 2g.48gb, torch 2.13+cu130; `scripts/gpu_smoke.py fuxi_ens`): PASS.** The 9.9 GB weights were fetched *on the Space*
from Zenodo with `scripts/fetch_fuxi_ens.py` (parallel HTTP range requests + local inflate: ≈ 430–480 s versus ≈ 3 MB/s ≈ 55 min for a single stream; `/tmp` has ≈ 3 TB free) and strict-loaded into CPU RAM
(0.8 s, memory-mapped); then, inside the 120 s ZeroGPU window, moved to the GPU in **fp32 (no reduced precision was needed)**: 3 s, 11.5 GB resident, 16.0 GB peak, one forward 2.5 s.
Same official `input.nc` and the same fixed noise (torch CPU generator, seed 1234) as the CPU run; compared on the committed stride-12 slice of the onnxruntime output (all 78 channels × both steps):
max |Δ| = 0.078 (max |Δ|/channel-std = 5.9e-4, tolerance 1e-2·std); step-0 pass-through equal; a second member with different noise differs. TF32 disabled. The Space also passes the 10 synthetic unit tests (2 file-dependent tests skipped there). Only this single forward step was run on the GPU (no 15-day rollout, no bf16/fp16).
Caveat: the Space weights are preloaded in a background thread (`fetch_fuxi_ens` endpoint) because a redeploy wipes `/tmp` and downloading does not fit into one 120 s GPU call.

## StormCast — native PyTorch regression U-Net + EDM diffusion U-Net (official weights load; bit-identical to PhysicsNeMo on CPU)

`weatherai/models/stormcast` re-implements NVIDIA's StormCast v1 (ERA5 → HRRR, 3 km, 512×640 grid, 99 state + 26 conditioning + 2 invariant channels)
without any PhysicsNeMo dependency. Weights: HF [`nvidia/stormcast-v1-era5-hrrr`](https://huggingface.co/nvidia/stormcast-v1-era5-hrrr) (Apache-2.0, ungated; not committed here).

```python
from weatherai.models.stormcast import StormCast_lite, load_official
from weatherai.models.stormcast.convert import download

m = StormCast_lite()                                      # tiny trainable config (64x64, 8 state channels), same code path
y = m(x_state, x_cond)                                    # one 1-hour step: regression mean + 18/4-step Heun diffusion residual

download("/workspace/ckpt/stormcast")                     # ~800 MB from the Hub (HF_TOKEN only needed if you are rate limited)
model = load_official("/workspace/ckpt/stormcast")        # strict load of both networks + means/stds/invariants
nxt = model(x_hrrr, x_cond_on_hrrr_grid)                  # (B,99,512,640) physical units, (B,26,512,640) coarse conditioning interpolated to HRRR
mean_only = model(x_hrrr, x_cond_on_hrrr_grid, deterministic_only=True)   # regression mean without the diffusion residual
```

* **What is verified (CPU, fp32, real official checkpoint, 512×640):** the native `StormCastUNet` and `EDMPrecond` match the PhysicsNeMo 2.2.2 modules loaded from the *same* `.mdlus` state dicts with **max|Δ| = 0.0** (regression net, and the EDM denoiser at σ = 0.002 / 0.5 / 5 / 800; random-normal inputs). Strict loading works (only the two empty `device_buffer` tensors are dropped); parameter counts 78,588,387 and 121,058,275.
  Small random-weight parity (64×64) and the 6-step Heun sampler against `physicsnemo.diffusion.samplers.sample` + `EDMNoiseScheduler` also pass (`tests/parity/stormcast`).
  Quirk reproduced on purpose: PhysicsNeMo's eval-mode GroupNorm uses the **unbiased** variance.
* **HF ZeroGPU (RTX PRO 6000 Blackwell MIG 2g.48gb, torch 2.13+cu130, 2026-10-02): PASS** — official weights downloaded on the Space, strict-loaded in 2.7 s; regression 512×640 forward 0.3–0.5 s; relative error vs the stored CPU reference slices 1.7e-6 (regression) / 7.7e-6 (EDM σ = 5) with TF32 off (8e-4 / 6.5e-3 with default TF32); a full step (regression + 18 Heun steps = 35 denoiser evaluations) took 10.3 s; peak GPU memory 5.2 GB; different seeds give different samples.
* **NOT verified:** forecast skill on real data — no real HRRR/GFS/ERA5 case was run (inputs were random tensors at physical mean/std), no multi-hour rollout, no comparison with the Earth2Studio pipeline end-to-end, no reproduction of paper scores, no bf16/fp16, no training of the official-size networks (only the lite config was trained for gradient flow). The HRRR/ERA5 data pipeline (hybrid-level fields, interpolation of the global conditioning to the HRRR grid) is not included.

## ArchesWeatherGen — native PyTorch 3D-attention forecaster + flow-matching generator (official weights load; parity with geoarches)

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

## ACE2-ERA5 — native PyTorch spherical Fourier neural operator (official weights load; matches `fme` rollouts)

`weatherai/models/ace2` re-implements Ai2's ACE2-ERA5 (180×360 Gauss–Legendre grid, 44 inputs → 50 outputs, 6 h step) without depending on `fme` or `torch-harmonics`:
`sht.py` (real SHT/iSHT on Gauss–Legendre and equiangular grids), `sfno.py` (SFNO: 8 blocks, embed 384, instance norm, per-degree complex channel mixing = Driscoll–Healy convolution, inner linear skip + outer identity skip, big skip), `model.py` (the stepper around the network) and `convert.py`.
Weights: HF [`allenai/ACE2-ERA5`](https://huggingface.co/allenai/ACE2-ERA5) (Apache-2.0, not committed).

```python
from weatherai.models.ace2 import download, load_official
from weatherai.models.ace2.data import load_case

download("/workspace/ckpt/ace2")                       # 1.8 GB checkpoint + 2020 initial conditions + forcing (0.55 GB)
m = load_official("/workspace/ckpt/ace2")              # strict-load; 455,831,040 network parameters
ic, forcing, times = load_case("/workspace/ckpt/ace2", ic_index=5, n_steps=32)   # 2020-06-01T00Z, 8 days
outs = m.rollout(ic, forcing, 32)                      # list of dicts name -> [1, 180, 360], physical units (all 50 outputs)
```

* **Stepper (taken from `fme`'s `step_with_adjustments` + the checkpoint's config, not guessed):** normalise → SFNO → denormalise → clamp the 16 `force_positive` fields → pin the global-mean dry-air surface pressure to its initial value (float64) → close the global moisture budget by rescaling precipitation and recomputing the advective tendency → overwrite surface temperature where `round(ocean_fraction)==1` with the next-step forcing. `DSWRFtoa` is read at t+1.
* **Verified (CPU fp32):** SHT/iSHT equal `torch_harmonics` 0.8.0 (max relative difference ≤3e-8 forward, 0 inverse; Gauss–Legendre 180×360, equiangular and small grids); a random-weight lite SFNO matches `fme`'s `SphericalFourierNeuralOperatorNet` to 1e-5; with the official checkpoint a **12-step (3-day) rollout from the shipped 2020-01-01 initial condition matches the official `fme` Stepper**: max relative error over all 50 output fields 2.6e-7 after step 1, 4.6e-6 after step 2, 7.9e-5 after step 12 (float32 round-off growth; surface pressure differs by ≤0.03 Pa). Unit tests: conservation of dry air and global moisture budget by construction, SST prescription, gradients.
* **Real case (CPU, one 8-day rollout from 2020-06-01T00Z, lat-weighted RMSE vs WeatherBench2 ERA5 interpolated to 1.5°; ACE2 vs persistence):** day 1: h500 7.27 vs 56.9 m, TMP850 0.739 vs 2.86 K, TMP2m 0.687 vs 1.76 K; day 3: h500 17.8 vs 92.3 m, TMP850 1.35 vs 3.96 K, TMP2m 1.09 vs 2.35 K; day 7: h500 53.3 vs 112 m, TMP850 2.62 vs 4.77 K, TMP2m 1.98 vs 3.1 K. Global dry air drifted by 3.2e-05 Pa over 8 days; global-mean precipitation 2.90 mm/day; CPU time 236 s for 32 steps.
* **HF ZeroGPU (RTX PRO 6000 Blackwell MIG 2g.48gb, torch 2.13+cu130, 2026-10-02): PASS** — strict load 3.6 s; relative error vs the stored CPU slices 1.5e-06 (1 step) / 6.5e-06 (4 steps) with TF32 off; 32-step (8-day) rollout 3.53 s; dry-air drift over 8 days 3.1e-05 Pa; peak GPU memory 3.1 GB; GPU RMSEs reproduce the CPU report (day-1 h500 7.266 m, day-7 h500 53.278 m).
* **NOT verified:** multi-year / climate-scale stability and the paper's AMIP-style forced-response results (the repository's 100-year inference was not run), skill statistics over more than one case, comparison with the paper's scores, other ACE2 checkpoints (ACE2-SOM, ACE2-EAMv3), training, bf16/fp16. The forecast case uses the interpolation of 1° output to ERA5's 1.5° grid, which smooths slightly. The native SHT is O(L²) memory per transform table (about 24 MB per direction in fp32).

## NeuralGCM precipitation — official checkpoints via official code, trainable loss

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

## GenCast — native PyTorch denoiser (official architecture, official Mini weights load) + EDM/DPM-Solver++ sampler

GenCast (*Nature* 2025) is a conditional EDM diffusion model whose denoiser is a GraphCast-style
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

Aardvark Weather (*Nature* 2025; official code CC0) = observation **encoder** → **processor** → station **decoder**.
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

## WeatherNext Cyclones Mini — native PyTorch network (official Mini weights load), JAX wrapper kept as oracle

WN-C / WeatherNext 2 (FGN, a *Functional Generative Network*) is the same family as GenCast/GraphCast (grid → icosahedral mesh → 16-layer k-hop
transformer → grid) but with its own wiring: instead of a diffusion noise level, a 32-channel N(0,1) vector conditions every normalisation, the
encoders/decoder are one big split-input / split-output linear over all variables, and the GNN edge MLPs use the "pre-gather matmul" form.
`weatherai.models.weathernext_cyclones` now contains an **explicit native PyTorch port of that network** (`network.py`; transformer, attention
and graph code are shared with the GenCast port), with the official `WeatherNextCyclones_Mini_<2024` weights converted 1:1 and
`WeatherNextCyclonesWrapper` (official JAX, Python ≥ 3.12) retained only as a test/oracle path.

```python
import numpy as np, torch
from weatherai.models.weathernext_cyclones import (
    WeatherNextCyclonesNet_from_official, stack_grid_inputs, stack_mesh_inputs, unstack_outputs, download_checkpoint)

lat = np.arange(-90, 90.5, 1.0, dtype=np.float32); lon = np.arange(0, 360, 1.0, dtype=np.float32)   # official 1° grid, 181x360
net = WeatherNextCyclonesNet_from_official(download_checkpoint(), lat, lon, attention_type="triblockdiag")   # 56.7 M params, strict load
# inputs: *normalised* fields in the official layout (see stack_grid_inputs), noise ~ N(0,1) of shape (batch, 32)
gx = stack_grid_inputs(inputs, forcings, (181, 360)); mx = stack_mesh_inputs(inputs, forcings)
y = net(gx, mx, torch.randn(1, 32))                  # (B, H*W, 101) one 6 h step, all 29 target variables
fields = unstack_outputs(y, net.cfg, (181, 360))     # {name: (B,1,[13,]181,360)}
```

Trainable, config-driven variant (same modules, ~0.2 M params, fair-CRPS ensemble loss as in the official `fgn.Predictor`):
`WeatherNextCyclonesNative_lite()` → `m.loss(grid_x, mesh_x, target, num_samples=2)`; example in `tests/models/weathernext_cyclones/test_wnc_native.py`.

**Status — verified:**
* All 376 official parameter arrays (56 684 133 weights) are consumed exactly once by `convert_official_params` and strict-load.
* Network vs the **official JAX `ForwardPass`** (official Mini weights, random inputs + noise, plain `mha`; JAX/Haiku run on CPU):
  * float64 reference, mesh_splits=2, k_hop=3, 13×24 grid, batch 2 (the *reference* is run with x64 and its bf16-oriented fp32 up-casts disabled, so rounding
    cannot hide a structural error): native vs official max |Δ| **2.2e-12** (dense attention) / **2.1e-12** (banded attention) — test tolerance 1e-9.
  * float32 official run, same small config: max |Δ| 1.4e-3 (outputs up to ~26; official-JAX f32 vs official-JAX f64 differ by 9.3e-4) — test tolerance 5e-3.
  * **trained size** (mesh_splits=5 → 10 242 mesh nodes, k_hop 16, 181×360 grid, banded attention, float32): max |Δ| 2.8e-3 on outputs up to ~25
    (per variable ≤ 2.9e-3; probabilities 3.5e-6) — test tolerance 5e-3 (the 65 MB reference is not committed; regenerate with `scripts/wnc_forward_reference.py full`).
  * graph edge sets (ball query grid→mesh 101 892 edges; closest triangle mesh→grid 195 480 edges at full size, 576/936 at the small size) equal the official arrays.
* HF ZeroGPU (RTX PRO 6000 Blackwell MIG, torch 2.13+cu130) `--pretrained` smoke: PASS — strict load, float64 max |Δ| 1.9e-12 / 1.6e-12 (dense / banded), float32 max |Δ|
  7.6e-4 / 1.2e-3 (TF32 off) and 8.3e-4 / 1.3e-3 (TF32 default), peak 533 MB; Space unit tests: 5 passed, 10 skipped (the skips need local official ckpt or the JAX package; parity is covered by the smoke).

**Not verified / not ported:** the `fgn.Predictor` wrapper stack (input/residual normalisation constants, NaN cleaning of SST, autoregressive rollout, the
2-sample training ensemble), data pipeline/ERA5-HRES loading, the **cyclone tracker** (`cyclones/direct_tracker.py`) and IBTrACS pipeline; multi-step forecasts of the
native model; the 0.25° / WN2 / `<2023` checkpoints; skill. The comparison is network-level on random inputs, not forecast-level on real data. The official TPU
`splash_mha` kernel is not used (neither on our side nor in the reference). Stochastic sampling: noise is drawn iid N(0,1) as in the official `gaussian_noise_generator`.

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
- Adapted code origin (MIT): https://github.com/lizhuoq/WeatherLearn

### Papers / 论文

Journal versions were checked against Crossref (DOIs resolve); preprints are listed for reference.
期刊版本已对照 Crossref 核实（DOI 可解析）；预印本仅供参考。

- Pangu-Weather — *Nature* 2023, [doi:10.1038/s41586-023-06185-3](https://doi.org/10.1038/s41586-023-06185-3) · preprint [arXiv:2211.02556](https://arxiv.org/abs/2211.02556)
- FuXi — *npj Climate and Atmospheric Science* 2023, [doi:10.1038/s41612-023-00512-1](https://doi.org/10.1038/s41612-023-00512-1) · preprint [arXiv:2306.12873](https://arxiv.org/abs/2306.12873)
- FengWu — preprint [arXiv:2304.02948](https://arxiv.org/abs/2304.02948) (2023) · journal version (retitled) *Communications Earth & Environment* 2025, [doi:10.1038/s43247-025-02502-y](https://doi.org/10.1038/s43247-025-02502-y)
- GraphCast — *Science* 2023, [doi:10.1126/science.adi2336](https://doi.org/10.1126/science.adi2336) · preprint [arXiv:2212.12794](https://arxiv.org/abs/2212.12794)
- NeuralGCM — *Nature* 2024, [doi:10.1038/s41586-024-07744-y](https://doi.org/10.1038/s41586-024-07744-y)
- GenCast — *Nature* 2025, [doi:10.1038/s41586-024-08252-9](https://doi.org/10.1038/s41586-024-08252-9)
- Aurora — *Nature* 2025, [doi:10.1038/s41586-025-09005-y](https://doi.org/10.1038/s41586-025-09005-y)
- Aardvark Weather — *Nature* 2025, [doi:10.1038/s41586-025-08897-0](https://doi.org/10.1038/s41586-025-08897-0)
- WeatherNext Cyclones (WN-C) — *Nature* 2026, [doi:10.1038/s41586-026-10953-2](https://doi.org/10.1038/s41586-026-10953-2) · FGN report [arXiv:2506.10772](https://arxiv.org/abs/2506.10772) (preprint)
- FuXi-ENS — *Science Advances* 2025, [doi:10.1126/sciadv.adu2854](https://doi.org/10.1126/sciadv.adu2854)
- StormCast — *Science Advances* 2026, [doi:10.1126/sciadv.adv0423](https://doi.org/10.1126/sciadv.adv0423)
- ArchesWeatherGen — *Science Advances* 2026, [doi:10.1126/sciadv.adx2372](https://doi.org/10.1126/sciadv.adx2372)
- ACE2 — *npj Climate and Atmospheric Science* 2025, [doi:10.1038/s41612-025-01090-0](https://doi.org/10.1038/s41612-025-01090-0)
- NeuralGCM precipitation — *Science Advances* 2026, [doi:10.1126/sciadv.adv6891](https://doi.org/10.1126/sciadv.adv6891)
- FuXi-DA (planned) — *npj Climate and Atmospheric Science* 2025, [doi:10.1038/s41612-025-01039-3](https://doi.org/10.1038/s41612-025-01039-3)
- NowcastNet (planned) — *Nature* 2023, [doi:10.1038/s41586-023-06184-4](https://doi.org/10.1038/s41586-023-06184-4)

## Roadmap / 后续

- [ ] Pretrained weight download helpers (no invented checkpoints in-repo)
- [ ] Training / finetune recipes (kept out of v0.1 for a clean starter zoo)
- [x] GraphCast small-scale JAX parity (mesh / MLP / one GraphNet layer)
- [x] GraphCast stage parity (graphs, Grid2Mesh, processor N≤16, Mesh2Grid) vs JAX + PhysicsNeMo (tiny cases; NV processor aggregation is an inherent, documented difference)
- [ ] GraphCast checkpoint loading / full-scale parity
- [ ] More models and evaluation utilities

### Planned models / 计划加入的模型

Citations checked against Crossref; "code" links point to the authors' official releases (not vendored here). Batch 2 (rows after FuXi-DA, selected 2026-10-02) is **not blocked**: official code and/or weights are public. Plan: [docs/model_status.md](docs/model_status.md#batch-2--nature--science-family-models-plan); survey: [docs/candidate_models_nature_science.md](docs/candidate_models_nature_science.md).

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
| **ORCA-DL** | *Science Advances* 2025, [doi:10.1126/sciadv.adu2488](https://doi.org/10.1126/sciadv.adu2488) | [OpenEarthLab/ORCA-DL](https://github.com/OpenEarthLab/ORCA-DL) (no LICENSE file) · HF dataset `JayKuo/ORCA-DL-data` — *planned, batch 2* |
| **TropiCycloneNet** | *Nature Communications* 2025, [doi:10.1038/s41467-025-61087-4](https://doi.org/10.1038/s41467-025-61087-4) | [xiaochengfuhuo/TropiCycloneNet](https://github.com/xiaochengfuhuo/TropiCycloneNet) (no LICENSE file) · [Zenodo 15024028](https://doi.org/10.5281/zenodo.15024028) (CC-BY-4.0) — *planned, batch 2* |
| **FuXi-S2S** | *Nature Communications* 2024, [doi:10.1038/s41467-024-50714-1](https://doi.org/10.1038/s41467-024-50714-1) | ONNX only · [Zenodo 15718402](https://zenodo.org/records/15718402) (**CC-BY-NC-ND-4.0**, weights not redistributed) — *planned, batch 2* |
| **UniCM** | *Nature Machine Intelligence* 2026, [doi:10.1038/s42256-026-01245-5](https://doi.org/10.1038/s42256-026-01245-5) | [tsinghua-fib-lab/UniCM-Global-Climate-Modes](https://github.com/tsinghua-fib-lab/UniCM-Global-Climate-Modes) (MIT) · [Zenodo 19173780](https://doi.org/10.5281/zenodo.19173780) — *planned, batch 2* |
| **GenFocal** | *Nature Machine Intelligence* 2026 (online 2026-09-28), [doi:10.1038/s42256-026-01308-7](https://doi.org/10.1038/s42256-026-01308-7) | [swirl-dynamics/genfocal](https://github.com/google-research/swirl-dynamics/tree/main/swirl_dynamics/projects/genfocal) (Apache-2.0, JAX) · [Zenodo 21285802](https://zenodo.org/records/21285802) (CC-BY-4.0, 8.7 + 39 GB) — *planned, batch 2* |

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
- [~] Batch 2 (one commit per model; status in docs/model_status.md): **done** — StormCast (parity + HF GPU smoke, no real-data run), ArchesWeatherGen (geoarches parity, HF GPU smoke, one real case), ACE2 (fme rollout parity, HF GPU smoke, one 8-day real case), NeuralGCM precipitation (official checkpoints run, trainable loss, one real case, HF GPU smoke; no native-torch port); **remaining** — ORCA-DL, TropiCycloneNet, FuXi-S2S, UniCM, GenFocal

## Disclaimer

Research / educational code. Forecast skill depends on data, training, and
weights — this repo ships **architectures**, not operational NWP replacements.
