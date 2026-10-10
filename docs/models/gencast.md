# GenCast — native PyTorch denoiser (official architecture, official Mini weights load) + EDM/DPM-Solver++ sampler

[← Model zoo](../../README.md) · [All model pages](README.md) · **Paper:** *Nature* 2025, [doi:10.1038/s41586-024-08252-9](https://doi.org/10.1038/s41586-024-08252-9)

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
(same code path, larger graphs); the lite config is not a trained model. Details: [docs/model_status.md](../model_status.md).
