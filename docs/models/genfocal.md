# GenFocal — climate debiasing and generative regional downscaling

## Update: two-stage interface, 19 October 2026

`from weatherai import GenFocal, GenFocal_lite` now exposes a composition of
the ReFlow debiasing network and a conditional EDM super-resolution network.
The native SR adds noise-level preconditioning, weighted denoising loss and
a single-segment Heun sampler. Both native U-ViTs remain approximations;
they are **not compatible with official Orbax checkpoints**. Default lite
weights are random. This is an architectural/testing implementation, not a
trained regional climate product.

```python
from weatherai import GenFocal_lite
model = GenFocal_lite(channels=4).eval()
# x, cond: normalized B,T,lon,lat,C tensors, with channel:mean/channel:std.
result = model.downscale(x, cond, output_size=(8, 32, 16),
                         flow_steps=8, diffusion_steps=16)
```

The native interpolation uses bilinear space and replicated time; official
daily/hourly preprocessing, positional embeddings, filtered resizes, local
regional padding, classifier-free guidance, multi-device overlap consensus
and full official network weight parity are not claimed. Physical output
requires an explicit physical baseline plus residual climatology mean/std.

`GenFocalOfficial` in `weatherai.models.genfocal` builds the **actual upstream**
ReFlow or EDM U-ViT from official JSON/YAML configs and optionally restores
Orbax state/EMA. It is lazy and separate from the lite PyTorch API. Inspected
upstream: `google-research/swirl-dynamics@0beb5d20a5b31969e12069ba7d86fc0537e08528`.
The wrapper follows upstream construction/restoration code, but full checkpoint
inference has not been executed here. Use upstream's recommended Linux/Python
3.12 environment; the existing Space uses Python 3.10, below upstream's >=3.11.

The previous archive-size obstacle has an alternative: the public GCS prefix
`gs://genfocal/super_resolution/checkpoints/conus_7d_t2m_w10m_q1000_msl/`
contains 12 actual files, **2.022 GiB** (2.171 GB), excluding directory markers.
`python scripts/fetch_genfocal.py` inventories it; `--download --out PATH`
downloads that one subtree with size/checksum checks. The verified manifest is
[recorded here](../results/genfocal_conus_checkpoint_manifest.json). No old
content was deleted and no weights have been loaded in the native models.

The complete public archives remain [Zenodo 21285802](https://zenodo.org/records/21285802):
8,737,960,532-byte SR and 39,143,363,979-byte debiasing tarballs (CC-BY-4.0).
[Code archive](https://zenodo.org/records/21268077), Apache-2.0.
See [three-paper integration and test scope](../research_20261019.md).

## Historical flow-core verification (2 October 2026)

[← Model zoo](../../README.md) · [All model pages](README.md) · **Paper:** *Nature Machine Intelligence* 2026, [doi:10.1038/s42256-026-01308-7](https://doi.org/10.1038/s42256-026-01308-7)

`weatherai/models/genfocal` ports the debiasing-stage maths of GenFocal (*Nature Machine Intelligence* 2026): conditional rectified-flow (ReFlow) loss `mean(w·((x₁−x₀)−v(x_t,t,cond))²)` with `x_t = t·x₁ + (1−t)·x₀`, the RK4 sampler (forward and reversed flow) with output de-normalisation, plus a **lite** trainable 3-D U-ViT flow network (`GenFocalNet`: FiLM residual blocks, axial spatial/temporal attention with QK-norm, periodic longitude, channel-mean/std conditioning, zero-init output). The network is a structural stand-in: it is **not** weight-compatible with the official Flax `UNet3d`. The super-resolution stage (EDM denoiser) is not ported. Code: GitHub google-research/swirl-dynamics (Apache-2.0); checkpoints Zenodo 21285802 (CC-BY-4.0), not downloaded.

```python
from weatherai.models.genfocal import GenFocalNet, GenFocalNetConfig, reflow_loss, sample_flow
net = GenFocalNet(GenFocalNetConfig.lite(4))                                    # inputs (batch, time, lon, lat, channels)
loss = reflow_loss(net, x0, x1, {"channel:mean": cm, "channel:std": cs})        # x0 = climate model, x1 = ERA5 (normalised)
y = sample_flow(net, x0, {"channel:mean": cm, "channel:std": cs}, num_steps=8)  # RK4, official schedule
```

* **Verified (CPU, fp64 native vs fp32 official; oracle = unchanged `ConditionalReFlowModel.loss_fn` and `inference_utils.sampling_from_batch` on a tiny Flax network, stored reference; `scripts/genfocal_make_ref.py`, `scripts/genfocal_parity.py`, `docs/results/genfocal_parity.json`):** loss rel. error ≤5.8e-7 (uniform / logistic time draw, with and without weighted norm); 4- and 8-step RK4 samples, forward and reversed, rel. L2 ≤7.5e-8 (max abs ≤1.3e-6); 6 unit tests (also periodic-longitude equivariance and a training-loss decrease of the lite net).
* **Official-code quirks reproduced:** `tspan = arange(0, 1, 1/N)` has N points, so the sampler takes N−1 steps and returns the state at t = 1−1/N, not t = 1 (`to_end=True` integrates to 1); `lognormal_sampler` is `logistic(std·u + mean)`, not log-normal.
* **HF ZeroGPU (2026-10-02), random weights:** **PASS** — lite net (3.3 M params, 8×64×32×10, 64-128-128 ch) train step + backward 0.23 s, 8-step RK4 sample 0.26 s, peak 1.0 GB; lite output vs CPU max abs 2.5e-3 (not isolated; default cuDNN conv precision suspected).
* **Blocker for the real model:** Zenodo 21285802 holds `super_resolution_checkpoints.tar.gz` (8.74 GB) and `debiasing_checkpoints.tar.gz` (39.1 GB), Orbax/OCDBT (tensorstore) checkpoints of the Flax `UNet3d` / `UNet` (e.g. `conus_7d_t2m_w10m_q1000_msl/300000`, 491 MB blobs); the debiasing config is a 6-level 128-channel 3-D U-ViT (`resize_to_shape` (256, 128), 8 snapshots, 10 channels, 128 heads). Not done: the 39 GB archive exceeds the box disk (~25 GB free) and the Space's storage; porting the official `UNet3d` / `FilteredResize` / `InterpConvMerge` / axial attention to read those weights; ERA5/LENS2 inputs (Google Cloud buckets). No real-data debiasing or downscaling was run.
* **NOT verified / not done:** the official network (no weight load, no parity), the super-resolution stage, real-data skill, the evaluation scripts.
