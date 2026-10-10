# GenFocal — lite native PyTorch flow core (parity vs official JAX), full checkpoints blocked

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
