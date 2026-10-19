# GenFocal — official pretrained climate debiasing and downscaling

[Paper](https://doi.org/10.1038/s42256-026-01308-7) · [Author code](https://github.com/google-research/swirl-dynamics/tree/0beb5d20a5b31969e12069ba7d86fc0537e08528/swirl_dynamics/projects/genfocal) · [Published checkpoints](https://zenodo.org/records/21285802)

`GenFocal` uses the author's unchanged JAX/Flax implementation and published
parameters. `GenFocal_lite` is a separate, untrained PyTorch approximation.
The main entry never falls back to lite or initializes random replacement weights.

```python
from weatherai import GenFocal, GenFocal_lite
sr = GenFocal.from_pretrained("super_resolution")
flow = GenFocal.from_pretrained("debiasing")
# Author-normalized arrays and conditioning; this is a network evaluation:
y = sr(x, sigma, cond)
# Optional explicitly untrained experimentation:
lite = GenFocal_lite(channels=4)
```

Use Linux/Python 3.12 and the pinned environment in
[hf_official/requirements.txt](../../hf_official/requirements.txt).
Set `WEATHERAI_CACHE` to a large local directory outside synchronized folders.
Author source is fetched at commit `0beb5d20a5b31969e12069ba7d86fc0537e08528`
and checked for modifications. Published OCDBT tensors are fetched directly
from the public `genfocal` GCS bucket via HTTP range reads, with shape, dtype
and SHA-256 records. Optimizer state and duplicate checkpoints are excluded.
Flax module names containing literal dots are restored from Orbax key metadata.

| Stage | Released checkpoint (step 300000) | Parameter tree |
|---|---|---|
| ReFlow debiasing | `debiasing/checkpoints/reflow_uvit` | raw `params`, as in author's sampling code |
| CONUS super-resolution | `super_resolution/checkpoints/conus_7d_t2m_w10m_q1000_msl` | `ema_state.ema`, as in author's inference |

Correction to the earlier documentation: the **released debiasing checkpoint**
uses channels **768, 768, 768, 1024, 1280, 1536**, not the 128-channel example
configuration. Its published run config is loaded from the checkpoint directory.
The raw debiasing tree contains 2,656,553,626 parameters (10.63 GB).
SR uses the author's `conus_sample.yaml` (128/256/384/512 channels, 84 two-hourly
steps, four output variables). This distinction is essential for weight compatibility.

This entry loads/evaluates each official network. Full climate trajectories also
require the author's input normalization, ReFlow integration, diffusion sampling,
regional baseline and residual statistics, overlap handling and physical-unit
reconstruction. A single denoiser/vector-field call is not a completed forecast.
Both complete pretrained networks were executed on HF ZeroGPU: ReFlow 146.13 s
(synthetic normalized input), CONUS SR 52.99 s (real ERA5 conditioning, one denoiser
call). All outputs were finite. See [current validation evidence and limitations](../research_20261019.md).

Official code is Apache-2.0; released checkpoints are CC-BY-4.0. Original weights
remain with the authors and are not redistributed in Git. Previous lite loss/RK4
parity results remain in `docs/results/genfocal_parity.json`; they do not establish
full pretrained model parity or climate skill.
