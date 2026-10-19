# Gray-swan translocation: forecast diagnostics

[Model pages](README.md) 路 [Three-paper integration](../research_20261019.md)

Sun et al., **Predicting regional gray swans via translocation: AI weather models
and Dubai's unprecedented 2024 rainfall**, Science Advances, 7 October 2026.
[Paper](https://doi.org/10.1126/sciadv.ady9406).

Classification: extreme-event generalization and interpretability. This paper
does **not** introduce a sixth/new forecasting architecture. It evaluates
GraphCast, FuXi, AIFS, Pangu-Weather and FourCastNet-v2, and investigates
GraphCast/AIFS effective receptive fields (ERFs) and fine-tuning experiments.

Official resources from the paper:

| Resource | Code / model |
|---|---|
| GraphCast | [DeepMind code and weight links](https://github.com/google-deepmind/graphcast) |
| FuXi | [Original cascade code (pinned)](https://github.com/tpys/FuXi/tree/9292fe0692156a01cd3d62bcb427cc3798cf8add), [author ONNX models / sample data](https://doi.org/10.5281/zenodo.10401602) |
| AIFS | [ecmwf/aifs-single-1.0](https://huggingface.co/ecmwf/aifs-single-1.0) |
| Pangu | [Original author code and four ONNX weight links](https://github.com/198808xc/Pangu-Weather), [6h checkpoint](https://drive.google.com/file/d/1a4XTktkZa5GCtjQxDJb_fNaqTAUiEJu4/view) |
| FourCastNet-v2 | [NVIDIA inference code](https://github.com/NVIDIA/earth2mip/blob/main/earth2mip/networks/fcnv2_sm.py), [official NGC v0.2 weights + statistics](https://api.ngc.nvidia.com/v2/models/nvidia/modulus/modulus_fcnv2_sm/versions/v0.2/files/fcnv2_sm.zip) |
| GraphCast fine-tuning | [csubich/graphcast](https://github.com/csubich/graphcast) |
| Dynamically similar event tracking | [cyclonetracking](https://github.com/alexcrawford0927/cyclonetracking) |
| Paper ERF notebook and data | [Zenodo 18111128](https://zenodo.org/records/18111128) |

Zenodo lists the 19,427-byte notebook `graphcast-ERF-cleaned.ipynb`, README,
and three ERF data ZIP files (~3.14, 2.03 and 2.41 GB). These are diagnostics,
not newly released model checkpoints. Model licenses follow each backbone.
The current FuXi repository advertises FuXi 2.1; that newer release must not be
silently substituted for the original cascade used in this study. The table
therefore links the original code snapshot and its author-hosted archive.
Only the GraphCast route was executed for this integration; the other official
backbone artifacts are discovery references, not newly validated implementations.

```python
from weatherai import Translocation
model = Translocation.from_pretrained()  # official 0.25-degree, 13-level GraphCast
inputs, targets, forcings = model.prepare(era5_dataset, lead_time="6h")
forecast = model(inputs, targets * float("nan"), forcings)
erf = model.effective_receptive_field(
    inputs, targets_template=targets * float("nan"), forcings=forcings,
    variable="total_precipitation_6hr",
    location={"batch": 0, "time": -1, "lat": 260, "lon": 220})
```

The main entry downloads the official GraphCast checkpoint and normalization
statistics from `dm_graphcast` and imports unchanged author code at
`08cf73625c9d12bd9aaa038868bcb2fe488f2a22`. This historical revision retains the
`graphcast` package before the upstream repository's WeatherNext reorganization.
Variants are `operational` (0.25°, 13 levels), `37levels` (0.25°, 37 levels), and
`small` (official pretrained 1°, 13 levels; not the paper's main resolution).
Code is Apache-2.0; GraphCast weights retain CC-BY-NC-SA-4.0.

The wrapper order follows the study's released ERF notebook: GraphCast,
Bfloat16Cast, InputsAndResiduals and autoregressive.Predictor. Signed input
sensitivities use JAX reverse-mode AD on a selected physical output. Coordinates,
variable, lead and normalization must match the intended event. ERF alone does
not establish causality. Full 0.25° operational forecasting and the actual Small-model ERF adapter passed
ZeroGPU execution. [Current test evidence](../research_20261019.md).

`Translocation_lite()` retains the independent random-weight PyTorch toy model.
For custom PyTorch diagnostic experiments, import `TranslocationLite` from
`weatherai.models.translocation`. Neither is the pretrained main entry.
The complete Dubai reconstruction, event tracking and fine-tuning ablations
have not been reproduced. No study-specific fine-tuned weights were found in
the cited ERF archive; its large ZIP files contain diagnostic data.
