# Gray-swan translocation: forecast diagnostics

[Model pages](README.md) · [Three-paper integration](../research_20261019.md)

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
| FuXi | [Code and ONNX weight links](https://github.com/tpys/FuXi) |
| AIFS | [ecmwf/aifs-single-1.0](https://huggingface.co/ecmwf/aifs-single-1.0) |
| Pangu / FourCastNet-v2 | [ECMWF ai-models integrations](https://github.com/ecmwf-lab/ai-models) |
| GraphCast fine-tuning | [csubich/graphcast](https://github.com/csubich/graphcast) |
| Dynamically similar event tracking | [cyclonetracking](https://github.com/alexcrawford0927/cyclonetracking) |
| Paper ERF notebook and data | [Zenodo 18111128](https://zenodo.org/records/18111128) |

Zenodo lists the 19,427-byte notebook `graphcast-ERF-cleaned.ipynb`, README,
and three ERF data ZIP files (~3.14, 2.03 and 2.41 GB). These are diagnostics,
not newly released model checkpoints. Model licenses follow each backbone.

```python
from weatherai import Translocation, Translocation_lite
model = Translocation_lite()  # random-weight small GraphCast
# For scientific use: model = Translocation(your_differentiable_forecaster).eval()
result = model.effective_receptive_field(x, channel=0, location=(3, 4))
# Or region_mask=(H,W): select the strongest predicted value inside the region.
signed_erf = result['gradient']  # B,C,H,W, with original signs preserved
```

The wrapper delegates forecasts to its supplied backbone. The diagnostic uses
the derivative of one selected output with respect to every input variable,
as in the paper's Dubai analysis. It does not change/accumulate parameter
gradients. Tests check signed analytical derivatives and a GraphCast directional
finite-difference oracle. Real use requires the correct physical output,
normalization, precipitation channel and coordinates. Lite channel 0 has no
physical variable identity. A nonlocal ERF is not by itself evidence of causal
translocation. The paper's weather reconstruction, event tracking and
fine-tuning ablations have not been reproduced.
