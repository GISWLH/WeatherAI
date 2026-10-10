# FuXi — cube embedding + Swin-V2 U-Transformer (PyTorch)

[← Model zoo](../../README.md) · [All model pages](README.md) · **Paper:** *npj Climate and Atmospheric Science* 2023, [doi:10.1038/s41612-023-00512-1](https://doi.org/10.1038/s41612-023-00512-1)

`weatherai.models.fuxi` (adapted from [WeatherLearn](https://github.com/lizhuoq/WeatherLearn), MIT): space-time cube embedding of
two input steps, a Swin-V2 U-Transformer and a fully connected head. Architecture only — the official FuXi weights are
released as ONNX and are not loaded here. The ensemble and sub-seasonal members of the family are separate, verified ports:
[FuXi-ENS](fuxi_ens.md) and [FuXi-S2S](fuxi_s2s.md).

```python
from weatherai.models import FuXi
m = FuXi()
```

Tests: `tests/models/fuxi`.
