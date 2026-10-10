# Pangu-Weather — 3-D Earth-specific transformer (PyTorch)

[← Model zoo](../../README.md) · [All model pages](README.md) · **Paper:** *Nature* 2023, [doi:10.1038/s41586-023-06185-3](https://doi.org/10.1038/s41586-023-06185-3)

`weatherai.models.pangu` is adapted from [WeatherLearn](https://github.com/lizhuoq/WeatherLearn) (MIT): patch embedding of the
surface (4 + 3 masks) and upper-air (5 variables × 13 levels) fields, a U-shaped stack of 3-D Earth-specific window-attention
blocks with an Earth position bias, and patch recovery. `Pangu` is the paper size (721 × 1440, embed 192); `Pangu_lite`
uses 8 × 8 patches for smaller grids. The same blocks are reused by [CAS-Canglong](canglong.md).

```python
from weatherai.models import Pangu, Pangu_lite
m = Pangu_lite()
surface, upper_air = m(surface, surface_mask, upper_air)   # (B,4,721,1440), (3,721,1440), (B,5,13,721,1440)
```

Not included: the official ONNX weights (no PyTorch checkpoint is published), training recipes. Tests: `tests/models/pangu`.
