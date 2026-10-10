# FengWu — multi-modal encode–fuse–decode (PyTorch)

[← Model zoo](../../README.md) · [All model pages](README.md) · **Paper:** *Communications Earth & Environment* 2025, [doi:10.1038/s43247-025-02502-y](https://doi.org/10.1038/s43247-025-02502-y)

`FengWu` follows the 2023 preprint architecture (separate encoders per variable group, a cross-modal fuser, decoders, optional
uncertainty head); `FengWu_lite` is a CPU-friendly default (64 × 128, 69 channels). Architecture notes and status:
[fengwu_specs.md](../fengwu_specs.md), [fengwu_status.md](../fengwu_status.md). No official weights are published.

```python
import torch
from weatherai.models import FengWu_lite
y = FengWu_lite()(torch.randn(1, 69, 64, 128))   # (1, 69, 64, 128)
```
