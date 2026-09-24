# Examples

Minimal stubs. Full training / ERA5 pipelines are intentionally omitted from
v0.1 so the repository stays a clean **model zoo**.

```python
import torch
from weatherai.models import FengWu_lite, FuXi, Pangu_lite

# FengWu lite forward
m = FengWu_lite()
y = m(torch.randn(1, 69, 64, 128))
print(y.shape)

# Construct other models (shapes depend on each architecture’s defaults)
_ = FuXi
_ = Pangu_lite
```

See the root [README](../README.md) and [docs/fengwu_specs.md](../docs/fengwu_specs.md).
