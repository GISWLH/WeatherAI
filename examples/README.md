# Examples

Minimal stubs and smoke scripts. Full ERA5 pipelines stay optional so the
repository remains a clean **model zoo**.

```python
import torch
from weatherai.models import FengWu_lite, FuXi, GraphCast_lite, Pangu_lite

# FengWu lite forward
m = FengWu_lite()
y = m(torch.randn(1, 69, 64, 128))
print(y.shape)

# GraphCast lite forward
gc = GraphCast_lite()
print(gc(torch.randn(1, 4, 32, 64)).shape)

_ = FuXi
_ = Pangu_lite
```

Synthetic GraphCast train smoke:

```bash
python examples/graphcast/train_smoke.py --steps 2 --device cpu
```

See the root [README](../README.md), [docs/fengwu_specs.md](../docs/fengwu_specs.md),
and [docs/graphcast_specs.md](../docs/graphcast_specs.md).
