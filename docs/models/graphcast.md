# GraphCast — grid ↔ icosahedral-mesh GNN (PyTorch)

[← Model zoo](../../README.md) · [All model pages](README.md) · **Paper:** *Science* 2023, [doi:10.1126/science.adi2336](https://doi.org/10.1126/science.adi2336)

Independent re-implementation of the encode (Grid2Mesh) – process (multi-mesh message passing) – decode (Mesh2Grid) design.
Stage-wise parity against DeepMind's JAX code and NVIDIA PhysicsNeMo at mesh levels 0–1 (graphs, Grid2Mesh, processor
N ≤ 16, Mesh2Grid): [graphcast_parity.md](../graphcast_parity.md). Architecture notes: [graphcast_specs.md](../graphcast_specs.md).
No official checkpoint is loaded and there is no whole-model parity claim.

```python
import torch
from weatherai.models import GraphCast_lite
y = GraphCast_lite()(torch.randn(1, 4, 32, 64))   # tiny mesh (level 1), residual next step
```

Training smoke example: `examples/graphcast/train_smoke.py`.
