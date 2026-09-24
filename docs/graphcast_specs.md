# GraphCast architecture specs (Lam et al., Science 2023 / arXiv:2212.12794)

Sources:
- Paper: https://arxiv.org/abs/2212.12794
- Science: https://www.science.org/doi/10.1126/science.adi2336

This document describes WeatherAI’s **independent PyTorch skeleton**
(`weatherai.models.graphcast`). It is **not** a vendored copy of DeepMind’s
JAX GraphCast or NVIDIA PhysicsNeMo GraphCast (Apache-2.0).

---

## 1. Problem / I/O

| Item | Paper GraphCast | WeatherAI skeleton |
|------|-----------------|--------------------|
| Grid | 0.25°, **721 × 1440** | Configurable `img_size` (lite default 32×64) |
| Channels | **227** = 5 surface + 6×37 pressure | Configurable `in_channels` (lite default **4**; zoo often 69) |
| Mapping | \(X^{t-1}, X^{t} \mapsto X^{t+1}\) (6 h) with forcing | Single-frame `(B,C,H,W)` → residual next step |
| Output | Residual on grid | `y = x + Δ` (channel-aligned) |

---

## 2. Network: encode–process–decode on multi-mesh

| Block | Paper | Skeleton in `weatherai/models/graphcast` |
|-------|-------|------------------------------------------|
| Mesh | Refined icosahedron **level 6** (40,962 nodes); **multi-mesh** = union of edges levels 0…6 | `mesh_level` default **2** (162 nodes); lite **1** (42). `use_multi_mesh` flag |
| Grid2Mesh | Bipartite radius edges; 1 GNN step; embed grid+mesh | k-NN bipartite (`g2m_k`); `BipartiteGraphNetBlock` |
| Processor | **16** unshared mesh GNN layers | `processor_layers` (default 4 / lite 2) |
| Mesh2Grid | Bipartite; 1 GNN step | k-NN (`m2g_k`); bipartite block |
| Head | MLP → grid channels | `grid_out` MLP → `out_channels` |
| Engine | JAX + typed GraphNets | **Pure PyTorch + numpy** (no PyG / DGL) |

### Mesh node counts

| Level | Nodes |
|------:|------:|
| 0 | 12 |
| 1 | 42 |
| 2 | 162 |
| 3 | 642 |
| 6 | 40,962 |

Formula: \(N = 10 \cdot 4^{L} + 2\).

### Configurable hyperparameters

| Name | `GraphCast` default | `GraphCast_lite` |
|------|---------------------|------------------|
| `img_size` | (64, 128) | (32, 64) |
| `in_channels` | 69 | 4 |
| `mesh_level` | 2 | 1 |
| `processor_layers` | 4 | 2 |
| `hidden_dim` | 64 | 32 |
| `use_multi_mesh` | True | False |

Paper reference scale (not the default ctor): mesh_level≈6, processor_layers=16, latent≈512, 721×1440, 227 channels.

---

## 3. Gaps vs full paper

- **No** two-frame input / forcing variables / solar radiation clocks
- **No** autoregressive rollout training or curriculum
- **No** latitude-weighted loss / per-variable pressure weighting
- **No** pretrained checkpoints
- Bipartite edges use **k-NN**, not paper radius queries
- Defaults are **smoke-sized**; scale `mesh_level` / `hidden_dim` toward paper for research
- InteractionNetwork MLPs are a simplified faithful sketch, not a line-by-line port

---

## 4. Public API

```python
from weatherai.models import GraphCast, GraphCast_lite
import torch

m = GraphCast_lite()                 # (B,4,32,64)
y = m(torch.randn(1, 4, 32, 64))

m69 = GraphCast_lite(in_channels=69) # FengWu-stack width on tiny grid
y69 = m69(torch.randn(1, 69, 32, 64))
```

---

## 5. JAX submodule parity

Small-scale numerical parity vs DeepMind JAX (mesh level 0/1, MLP+LayerNorm,
one InteractionNetwork layer) is tracked in [graphcast_parity.md](graphcast_parity.md).
**Full-model / checkpoint equality is not claimed.**

