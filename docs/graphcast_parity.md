# GraphCast parity: WeatherAI (PyTorch) ↔ DeepMind JAX ↔ NVIDIA PhysicsNeMo

**Status (2026-09-25).** Tested stage by stage on tiny, seeded, fixed cases:
Grid2Mesh → multi-mesh processor (N = 1, 2, 4, 16) → Mesh2Grid (+ decoder).
Full-model training behaviour and **checkpoint** parity are **not** claimed and
have not been tested. Nothing here loads official weights. The weights are
seeded random values mapped identically into all three implementations.

References (tests only, never vendored into `weatherai/`):

| Framework | Source | Revision used | How it runs here |
|-----------|--------|---------------|------------------|
| DeepMind JAX GraphCast | `google-deepmind/graphcast` (WeatherNext layout, `weathernext/weathernext1_graph/graphcast.py`) | `f2f2c51` (2026-09-04) | real `GraphCast` stage methods `_run_grid2mesh_gnn` / `_run_mesh_gnn` / `_run_mesh2grid_gnn`, real graph builders |
| NVIDIA PhysicsNeMo | `NVIDIA/physicsnemo` (`physicsnemo/models/graphcast`, `physicsnemo/nn/module/gnn_layers`) | `426f755` (2026-09-22) | `GraphCastEncoderEmbedder`, `MeshGraphEncoder`, `GraphCastProcessor`, `GraphCastDecoderEmbedder`, `MeshGraphDecoder`, `MeshGraphMLP`, `utils.graph.Graph`; PyG backend on CPU (no DGL, no CUDA) |

Tolerances: fp32 `atol=1e-5`, `rtol=1e-4` elementwise (`|a-b| <= atol + rtol*|b|`)
everywhere. They were never relaxed. fp64 is used only where fp32 rounding itself
exceeds the tolerance (N = 16, see below).

## Stage × framework matrix

Max over the cases in `tests/parity/graphcast/test_stage_parity.py::CASES`
(grids 4×8, 6×12, 5×8 incl. poles; mesh level 0/1; latent 8/16; N ∈ {1,2,4,16};
node **and** edge latents compared). Reproduce with
`python -m tests.parity.graphcast.report`. "max rel" is dominated by
near-zero entries; the pass criterion is the elementwise one above.

| Stage | ours ↔ JAX (max abs / max rel) | ours ↔ PhysicsNeMo | JAX ↔ PhysicsNeMo |
|-------|-------------------------------|--------------------|-------------------|
| Graph: multi-mesh edges | **pass**: identical indices & order; features ≤ 1.4e-14 | **pass**: same edge set (PyG reorders); features match after column permutation | **pass**: same set; features ≤ 1e-5 after permutation |
| Graph: Grid2Mesh edges (radius 0.6 × longest finest edge) | **pass**: identical (L0–L3, 5 grids) | **pass**: same set | **pass**: same set (NV's 4-NN cap never binds; max in-degree 3) |
| Graph: Mesh2Grid edges (containing triangle) | **pass** with `m2g_backend="trimesh"` (identical); default `"numpy"` differs **only on exact shared-edge ties** (asserted) | known-diff (see D2) | **known-diff D2**, pinned exact counts |
| Graph: node features | **pass** (identical) | known-diff D3 | **known-diff D3** |
| Graph: edge features at pole grid points | **pass** | known-diff D4 | **known-diff D4** |
| **Grid2Mesh** (embed + 1 bipartite step; mesh, grid, edge latents) | **pass** 3.1e-6 / 1.2e-3 | **pass** 9.5e-7 / 4.5e-4 (concat-trick 1.1e-6) | **pass** 3.1e-6 / 1.2e-3 (concat-trick 3.8e-6) |
| **Processor N=1** | **pass** 9.5e-7 / 5.1e-3 | **known-diff D1** 6.8e-1 (`aggregate="updated"`: **pass** 2.4e-7) | **known-diff D1** 1.4e0 (mesh edges still pass at N=1) |
| **Processor N=2** | **pass** 2.0e-6 / 6.6e-4 | **known-diff D1** 2.8e0 (`updated`: **pass** 9.5e-7) | **known-diff D1** 4.9e0 |
| **Processor N=4** | **pass** 7.5e-6 / 1.8e-4 | **known-diff D1** 5.1e0 (`updated`: **pass** 2.1e-6) | **known-diff D1** 5.5e0 |
| **Processor N=16** | **pass (fp64)** 6.6e-14 / 2.2e-12; fp32: 4.2e-5 = same size as JAX's own fp32 error (4.8e-5 vs JAX-fp64), asserted ≤ 2× | **known-diff D1** 1.1e1 (`updated`: **pass** 8.5e-6) | **known-diff D1** 1.5e1 |
| **Mesh2Grid** (+ decoder MLP; outputs, edge latents) | **pass** 1.2e-6 / 2.0e-3 | **pass** 8.3e-7 / 5.9e-5 (concat-trick 1.9e-6) | **pass** 1.2e-6 / 2.0e-3 (concat-trick 1.8e-6) |
| **Chained end-to-end** (G2M→P→M2G, and public `forward()`) | **pass** N ≤ 4 fp32 (out 1.9e-6); N = 16 fp64 (out 6.9e-15) | `updated` mode: **pass** (out 1.2e-6) | **known-diff** (inherits D1), `xfail(strict)` |

"ours ↔ PhysicsNeMo" runs without JAX: synthetic Haiku-named weights plus
WeatherAI graphs (`_refs.synthetic_reference`). "JAX ↔ PhysicsNeMo" feeds JAX's
weights and JAX's graphs into PhysicsNeMo's modules. Isolated tests feed each
stage the reference's inputs for that stage. Chained tests run each
implementation end to end.

### PhysicsNeMo configuration used to match JAX

These settings are **configurable**, so they are set to match JAX and then tested:

- `activation_fn=nn.SiLU()` (GraphCast uses `"swish"`), `hidden_layers=1`,
  `norm_type="LayerNorm"`, `aggregation="sum"`. The decoder `finale` has `norm_type=None`.
- `do_concat_trick` is tested **both** False and True. It is the same maths
  (split first Linear), and both pass.
- Node-MLP input order: PhysicsNeMo concatenates `[agg, node]`, JAX
  `[node, agg]`. This is handled by permuting the first-layer weight columns (pure relabelling).
- Mesh-node embedder: JAX feeds `[zeros(C), feat]`, PhysicsNeMo `feat`. The zero
  block contributes nothing, so the JAX weight rows `C:` are used.
- Grid structural features: PhysicsNeMo expects them as ordinary input
  channels. The tests append JAX's `[sin lat, cos lon, sin lon]` to the inputs.
- `MeshGraphEncoder` does not return edge latents. The tests recompute
  `e + encoder.edge_mlp(e, (grid, mesh), graph)` with PhysicsNeMo's own module.
- Processor built as `GraphCastProcessor(processor_layers=N)` directly, because
  `GraphCastNet` splits layers into encoder/processor/decoder parts, which needs ≥ 3 layers.
- `torch.compile` is disabled in the test process (the box has no C++ toolchain).
  This only affects compilation; the maths is unchanged.

### Inherent differences (cannot be configured away)

- **D1: processor aggregation.** PhysicsNeMo `MeshNodeBlock` sums the
  *residual-updated* edge latents `e + Δe` returned by `MeshEdgeBlock`. JAX
  GraphCast (jraph `InteractionNetwork` inside `DeepTypedGraphNet`) sums the
  edge-MLP output `Δe`. Nothing in `GraphCastProcessor` / `GraphCastNet` changes
  this. Grid2Mesh and Mesh2Grid are unaffected, because PhysicsNeMo's
  encoder/decoder aggregate the raw edge-MLP output, as JAX does.
  *Pinned by:*
  - `test_physicsnemo_vs_jax_processor` and `..._chained_output`: `xfail(strict=True)`.
  - `test_physicsnemo_processor_known_difference_pinned`: asserts that the
    difference is > 1e-2 **and** that PhysicsNeMo equals WeatherAI with
    `GraphNetBlock(aggregate="updated")` within the normal tolerance, which
    identifies the exact cause.
  - At N=1 the edge latents still agree, because the first edge update is identical.
- **D2: Mesh2Grid connectivity.** PhysicsNeMo connects each grid point to the
  face whose **centroid** is nearest (sklearn 1-NN over the multimesh faces).
  JAX uses the face that **contains** the point (`trimesh.nearest.on_surface`).
  Coarse faces are never chosen in practice. The differing rows are either exact
  shared-edge ties or genuinely non-containing faces. Pinned counts
  (differing rows, of which non-containing), `test_physicsnemo_graphs_vs_jax`:
  4×8 L0/1/2: (0,0)/(1,0)/(0,0); 6×12: (3,0)/(6,0)/(2,0); poles 5×8:
  (3,0)/(3,0)/(7,4); 16×32: (4,0)/(24,16)/(31,24).
- **D3: node features.** PhysicsNeMo's PyG `add_node_features` calls
  `xyz2latlon(pos)` (default `unit="deg"`) and then `torch.cos` / `torch.sin` on
  the **degree** values: `(cos(lat°), sin(lon°), cos(lon°))`, with degrees used
  as radians. JAX uses `(sin lat, cos lon, sin lon)` in radians. Pinned by an
  independent float32 re-derivation (passes at the normal tolerance). This looks
  unintended upstream; we report it and do not follow it.
- **D4: pole receivers.** For grid points at |lat| = 90° the receiver-local
  frame is degenerate. JAX (and WeatherAI) use the grid's given longitude;
  PhysicsNeMo re-derives the longitude from float32 xyz with `arctan2`
  (cos 90° ≈ −4e-8), so the Mesh2Grid edge features at pole receivers are
  rotated. Non-pole receivers agree (≤ 1.6e-6 on unit directions). Pinned in
  `test_physicsnemo_graphs_vs_jax`.
- PhysicsNeMo edge features are `(dx, dy, dz, |d|)` versus JAX `(|d|, dx, dy, dz)`.
  This is a column permutation only; tests compare after permuting.

## What changed in WeatherAI to match JAX

- **Activation fix.** The previous revision (`22ae436`) switched the MLP default to
  **ReLU** because of `DeepTypedGraphNet`'s default. **That was a mistake.**
  `graphcast.GraphCast` passes `activation="swish"` to all three GNNs. The
  default is now **SiLU** again. ReLU remains available and is still tested as a
  variant in `test_mlp_parity` / `test_graphnet_parity`.
- **Graphs** (`mesh.py`):
  - float32 icosahedral vertices, bit-identical to JAX for L0–4.
  - Grid2Mesh uses a radius query (`radius_query_fraction_edge_length=0.6`
    × longest finest-mesh edge; squared-distance compare, sorted by (grid, mesh)).
  - Mesh2Grid uses the containing triangle of the finest mesh, 3 edges per grid
    point in face-vertex order.
  - `m2g_backend="numpy"` (default, no extra dependency; ties go to the lowest
    face index) or `"trimesh"` (DeepMind's call, identical including tie-breaks).
  - Structural features follow JAX exactly: node `[cos colat, cos lon, sin lon]`;
    edge `[|d|, d] / max|d|`, with d taken in the receiver-local frame and the
    angles formed in float32 as JAX does.
  - The k-NN arguments `g2m_k` / `m2g_k` are deprecated and ignored (they emit a
    `DeprecationWarning`).
- **Layers** (`layers.py`):
  - `BipartiteGraphNetBlock`: source update is `src + MLP(src)` (no aggregation,
    as in GraphCast Grid2Mesh). The destination update is `[dst, Σ Δe]`.
  - `GraphNetBlock(aggregate="delta" | "updated")`: "delta" is the canonical JAX
    path; "updated" exists only to pin D1.
- **Model** (`graphcast.py`):
  - The mesh-node embedder takes `[zeros(C), feat]`, and there are separate edge
    embedders for g2m, mesh and m2g.
  - The output head is an MLP without LayerNorm (the JAX decoder).
  - New stage methods `run_grid2mesh` / `run_processor` / `run_mesh2grid`
    mirror the JAX stages. New arguments: `grid_lat`/`grid_lon`, `m2g_backend`,
    `activation`, and `graphs=` for injecting prebuilt graphs.
  - The `GraphCast` / `GraphCast_lite` API and the `(B, C, H, W)` residual
    forward are unchanged. The buffer and parameter layout changed, so state
    dicts from `22ae436` do not load.

### fp32 at N = 16

With identical structural features in fp64, WeatherAI matches the real JAX
GraphCast to ≤ 1.5e-13 end to end at N = 16 (JAX's grid2mesh `f32_aggregation`
is switched off for this fp64 debug run only). In fp32, JAX itself deviates
from its fp64 result by 3–5e-5 on processor edge latents (magnitude ~15) after
16 residual layers, and WeatherAI's fp32 error is the same size. So the N = 16
exact-tolerance checks run in fp64, and the fp32 run is asserted to be no less
accurate than JAX-fp32 (≤ 2× its error against JAX-fp64).

## Running

```bash
# JAX reference clone (+ WeatherNext deps):
git clone https://github.com/google-deepmind/graphcast.git /workspace/tmp/graphcast-jax
pip install -e ".[parity]"         # jax, haiku, jraph, xarray_jax, trimesh, rtree, scipy
pytest tests/models/graphcast tests/parity/graphcast -q
```

PhysicsNeMo tests need `torch_geometric` + `torch_scatter` + `torch_sparse`
wheels that match your torch. The main venv here (torch 2.14) has none, so those
tests **skip** there with their own marker (`requires_physicsnemo`), separate from
`requires_jax`. We use a separate "parity venv" (Python 3.13, torch 2.12.1+cpu,
PyG 2.8 + `torch_scatter 2.1.2+pt212cpu` / `torch_sparse 0.6.18+pt212cpu`,
jax 0.11.2, `physicsnemo` installed `-e --no-deps` from a clone plus its light
runtime deps, `weatherai -e --no-deps`):

```bash
git clone https://github.com/NVIDIA/physicsnemo.git /workspace/tmp/physicsnemo
/workspace/tmp/parity-venv/bin/python -m pytest tests/models/graphcast tests/parity/graphcast -q
/workspace/tmp/parity-venv/bin/python -m tests.parity.graphcast.report   # matrix numbers
```

Results on 2026-09-25 (CPU): parity venv **161 passed, 10 xfailed (strict)**;
main venv **93 passed, 78 skipped** (all PhysicsNeMo).

Test files (`tests/parity/graphcast/`): `test_mesh_parity.py`,
`test_mlp_parity.py`, `test_graphnet_parity.py`, `test_graph_parity.py`
(graph construction, three-way), `test_stage_parity.py` (stages, three-way),
`_refs.py` (reference runners + weight mappers), `report.py`.

## Not claimed

- Loading DeepMind / NVIDIA checkpoints, full 0.25° / level-6 / 16×512 model
  equality, the two-frame input pipeline, normalisation, and rollouts.
- Mesh levels > 3 for graphs, and > 1 for the network stages, are not tested.

See also: [graphcast_specs.md](graphcast_specs.md).
