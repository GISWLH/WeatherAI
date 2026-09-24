# GraphCast submodule parity (WeatherAI Torch ↔ DeepMind JAX)

**Honest status (2026-09):** full-model and checkpoint numerical equality are
**not** claimed. Small-scale submodule parity **is** automated for:

| Unit | Status | Tolerances (float32, CPU) |
|------|--------|---------------------------|
| **A. Mesh / icosahedron** (level 0 & 1) | **Done** — node XYZ, faces, directed edges match JAX | vertices `atol≈1e-5`; faces/edges exact |
| **B. MLP + LayerNorm** | **Done** — same Haiku weights → same Torch output | `atol=1e-5`, `rtol=1e-4` |
| **C. One GraphNet / InteractionNetwork layer** | **Done** — identical `edge_index`, features, mapped weights | `atol=1e-5`, `rtol=1e-4` |
| Grid2Mesh / Mesh2Grid bipartite | **Not** claimed | — |
| Full processor stack (16 layers) | **Not** claimed | — |
| Full encode–process–decode + HF/JAX checkpoints | **Not** claimed | — |

## Reference clone (tests only)

```bash
git clone --depth 1 https://github.com/google-deepmind/graphcast.git /workspace/tmp/graphcast-jax
```

As of 2026 the GitHub repo hosts the **WeatherNext** package layout; GraphCast
helpers live under `weathernext/utils/` (e.g. `icosahedral_mesh.py`,
`legacy/deep_typed_graph_net.py`). WeatherAI does **not** vendor those sources
into `weatherai/` (MIT). Parity tests import them from the clone path
(`WEATHERAI_GRAPHCAST_JAX_ROOT`, default `/workspace/tmp/graphcast-jax`).

Install optional deps:

```bash
pip install -e ".[parity]"
pytest tests/parity/graphcast -q
```

Tests **skip** if JAX/Haiku/jraph or the clone are missing.

## What we aligned in Torch

1. **Mesh** (`weatherai/models/graphcast/mesh.py`):
   - Same icosahedron vertex generation order and face index table as DeepMind.
   - Default `pole_parallel_faces=True` (Y-rotation so nodes are not exactly at
     geographic poles) — GraphCast JAX default.
   - 1→4 refinement child-face order matches JAX so refined indices align.
   - Directed mesh edges via `faces_to_edges` (3 directed edges per triangle).
2. **MLP** (`layers.MLP`): ReLU (not SiLU) + LayerNorm after the last Linear —
   matches Haiku `MLP` → `LayerNorm` used in GraphCast’s typed GraphNet.
3. **GraphNetBlock**: aggregates **edge deltas** before residual add, matching
   jraph `InteractionNetwork` + residual in `DeepTypedGraphNet._process_step`.

Weight mapping: Haiku Linear `w` is `(in, out)`; Torch `weight` is `(out, in)`
→ copy `torch_weight = jax_w.T`.

## Observed errors (local CPU)

Typical max abs error on the checked-in parity fixtures:

- MLP: ~`3e-7`
- GraphNet nodes/edges: ~`1e-6`

## Intentional / remaining differences

- Bipartite Grid2Mesh / Mesh2Grid still use **k-NN**, not paper radius queries.
- Lat-lon grid construction is WeatherAI’s; not parity-tested against JAX grids.
- Public `GraphCast` / `GraphCast_lite` remain smoke-sized; they use the aligned
  blocks but are **not** end-to-end equal to official checkpoints.
- Multi-mesh face concatenation order matches JAX `merge_meshes` at level ≤1;
  higher levels should match by construction but are not asserted in CI yet.

## Next expansion step

1. Grid2Mesh bipartite layer parity (fixed edge_index + mapped weights).
2. Then Mesh2Grid.
3. Then unshared processor stack depth > 1 on mesh_level=0/1, latent 8–16.
4. Only later: full model I/O and checkpoint loading (separate effort).

See also: [graphcast_specs.md](graphcast_specs.md).
