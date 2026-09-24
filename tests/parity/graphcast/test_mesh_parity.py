"""Mesh / icosahedron parity vs DeepMind JAX helpers (levels 0 and 1)."""

from __future__ import annotations

import numpy as np

from weatherai.models.graphcast.mesh import (
    build_mesh_faces,
    build_mesh_hierarchy,
    faces_to_edges,
    mesh_num_nodes,
)

from ._helpers import ATOL, ensure_jax_ref_on_path, max_abs_rel, requires_jax_parity


@requires_jax_parity
def test_node_counts_level_0_1():
    assert mesh_num_nodes(0) == 12
    assert mesh_num_nodes(1) == 42


@requires_jax_parity
def test_vertices_and_faces_match_jax_level_0_1():
    ensure_jax_ref_on_path()
    from weathernext.utils import icosahedral_mesh as jax_mesh

    for level in (0, 1):
        hier = jax_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits=level)
        j = hier[-1]
        our_hier = build_mesh_hierarchy(level, pole_parallel_faces=True)
        our_v, our_f = our_hier[-1]

        assert our_v.shape[0] == j.vertices.shape[0] == mesh_num_nodes(level)
        assert our_f.shape == j.faces.shape

        abs_e, rel_e = max_abs_rel(our_v.astype(np.float32), np.asarray(j.vertices))
        assert abs_e < 1e-5, f"level={level} vertex max abs {abs_e}"
        assert np.array_equal(our_f.astype(np.int32), np.asarray(j.faces))

        # Directed edges from faces (order + connectivity)
        js, jr = jax_mesh.faces_to_edges(j.faces)
        os, or_ = faces_to_edges(our_f)
        assert np.array_equal(os, np.asarray(js))
        assert np.array_equal(or_, np.asarray(jr))


@requires_jax_parity
def test_multimesh_edge_cardinality_level_1():
    """Union of undirected edges over levels 0..1 matches JAX merge_meshes."""
    ensure_jax_ref_on_path()
    from weathernext.utils import icosahedral_mesh as jax_mesh

    hier = jax_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits=1)
    merged = jax_mesh.merge_meshes(hier)
    js, jr = jax_mesh.faces_to_edges(merged.faces)
    j_und = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in zip(js.tolist(), jr.tolist())}

    _nodes, faces = build_mesh_faces(1, use_multi_mesh=True, pole_parallel_faces=True)
    os, or_ = faces_to_edges(faces)
    o_und = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in zip(os.tolist(), or_.tolist())}

    assert o_und == j_und
    assert len(o_und) == 150  # 30 (L0) + 120 (L1 finest)
    # Full directed arrays match when face concatenation order matches
    assert np.array_equal(faces.astype(np.int32), merged.faces)
    assert np.array_equal(os, np.asarray(js))
    assert np.array_equal(or_, np.asarray(jr))
    _ = ATOL  # documented tolerance namespace for mesh floats above
