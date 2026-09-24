"""Icosahedral multi-mesh and lat-lon grid graph construction (numpy).

Independent WeatherAI reimplementation inspired by Lam et al. GraphCast
(Science 2023 / arXiv:2212.12794). Geometry at mesh_level 0/1 is aligned with
DeepMind's published icosahedral helpers (``weathernext.utils.icosahedral_mesh``
in google-deepmind/graphcast, Apache-2.0) for small-scale parity tests — we do
**not** vendor their sources into this MIT package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

_PHI = (1.0 + np.sqrt(5.0)) / 2.0

# Counter-clockwise faces for the Wikipedia / DeepMind regular-icosahedron
# vertex ordering (see get_icosahedron). Kept as a compact index table so
# node XYZ and edge connectivity can match the JAX reference at level 0/1.
_ICOSAHEDRON_FACES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),
    (0, 6, 1),
    (8, 0, 2),
    (8, 4, 0),
    (3, 8, 2),
    (3, 2, 7),
    (7, 2, 1),
    (0, 4, 6),
    (4, 11, 6),
    (6, 11, 5),
    (1, 5, 7),
    (4, 10, 11),
    (4, 8, 10),
    (10, 8, 3),
    (10, 3, 9),
    (11, 10, 9),
    (11, 9, 5),
    (5, 9, 7),
    (9, 3, 7),
    (1, 6, 5),
)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def _rotation_matrix_y(angle: float) -> np.ndarray:
    """Active Y-axis rotation (matches ``scipy.spatial.transform.Rotation``)."""
    c, s = float(np.cos(angle)), float(np.sin(angle))
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def icosahedron_vertices(pole_parallel_faces: bool = True) -> np.ndarray:
    """Return (12, 3) unit-sphere vertices of a regular icosahedron.

    Vertex generation order and optional pole-parallel rotation match the
    DeepMind JAX helper (default ``pole_parallel_faces=True`` avoids nodes
    exactly at the geographic poles).
    """
    verts: List[Tuple[float, float, float]] = []
    for c1 in (1.0, -1.0):
        for c2 in (_PHI, -_PHI):
            verts.append((c1, c2, 0.0))
            verts.append((0.0, c1, c2))
            verts.append((c2, 0.0, c1))
    vertices = np.asarray(verts, dtype=np.float64)
    vertices /= np.linalg.norm([1.0, _PHI])

    if pole_parallel_faces:
        angle_between_faces = 2.0 * np.arcsin(_PHI / np.sqrt(3.0))
        rotation_angle = (np.pi - angle_between_faces) / 2.0
        vertices = vertices @ _rotation_matrix_y(rotation_angle)

    return _unit(vertices).astype(np.float64)


def icosahedron_faces(vertices: np.ndarray | None = None) -> np.ndarray:
    """Return (20, 3) triangular faces (CCW from outside).

    ``vertices`` is accepted for API compatibility; face indices are the fixed
    table for ``icosahedron_vertices()`` ordering (not rediscovered via k-NN).
    """
    del vertices  # ordering is defined by icosahedron_vertices()
    return np.asarray(_ICOSAHEDRON_FACES, dtype=np.int64)


def mesh_num_nodes(level: int) -> int:
    """Nodes on a level-``level`` refined icosahedron: ``10 * 4^L + 2``."""
    if level < 0:
        raise ValueError(f"mesh_level must be >= 0, got {level}")
    return 10 * (4**level) + 2


def faces_to_edges(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Directed edges from triangular faces: i→j, j→k, k→i per face.

    Same convention as DeepMind ``icosahedral_mesh.faces_to_edges``. For a
    closed oriented surface this yields a bidirectional edge set.
    """
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError(f"faces must be [F,3], got {faces.shape}")
    senders = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    receivers = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    return senders.astype(np.int64), receivers.astype(np.int64)


def refine_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One 1→4 triangle subdivision with sphere reprojection.

    Face child order matches DeepMind ``_two_split_unit_sphere_triangle_faces``
    so refined vertex indices line up with the JAX hierarchy.

    Returns new_vertices, new_faces, and undirected edges of the refined mesh.
    """
    edge_mid: Dict[Tuple[int, int], int] = {}
    verts: List[np.ndarray] = [vertices[i].copy() for i in range(len(vertices))]

    def midpoint(i: int, j: int) -> int:
        key = (i, j) if i < j else (j, i)
        if key in edge_mid:
            return edge_mid[key]
        m = _unit(0.5 * (verts[i] + verts[j]))
        idx = len(verts)
        verts.append(m)
        edge_mid[key] = idx
        return idx

    new_faces: List[Tuple[int, int, int]] = []
    for ind1, ind2, ind3 in faces:
        ind1, ind2, ind3 = int(ind1), int(ind2), int(ind3)
        ind12 = midpoint(ind1, ind2)
        ind23 = midpoint(ind2, ind3)
        ind31 = midpoint(ind3, ind1)
        new_faces.extend(
            [
                (ind1, ind12, ind31),
                (ind12, ind2, ind23),
                (ind31, ind23, ind3),
                (ind12, ind23, ind31),
            ]
        )

    new_vertices = np.stack(verts, axis=0).astype(np.float64)
    new_faces_arr = np.asarray(new_faces, dtype=np.int64)
    e_set = set()
    for a, b, c in new_faces_arr:
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            e_set.add((u, v) if u < v else (v, u))
    new_edges = np.asarray(sorted(e_set), dtype=np.int64)
    return new_vertices, new_faces_arr, new_edges


def build_mesh_hierarchy(
    mesh_level: int = 2,
    pole_parallel_faces: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return ``[(vertices, faces), ...]`` from level 0 .. ``mesh_level``."""
    nodes = icosahedron_vertices(pole_parallel_faces=pole_parallel_faces)
    faces = icosahedron_faces(nodes)
    out: List[Tuple[np.ndarray, np.ndarray]] = [(nodes, faces)]
    for _ in range(mesh_level):
        nodes, faces, _ = refine_mesh(nodes, faces)
        out.append((nodes, faces))
    return out


def build_multimesh(
    mesh_level: int = 2,
    pole_parallel_faces: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Build finest-level nodes + multi-mesh undirected edges.

    Paper GraphCast uses mesh_level ≈ 6 (40,962 nodes). This skeleton defaults
    to smaller levels for CPU / ZeroGPU smoke tests.

    Undirected multi-mesh edges are the union of undirected edges over levels
    0..mesh_level (same cardinality as DeepMind ``merge_meshes`` + undirected).
    """
    hierarchy = build_mesh_hierarchy(mesh_level, pole_parallel_faces=pole_parallel_faces)
    edges_per_level: List[np.ndarray] = []
    all_edges: set = set()
    for _nodes, faces in hierarchy:
        e_set = set()
        for a, b, c in faces:
            for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
                key = (u, v) if u < v else (v, u)
                e_set.add(key)
                all_edges.add(key)
        edges_per_level.append(np.asarray(sorted(e_set), dtype=np.int64))

    nodes = hierarchy[-1][0]
    expected = mesh_num_nodes(mesh_level)
    if nodes.shape[0] != expected:
        raise RuntimeError(
            f"Expected {expected} mesh nodes at level {mesh_level}, got {nodes.shape[0]}"
        )

    multi_edges = np.asarray(sorted(all_edges), dtype=np.int64)
    return nodes.astype(np.float64), multi_edges, edges_per_level


def build_mesh_faces(
    mesh_level: int = 2,
    use_multi_mesh: bool = True,
    pole_parallel_faces: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Finest vertices and face array (merged across levels if multi-mesh)."""
    hierarchy = build_mesh_hierarchy(mesh_level, pole_parallel_faces=pole_parallel_faces)
    nodes = hierarchy[-1][0]
    if use_multi_mesh:
        faces = np.concatenate([f for _, f in hierarchy], axis=0)
    else:
        faces = hierarchy[-1][1]
    return nodes.astype(np.float64), faces.astype(np.int64)


def latlon_grid_nodes(img_size: Tuple[int, int]) -> np.ndarray:
    """Equiangular lat-lon grid nodes as unit-sphere xyz, shape (H*W, 3)."""
    h, w = int(img_size[0]), int(img_size[1])
    lats = np.linspace(-90.0 + 90.0 / h, 90.0 - 90.0 / h, h)
    lons = np.linspace(0.0, 360.0, w, endpoint=False) + (180.0 / w)
    lat_r = np.deg2rad(lats)
    lon_r = np.deg2rad(lons)
    lat_g, lon_g = np.meshgrid(lat_r, lon_r, indexing="ij")
    cos_lat = np.cos(lat_g)
    x = cos_lat * np.cos(lon_g)
    y = cos_lat * np.sin(lon_g)
    z = np.sin(lat_g)
    return np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float64)


def _directed_from_undirected(edges: np.ndarray) -> np.ndarray:
    if edges.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.concatenate([edges, edges[:, ::-1]], axis=0)


def edge_relative_features(
    src_xyz: np.ndarray,
    dst_xyz: np.ndarray,
    senders: np.ndarray,
    receivers: np.ndarray,
) -> np.ndarray:
    """Relative edge features Δxyz + length → (E, 4)."""
    s = src_xyz[senders]
    r = dst_xyz[receivers]
    delta = s - r
    length = np.linalg.norm(delta, axis=-1, keepdims=True)
    return np.concatenate([delta, length], axis=-1).astype(np.float32)


def knn_bipartite_edges(
    src_xyz: np.ndarray,
    dst_xyz: np.ndarray,
    k: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Connect each dst node to k nearest src nodes on the sphere."""
    dots = dst_xyz @ src_xyz.T
    k_eff = min(int(k), src_xyz.shape[0])
    idx = np.argpartition(-dots, kth=k_eff - 1, axis=1)[:, :k_eff]
    row = np.arange(dst_xyz.shape[0])[:, None]
    order = np.argsort(-dots[row, idx], axis=1)
    idx = idx[row, order]
    receivers = np.repeat(np.arange(dst_xyz.shape[0], dtype=np.int64), k_eff)
    senders = idx.reshape(-1).astype(np.int64)
    return senders, receivers


@dataclass
class GraphCastGraphs:
    """Static graph arrays for one (img_size, mesh_level) configuration."""

    mesh_nodes: np.ndarray
    grid_nodes: np.ndarray
    mesh_senders: np.ndarray
    mesh_receivers: np.ndarray
    mesh_edge_attr: np.ndarray
    g2m_senders: np.ndarray
    g2m_receivers: np.ndarray
    g2m_edge_attr: np.ndarray
    m2g_senders: np.ndarray
    m2g_receivers: np.ndarray
    m2g_edge_attr: np.ndarray
    mesh_level: int
    img_size: Tuple[int, int]
    g2m_k: int
    m2g_k: int


def build_graphs(
    img_size: Tuple[int, int] = (32, 64),
    mesh_level: int = 2,
    g2m_k: int = 4,
    m2g_k: int = 4,
    use_multi_mesh: bool = True,
    pole_parallel_faces: bool = True,
) -> GraphCastGraphs:
    """Construct mesh, grid, and bipartite / mesh edge arrays.

    Mesh directed edges come from ``faces_to_edges`` on (multi-)mesh faces so
    ordering matches the JAX reference when the same hierarchy is used.
    """
    mesh_xyz, faces = build_mesh_faces(
        mesh_level, use_multi_mesh=use_multi_mesh, pole_parallel_faces=pole_parallel_faces
    )
    mesh_senders, mesh_receivers = faces_to_edges(faces)
    mesh_edge_attr = edge_relative_features(mesh_xyz, mesh_xyz, mesh_senders, mesh_receivers)

    grid_xyz = latlon_grid_nodes(img_size)
    g2m_senders, g2m_receivers = knn_bipartite_edges(grid_xyz, mesh_xyz, k=g2m_k)
    g2m_edge_attr = edge_relative_features(grid_xyz, mesh_xyz, g2m_senders, g2m_receivers)
    m2g_senders, m2g_receivers = knn_bipartite_edges(mesh_xyz, grid_xyz, k=m2g_k)
    m2g_edge_attr = edge_relative_features(mesh_xyz, grid_xyz, m2g_senders, m2g_receivers)

    return GraphCastGraphs(
        mesh_nodes=mesh_xyz.astype(np.float32),
        grid_nodes=grid_xyz.astype(np.float32),
        mesh_senders=mesh_senders,
        mesh_receivers=mesh_receivers,
        mesh_edge_attr=mesh_edge_attr,
        g2m_senders=g2m_senders,
        g2m_receivers=g2m_receivers,
        g2m_edge_attr=g2m_edge_attr,
        m2g_senders=m2g_senders,
        m2g_receivers=m2g_receivers,
        m2g_edge_attr=m2g_edge_attr,
        mesh_level=mesh_level,
        img_size=(int(img_size[0]), int(img_size[1])),
        g2m_k=int(g2m_k),
        m2g_k=int(m2g_k),
    )


def mesh_level_summary(max_level: int = 6) -> Sequence[Tuple[int, int]]:
    """Return ``(level, num_nodes)`` pairs for documentation."""
    return [(L, mesh_num_nodes(L)) for L in range(max_level + 1)]
