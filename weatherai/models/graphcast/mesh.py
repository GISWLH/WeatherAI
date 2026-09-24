"""Icosahedral multi-mesh and lat-lon grid graph construction (numpy).

Original WeatherAI helpers inspired by Lam et al. GraphCast (Science 2023 /
arXiv:2212.12794). Independent reimplementation — not copied from DeepMind JAX
or NVIDIA PhysicsNeMo sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

_PHI = (1.0 + np.sqrt(5.0)) / 2.0


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def icosahedron_vertices() -> np.ndarray:
    """Return (12, 3) unit-sphere vertices of a regular icosahedron."""
    verts: List[List[float]] = []
    for x in (-1.0, 1.0):
        for y in (-1.0, 1.0):
            verts.append([0.0, x, y * _PHI])
            verts.append([x, y * _PHI, 0.0])
            verts.append([x * _PHI, 0.0, y])
    return _unit(np.asarray(verts, dtype=np.float64))


def icosahedron_faces(vertices: np.ndarray | None = None) -> np.ndarray:
    """Return (20, 3) triangular faces from nearest-neighbor edges."""
    verts = icosahedron_vertices() if vertices is None else vertices
    n = verts.shape[0]
    dist = np.linalg.norm(verts[:, None, :] - verts[None, :, :], axis=-1)
    # Shortest nonzero distance = edge length of the regular icosahedron
    iu = np.triu_indices(n, 1)
    edge_len = float(dist[iu].min())
    edges = np.argwhere((dist > 0) & (dist <= edge_len * 1.01))
    edges = edges[edges[:, 0] < edges[:, 1]]
    edge_set = {tuple(map(int, e)) for e in edges}
    faces: List[Tuple[int, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edge_set:
                continue
            for k in range(j + 1, n):
                if (i, k) in edge_set and (j, k) in edge_set:
                    normal = np.cross(verts[j] - verts[i], verts[k] - verts[i])
                    if np.dot(normal, verts[i]) < 0:
                        faces.append((i, k, j))
                    else:
                        faces.append((i, j, k))
    if len(faces) != 20:
        raise RuntimeError(f"Expected 20 icosahedron faces, got {len(faces)}")
    return np.asarray(faces, dtype=np.int64)


def mesh_num_nodes(level: int) -> int:
    """Nodes on a level-``level`` refined icosahedron: ``10 * 4^L + 2``."""
    if level < 0:
        raise ValueError(f"mesh_level must be >= 0, got {level}")
    return 10 * (4**level) + 2


def refine_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One 1→4 triangle subdivision with sphere reprojection.

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
    for a, b, c in faces:
        a, b, c = int(a), int(b), int(c)
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        new_faces.extend([(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)])

    new_vertices = np.stack(verts, axis=0).astype(np.float64)
    new_faces_arr = np.asarray(new_faces, dtype=np.int64)
    e_set = set()
    for a, b, c in new_faces_arr:
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            e_set.add((u, v) if u < v else (v, u))
    new_edges = np.asarray(sorted(e_set), dtype=np.int64)
    return new_vertices, new_faces_arr, new_edges


def build_multimesh(mesh_level: int = 2) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Build finest-level nodes + multi-mesh undirected edges.

    Paper GraphCast uses mesh_level ≈ 6 (40,962 nodes). This skeleton defaults
    to smaller levels for CPU / ZeroGPU smoke tests.
    """
    nodes = icosahedron_vertices()
    faces = icosahedron_faces(nodes)
    e0 = set()
    for a, b, c in faces:
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            e0.add((u, v) if u < v else (v, u))
    edges_per_level: List[np.ndarray] = [np.asarray(sorted(e0), dtype=np.int64)]
    all_edges = set(e0)

    for _ in range(mesh_level):
        nodes, faces, new_e = refine_mesh(nodes, faces)
        edges_per_level.append(new_e)
        for u, v in new_e:
            all_edges.add((int(u), int(v)))

    expected = mesh_num_nodes(mesh_level)
    if nodes.shape[0] != expected:
        raise RuntimeError(f"Expected {expected} mesh nodes at level {mesh_level}, got {nodes.shape[0]}")

    multi_edges = np.asarray(sorted(all_edges), dtype=np.int64)
    return nodes.astype(np.float64), multi_edges, edges_per_level


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
) -> GraphCastGraphs:
    """Construct mesh, grid, and bipartite / mesh edge arrays."""
    mesh_xyz, multi_edges, edges_per_level = build_multimesh(mesh_level)
    undirected = multi_edges if use_multi_mesh else edges_per_level[-1]
    directed = _directed_from_undirected(undirected)
    mesh_senders = directed[:, 0]
    mesh_receivers = directed[:, 1]
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
