"""Static graphs of the official GenCast denoiser (numpy/scipy), built with WeatherAI's GraphCast helpers.

Differences to the GraphCast graphs in :mod:`weatherai.models.graphcast.mesh` (all taken from
``weathernext/weathernext1_gen/denoiser.py``):

* the mesh processor graph is the **finest** icosahedron only (no multi-mesh edges); the transformer
  only uses its adjacency (k-hop attention mask), mesh edge features are never computed;
* the mesh nodes are **permuted to a banded adjacency** (reverse Cuthill-McKee) before anything else
  (the official sparse attention needs a banded mask; we keep the permutation because it defines the
  mesh-node order that e.g. mesh latents / debugging dumps are in);
* node features ``[cos(colat), cos(lon), sin(lon)]`` and 4-d edge features ``[|d|, d]/max|d|`` in the
  receiver-local frame, exactly as in GraphCast (re-used from ``graphcast.mesh``);
* grid→mesh edges: radius query ``radius_query_fraction_edge_length * max(finest mesh edge length)``
  (scipy ``cKDTree.query_ball_point`` like the official code), mesh→grid: the 3 vertices of the closest
  triangle of the finest mesh.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..graphcast.mesh import (
    build_mesh_hierarchy,
    cartesian_to_lat_lon,
    edge_spatial_features,
    faces_to_edges,
    grid_lat_lon_to_xyz,
    grid_node_lat_lon,
    in_mesh_triangle_edges,
    node_spatial_features,
)


@dataclass
class GenCastGraphs:
    mesh_vertices: np.ndarray  # (Nm, 3) float32, banded order
    mesh_faces: np.ndarray  # (F, 3) int64, banded order
    mesh_lat: np.ndarray
    mesh_lon: np.ndarray
    mesh_senders: np.ndarray  # finest-mesh directed edges (only used for the attention mask)
    mesh_receivers: np.ndarray
    mesh_permutation: np.ndarray  # new_index -> old (icosahedron-refinement) index
    grid_lat: np.ndarray
    grid_lon: np.ndarray
    grid_node_features: np.ndarray  # (Ng, 3)
    mesh_node_features: np.ndarray  # (Nm, 3)
    g2m_senders: np.ndarray  # grid idx
    g2m_receivers: np.ndarray  # mesh idx
    g2m_edge_attr: np.ndarray  # (E, 4)
    m2g_senders: np.ndarray  # mesh idx
    m2g_receivers: np.ndarray  # grid idx
    m2g_edge_attr: np.ndarray
    query_radius: float
    mesh_size: int

    @property
    def num_mesh_nodes(self) -> int:
        return self.mesh_vertices.shape[0]

    @property
    def num_grid_nodes(self) -> int:
        return self.grid_node_features.shape[0]


def permute_mesh_to_banded(vertices: np.ndarray, faces: np.ndarray):
    """Reverse Cuthill-McKee relabelling of the mesh (official ``_permute_mesh_to_banded``)."""
    import scipy.sparse as sp
    import scipy.sparse.csgraph  # noqa: F401

    n = vertices.shape[0]
    s, r = faces_to_edges(faces)
    adj = sp.csr_matrix((np.ones(len(s)), (s, r)), shape=(n, n))  # same sparsity structure as the official adj[s, r] = 1
    perm = sp.csgraph.reverse_cuthill_mckee(adj, symmetric_mode=True)
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n)
    return vertices[perm], inv[np.asarray(faces)], perm.astype(np.int64)


def khop_attention_mask(senders: np.ndarray, receivers: np.ndarray, n: int, k: int) -> np.ndarray:
    """Dense bool (n, n): ``True`` where node j is within ``k`` hops of node i (self included).

    Same set as the official ``(A + I) ** k`` boolean matrix power (``attention_k_hop``).
    """
    import scipy.sparse as sp

    a = sp.csr_matrix((np.ones(len(senders), dtype=np.float32), (senders, receivers)), shape=(n, n))
    a = ((a + sp.identity(n, dtype=np.float32, format="csr")) > 0).astype(np.float32)
    result = sp.identity(n, dtype=np.float32, format="csr")
    base, e = a, int(k)
    while e:  # binary exponentiation with re-binarisation (keeps it sparse-ish & exact for reachability)
        if e & 1:
            result = ((result @ base) > 0).astype(np.float32)
        base = ((base @ base) > 0).astype(np.float32)
        e >>= 1
    return result.toarray() > 0


def build_gencast_graphs(
    mesh_size: int,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    radius_query_fraction_edge_length: float = 0.6,
    m2g_backend: str = "auto",
) -> GenCastGraphs:
    """Build the three GenCast graphs for a lat/lon grid (1-D degrees, any size, flattened lat-major)."""
    import scipy.spatial

    grid_lat = np.asarray(grid_lat, dtype=np.float32)
    grid_lon = np.asarray(grid_lon, dtype=np.float32)
    v0, f0 = build_mesh_hierarchy(mesh_size)[-1]
    vertices, faces, perm = permute_mesh_to_banded(v0, f0)
    ms, mr = faces_to_edges(faces)

    mesh_lat, mesh_lon = cartesian_to_lat_lon(vertices)
    mesh_lat, mesh_lon = mesh_lat.astype(np.float32), mesh_lon.astype(np.float32)
    g_lat, g_lon = grid_node_lat_lon(grid_lat, grid_lon)
    grid_xyz = grid_lat_lon_to_xyz(grid_lat, grid_lon)

    radius = np.linalg.norm(vertices[ms] - vertices[mr], axis=-1).max() * radius_query_fraction_edge_length
    tree = scipy.spatial.cKDTree(vertices)
    neigh = tree.query_ball_point(x=grid_xyz, r=radius)
    gs = np.concatenate([np.repeat(i, len(n)) for i, n in enumerate(neigh)]).astype(np.int64)
    gr = np.concatenate([np.asarray(n, dtype=np.int64) for n in neigh]).astype(np.int64)
    g2m_edge = edge_spatial_features(g_lat, g_lon, mesh_lat, mesh_lon, gs, gr)

    if m2g_backend == "auto":  # trimesh reproduces the official tie-breaking exactly; numpy is the dependency-free fallback
        try:
            import trimesh  # noqa: F401
            import rtree  # noqa: F401

            m2g_backend = "trimesh"
        except Exception:
            m2g_backend = "numpy"
    m2g_grid, m2g_mesh = in_mesh_triangle_edges(grid_xyz, vertices, faces, backend=m2g_backend)
    m2g_edge = edge_spatial_features(mesh_lat, mesh_lon, g_lat, g_lon, m2g_mesh, m2g_grid)

    return GenCastGraphs(
        mesh_vertices=vertices.astype(np.float32),
        mesh_faces=faces,
        mesh_lat=mesh_lat,
        mesh_lon=mesh_lon,
        mesh_senders=ms,
        mesh_receivers=mr,
        mesh_permutation=perm,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        grid_node_features=node_spatial_features(g_lat, g_lon).astype(np.float32),
        mesh_node_features=node_spatial_features(mesh_lat, mesh_lon).astype(np.float32),
        g2m_senders=gs,
        g2m_receivers=gr,
        g2m_edge_attr=g2m_edge.astype(np.float32),
        m2g_senders=m2g_mesh,
        m2g_receivers=m2g_grid,
        m2g_edge_attr=m2g_edge.astype(np.float32),
        query_radius=float(radius),
        mesh_size=int(mesh_size),
    )
