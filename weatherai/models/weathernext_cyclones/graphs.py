"""Static graphs of the official WeatherNext Cyclones (FGN) network.

Same construction as GenCast's (:mod:`weatherai.models.gencast.graphs`: finest icosahedron only, reverse
Cuthill-McKee banded node order, grid->mesh *ball query* with radius ``0.6 * longest mesh edge``,
mesh->grid *closest triangle* (3 edges per grid point), 4-d receiver-local edge features), with two
differences taken from ``weathernext/utils/points_mesh_gnn.py``:

* edges are **sorted by receiver** before the GNN sees them (we use a stable sort; the official unstable ``np.argsort`` orders
  edges *within* one receiver in a numpy-version-dependent way, which does not change the result of the sum aggregation);
* node inputs are **not** the GraphCast structural features: the encoders get ``sin(lat), sin(lon), cos(lon)``
  of each node (returned here as ``grid_spatial`` / ``mesh_spatial``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..gencast.graphs import GenCastGraphs, build_gencast_graphs
from ..graphcast.mesh import edge_spatial_features, faces_to_edges, in_mesh_triangle_edges


@dataclass
class WNCGraphs:
    base: GenCastGraphs  # mesh geometry / attention-mask edges (finest mesh)
    grid_spatial: np.ndarray  # (Ng, 3) sin_lat, sin_lon, cos_lon (float32)
    mesh_spatial: np.ndarray  # (Nm, 3)
    g2m_senders: np.ndarray  # grid idx, sorted by receiver
    g2m_receivers: np.ndarray  # mesh idx
    g2m_edge_attr: np.ndarray  # (E, 4) distance, rel_x, rel_y, rel_z
    m2g_senders: np.ndarray  # mesh idx
    m2g_receivers: np.ndarray  # grid idx, sorted
    m2g_edge_attr: np.ndarray

    @property
    def num_grid_nodes(self) -> int:
        return self.base.num_grid_nodes

    @property
    def num_mesh_nodes(self) -> int:
        return self.base.num_mesh_nodes


def _spatial(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    return np.stack([np.sin(lat), np.sin(lon), np.cos(lon)], axis=-1).astype(np.float32)


def _unit_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """float32 lat/lon → float32 xyz exactly as the official ``lat_lon_to_cartesian`` (node positions are re-derived
    from float32 lat/lon, not taken from the mesh vertices; this matters for exact tie-breaking of the edges)."""
    phi, theta = np.deg2rad(lon_deg), np.deg2rad(90 - lat_deg)
    return np.stack([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)], axis=-1)


def build_wnc_graphs(mesh_splits: int, grid_lat: np.ndarray, grid_lon: np.ndarray, radius_fraction: float = 0.6,
                     m2g_backend: str = "auto") -> WNCGraphs:
    """Graphs for a lat/lon grid (1-D degrees, flattened lat-major) and ``mesh_splits`` 2-way icosahedron splits."""
    import scipy.spatial

    grid_lat = np.asarray(grid_lat, np.float32)
    grid_lon = np.asarray(grid_lon, np.float32)
    g = build_gencast_graphs(mesh_splits, grid_lat, grid_lon, radius_fraction, m2g_backend=m2g_backend)  # mesh / permutation
    glat = np.repeat(grid_lat, len(grid_lon))
    glon = np.tile(grid_lon, len(grid_lat))
    mesh_xyz = _unit_xyz(g.mesh_lat, g.mesh_lon)
    grid_xyz = _unit_xyz(glat, glon)
    ms, mr = faces_to_edges(g.mesh_faces)
    radius = radius_fraction * np.linalg.norm(mesh_xyz[ms] - mesh_xyz[mr], axis=-1).max()
    neigh = scipy.spatial.cKDTree(mesh_xyz).query_ball_point(x=grid_xyz, r=radius)
    gs = np.concatenate([np.repeat(i, len(n)) for i, n in enumerate(neigh)]).astype(np.int64)
    gr = np.concatenate([np.asarray(n, dtype=np.int64) for n in neigh])
    o = np.argsort(gr, kind="stable")  # official: np.argsort(receivers) (unstable → order inside a receiver is numpy-version dependent; sums are order-free)
    gs, gr = gs[o], gr[o]
    g2m = edge_spatial_features(glat, glon, g.mesh_lat, g.mesh_lon, gs, gr)

    if m2g_backend == "auto":
        try:
            import rtree  # noqa: F401
            import trimesh  # noqa: F401

            m2g_backend = "trimesh"
        except Exception:
            m2g_backend = "numpy"
    m2g_grid, m2g_mesh = in_mesh_triangle_edges(grid_xyz, mesh_xyz, g.mesh_faces, backend=m2g_backend)
    o = np.argsort(m2g_grid, kind="stable")
    m2g_grid, m2g_mesh = m2g_grid[o], m2g_mesh[o]
    m2g = edge_spatial_features(g.mesh_lat, g.mesh_lon, glat, glon, m2g_mesh, m2g_grid)
    return WNCGraphs(
        base=g, grid_spatial=_spatial(glat, glon), mesh_spatial=_spatial(g.mesh_lat, g.mesh_lon),
        g2m_senders=gs, g2m_receivers=gr, g2m_edge_attr=g2m,
        m2g_senders=m2g_mesh, m2g_receivers=m2g_grid, m2g_edge_attr=m2g,
    )
