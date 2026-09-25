"""Icosahedral multi-mesh, grid<->mesh connectivity and spatial features (numpy).

Independent WeatherAI reimplementation of the graph construction described in
Lam et al., GraphCast (Science 2023 / arXiv:2212.12794). The conventions are
chosen to be numerically interchangeable with DeepMind's published JAX
reference (google-deepmind/graphcast, Apache-2.0) so that parity tests can
compare stage by stage; none of their source is vendored here.

Conventions (all verified by ``tests/parity/graphcast``):

* Mesh: regular icosahedron rotated so top/bottom faces are pole-parallel,
  refined by 1->4 splits; vertices are float32 like the reference.
* Multi-mesh edges: ``faces_to_edges`` over the concatenated faces of levels
  0..L (each triangle contributes i->j, j->k, k->i).
* Grid2Mesh: every (grid, mesh) pair with straight-line distance
  ``<= radius_query_fraction_edge_length * max_finest_edge_length``,
  ordered by grid index then ascending mesh index.
* Mesh2Grid: each grid point connects to the 3 vertices of the finest-mesh
  triangle closest to it (containing triangle), in face vertex order.
* Node features: ``[cos(colatitude), cos(lon), sin(lon)]``.
* Edge features: ``[|d|, d_x, d_y, d_z] / max|d|`` where ``d`` is
  sender - receiver after rotating both into the receiver's local frame
  (receiver moved to lat=0, lon=0).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_PHI = (1.0 + np.sqrt(5.0)) / 2.0

# Counter-clockwise faces for the vertex ordering produced by
# ``icosahedron_vertices`` (standard (±1, ±phi, 0) cyclic construction).
_ICOSAHEDRON_FACES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2), (0, 6, 1), (8, 0, 2), (8, 4, 0), (3, 8, 2),
    (3, 2, 7), (7, 2, 1), (0, 4, 6), (4, 11, 6), (6, 11, 5),
    (1, 5, 7), (4, 10, 11), (4, 8, 10), (10, 8, 3), (10, 3, 9),
    (11, 10, 9), (11, 9, 5), (5, 9, 7), (9, 3, 7), (1, 6, 5),
)


# --------------------------------------------------------------------------
# Rotations
# --------------------------------------------------------------------------

def _rot_z(angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    z, o = np.zeros_like(angle), np.ones_like(angle)
    return np.stack(
        [np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2
    )


def _rot_y(angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    z, o = np.zeros_like(angle), np.ones_like(angle)
    return np.stack(
        [np.stack([c, z, s], -1), np.stack([z, o, z], -1), np.stack([-s, z, c], -1)], -2
    )


# --------------------------------------------------------------------------
# Icosahedral mesh
# --------------------------------------------------------------------------

def icosahedron_vertices(pole_parallel_faces: bool = True) -> np.ndarray:
    """(12, 3) float32 unit vertices of a regular icosahedron."""
    verts: List[Tuple[float, float, float]] = []
    for c1 in (1.0, -1.0):
        for c2 in (_PHI, -_PHI):
            verts.append((c1, c2, 0.0))
            verts.append((0.0, c1, c2))
            verts.append((c2, 0.0, c1))
    # float32 construction then float64 normalisation / rotation, cast back to
    # float32 (same dtype flow as the reference implementation).
    vertices = np.asarray(verts, dtype=np.float32)
    # Normalise with a float32 result (in-place semantics), then rotate in f64.
    vertices = (vertices / np.linalg.norm([1.0, _PHI])).astype(np.float32)
    if pole_parallel_faces:
        angle_between_faces = 2.0 * np.arcsin(_PHI / np.sqrt(3.0))
        rotation_angle = (np.pi - angle_between_faces) / 2.0
        vertices = vertices @ _rot_y(np.asarray(rotation_angle, dtype=np.float64))
    return vertices.astype(np.float32)


def icosahedron_faces(vertices: Optional[np.ndarray] = None) -> np.ndarray:
    """(20, 3) CCW faces for ``icosahedron_vertices`` ordering."""
    del vertices
    return np.asarray(_ICOSAHEDRON_FACES, dtype=np.int64)


def mesh_num_nodes(level: int) -> int:
    """Nodes on a level-``level`` refined icosahedron: ``10 * 4^L + 2``."""
    if level < 0:
        raise ValueError(f"mesh_level must be >= 0, got {level}")
    return 10 * (4**level) + 2


def faces_to_edges(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Directed edges i->j, j->k, k->i for every triangle (blocks by column)."""
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError(f"faces must be [F,3], got {faces.shape}")
    senders = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    receivers = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    return senders, receivers


def refine_mesh(
    vertices: np.ndarray, faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One 1->4 split with sphere reprojection (child order preserves CCW).

    Returns ``(new_vertices, new_faces, undirected_edges)``.
    """
    vertices = np.asarray(vertices)
    edge_mid: Dict[Tuple[int, int], int] = {}
    verts: List[np.ndarray] = list(vertices)

    def midpoint(i: int, j: int) -> int:
        key = (i, j) if i < j else (j, i)
        idx = edge_mid.get(key)
        if idx is None:
            m = vertices[[key[0], key[1]]].mean(0)
            m = m / np.linalg.norm(m)
            idx = len(verts)
            verts.append(m)
            edge_mid[key] = idx
        return idx

    new_faces: List[Tuple[int, int, int]] = []
    for i1, i2, i3 in faces:
        i1, i2, i3 = int(i1), int(i2), int(i3)
        i12, i23, i31 = midpoint(i1, i2), midpoint(i2, i3), midpoint(i3, i1)
        new_faces.extend([(i1, i12, i31), (i12, i2, i23), (i31, i23, i3), (i12, i23, i31)])

    new_vertices = np.asarray(verts, dtype=vertices.dtype)
    new_faces_arr = np.asarray(new_faces, dtype=np.int64)
    s, r = faces_to_edges(new_faces_arr)
    und = np.unique(np.stack([np.minimum(s, r), np.maximum(s, r)], 1), axis=0)
    return new_vertices, new_faces_arr, und


def build_mesh_hierarchy(
    mesh_level: int = 2, pole_parallel_faces: bool = True
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """``[(vertices, faces), ...]`` for levels 0..``mesh_level``."""
    if mesh_level < 0:
        raise ValueError(f"mesh_level must be >= 0, got {mesh_level}")
    v = icosahedron_vertices(pole_parallel_faces)
    f = icosahedron_faces(v)
    out = [(v, f)]
    for _ in range(mesh_level):
        v, f, _ = refine_mesh(v, f)
        out.append((v, f))
    return out


def build_mesh_faces(
    mesh_level: int = 2, use_multi_mesh: bool = True, pole_parallel_faces: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Finest vertices and faces (concatenated over levels if multi-mesh)."""
    hierarchy = build_mesh_hierarchy(mesh_level, pole_parallel_faces)
    vertices = hierarchy[-1][0]
    if use_multi_mesh:
        faces = np.concatenate([f for _, f in hierarchy], axis=0)
    else:
        faces = hierarchy[-1][1]
    return vertices, faces


def build_multimesh(
    mesh_level: int = 2, pole_parallel_faces: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Finest vertices, undirected multi-mesh edges, undirected edges per level."""
    hierarchy = build_mesh_hierarchy(mesh_level, pole_parallel_faces)
    per_level = []
    for _, f in hierarchy:
        s, r = faces_to_edges(f)
        per_level.append(np.unique(np.stack([np.minimum(s, r), np.maximum(s, r)], 1), axis=0))
    multi = np.unique(np.concatenate(per_level, 0), axis=0)
    return hierarchy[-1][0], multi, per_level


def max_edge_length(vertices: np.ndarray, faces: np.ndarray) -> np.floating:
    s, r = faces_to_edges(faces)
    return np.linalg.norm(vertices[s] - vertices[r], axis=-1).max()


# --------------------------------------------------------------------------
# Coordinates
# --------------------------------------------------------------------------

def lat_lon_deg_to_spherical(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(phi=lon rad, theta=colatitude rad)."""
    return np.deg2rad(lon), np.deg2rad(90 - lat)


def spherical_to_cartesian(phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)], axis=-1
    )


def cartesian_to_lat_lon(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Unit-sphere xyz -> (lat, lon) degrees, lon in [0, 360)."""
    phi = np.arctan2(xyz[..., 1], xyz[..., 0])
    with np.errstate(invalid="ignore"):
        theta = np.arccos(xyz[..., 2])
    return 90 - np.rad2deg(theta), np.mod(np.rad2deg(phi), 360)


def default_grid_lat_lon(img_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Cell-centred equiangular grid used by WeatherAI: lat ascending, lon from 180/W.

    Returns float32 1-D ``(lat[H], lon[W])`` in degrees.
    """
    h, w = int(img_size[0]), int(img_size[1])
    lat = np.linspace(-90.0 + 90.0 / h, 90.0 - 90.0 / h, h)
    lon = np.linspace(0.0, 360.0, w, endpoint=False) + 180.0 / w
    return lat.astype(np.float32), lon.astype(np.float32)


def grid_lat_lon_to_xyz(grid_lat: np.ndarray, grid_lon: np.ndarray) -> np.ndarray:
    """1-D lat [H], lon [W] -> (H*W, 3) xyz, lat-major (row = i_lat*W + i_lon)."""
    grid_lat = np.asarray(grid_lat, dtype=np.float32)
    grid_lon = np.asarray(grid_lon, dtype=np.float32)
    phi, theta = np.meshgrid(np.deg2rad(grid_lon), np.deg2rad(90 - grid_lat))
    return spherical_to_cartesian(phi, theta).reshape(-1, 3)


def grid_node_lat_lon(grid_lat: np.ndarray, grid_lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Flattened (lat-major) per-node lat, lon, float32."""
    lon_g, lat_g = np.meshgrid(
        np.asarray(grid_lon, dtype=np.float32), np.asarray(grid_lat, dtype=np.float32)
    )
    return lat_g.reshape(-1).astype(np.float32), lon_g.reshape(-1).astype(np.float32)


def latlon_grid_nodes(img_size: Tuple[int, int]) -> np.ndarray:
    """Default-grid xyz, shape (H*W, 3)."""
    return grid_lat_lon_to_xyz(*default_grid_lat_lon(img_size))


# --------------------------------------------------------------------------
# Grid <-> mesh connectivity
# --------------------------------------------------------------------------

def radius_query_edges(
    grid_xyz: np.ndarray, mesh_xyz: np.ndarray, radius: float, chunk: int = 4096
) -> Tuple[np.ndarray, np.ndarray]:
    """All (grid, mesh) pairs with Euclidean distance <= ``radius``.

    Ordered by grid index then ascending mesh index. Distances are evaluated in
    float64. Returns ``(grid_indices, mesh_indices)``.
    """
    g = np.asarray(grid_xyz, dtype=np.float64)
    m = np.asarray(mesh_xyz, dtype=np.float64)
    r2 = float(radius) ** 2
    gi, mi = [], []
    for start in range(0, g.shape[0], chunk):
        d2 = np.sum((g[start:start + chunk, None, :] - m[None, :, :]) ** 2, axis=-1)
        rows, cols = np.nonzero(d2 <= r2)  # row-major -> sorted by (grid, mesh)
        gi.append(rows + start)
        mi.append(cols)
    return (
        np.concatenate(gi).astype(np.int64),
        np.concatenate(mi).astype(np.int64),
    )


def _closest_point_sq_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Squared distance from points p[n,1,3] to triangles (a,b,c)[1,f,3] -> [n,f]."""
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    dot = lambda u, v: np.sum(u * v, axis=-1)  # noqa: E731
    d1, d2 = dot(ab, ap), dot(ac, ap)
    d3, d4 = dot(ab, bp), dot(ac, bp)
    d5, d6 = dot(ab, cp), dot(ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    with np.errstate(divide="ignore", invalid="ignore"):
        v_ab = d1 / (d1 - d3)
        w_ac = d2 / (d2 - d6)
        w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        denom = 1.0 / (va + vb + vc)
    v_in, w_in = vb * denom, vc * denom
    zero = np.zeros_like(d1)
    conds = [
        (d1 <= 0) & (d2 <= 0),                                 # vertex a
        (d3 >= 0) & (d4 <= d3),                                # vertex b
        (vc <= 0) & (d1 >= 0) & (d3 <= 0),                     # edge ab
        (d6 >= 0) & (d5 <= d6),                                # vertex c
        (vb <= 0) & (d2 >= 0) & (d6 <= 0),                     # edge ac
        (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),       # edge bc
    ]
    # Closest point = a + s*ab + t*ac for every region.
    s_coef = np.select(conds, [zero, zero + 1, v_ab, zero, zero, 1 - w_bc], v_in)
    t_coef = np.select(conds, [zero, zero, zero, zero + 1, w_ac, w_bc], w_in)
    q = a + s_coef[..., None] * ab + t_coef[..., None] * ac
    return np.sum((p - q) ** 2, axis=-1)


def closest_triangle_sq_distances(
    grid_xyz: np.ndarray, mesh_xyz: np.ndarray, faces: np.ndarray
) -> np.ndarray:
    """Dense (Ng, F) squared point-to-triangle distances (debug / tests)."""
    g = np.asarray(grid_xyz, dtype=np.float64)
    v = np.asarray(mesh_xyz, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    a, b, c = (v[faces[:, k]][None] for k in range(3))
    return _closest_point_sq_dist(g[:, None, :], a, b, c)


def in_mesh_triangle_edges(
    grid_xyz: np.ndarray,
    mesh_xyz: np.ndarray,
    faces: np.ndarray,
    chunk: int = 1024,
    backend: str = "numpy",
) -> Tuple[np.ndarray, np.ndarray]:
    """Connect every grid point to the 3 vertices of its closest mesh triangle.

    Returns ``(grid_indices, mesh_indices)`` with 3 edges per grid point, in
    face vertex order.

    ``backend``:
      * ``"numpy"`` (default): exact closest-point-on-triangle search; exact
        ties (grid point on an edge shared by two faces) resolve to the lowest
        face index. Deterministic, no extra dependency.
      * ``"trimesh"``: delegate to ``trimesh.Trimesh.nearest.on_surface`` (the
        call used by the DeepMind reference). Reproduces its tie-breaking
        exactly; requires ``trimesh`` + ``rtree``.
    Both backends agree on every grid point whose closest face is unique.
    """
    faces = np.asarray(faces, dtype=np.int64)
    if backend == "trimesh":
        import trimesh  # optional dependency (MIT)

        tm = trimesh.Trimesh(vertices=np.asarray(mesh_xyz), faces=faces, process=False)
        _, _, best = tm.nearest.on_surface(np.asarray(grid_xyz))
        best = np.asarray(best, dtype=np.int64)
        mesh_idx = faces[best].reshape(-1)
        grid_idx = np.repeat(np.arange(len(best), dtype=np.int64), 3)
        return grid_idx, mesh_idx
    if backend != "numpy":
        raise ValueError(f"Unknown mesh2grid backend {backend!r}")
    g = np.asarray(grid_xyz, dtype=np.float64)
    v = np.asarray(mesh_xyz, dtype=np.float64)
    a, b, c = (v[faces[:, k]][None] for k in range(3))
    best = np.empty(g.shape[0], dtype=np.int64)
    for start in range(0, g.shape[0], chunk):
        d2 = _closest_point_sq_dist(g[start:start + chunk, None, :], a, b, c)
        best[start:start + chunk] = np.argmin(d2, axis=1)
    mesh_idx = faces[best].reshape(-1)
    grid_idx = np.repeat(np.arange(g.shape[0], dtype=np.int64), 3)
    return grid_idx, mesh_idx


# --------------------------------------------------------------------------
# Spatial features
# --------------------------------------------------------------------------

def node_spatial_features(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """``[cos(colat), cos(lon), sin(lon)]`` per node (float32 in -> float32)."""
    phi, theta = lat_lon_deg_to_spherical(lat, lon)
    return np.stack([np.cos(theta), np.cos(phi), np.sin(phi)], axis=-1)


def _local_rotation_matrices(phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Per-node matrices moving the node to lon=0 (about z) then lat=0 (about y).

    The angles are formed in the input dtype (float32 for GraphCast lat/lon),
    exactly like the DeepMind reference, before building float64 matrices;
    this keeps the structural features within ~1e-15 of the reference instead
    of ~1 float32 ulp.
    """
    azimuthal = -phi
    polar = -theta + np.pi / 2
    return _rot_y(np.asarray(polar, dtype=np.float64)) @ _rot_z(np.asarray(azimuthal, dtype=np.float64))


def edge_spatial_features(
    senders_lat: np.ndarray,
    senders_lon: np.ndarray,
    receivers_lat: np.ndarray,
    receivers_lon: np.ndarray,
    senders: np.ndarray,
    receivers: np.ndarray,
    edge_normalization_factor: Optional[float] = None,
) -> np.ndarray:
    """``[|d|, d] / norm`` with d = sender - receiver in the receiver-local frame.

    ``norm`` defaults to the maximum ``|d|`` over the edge set. Output float64.
    """
    s_phi, s_theta = lat_lon_deg_to_spherical(senders_lat, senders_lon)
    r_phi, r_theta = lat_lon_deg_to_spherical(receivers_lat, receivers_lon)
    s_pos = spherical_to_cartesian(s_phi, s_theta)
    r_pos = spherical_to_cartesian(r_phi, r_theta)
    rot = _local_rotation_matrices(r_phi, r_theta)[receivers]
    rel = np.einsum("...ji,...i->...j", rot, s_pos[senders]) - np.einsum(
        "...ji,...i->...j", rot, r_pos[receivers]
    )
    dist = np.linalg.norm(rel, axis=-1, keepdims=True)
    norm = dist.max() if edge_normalization_factor is None else edge_normalization_factor
    return np.concatenate([dist / norm, rel / norm], axis=-1)


# --------------------------------------------------------------------------
# Full graph bundle
# --------------------------------------------------------------------------

@dataclass
class GraphCastGraphs:
    """Static graph arrays for one (grid, mesh_level) configuration."""

    grid_lat: np.ndarray
    grid_lon: np.ndarray
    mesh_nodes: np.ndarray        # (Nm, 3) xyz
    grid_nodes: np.ndarray        # (Ng, 3) xyz
    mesh_node_features: np.ndarray  # (Nm, 3)
    grid_node_features: np.ndarray  # (Ng, 3)
    mesh_senders: np.ndarray
    mesh_receivers: np.ndarray
    mesh_edge_attr: np.ndarray    # (Em, 4)
    g2m_senders: np.ndarray       # grid indices
    g2m_receivers: np.ndarray     # mesh indices
    g2m_edge_attr: np.ndarray
    m2g_senders: np.ndarray       # mesh indices
    m2g_receivers: np.ndarray     # grid indices
    m2g_edge_attr: np.ndarray
    mesh_level: int
    img_size: Tuple[int, int]
    query_radius: float


def build_graphs(
    img_size: Tuple[int, int] = (32, 64),
    mesh_level: int = 2,
    use_multi_mesh: bool = True,
    radius_query_fraction_edge_length: float = 0.6,
    grid_lat: Optional[np.ndarray] = None,
    grid_lon: Optional[np.ndarray] = None,
    mesh2grid_edge_normalization_factor: Optional[float] = None,
    m2g_backend: str = "numpy",
    g2m_k: Optional[int] = None,
    m2g_k: Optional[int] = None,
) -> GraphCastGraphs:
    """Build grid2mesh / multi-mesh / mesh2grid graphs with GraphCast features.

    ``grid_lat`` / ``grid_lon`` (degrees, 1-D) override the default
    cell-centred grid derived from ``img_size``. ``g2m_k`` / ``m2g_k`` are
    accepted for backwards compatibility only and are ignored (k-NN
    connectivity was replaced by the GraphCast radius / containing-triangle
    rules). ``m2g_backend`` selects the containing-triangle search (see
    ``in_mesh_triangle_edges``).
    """
    if g2m_k is not None or m2g_k is not None:
        warnings.warn(
            "g2m_k / m2g_k are deprecated and ignored: Grid2Mesh uses a radius query "
            "and Mesh2Grid uses containing-triangle connectivity (GraphCast).",
            DeprecationWarning,
            stacklevel=2,
        )
    if grid_lat is None or grid_lon is None:
        grid_lat, grid_lon = default_grid_lat_lon(img_size)
    grid_lat = np.asarray(grid_lat, dtype=np.float32)
    grid_lon = np.asarray(grid_lon, dtype=np.float32)
    img_size = (int(grid_lat.shape[0]), int(grid_lon.shape[0]))

    hierarchy = build_mesh_hierarchy(mesh_level)
    mesh_xyz, finest_faces = hierarchy[-1]
    if use_multi_mesh:
        proc_faces = np.concatenate([f for _, f in hierarchy], axis=0)
    else:
        proc_faces = finest_faces

    mesh_lat, mesh_lon = cartesian_to_lat_lon(mesh_xyz)
    mesh_lat, mesh_lon = mesh_lat.astype(np.float32), mesh_lon.astype(np.float32)
    g_lat, g_lon = grid_node_lat_lon(grid_lat, grid_lon)
    grid_xyz = grid_lat_lon_to_xyz(grid_lat, grid_lon)

    # Processor multi-mesh.
    ms, mr = faces_to_edges(proc_faces)
    mesh_edge = edge_spatial_features(mesh_lat, mesh_lon, mesh_lat, mesh_lon, ms, mr)

    # Grid2Mesh radius query (radius relative to the finest mesh edge length).
    radius = max_edge_length(mesh_xyz, finest_faces) * radius_query_fraction_edge_length
    gs, gr = radius_query_edges(grid_xyz, mesh_xyz, radius)
    g2m_edge = edge_spatial_features(g_lat, g_lon, mesh_lat, mesh_lon, gs, gr)

    # Mesh2Grid containing triangle on the finest mesh.
    m2g_grid, m2g_mesh = in_mesh_triangle_edges(
        grid_xyz, mesh_xyz, finest_faces, backend=m2g_backend
    )
    m2g_edge = edge_spatial_features(
        mesh_lat, mesh_lon, g_lat, g_lon, m2g_mesh, m2g_grid,
        edge_normalization_factor=mesh2grid_edge_normalization_factor,
    )

    return GraphCastGraphs(
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        mesh_nodes=mesh_xyz.astype(np.float32),
        grid_nodes=grid_xyz.astype(np.float32),
        mesh_node_features=node_spatial_features(mesh_lat, mesh_lon).astype(np.float32),
        grid_node_features=node_spatial_features(g_lat, g_lon).astype(np.float32),
        mesh_senders=ms,
        mesh_receivers=mr,
        mesh_edge_attr=mesh_edge.astype(np.float32),
        g2m_senders=gs,
        g2m_receivers=gr,
        g2m_edge_attr=g2m_edge.astype(np.float32),
        m2g_senders=m2g_mesh,
        m2g_receivers=m2g_grid,
        m2g_edge_attr=m2g_edge.astype(np.float32),
        mesh_level=int(mesh_level),
        img_size=img_size,
        query_radius=float(radius),
    )


def mesh_level_summary(max_level: int = 6) -> Sequence[Tuple[int, int]]:
    """``(level, num_nodes)`` pairs for documentation."""
    return [(L, mesh_num_nodes(L)) for L in range(max_level + 1)]
