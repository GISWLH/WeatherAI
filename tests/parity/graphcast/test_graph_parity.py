"""Graph construction parity: WeatherAI vs DeepMind JAX GraphCast vs PhysicsNeMo.

Covers grid2mesh (radius query), multi-mesh processor edges, mesh2grid
(containing triangle) connectivity and the node / edge structural features.
"""

from __future__ import annotations

import numpy as np
import pytest

from weatherai.models.graphcast import mesh as M

from . import _refs as R
from ._helpers import assert_close, requires_jax, requires_physicsnemo, requires_trimesh

GRIDS = ["4x8", "6x12", "poles5x8", "16x32", "poles19x36"]
LEVELS = [0, 1, 2, 3]


def _pairs(s, r):
    return set(zip(np.asarray(s).tolist(), np.asarray(r).tolist()))


def _ours(c: R.Case, backend: str):
    lat, lon = c.lat_lon()
    return M.build_graphs(mesh_level=c.mesh_level, grid_lat=lat, grid_lon=lon, m2g_backend=backend)


def _jax(grid: str, level: int):
    return R.jax_reference(R.case(grid, level, 8, 1))


@requires_jax
@requires_trimesh
@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("grid", GRIDS)
def test_ours_trimesh_backend_equals_jax_graphs(grid, level):
    """With m2g_backend='trimesh' every edge index is identical (order included)."""
    c = R.case(grid, level, 8, 1)
    ref, ours = _jax(grid, level), _ours(c, "trimesh")
    for name in ("g2m", "mesh", "m2g"):
        np.testing.assert_array_equal(getattr(ours, f"{name}_senders"), ref[f"{name}_senders"], err_msg=name)
        np.testing.assert_array_equal(getattr(ours, f"{name}_receivers"), ref[f"{name}_receivers"], err_msg=name)
        assert_close(f"{name}_edge_attr", getattr(ours, f"{name}_edge_attr"), ref[f"{name}_edge_attr"])
    assert_close("grid_node_features", ours.grid_node_features, ref["grid_node_features"])
    assert_close("mesh_node_features", ours.mesh_node_features, ref["mesh_node_features"])


@requires_jax
@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("grid", GRIDS)
def test_ours_numpy_backend_differs_from_jax_only_on_exact_ties(grid, level):
    """Default dependency-free backend: grid2mesh / mesh identical; mesh2grid rows
    differ only where the grid point lies on an edge shared by two faces (both
    faces contain the point; trimesh's pick depends on rtree candidate order)."""
    c = R.case(grid, level, 8, 1)
    ref, ours = _jax(grid, level), _ours(c, "numpy")
    for name in ("g2m", "mesh"):
        np.testing.assert_array_equal(getattr(ours, f"{name}_senders"), ref[f"{name}_senders"])
        np.testing.assert_array_equal(getattr(ours, f"{name}_receivers"), ref[f"{name}_receivers"])
    np.testing.assert_array_equal(ours.m2g_receivers, ref["m2g_receivers"])
    a, b = ours.m2g_senders.reshape(-1, 3), ref["m2g_senders"].reshape(-1, 3)
    rows = np.where((a != b).any(1))[0]
    vx = ours.mesh_nodes.astype(np.float64)
    for i in rows:
        p = ours.grid_nodes[i:i + 1]
        da = M.closest_triangle_sq_distances(p, vx, a[i:i + 1])[0, 0]
        db = M.closest_triangle_sq_distances(p, vx, b[i:i + 1])[0, 0]
        assert abs(da - db) < 1e-12, (i, da, db)


# ------------------------------------------------------------- PhysicsNeMo
# Rows of PhysicsNeMo's mesh2grid that differ from DeepMind's containing-triangle
# rule: (total differing rows, of which genuinely non-containing faces).
# PhysicsNeMo picks the face with the nearest *centroid* (sklearn 1-NN over the
# multimesh faces); the rest of the differing rows are exact shared-edge ties.
NV_M2G_DIFF = {
    ("4x8", 0): (0, 0), ("4x8", 1): (1, 0), ("4x8", 2): (0, 0),
    ("6x12", 0): (3, 0), ("6x12", 1): (6, 0), ("6x12", 2): (2, 0),
    ("poles5x8", 0): (3, 0), ("poles5x8", 1): (3, 0), ("poles5x8", 2): (7, 4),
    ("16x32", 0): (4, 0), ("16x32", 1): (24, 16), ("16x32", 2): (31, 24),
}


@requires_jax
@requires_physicsnemo
@pytest.mark.parametrize("grid,level", sorted(NV_M2G_DIFF))
def test_physicsnemo_graphs_vs_jax(grid, level):
    c = R.case(grid, level, 8, 1)
    ref = _jax(grid, level)
    _, mesh, g2m, m2g = R.physicsnemo_graph(c)

    # Multi-mesh: identical edge set (PyG to_undirected reorders); features are
    # (dx, dy, dz, |d|) vs JAX (|d|, dx, dy, dz) -> permutation only.
    ms, mr = mesh.edge_index.numpy()
    assert _pairs(ms, mr) == _pairs(ref["mesh_senders"], ref["mesh_receivers"])
    _check_edge_feats("mesh", ms, mr, mesh.edge_attr.numpy(), ref)

    # Grid2Mesh: 4-NN capped radius query equals the plain radius query here
    # (max JAX in-degree per grid node <= 3, so the cap never triggers).
    gs, gr = g2m[g2m.edge_types[0]].edge_index.numpy()
    assert np.bincount(ref["g2m_senders"]).max() <= 4
    assert _pairs(gs, gr) == _pairs(ref["g2m_senders"], ref["g2m_receivers"])
    _check_edge_feats("g2m", gs, gr, g2m.edge_attr.numpy(), ref)

    # Mesh2Grid: KNOWN DIFFERENCE (nearest face centroid vs containing triangle).
    ns, nr = m2g[m2g.edge_types[0]].edge_index.numpy()
    np.testing.assert_array_equal(nr, ref["m2g_receivers"])
    a, b = ns.reshape(-1, 3), ref["m2g_senders"].reshape(-1, 3)
    rows = np.where((np.sort(a, 1) != np.sort(b, 1)).any(1))[0]
    vx, _ = M.build_mesh_hierarchy(level)[-1]
    lat, lon = c.lat_lon()
    gx = M.grid_lat_lon_to_xyz(lat, lon)
    genuine = 0
    for i in rows:
        da = M.closest_triangle_sq_distances(gx[i:i + 1], vx, a[i:i + 1])[0, 0]
        db = M.closest_triangle_sq_distances(gx[i:i + 1], vx, b[i:i + 1])[0, 0]
        assert da >= db - 1e-12  # JAX's face is always a closest one
        genuine += int(da - db > 1e-12)
    assert (len(rows), genuine) == NV_M2G_DIFF[(grid, level)]
    # Edge features on common rows. KNOWN DIFFERENCE at pole grid points
    # (|lat| = 90): the receiver-local frame is degenerate; JAX (and WeatherAI)
    # use the grid's given longitude, PhysicsNeMo re-derives it from float32
    # xyz via arctan2 (cos(90deg) ~ -4e-8), so the frame rotates arbitrarily.
    grid_lat = np.repeat(lat, lon.size)
    pole = np.abs(grid_lat) == 90.0
    same = np.setdiff1d(np.arange(c.num_grid), rows)
    keep = np.isin(nr, same) & ~pole[nr]
    _check_edge_feats("m2g", ns[keep], nr[keep], m2g.edge_attr.numpy()[keep], ref)
    if pole.any():
        at_pole = np.isin(nr, same) & pole[nr]
        with np.errstate(invalid="ignore", divide="ignore"):
            d = _edge_dir_diff(ns[at_pole], nr[at_pole], m2g.edge_attr.numpy()[at_pole], ref)
        assert np.nanmax(d) > 0.1

    # Node features: KNOWN DIFFERENCE. PhysicsNeMo's PyG backend computes
    # ``xyz2latlon(pos)`` (default unit="deg") and then applies torch.cos/sin to
    # the *degree* values, i.e. (cos(lat_deg), sin(lon_deg), cos(lon_deg)) with
    # degrees interpreted as radians. JAX uses (sin lat, cos lon, sin lon) in
    # radians. Pinned via an independent float32 re-derivation.
    import torch

    pos = torch.from_numpy(M.build_mesh_hierarchy(level)[-1][0].astype(np.float32))
    lat_deg = torch.arcsin(pos[:, 2]) * (180.0 / np.pi)
    lon_deg = torch.arctan2(pos[:, 1], pos[:, 0]) * (180.0 / np.pi)
    expected_nv = torch.stack([torch.cos(lat_deg), torch.sin(lon_deg), torch.cos(lon_deg)], -1)
    assert_close("NV mesh node features == trig(degrees)", mesh.x.numpy(), expected_nv.numpy())
    jx = ref["mesh_node_features"]
    assert np.abs(mesh.x.numpy()[:, 0] - jx[:, 0]).max() > 0.5  # really different from JAX


def _edge_dir_diff(s, r, nv_attr, ref):
    idx = {k: i for i, k in enumerate(zip(ref["m2g_senders"].tolist(), ref["m2g_receivers"].tolist()))}
    order = np.array([idx[k] for k in zip(np.asarray(s).tolist(), np.asarray(r).tolist())], dtype=np.int64)
    a = np.asarray(nv_attr, dtype=np.float64)[:, [3, 0, 1, 2]]
    b = ref["m2g_edge_attr"][order].astype(np.float64)
    return np.abs(a[:, 1:] / a[:, :1] - b[:, 1:] / b[:, :1]).max(axis=1)


def _check_edge_feats(name, s, r, nv_attr, ref):
    """Compare PhysicsNeMo (dx, dy, dz, |d|) features with JAX (|d|, dx, dy, dz)
    on the listed edges. If the edge sets are equal the max-length normalisation
    is identical and features are compared directly; otherwise (mesh2grid with
    different rows) only the scale-free unit direction d/|d| is compared."""
    idx = {k: i for i, k in enumerate(zip(ref[f"{name}_senders"].tolist(), ref[f"{name}_receivers"].tolist()))}
    order = np.array([idx[k] for k in zip(np.asarray(s).tolist(), np.asarray(r).tolist())], dtype=np.int64)
    jax_attr = ref[f"{name}_edge_attr"][order].astype(np.float64)
    nv_perm = np.asarray(nv_attr, dtype=np.float64)[:, [3, 0, 1, 2]]
    if len(order) == len(ref[f"{name}_senders"]):
        assert_close(f"{name}_edge_attr (NV, permuted)", nv_perm, jax_attr)
    else:
        # Zero-length edges (grid point coincides with a mesh vertex) have no
        # direction: require both lengths to be ~0 and compare the rest.
        zero = jax_attr[:, 0] < 1e-6
        assert (nv_perm[zero, 0] < 1e-6).all()
        a, b = nv_perm[~zero], jax_attr[~zero]
        assert_close(f"{name}_edge_dir (NV)", a[:, 1:] / a[:, :1], b[:, 1:] / b[:, :1])
