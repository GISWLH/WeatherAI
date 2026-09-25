"""Stage-wise GraphCast parity: Grid2Mesh → Processor (N layers) → Mesh2Grid.

Three-way: WeatherAI ↔ DeepMind JAX GraphCast, WeatherAI ↔ NVIDIA PhysicsNeMo,
JAX ↔ PhysicsNeMo. Identical (randomised) weights are mapped into every
implementation; fp32 tolerances atol=1e-5, rtol=1e-4 (see _helpers).

"isolated" = each stage is fed the *reference's* inputs for that stage;
"chained"  = each implementation runs end-to-end on the same raw inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from . import _refs as R
from ._helpers import ATOL, assert_close, max_abs_rel, requires_jax, requires_physicsnemo

CASES = [
    R.case("4x8", 0, 8, 1),
    R.case("4x8", 1, 16, 2),
    R.case("6x12", 0, 16, 4),
    R.case("6x12", 1, 8, 4),
    R.case("poles5x8", 1, 8, 2),
    R.case("4x8", 0, 8, 16),
    R.case("6x12", 1, 8, 16),
]
PROC_REASON = (
    "INHERENT: PhysicsNeMo MeshNodeBlock aggregates the residual-updated edge latents "
    "(e + Δe) from MeshEdgeBlock, JAX GraphCast / jraph aggregate the edge-MLP output Δe. "
    "Not configurable in GraphCastProcessor."
)
STAGES = list(R.STAGE_KEYS)


def _cmp(stage, got, ref, label):
    for k in R.STAGE_KEYS[stage]:
        assert_close(f"{label}:{stage}:{k}", got[k], ref[k])


# ---------------------------------------------------------------- ours ↔ JAX
# Exact-tolerance checks: fp32 for N <= 4. For N = 16 the fp32 rounding that
# accumulates over 16 residual layers is ~3-5e-5 on edge latents of magnitude
# ~15 -- in JAX itself (JAX-fp32 vs JAX-fp64) just as in WeatherAI -- so the
# N = 16 exact checks (isolated and chained) run in fp64 (same atol/rtol;
# measured <= 1.5e-13) and the fp32 N = 16 run is held to "no less accurate
# than JAX-fp32" below.
_FP64 = lambda c: R.Case(c.name, c.grid, c.mesh_level, c.latent, c.n_steps, c.n_in, c.n_out, c.seed, "float64")
CHAINED = [c for c in CASES if c.n_steps <= 4] + [_FP64(c) for c in CASES if c.n_steps > 4] + [_FP64(CASES[1])]
ISOLATED = [c for c in CASES if c.n_steps <= 4] + [_FP64(c) for c in CASES if c.n_steps > 4]


@requires_jax
@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("c", ISOLATED, ids=str)
def test_ours_vs_jax_isolated(c, stage):
    ref = R.jax_reference(c)
    model = R.ours_model(c, ref)
    if stage == "grid2mesh":
        got = R.ours_grid2mesh(model, ref["x"])
    elif stage == "processor":
        got = R.ours_processor(model, ref["mesh_g2m"])
    else:
        got = R.ours_mesh2grid(model, ref["mesh_proc"], ref["grid_g2m"])
    _cmp(stage, got, ref, "ours-vs-jax")


@requires_jax
@pytest.mark.parametrize("c", CHAINED, ids=str)
def test_ours_vs_jax_chained_and_forward(c):
    import torch

    ref = R.jax_reference(c)
    model = R.ours_model(c, ref)
    got = R.ours_chained(model, ref["x"])
    for stage in STAGES:
        _cmp(stage, got, ref, "ours-vs-jax-chained")
    # Public (B, C, H, W) forward uses lat-major node order like JAX.
    H, W = model.img_size
    x = torch.from_numpy(ref["x"].T.reshape(1, c.n_in, H, W).copy())
    with torch.no_grad():
        y = model(x)[0].reshape(c.n_out, H * W).T.numpy()
    assert_close("ours-vs-jax forward()", y, ref["out"])


@requires_jax
@pytest.mark.parametrize("c", [c for c in CASES if c.n_steps > 4], ids=str)
def test_ours_fp32_deep_processor_as_accurate_as_jax_fp32(c):
    """N=16, fp32: error of WeatherAI-fp32 w.r.t. JAX-fp64 must not exceed 2x
    the error of JAX-fp32 itself w.r.t. JAX-fp64 (per stage key), both chained
    and for the processor stage alone."""
    ref32, ref64 = R.jax_reference(c), R.jax_reference(_FP64(c))
    model = R.ours_model(c, ref32)
    got = R.ours_chained(model, ref32["x"])
    for k in ("mesh_g2m", "g2m_edges", "mesh_proc", "mesh_edges", "out", "m2g_edges"):
        ours_err, _ = max_abs_rel(got[k], ref64[k])
        jax_err, _ = max_abs_rel(ref32[k], ref64[k])
        assert ours_err <= 2 * max(jax_err, ATOL / 10), (k, ours_err, jax_err)
    # Processor alone: WeatherAI-fp32 on the fp64 chain's processor input
    # (cast to fp32), against the fp64 processor output.
    iso = R.ours_processor(model, ref64["mesh_g2m"].astype(np.float32))
    for k in ("mesh_proc", "mesh_edges"):
        ours_err, _ = max_abs_rel(iso[k], ref64[k])
        jax_err, _ = max_abs_rel(ref32[k], ref64[k])
        assert ours_err <= 2 * max(jax_err, ATOL / 10), (k, ours_err, jax_err)


@requires_jax
@pytest.mark.parametrize("c", [R.case("6x12c", 0, 16, 4)], ids=str)
def test_ours_default_numpy_backend_vs_jax_on_tie_free_grid(c):
    """On a grid without shared-edge ties the default backend is identical too."""
    ref = R.jax_reference(c)
    model = R.ours_model(c, ref, m2g_backend="numpy")
    np.testing.assert_array_equal(model.m2g_senders.numpy(), ref["m2g_senders"])
    got = R.ours_chained(model, ref["x"])
    for stage in STAGES:
        _cmp(stage, got, ref, "ours(numpy)-vs-jax")


# ----------------------------------------------------------- JAX ↔ PhysicsNeMo
@requires_jax
@requires_physicsnemo
@pytest.mark.parametrize("concat_trick", [False, True], ids=["concat", "concat_trick"])
@pytest.mark.parametrize("stage", ["grid2mesh", "mesh2grid"])
@pytest.mark.parametrize("c", CASES, ids=str)
def test_physicsnemo_vs_jax_isolated(c, stage, concat_trick):
    ref = R.jax_reference(c)
    got = R.nv_isolated(R.NVStages(c, ref, do_concat_trick=concat_trick), ref)
    _cmp(stage, got, ref, "nv-vs-jax")


@requires_jax
@requires_physicsnemo
@pytest.mark.xfail(strict=True, reason=PROC_REASON)
@pytest.mark.parametrize("c", CASES, ids=str)
def test_physicsnemo_vs_jax_processor(c):
    ref = R.jax_reference(c)
    got = R.nv_isolated(R.NVStages(c, ref), ref)
    _cmp("processor", got, ref, "nv-vs-jax")


@requires_jax
@requires_physicsnemo
@pytest.mark.parametrize("concat_trick", [False, True], ids=["concat", "concat_trick"])
@pytest.mark.parametrize("c", CASES, ids=str)
def test_physicsnemo_processor_known_difference_pinned(c, concat_trick):
    """Pin the *cause*: NV processor == WeatherAI with aggregate='updated' (e+Δe),
    and differs from JAX by far more than fp32 noise."""
    ref = R.jax_reference(c)
    got = R.nv_isolated(R.NVStages(c, ref, do_concat_trick=concat_trick), ref)
    abs_err, _ = max_abs_rel(got["mesh_proc"], ref["mesh_proc"])
    assert abs_err > 1e3 * ATOL, abs_err
    ours_upd = R.ours_processor(R.ours_model(c, ref, aggregate="updated"), ref["mesh_g2m"])
    _cmp("processor", got, ours_upd, "nv-vs-ours(updated)")
    if c.n_steps == 1:  # first edge update is identical, only nodes differ
        assert_close("nv-vs-jax:N=1 edges", got["mesh_edges"], ref["mesh_edges"])


@requires_jax
@requires_physicsnemo
@pytest.mark.xfail(strict=True, reason="end-to-end inherits the processor difference: " + PROC_REASON)
@pytest.mark.parametrize("c", CASES[:3], ids=str)
def test_physicsnemo_vs_jax_chained_output(c):
    ref = R.jax_reference(c)
    got = R.nv_chained(R.NVStages(c, ref), ref["x"])
    assert_close("nv-vs-jax chained out", got["out"], ref["out"])


# --------------------------------------------------------- ours ↔ PhysicsNeMo
# JAX-free: synthetic Haiku-named weights + WeatherAI (numpy-backend) graphs.
@requires_physicsnemo
@pytest.mark.parametrize("concat_trick", [False, True], ids=["concat", "concat_trick"])
@pytest.mark.parametrize("c", CASES, ids=str)
def test_ours_vs_physicsnemo(c, concat_trick):
    ref = R.synthetic_reference(c)
    nv = R.NVStages(c, ref, do_concat_trick=concat_trick)
    ours = R.ours_chained(R.ours_model(c, ref, m2g_backend="numpy"), ref["x"])
    # Grid2Mesh + Mesh2Grid (fed WeatherAI's stage inputs): must match.
    iso = R.nv_isolated(nv, dict(ours, x=ref["x"]))
    _cmp("grid2mesh", iso, ours, "nv-vs-ours")
    _cmp("mesh2grid", iso, ours, "nv-vs-ours")
    # Processor: canonical (JAX) path differs; the NV-semantics switch matches.
    abs_err, _ = max_abs_rel(iso["mesh_proc"], ours["mesh_proc"])
    assert abs_err > 1e3 * ATOL, abs_err
    upd_model = R.ours_model(c, ref, m2g_backend="numpy", aggregate="updated")
    _cmp("processor", iso, R.ours_processor(upd_model, ours["mesh_g2m"]), "nv-vs-ours(updated)")
    # Fully chained with NV semantics: identical end-to-end.
    _cmp("mesh2grid", R.nv_chained(nv, ref["x"]), R.ours_chained(upd_model, ref["x"]), "nv-vs-ours(updated)-chained")
