"""Print the GraphCast stage × framework parity matrix (markdown).

Run in an environment with JAX GraphCast + PhysicsNeMo (the "parity venv")::

    python -m tests.parity.graphcast.report

Numbers are max over the tiny cases in ``test_stage_parity.CASES`` (fp32
unless marked fp64), max-abs / max-rel as in ``_helpers.max_abs_rel``.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import _refs as R
from ._helpers import max_abs_rel
from .test_stage_parity import CASES, _FP64


def _agg(store, key, a, b):
    ab, rl = max_abs_rel(a, b)
    o = store[key]
    store[key] = (max(o[0], ab), max(o[1], rl))


def main() -> None:
    S = defaultdict(lambda: (0.0, 0.0))
    for c in CASES:
        ref = R.jax_reference(c)
        n = f"N={c.n_steps}"
        ours = R.ours_model(c, ref)
        iso_g2m = R.ours_grid2mesh(ours, ref["x"])
        iso_m2g = R.ours_mesh2grid(ours, ref["mesh_proc"], ref["grid_g2m"])
        for k in R.STAGE_KEYS["grid2mesh"]:
            _agg(S, ("ours-jax", "grid2mesh"), iso_g2m[k], ref[k])
        for k in R.STAGE_KEYS["mesh2grid"]:
            _agg(S, ("ours-jax", "mesh2grid"), iso_m2g[k], ref[k])
        if c.n_steps <= 4:
            iso_p = R.ours_processor(ours, ref["mesh_g2m"])
            ch = R.ours_chained(ours, ref["x"])
            for k in R.STAGE_KEYS["processor"]:
                _agg(S, ("ours-jax", f"processor {n}"), iso_p[k], ref[k])
            _agg(S, ("ours-jax", "chained out (N<=4)"), ch["out"], ref["out"])
        else:
            c64 = _FP64(c)
            r64 = R.jax_reference(c64)
            m64 = R.ours_model(c64, r64)
            iso_p = R.ours_processor(m64, r64["mesh_g2m"])
            ch64 = R.ours_chained(m64, r64["x"])
            for k in R.STAGE_KEYS["processor"]:
                _agg(S, ("ours-jax", f"processor {n} (fp64)"), iso_p[k], r64[k])
                _agg(S, ("ours-jax", f"processor {n} (fp32 info)"), R.ours_processor(ours, ref["mesh_g2m"])[k], ref[k])
                _agg(S, ("jax32-jax64", f"processor {n}"), ref[k], r64[k])
            _agg(S, ("ours-jax", "chained out N=16 (fp64)"), ch64["out"], r64["out"])
        # JAX <-> NV (JAX weights + JAX graphs)
        for trick in (False, True):
            nv = R.NVStages(c, ref, do_concat_trick=trick)
            iso = R.nv_isolated(nv, ref)
            t = " (concat_trick)" if trick else ""
            for k in R.STAGE_KEYS["grid2mesh"]:
                _agg(S, ("jax-nv", "grid2mesh" + t), iso[k], ref[k])
            for k in R.STAGE_KEYS["mesh2grid"]:
                _agg(S, ("jax-nv", "mesh2grid" + t), iso[k], ref[k])
            if not trick:
                for k in R.STAGE_KEYS["processor"]:
                    _agg(S, ("jax-nv", f"processor {n}"), iso[k], ref[k])
                upd = R.ours_processor(R.ours_model(c, ref, aggregate="updated"), ref["mesh_g2m"])
                for k in R.STAGE_KEYS["processor"]:
                    _agg(S, ("nv-ours(updated)", f"processor {n}"), iso[k], upd[k])
        # ours <-> NV (synthetic weights, WeatherAI graphs, no JAX involved)
        sref = R.synthetic_reference(c)
        so = R.ours_chained(R.ours_model(c, sref, m2g_backend="numpy"), sref["x"])
        for trick in (False, True):
            nv = R.NVStages(c, sref, do_concat_trick=trick)
            iso = R.nv_isolated(nv, dict(so, x=sref["x"]))
            t = " (concat_trick)" if trick else ""
            for k in R.STAGE_KEYS["grid2mesh"]:
                _agg(S, ("ours-nv", "grid2mesh" + t), iso[k], so[k])
            for k in R.STAGE_KEYS["mesh2grid"]:
                _agg(S, ("ours-nv", "mesh2grid" + t), iso[k], so[k])
            if not trick:
                for k in R.STAGE_KEYS["processor"]:
                    _agg(S, ("ours-nv", f"processor {n}"), iso[k], so[k])
                upd = R.ours_model(c, sref, m2g_backend="numpy", aggregate="updated")
                for k in R.STAGE_KEYS["processor"]:
                    _agg(S, ("ours(updated)-nv", f"processor {n}"), iso[k], R.ours_processor(upd, so["mesh_g2m"])[k])
                _agg(S, ("ours(updated)-nv", "chained out"), R.nv_chained(nv, sref["x"])["out"],
                     R.ours_chained(upd, sref["x"])["out"])
    for (pair, stage), (ab, rl) in sorted(S.items()):
        print(f"| {pair} | {stage} | {ab:.1e} | {rl:.1e} |")


if __name__ == "__main__":
    main()
