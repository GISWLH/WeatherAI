"""Helpers for GraphCast JAX↔Torch parity tests (not part of the MIT package)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pytest

# Official DeepMind clone (reference + tests only). Override with env var.
_DEFAULT_JAX_ROOT = Path("/workspace/tmp/graphcast-jax")
_JAX_ROOT = Path(os.environ.get("WEATHERAI_GRAPHCAST_JAX_ROOT", str(_DEFAULT_JAX_ROOT)))

ATOL = 1e-5
RTOL = 1e-4


def jax_stack_available() -> bool:
    try:
        import jax  # noqa: F401
        import haiku  # noqa: F401
        import jraph  # noqa: F401
    except ImportError:
        return False
    return jax_ref_available()


def jax_ref_available() -> bool:
    return (_JAX_ROOT / "weathernext" / "utils" / "icosahedral_mesh.py").is_file()


requires_jax_parity = pytest.mark.skipif(
    not jax_stack_available(),
    reason="JAX/Haiku/jraph + /workspace/tmp/graphcast-jax clone required "
    "(pip install -e '.[parity]'; clone google-deepmind/graphcast)",
)


def ensure_jax_ref_on_path() -> Path:
    if not jax_ref_available():
        raise FileNotFoundError(
            f"DeepMind GraphCast/WeatherNext clone not found at {_JAX_ROOT}. "
            "Clone https://github.com/google-deepmind/graphcast into that path "
            "or set WEATHERAI_GRAPHCAST_JAX_ROOT."
        )
    root = str(_JAX_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    return _JAX_ROOT


def flatten_haiku_params(params: Mapping[str, Any], prefix: str = "") -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, value in params.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(flatten_haiku_params(value, path))
        else:
            out[path] = np.asarray(value)
    return out


def load_haiku_mlp_into_torch(
    torch_mlp,
    flat: Mapping[str, np.ndarray],
    mlp_prefix: str,
    ln_prefix: Optional[str] = None,
) -> None:
    """Copy Haiku MLP (+ optional LayerNorm) weights into weatherai ``MLP``.

    Haiku Linear ``w`` is (in, out); Torch Linear ``weight`` is (out, in).
    """
    import torch

    w0 = flat[f"{mlp_prefix}/~/linear_0/w"]
    b0 = flat[f"{mlp_prefix}/~/linear_0/b"]
    w1 = flat[f"{mlp_prefix}/~/linear_1/w"]
    b1 = flat[f"{mlp_prefix}/~/linear_1/b"]
    # Sequential: Linear, Act, Linear, ...
    linears = [m for m in torch_mlp.net if isinstance(m, torch.nn.Linear)]
    if len(linears) < 2:
        raise ValueError(f"Expected >=2 Linear layers, got {len(linears)}")
    with torch.no_grad():
        linears[0].weight.copy_(torch.from_numpy(np.array(w0, copy=True).T))
        linears[0].bias.copy_(torch.from_numpy(np.array(b0, copy=True)))
        linears[1].weight.copy_(torch.from_numpy(np.array(w1, copy=True).T))
        linears[1].bias.copy_(torch.from_numpy(np.array(b1, copy=True)))
        if ln_prefix is not None and hasattr(torch_mlp.norm, "weight"):
            torch_mlp.norm.weight.copy_(torch.from_numpy(np.array(flat[f"{ln_prefix}/scale"], copy=True)))
            torch_mlp.norm.bias.copy_(torch.from_numpy(np.array(flat[f"{ln_prefix}/offset"], copy=True)))


def max_abs_rel(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = np.abs(a - b)
    rel = diff / np.maximum(np.abs(b), 1e-8)
    return float(diff.max()), float(rel.max())


def jax_graphcast_available() -> bool:
    """Real JAX GraphCast (weathernext1_graph) importable, incl. its graph builders."""
    if not jax_stack_available():
        return False
    try:
        ensure_jax_ref_on_path()
        import rtree  # noqa: F401  (trimesh nearest-on-surface)
        import scipy  # noqa: F401  (cKDTree radius query)
        import trimesh  # noqa: F401
        from weathernext.weathernext1_graph import graphcast  # noqa: F401
    except Exception:
        return False
    return True


requires_jax = pytest.mark.skipif(
    not jax_graphcast_available(),
    reason="DeepMind JAX GraphCast reference unavailable (needs jax/haiku/jraph/xarray_jax/"
    "trimesh/rtree/scipy + clone at WEATHERAI_GRAPHCAST_JAX_ROOT); pip install -e '.[parity]'",
)


def physicsnemo_available() -> bool:
    try:
        import torch_geometric  # noqa: F401
        import torch_scatter  # noqa: F401
        from physicsnemo.nn.module.gnn_layers import utils as nv_utils
        from physicsnemo.nn.module.gnn_layers.mesh_graph_encoder import MeshGraphEncoder  # noqa: F401
    except Exception:  # ImportError, or missing optional deps at import time
        return False
    return bool(getattr(nv_utils, "PYG_AVAILABLE", True))


requires_physicsnemo = pytest.mark.skipif(
    not physicsnemo_available(),
    reason="NVIDIA PhysicsNeMo + torch_geometric/torch_scatter/torch_sparse required "
    "(see docs/graphcast_parity.md, 'parity venv')",
)


def trimesh_available() -> bool:
    try:
        import rtree  # noqa: F401
        import trimesh  # noqa: F401
    except ImportError:
        return False
    return True


requires_trimesh = pytest.mark.skipif(
    not trimesh_available(), reason="trimesh + rtree required (DeepMind mesh2grid builder)"
)


def assert_close(name: str, actual, expected, atol: float = ATOL, rtol: float = RTOL) -> tuple[float, float]:
    """``np.testing.assert_allclose`` with a readable max-abs/max-rel message."""
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)
    assert a.shape == b.shape, f"{name}: shape {a.shape} != {b.shape}"
    abs_err, rel_err = max_abs_rel(a, b)
    np.testing.assert_allclose(
        a, b, atol=atol, rtol=rtol,
        err_msg=f"{name}: max_abs={abs_err:.3e} max_rel={rel_err:.3e}",
    )
    return abs_err, rel_err
