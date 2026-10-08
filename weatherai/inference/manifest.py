"""Run manifest: everything needed to say *what* produced a forecast file.

Stored next to every saved result (``<file>.manifest.json``): code revision, weights / normalisation file hashes, input and
output array hashes, the full run configuration (checkpoint, seeds, members, device, dtype, horizon, forcing provenance),
package versions, timings and peak memory. Hashing a 2 GB weight file takes a few seconds; results are cached by
(path, size, mtime) in ``$WEATHERAI_CACHE/hashes.json``.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import numpy as np

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _cache_path() -> str:
    d = os.environ.get("WEATHERAI_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "weatherai"))
    return os.path.join(d, "hashes.json")


def file_sha256(path: str, use_cache: bool = True) -> Dict[str, Any]:
    """{'path', 'size', 'sha256'} of a file (symlinks resolved)."""
    path = os.path.realpath(path)
    st = os.stat(path)
    key = f"{path}|{st.st_size}|{int(st.st_mtime)}"
    cache = {}
    if use_cache:
        try:
            with open(_cache_path()) as f:
                cache = json.load(f)
        except (OSError, ValueError):
            cache = {}
        if key in cache:
            return {"path": path, "size": st.st_size, "sha256": cache[key]}
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if use_cache:
        try:
            os.makedirs(os.path.dirname(_cache_path()), exist_ok=True)
            cache[key] = digest
            with open(_cache_path(), "w") as f:
                json.dump(cache, f)
        except OSError:
            pass
    return {"path": path, "size": st.st_size, "sha256": digest}


def array_sha256(a) -> str:
    """Hash of dtype + shape + bytes (NaNs included bit-exactly)."""
    a = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def code_revision() -> Dict[str, Any]:
    from .. import __version__
    out: Dict[str, Any] = {"package": "weatherai", "version": __version__, "path": _PKG_ROOT}
    try:
        rev = subprocess.run(["git", "-C", _PKG_ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if rev.returncode == 0:
            out["git_revision"] = rev.stdout.strip()
            st = subprocess.run(["git", "-C", _PKG_ROOT, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                text=True, timeout=10)
            out["git_dirty"] = bool(st.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def environment() -> Dict[str, Any]:
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__}
    for mod in ("torch", "onnxruntime", "onnx", "jax", "neuralgcm", "xarray"):
        m = sys.modules.get(mod)
        if m is not None:
            env[mod] = getattr(m, "__version__", "?")
    t = sys.modules.get("torch")
    if t is not None and t.cuda.is_available() and t.cuda.is_initialized():
        env["cuda_device"] = t.cuda.get_device_name()
    return env


def peak_memory() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        import resource
        out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)    # KiB on Linux
    except Exception:
        pass
    t = sys.modules.get("torch")
    if t is not None and t.cuda.is_available() and t.cuda.is_initialized():
        out["peak_gpu_mb"] = round(t.cuda.max_memory_allocated() / 2 ** 20, 1)
    return out


@dataclass
class RunManifest:
    model: str
    checkpoint: str
    backend: str
    config: Dict[str, Any]
    created_utc: str = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    code: Dict[str, Any] = field(default_factory=code_revision)
    weights: List[Dict[str, Any]] = field(default_factory=list)
    normalisation: List[Dict[str, Any]] = field(default_factory=list)
    inputs: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    timings_s: Dict[str, float] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=environment)
    notes: List[str] = field(default_factory=list)
    card: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)     # input checks etc.; not part of the run identity

    def config_hash(self) -> str:
        """Identity of the run request (used to resume chunked runs safely)."""
        ident = {"model": self.model, "checkpoint": self.checkpoint, "backend": self.backend, "config": self.config,
                 "weights": [w.get("sha256") for w in self.weights], "inputs": self.inputs}
        return hashlib.sha256(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()

    def finish(self):
        self.memory = peak_memory()
        self.environment = environment()
        return self

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["config_hash"] = self.config_hash()
        return d

    def write(self, path: str) -> str:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1, default=str)
        return path

    @classmethod
    def read(cls, path: str) -> "RunManifest":
        with open(path) as f:
            d = json.load(f)
        d.pop("config_hash", None)
        return cls(**d)
