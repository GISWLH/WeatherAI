"""Model-native forecast container.

``ForecastResult`` keeps each model's own arrays, coordinates and masks (no forced common tensor layout): ORCA-DL keeps
(member, lead[month], depth, lat, lon) on its 128x360 regional grid, FuXi-S2S keeps (member, lead[day], channel, lat, lon) on
121x240. Values outside a variable's mask are NaN (never zero-filled). ``to_xarray`` / ``save`` give a self-describing
netCDF plus ``<file>.manifest.json``; WeatherPre converts that to its common schema.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np

from .manifest import RunManifest, array_sha256


@dataclass
class ForecastResult:
    model: str
    data: Dict[str, np.ndarray]                    # variable -> array
    dims: Dict[str, Tuple[str, ...]]               # variable -> dimension names
    coords: Dict[str, Any]                         # name -> 1-D array, or (dims, array) for non-dimension coordinates
    masks: Dict[str, np.ndarray] = field(default_factory=dict)       # variable -> bool (True = valid), dims = trailing spatial dims
    var_attrs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    attrs: Dict[str, Any] = field(default_factory=dict)
    manifest: RunManifest | None = None

    def __post_init__(self):
        for v, a in self.data.items():
            if a.ndim != len(self.dims[v]):
                raise ValueError(f"{v}: array ndim {a.ndim} != dims {self.dims[v]}")

    def hash_outputs(self):
        if self.manifest is not None:
            self.manifest.outputs = {v: array_sha256(a) for v, a in self.data.items()}
        return self

    # ------------------------------------------------------------------------------------------------ conversion / IO
    def to_xarray(self):
        import xarray as xr
        coords = {}
        for k, c in self.coords.items():
            coords[k] = c if isinstance(c, tuple) else np.asarray(c)
        ds = xr.Dataset({v: (self.dims[v], a, self.var_attrs.get(v, {})) for v, a in self.data.items()}, coords=coords)
        for v, m in self.masks.items():
            sp = self.dims[v][-m.ndim:]
            ds[f"mask_{v}"] = (sp, m.astype("int8"), {"long_name": f"valid-data mask of {v}", "flag_values": "0 invalid, 1 valid"})
        attrs = {k: (v if isinstance(v, (str, int, float)) else str(v)) for k, v in self.attrs.items()}
        attrs["model"] = self.model
        if self.manifest is not None:
            attrs["config_hash"] = self.manifest.config_hash()
            attrs["code_revision"] = str(self.manifest.code.get("git_revision", self.manifest.code.get("version")))
        ds.attrs.update(attrs)
        return ds

    def save(self, path: str) -> str:
        """Write netCDF (needs xarray + netCDF4/h5netcdf) and the manifest JSON next to it."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.hash_outputs()
        tmp = path + ".part"
        self.to_xarray().to_netcdf(tmp)
        os.replace(tmp, path)
        if self.manifest is not None:
            self.manifest.write(path + ".manifest.json")
        return path
