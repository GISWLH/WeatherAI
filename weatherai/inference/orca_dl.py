"""ORCA-DL forecaster: GODAS-style monthly ocean state -> monthly forecasts of so, thetao, tos, uo, vo, zos.

Official conventions reproduced here (sources: OpenEarthLab/ORCA-DL README, demo.ipynb, dataset/godas.py, model_config.json):

* variable order (model channels): so(16) thetao(16) tos(1) uo(16) vo(16) zos(1) = 66 ocean channels; wind stress tauu, tauv;
* GODAS names: salt, pottmp, sst (= GODAS 5 m potential temperature in the official example), ucur, vcur, sshg, uflx, vflx;
* units: degC, g/kg, m/s, m, N/m^2 (GODAS pottmp K -> degC, salt kg/kg -> g/kg; ``convert_units``);
* depths: 10..1000 m (16 levels) by linear interpolation (``cdo intlevel``; ``interp_depth`` refuses to extrapolate);
* grid: 1 deg, lat -63.5..63.5, lon 0.5..359.5 (``cdo remapbil``; ``to_model_grid``);
* normalisation: (x - mean[m]) / std[m] with m = calendar month index of the initial month; output lead L de-normalised with
  month (m + L) % 12; NaN -> 0 after normalisation;
* rollout: direct leads 1..6 from the initial state, the lead-6 prediction becomes the next input; wind stress stays at the
  initial month (the model has no future-forcing input).

What WeatherAI adds: input checks with explicit errors, per-variable valid-data masks (outputs outside are NaN -- the
regional 63.5S-63.5N domain is never padded or zero-filled to look global), members = training seeds (same model, not
independent), sequential seed loading, and a run manifest.
"""
from __future__ import annotations

import datetime as _dt
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from .cards import ORCA_DEPTHS, ORCA_DL as CARD
from .errors import MissingInputError, UnitError, WeightsUnavailableError
from .manifest import RunManifest, array_sha256, file_sha256
from .result import ForecastResult

OCEAN_VARS = ("so", "thetao", "tos", "uo", "vo", "zos")
ATMO_VARS = ("tauu", "tauv")
GODAS_NAME = {"so": "salt", "thetao": "pottmp", "tos": "sst", "uo": "ucur", "vo": "vcur", "zos": "sshg", "tauu": "uflx", "tauv": "vflx"}
UNITS = {"so": "g/kg", "thetao": "degC", "tos": "degC", "uo": "m/s", "vo": "m/s", "zos": "m", "tauu": "N/m^2", "tauv": "N/m^2"}
# generous physical ranges in the model units; values outside almost always mean a missed unit conversion
RANGES = {"so": (2.0, 45.0), "thetao": (-3.0, 40.0), "tos": (-3.0, 40.0), "uo": (-5.0, 5.0), "vo": (-5.0, 5.0),
          "zos": (-5.0, 5.0), "tauu": (-5.0, 5.0), "tauv": (-5.0, 5.0)}
ORCA_LAT = np.linspace(-63.5, 63.5, 128)
ORCA_LON = np.linspace(0.5, 359.5, 360)
LONG_NAMES = {"so": "sea water salinity", "thetao": "sea water potential temperature",
              "tos": "ORCA-DL top-layer temperature (CMIP6 tos); near-surface proxy, NOT a satellite/bulk SST",
              "uo": "eastward sea water velocity", "vo": "northward sea water velocity", "zos": "sea surface height"}


# ------------------------------------------------------------------------------------------------- preprocessing helpers
def convert_units(var: str, values, units: str):
    """Return ``values`` in ORCA-DL units (``UNITS[var]``) from ``units``; raises ``UnitError`` for unknown conversions."""
    u = (units or "").strip().lower().replace(" ", "")
    target = UNITS[var].lower()
    a = np.asarray(values, dtype=np.float64)
    if u == target or (target == "degc" and u in ("degc", "c", "celsius", "deg_c", "°c")) or \
            (target == "g/kg" and u in ("g/kg", "psu", "1e-3", "pss-78")) or (target == "n/m^2" and u in ("n/m^2", "n/m2", "pa")):
        return a
    if target == "degc" and u in ("k", "kelvin", "degk", "deg_k"):
        return a - 273.15
    if target == "g/kg" and u in ("kg/kg", "1", "fraction"):
        return a * 1000.0
    if target in ("m/s", "m") and u in ("cm/s", "cm"):
        return a / 100.0
    if target == "n/m^2" and u in ("dyn/cm^2", "dyn/cm2"):
        return a / 10.0
    raise UnitError(f"ORCA-DL {var}: no conversion from {units!r} to {UNITS[var]!r}")


def interp_depth(values, src_depths: Sequence[float], dst_depths: Sequence[float] = ORCA_DEPTHS):
    """Linear interpolation along axis 0 (depth) like ``cdo intlevel``; NaN where a bracketing level is NaN; no extrapolation."""
    a = np.asarray(values, dtype=np.float64)
    src = np.asarray(src_depths, dtype=np.float64)
    dst = np.asarray(dst_depths, dtype=np.float64)
    if a.shape[0] != src.size:
        raise MissingInputError(f"depth axis has {a.shape[0]} levels but {src.size} depths given")
    if np.any(np.diff(src) <= 0):
        raise MissingInputError("source depths must be strictly increasing")
    if dst.min() < src.min() or dst.max() > src.max():
        raise MissingInputError(f"target depths {dst.min()}..{dst.max()} m outside source {src.min()}..{src.max()} m (no extrapolation)")
    i = np.clip(np.searchsorted(src, dst, side="right") - 1, 0, src.size - 2)
    w = (dst - src[i]) / (src[i + 1] - src[i])
    shape = (-1,) + (1,) * (a.ndim - 1)
    lo, hi, w = a[i], a[i + 1], w.reshape(shape)
    out = (1 - w) * lo + w * hi
    exact_lo, exact_hi = (w == 0), (w == 1)
    out = np.where(np.broadcast_to(exact_lo, out.shape), lo, out)
    out = np.where(np.broadcast_to(exact_hi, out.shape), hi, out)
    return out


def to_model_grid(da, lat=ORCA_LAT, lon=ORCA_LON, lat_name: str = "lat", lon_name: str = "lon"):
    """Bilinear regridding of an xarray DataArray to the ORCA-DL grid (periodic in longitude; NaN propagates, so coastal
    points next to land stay missing rather than being extrapolated)."""
    import xarray as xr
    da = da.sortby(lat_name)
    x = da[lon_name].values % 360.0
    da = da.assign_coords({lon_name: x}).sortby(lon_name)
    first, last = da.isel({lon_name: 0}), da.isel({lon_name: -1})
    da = xr.concat([last.assign_coords({lon_name: float(da[lon_name][-1]) - 360.0}), da,
                    first.assign_coords({lon_name: float(da[lon_name][0]) + 360.0})], lon_name)
    return da.interp({lat_name: np.asarray(lat), lon_name: np.asarray(lon)}, method="linear")


@dataclass
class ORCADLInputs:
    """One initial month. ``fields[var]``: (n_depth, H, W) for so/thetao/uo/vo, (H, W) for tos/zos/tauu/tauv, in ``UNITS``,
    already on the model grid and depths; NaN marks land / missing data."""
    fields: Dict[str, np.ndarray]
    init_month: str                                  # "YYYY-MM" of the monthly mean used as initial state
    source: str = ""                                 # e.g. "GODAS (PSL) 1980-01", "ORCA-DL official example"
    tos_definition: str = "GODAS 5 m potential temperature (official ORCA-DL convention)"

    @property
    def month_index(self) -> int:
        return int(self.init_month[5:7]) - 1

    @classmethod
    def from_official_example(cls, example_dir: str, init_month: str = "1980-01"):
        """The official ``example_data/*.nc`` (GODAS January 1980, already on the model grid / depths / units)."""
        import xarray as xr
        f = {}
        for v, g in GODAS_NAME.items():
            a = xr.open_dataset(os.path.join(example_dir, f"{g}.nc"))[g].values
            f[v] = np.asarray(a, dtype=np.float32).squeeze()
        return cls(f, init_month, source=f"ORCA-DL official example_data (GODAS {init_month})")


def check_inputs(inputs: ORCADLInputs, H: int = 128, W: int = 360, n_depth: int = len(ORCA_DEPTHS)) -> Dict[str, dict]:
    """Input checks that need no weights: variables, shapes on the model grid, finite coverage, physical ranges (units)."""
    if not (len(inputs.init_month) >= 7 and inputs.init_month[4] == "-" and 1 <= int(inputs.init_month[5:7]) <= 12):
        raise MissingInputError(f"init_month must be 'YYYY-MM', got {inputs.init_month!r}")
    miss = [v for v in OCEAN_VARS + ATMO_VARS if v not in inputs.fields]
    if miss:
        raise MissingInputError(f"ORCA-DL needs {OCEAN_VARS + ATMO_VARS}; missing {miss}")
    rep = {}
    for v in OCEAN_VARS + ATMO_VARS:
        a = np.asarray(inputs.fields[v])
        nz = 1 if v in ("tos", "zos", "tauu", "tauv") else n_depth
        if a.ndim == 2:
            a = a[None]
        if a.shape != (nz, H, W):
            raise MissingInputError(f"{v}: shape {a.shape}, expected ({nz}, {H}, {W}) on the model grid/depths")
        ok = np.isfinite(a)
        frac = float(ok[0].mean())
        if frac < 0.3:
            raise MissingInputError(f"{v}: only {frac:.0%} of the top level is finite")
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        rlo, rhi = RANGES[v]
        if lo < rlo or hi > rhi:
            raise UnitError(f"{v}: values {lo:.4g}..{hi:.4g} outside {rlo}..{rhi} {UNITS[v]} -- convert units first "
                            f"(weatherai.inference.orca_dl.convert_units)")
        rep[v] = {"finite_fraction_top": frac, "min": lo, "max": hi, "units": UNITS[v]}
    return rep


def _month_start(ym: str, add: int = 0) -> np.datetime64:
    y, m = int(ym[:4]), int(ym[5:7]) - 1 + add
    return np.datetime64(_dt.date(y + m // 12, m % 12 + 1, 1), "ns")


# ------------------------------------------------------------------------------------------------------------ forecaster
class ORCADLForecaster:
    """Load once, run many initial months. Members are checkpoint seeds; each seed is loaded, run and released in turn."""

    card = CARD

    def __init__(self, stats: Dict[str, Dict[str, np.ndarray]], loader, seeds: Iterable[int] = (1,), *, cfg=None,
                 depths: Sequence[float] = ORCA_DEPTHS, device: str = "cpu", dtype: str = "float32", files=None,
                 checkpoint: str = "official"):
        from ..models.orca_dl import ORCADLConfig
        self.cfg = cfg or ORCADLConfig.official()
        self.depths = tuple(float(d) for d in depths)
        self.seeds = tuple(int(s) for s in seeds)
        self.device, self.dtype = device, dtype
        self._loader = loader                        # seed -> nn.Module (eval)
        self.files = files or {"weights": {}, "stats": []}
        self.checkpoint = checkpoint
        self.H, self.W = self.cfg.lat_space[-1], self.cfg.lon_space[-1]
        self.lat = np.linspace(*self.cfg.lat_space)
        self.lon = np.linspace(*self.cfg.lon_space)
        nz = self.cfg.in_chans[0]
        if len(self.depths) != nz:
            raise ValueError(f"{len(self.depths)} depths for {nz} model levels")
        self.stats = {k: {v: self._stat_shape(v, a) for v, a in d.items()} for k, d in stats.items()}

    # ---------------------------------------------------------------------------------------------------- construction
    @classmethod
    def from_checkpoint(cls, root: str, seeds: Iterable[int] = (1,), device: str = "cpu", dtype: str = "float32",
                        hash_files: bool = True) -> "ORCADLForecaster":
        """``root`` = local copy of the HF dataset ``JayKuo/ORCA-DL-data`` (``model_weights/seed_N.bin`` + ``stat/``)."""
        seeds = tuple(int(s) for s in seeds)
        missing = [f"model_weights/seed_{s}.bin" for s in seeds if not os.path.exists(os.path.join(root, "model_weights", f"seed_{s}.bin"))]
        missing += [f"stat/{k}/{g}.npy" for k in ("mean", "std") for g in GODAS_NAME.values()
                    if not os.path.exists(os.path.join(root, "stat", k, f"{g}.npy"))]
        if missing:
            raise WeightsUnavailableError(
                f"ORCA-DL files missing under {root}: {missing[:4]}{' ...' if len(missing) > 4 else ''}. Download with "
                f"`python -c \"from weatherai.models.orca_dl import download; download('{root}', seeds={list(seeds)})\"` "
                "(HF dataset JayKuo/ORCA-DL-data; licence unstated -> do not redistribute).")
        from ..models.orca_dl.data import load_stats
        st = load_stats(root)
        stats = {k: {v: st[k][g] for v, g in GODAS_NAME.items()} for k in ("mean", "std")}
        files = {"weights": {s: os.path.join(root, "model_weights", f"seed_{s}.bin") for s in seeds},
                 "stats": [os.path.join(root, "stat", k, f"{g}.npy") for k in ("mean", "std") for g in GODAS_NAME.values()],
                 "hash": hash_files}

        def loader(seed):
            from ..models.orca_dl import load_official
            return load_official(root, seed=seed, device=device)

        return cls(stats, loader, seeds, device=device, dtype=dtype, files=files, checkpoint=f"{root}/model_weights/seed_N.bin")

    @classmethod
    def from_modules(cls, models: Dict[int, object], stats, cfg, depths, device="cpu", dtype="float32") -> "ORCADLForecaster":
        """Inject already-built modules (tests / user-trained configs)."""
        return cls(stats, lambda s: models[s], tuple(models), cfg=cfg, depths=depths, device=device, dtype=dtype,
                   checkpoint="in-memory modules")

    def _stat_shape(self, var, a):
        a = np.asarray(a, dtype=np.float64)
        nz = 1 if var in ("tos", "zos", "tauu", "tauv") else len(self.depths)
        if a.shape[0] != 12:
            raise MissingInputError(f"statistics of {var}: first axis must be 12 calendar months, got {a.shape}")
        return a.reshape(12, nz, self.H, self.W)

    # ---------------------------------------------------------------------------------------------------------- checks
    def check_inputs(self, inputs: ORCADLInputs) -> Dict[str, dict]:
        """Raise ``MissingInputError`` / ``UnitError`` on problems; return per-variable diagnostics."""
        return check_inputs(inputs, self.H, self.W, len(self.depths))

    def masks(self, inputs: ORCADLInputs) -> Dict[str, np.ndarray]:
        """Valid-data mask per output variable: finite at the top level of the input (True = ocean with data)."""
        out = {}
        for v in OCEAN_VARS:
            a = np.asarray(inputs.fields[v])
            out[v] = np.isfinite(a[0] if a.ndim == 3 else a)
        return out

    def normalise(self, inputs: ORCADLInputs):
        m = inputs.month_index
        def norm(v):
            a = np.asarray(inputs.fields[v], dtype=np.float64)
            a = a[None] if a.ndim == 2 else a
            return (a - self.stats["mean"][v][m]) / self.stats["std"][v][m]
        with np.errstate(invalid="ignore", divide="ignore"):
            ocean = np.concatenate([norm(v) for v in OCEAN_VARS], 0)
            atmo = np.concatenate([norm(v) for v in ATMO_VARS], 0)
        f = lambda x: np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)[None].astype(np.float32)
        return f(ocean), f(atmo)

    # ------------------------------------------------------------------------------------------------------------- run
    def run(self, inputs: ORCADLInputs, lead_months: int, seeds: Optional[Iterable[int]] = None,
            out_path: Optional[str] = None) -> ForecastResult:
        import torch
        if int(lead_months) < 1:
            raise ValueError("lead_months must be >= 1")
        n = int(lead_months)
        seeds = tuple(int(s) for s in (seeds or self.seeds))
        t0 = time.time()
        diag = self.check_inputs(inputs)
        masks = self.masks(inputs)
        ocean, atmo = self.normalise(inputs)
        dt = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[self.dtype]
        notes = []
        if self.dtype != "float32":
            notes.append(f"dtype {self.dtype}: precision not verified against the official fp32 code")
        if n > 24:
            notes.append(f"lead {n} months beyond the official predict.sh horizon (24); stability not verified")
        m0 = inputs.month_index
        split = np.cumsum([0] + list(self.cfg.out_chans))
        per_var = {v: [] for v in OCEAN_VARS}
        timings = {"check_s": time.time() - t0}
        for s in seeds:
            tl = time.time()
            model = self._loader(s)
            model = model.to(device=self.device, dtype=dt).eval()
            timings[f"load_seed{s}_s"] = time.time() - tl
            tr = time.time()
            with torch.no_grad():
                y = model(torch.from_numpy(ocean).to(self.device, dt), torch.from_numpy(atmo).to(self.device, dt),
                          predict_time_steps=n)[0].float().cpu().numpy()            # (n, 66, H, W) normalised
            timings[f"run_seed{s}_s"] = time.time() - tr
            del model
            if self.device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            for i, v in enumerate(OCEAN_VARS):
                p = y[:, split[i]:split[i + 1]].astype(np.float64)                     # (n, nz, H, W)
                months = [(m0 + L) % 12 for L in range(1, n + 1)]
                phys = p * self.stats["std"][v][months] + self.stats["mean"][v][months]
                phys = np.where(masks[v][None, None], phys, np.nan).astype(np.float32)
                per_var[v].append(phys if p.shape[1] > 1 else phys[:, 0])
        data, dims, var_attrs = {}, {}, {}
        for v in OCEAN_VARS:
            data[v] = np.stack(per_var[v])
            dims[v] = ("member", "lead", "depth", "lat", "lon") if data[v].ndim == 5 else ("member", "lead", "lat", "lon")
            var_attrs[v] = {"units": UNITS[v], "long_name": LONG_NAMES[v]}
        var_attrs["tos"].update(sst_definition="proxy", init_definition=inputs.tos_definition,
                                note="Do not merge with OISST/ERA5 SST without stating the proxy; evaluate against the same definition.")
        init = _month_start(inputs.init_month)
        vs = np.array([_month_start(inputs.init_month, L) for L in range(1, n + 1)])
        ve = np.array([_month_start(inputs.init_month, L + 1) for L in range(1, n + 1)])
        coords = {"member": np.array(seeds), "lead": np.arange(1, n + 1), "depth": np.array(self.depths), "lat": self.lat,
                  "lon": self.lon, "init_time": init, "valid_start": (("lead",), vs), "valid_end": (("lead",), ve)}
        attrs = dict(model="orca-dl", product=CARD.product, lead_units="calendar months",
                     time_semantics=CARD.time_semantics, member_kind="checkpoint_seed",
                     ensemble_note="members are training seeds of one model; count as ONE model, not as independent models",
                     domain="lat -63.5..63.5 only; NaN outside the input-valid ocean mask (never zero-filled)",
                     normalisation="monthly statistics; input month m, lead L de-normalised with (m+L)%12",
                     forcing_policy="wind stress of the initial month only (model design)", source=inputs.source)
        man = RunManifest("orca-dl", self.checkpoint, "torch-native",
                          config=dict(seeds=list(seeds), lead_months=n, device=self.device, dtype=self.dtype,
                                      init_month=inputs.init_month, source=inputs.source, depths=list(self.depths)),
                          notes=notes, card=CARD.key)
        if self.files.get("hash"):
            man.weights = [file_sha256(self.files["weights"][s]) for s in seeds]
            man.normalisation = [file_sha256(p) for p in self.files["stats"]]
        man.inputs = {v: array_sha256(inputs.fields[v]) for v in OCEAN_VARS + ATMO_VARS}
        timings["total_s"] = time.time() - t0
        man.timings_s = timings
        man.finish()
        man.diagnostics["inputs"] = diag
        res = ForecastResult("orca-dl", data, dims, coords, masks, var_attrs, attrs, man).hash_outputs()
        if out_path:
            res.save(out_path)
        return res
