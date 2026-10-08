"""FuXi-S2S forecaster: two ERA5 daily means -> up to 42 daily forecasts, stochastic members.

Backends
--------
``onnx`` (default, reference): the official ``fuxi_s2s.onnx`` run by onnxruntime exactly as the official ``inference.py``
    (session options, ``step`` input = lead index 0..41, the whole output fed back as the next input).
    ``noise="seeded"`` (default) replaces the graph's two ``RandomNormalLike`` nodes by graph inputs and feeds standard-normal
    draws from ``numpy.random.default_rng([seed, member, step])``: same distribution as the internal sampler, but
    reproducible and identical for both backends. ``noise="internal"`` keeps the unmodified graph and only calls
    ``onnxruntime.set_seed`` (reproducibility then depends on the ORT build / provider).
``native``: the WeatherAI PyTorch port (``weatherai.models.fuxi_s2s``), verified against onnxruntime for 1 step and a
    3-step chain only (docs/results/fuxi_s2s_parity.json). Use it for gradients / research, not as the reference.

Conventions (verified: docs/results/fuxi_s2s_input_check.json)
---------------------------------------------------------------
* inputs: daily means of hourly ERA5 (00..23 UTC; accumulations = mean of the hourly accumulations ending 01..24 UTC),
  1.5 deg, lat 90 -> -90, channels ``CHANNELS``; sst NaN over land (official ``data/mask.nc``);
* ``tp`` channel = clip(1000 * daily-mean hourly accumulation [m], 0, 1000) = **mm/h**; x24 = mm/day;
  ``ttr`` = hourly accumulation / 3600 = W m-2;
* output step k (0-based) = daily mean of init + (k+1) days; hard limit 42 steps (lead encoding trained for 0..41).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .cards import FUXI_S2S as CARD, FUXI_TP_NOTE
from .errors import BackendUnavailableError, HorizonError, MissingInputError, UnitError, WeightsUnavailableError
from .manifest import RunManifest, array_sha256, file_sha256
from .result import ForecastResult

MAX_LEAD_DAYS = 42
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
CHANNELS = ([f"{v}{l}" for v in "ztuvq" for l in LEVELS]
            + ["t2m", "d2m", "sst", "ttr", "10u", "10v", "100u", "100v", "msl", "tcwv", "tp"])
UNITS = {**{f"z{l}": "m2 s-2" for l in LEVELS}, **{f"t{l}": "K" for l in LEVELS}, **{f"u{l}": "m s-1" for l in LEVELS},
         **{f"v{l}": "m s-1" for l in LEVELS}, **{f"q{l}": "kg kg-1" for l in LEVELS},
         "t2m": "K", "d2m": "K", "sst": "K", "ttr": "W m-2", "10u": "m s-1", "10v": "m s-1", "100u": "m s-1", "100v": "m s-1",
         "msl": "Pa", "tcwv": "kg m-2", "tp": "mm h-1"}
LAT = np.linspace(90.0, -90.0, 121)
LON = np.arange(240) * 1.5
_NAMES = [("geopotential", "z"), ("temperature", "t"), ("u_component_of_wind", "u"), ("v_component_of_wind", "v"),
          ("specific_humidity", "q"), ("2m_temperature", "t2m"), ("2m_dewpoint_temperature", "d2m"),
          ("sea_surface_temperature", "sst"), ("top_net_thermal_radiation", "ttr"), ("10m_u_component_of_wind", "10u"),
          ("10m_v_component_of_wind", "10v"), ("100m_u_component_of_wind", "100u"), ("100m_v_component_of_wind", "100v"),
          ("mean_sea_level_pressure", "msl"), ("total_column_water_vapour", "tcwv"), ("total_precipitation", "tp")]
# (channel, lo, hi, hint) plausibility of raw inputs; area-weighted global means are checked for tp separately
_CHECKS = [("z500", 4.4e4, 6.2e4, "geopotential in m2 s-2 (not geopotential height in m)"),
           ("t850", 200.0, 330.0, "temperature in K"), ("q1000", 0.0, 0.06, "specific humidity in kg/kg"),
           ("t2m", 180.0, 345.0, "K"), ("d2m", 160.0, 320.0, "K"), ("sst", 265.0, 310.0, "K (not degC)"),
           ("ttr", -500.0, 0.0, "W m-2 = ERA5 hourly accumulation [J m-2] / 3600"), ("msl", 8.5e4, 1.1e5, "Pa (not hPa)"),
           ("tcwv", 0.0, 110.0, "kg m-2"), ("tp", 0.0, 1000.0, "mm/h daily-mean rate (official clip(m*1000, 0, 1000))")]


@dataclass
class FuXiS2SInputs:
    x: np.ndarray                         # (2, 76, 121, 240) float32, raw units, channel order CHANNELS
    times: Sequence                       # (init - 1 day, init), daily-mean labels
    source: str = ""
    land_mask: Optional[np.ndarray] = None   # (121, 240) bool, True = ocean (sst defined); default: finite sst at init

    @property
    def init(self) -> np.datetime64:
        return np.datetime64(self.times[-1], "D")

    @classmethod
    def from_official_files(cls, sample_dir: str, mask_path: Optional[str] = None, source: str = ""):
        """Official ``data/sample/<long_name>.nc`` layout (as ``data_util.make_input``) + ``mask.nc`` land mask."""
        import xarray as xr
        chans, times = [], None
        for long, short in _NAMES:
            v = xr.open_dataarray(os.path.join(sample_dir, f"{long}.nc"))
            if short == "tp":
                v = np.clip(v * 1000, 0, 1000)
            elif short == "ttr":
                v = v / 3600
            if v.level.values[0] != 1000:
                v = v.reindex(level=v.level[::-1])
            if float(v.lat[0]) != 90.0:
                raise MissingInputError(f"{long}: latitude must run 90 -> -90 (official layout)")
            chans.append(v.values.astype(np.float32).reshape(v.shape[0], -1, *v.shape[-2:]))
            times = v.time.values
        x = np.concatenate(chans, 1)
        mask = None
        if mask_path:
            mask = xr.open_dataarray(mask_path).values.astype(bool)
            i = CHANNELS.index("sst")
            x[:, i] = np.where(mask, x[:, i], np.nan)
        return cls(x, [np.datetime64(t, "D") for t in times], source or f"official-layout files {sample_dir}", mask)

    @classmethod
    def from_dataarray(cls, da, mask=None, source: str = ""):
        """Official ``input.nc`` DataArray (time, channel, lat, lon), raw units."""
        ch = [str(c) for c in da.channel.values]
        if ch != CHANNELS:
            raise MissingInputError(f"channel order differs from the official one (first mismatch: "
                                    f"{next((a, b) for a, b in zip(ch, CHANNELS) if a != b) if len(ch) == 76 else len(ch)})")
        x = da.values.astype(np.float32)
        if mask is not None:
            i = CHANNELS.index("sst")
            x[:, i] = np.where(mask, x[:, i], np.nan)
        return cls(x, [np.datetime64(t, "D") for t in da.time.values], source, mask)


def check_inputs(inp: FuXiS2SInputs) -> Dict[str, dict]:
    x = inp.x
    if x.shape != (2, 76, 121, 240):
        raise MissingInputError(f"input shape {x.shape}, expected (2, 76, 121, 240)")
    if len(inp.times) != 2 or (np.datetime64(inp.times[1], "D") - np.datetime64(inp.times[0], "D")) != np.timedelta64(1, "D"):
        raise MissingInputError(f"need two consecutive daily means (init-1d, init); got {list(inp.times)}")
    isst = CHANNELS.index("sst")
    bad = [CHANNELS[c] for c in range(76) if c != isst and not np.isfinite(x[:, c]).all()]
    if bad:
        raise MissingInputError(f"non-finite values in {bad[:5]} (only sst may be NaN, over land)")
    sst_nan = float(np.isnan(x[-1, isst]).mean())
    if not 0.2 < sst_nan < 0.5:
        raise MissingInputError(f"sst land fraction {sst_nan:.2f}: apply the official land mask (data/mask.nc -> NaN over land)")
    rep = {}
    for ch, lo, hi, hint in _CHECKS:
        a = x[:, CHANNELS.index(ch)]
        mn, mx = float(np.nanmin(a)), float(np.nanmax(a))
        if mn < lo or mx > hi:
            raise UnitError(f"{ch}: values {mn:.4g}..{mx:.4g} outside {lo:.4g}..{hi:.4g}; expected {hint}")
        rep[ch] = {"min": mn, "max": mx}
    w = np.cos(np.deg2rad(LAT))[:, None] * np.ones((1, 240))
    tpm = float((x[-1, CHANNELS.index("tp")] * w).sum() / w.sum())
    if not 0.02 < tpm < 0.4:
        raise UnitError(f"tp global mean {tpm:.4g} mm/h: expected ~0.1 mm/h (daily-mean rate). x24 too large -> mm/day was given; "
                        "~1e-4 -> metres were given without x1000")
    rep["tp_global_mean_mm_per_h"] = tpm
    rep["sst_land_fraction"] = sst_nan
    return rep


def noise_for(seed: int, member: int, step: int, shapes) -> List[np.ndarray]:
    """Standard-normal draws for one (seed, member, step); same for the ONNX and native backends."""
    rng = np.random.default_rng([int(seed), int(member), int(step)])
    return [rng.standard_normal(tuple(s), dtype=np.float32) for s in shapes]


# ------------------------------------------------------------------------------------------------------------- runners
class _OnnxRunner:
    backend = "onnx-official"

    def __init__(self, onnx_path: str, device: str = "cpu", threads: Optional[int] = None, noise: str = "seeded",
                 cache_dir: Optional[str] = None, noise_shapes=None):
        try:
            import onnx
            import onnxruntime as ort
        except ImportError as e:
            raise BackendUnavailableError("FuXi-S2S ONNX backend needs `pip install onnx onnxruntime` "
                                          "(or onnxruntime-gpu); install weatherai[onnx]") from e
        if noise not in ("seeded", "internal"):
            raise ValueError("noise must be 'seeded' or 'internal'")
        self.ort, self.noise, self.device = ort, noise, device
        self.onnx_path = os.path.abspath(onnx_path)
        path = self.onnx_path
        self.noise_names, self.noise_shapes, self.noise_dtypes = [], [], []
        if noise == "seeded":
            path = self._seeded_graph(onnx, cache_dir, noise_shapes)
        self.graph_path = path
        so = ort.SessionOptions()
        so.enable_cpu_mem_arena = False                               # official inference.py settings
        so.enable_mem_pattern = False
        so.enable_mem_reuse = False
        if threads:
            so.intra_op_num_threads = int(threads)
        if device.startswith("cuda"):
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise BackendUnavailableError("onnxruntime has no CUDAExecutionProvider here; install onnxruntime-gpu or use device='cpu'")
            providers = [("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested",
                                                    "device_id": int(device.split(":")[1]) if ":" in device else 0})]
        else:
            providers = ["CPUExecutionProvider"]
        ort.set_default_logger_severity(3)
        self.sess = ort.InferenceSession(path, so, providers=providers)
        self.input_names = [i.name for i in self.sess.get_inputs()]

    def _seeded_graph(self, onnx, cache_dir, noise_shapes):
        import hashlib
        from onnx import TensorProto, helper, shape_inference
        m = onnx.load(self.onnx_path, load_external_data=False)
        g = m.graph
        nodes = [n for n in g.node if n.op_type == "RandomNormalLike"]
        if len(nodes) != 2:
            raise BackendUnavailableError(f"expected 2 RandomNormalLike nodes in {self.onnx_path}, found {len(nodes)}")
        shapes = {}
        try:
            inf = shape_inference.infer_shapes(m)
            for vi in list(inf.graph.value_info) + list(inf.graph.input):
                dims = [d.dim_value if d.HasField("dim_value") else None for d in vi.type.tensor_type.shape.dim]
                shapes[vi.name] = dims
        except Exception:
            pass
        defaults = noise_shapes or [(128, 12), (128, 7200)]
        for k, n in enumerate(nodes):
            like = shapes.get(n.input[0])
            shp = tuple(like) if like and all(d for d in like) else tuple(defaults[k])
            dt = next((a.i for a in n.attribute if a.name == "dtype"), TensorProto.FLOAT)
            g.node.remove(n)
            g.input.append(helper.make_tensor_value_info(n.output[0], dt, list(shp)))
            self.noise_names.append(n.output[0])
            self.noise_shapes.append(shp)
            self.noise_dtypes.append(helper.tensor_dtype_to_np_dtype(dt))
        data = m.SerializeToString()
        tag = hashlib.sha256(data).hexdigest()[:16]
        root = cache_dir or os.path.join(os.environ.get("WEATHERAI_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "weatherai")),
                                         "fuxi_s2s_seeded", tag)
        os.makedirs(root, exist_ok=True)
        src_dir = os.path.dirname(self.onnx_path)
        for init in g.initializer:                                     # external weights stay where they are: symlink
            for e in init.external_data:
                if e.key == "location":
                    dst = os.path.join(root, e.value)
                    if not os.path.lexists(dst):
                        os.makedirs(os.path.dirname(dst) or root, exist_ok=True)
                        os.symlink(os.path.join(src_dir, e.value), dst)
        out = os.path.join(root, "fuxi_s2s_seeded.onnx")
        if not os.path.exists(out):
            with open(out + ".part", "wb") as f:
                f.write(data)
            os.replace(out + ".part", out)
        return out

    def start_member(self, seed: int, member: int):
        if self.noise == "internal":
            self.ort.set_seed(int(seed) * 100003 + int(member))

    def step(self, x: np.ndarray, k: int, seed: int, member: int) -> np.ndarray:
        feeds = {"input": x}
        if "step" in self.input_names:
            feeds["step"] = np.array([k], dtype=np.float32)
        if self.noise == "seeded":
            for name, arr, dt in zip(self.noise_names, noise_for(seed, member, k, self.noise_shapes), self.noise_dtypes):
                feeds[name] = arr.astype(dt)
        return self.sess.run(None, feeds)[0]


class _NativeRunner:
    backend = "torch-native"

    def __init__(self, onnx_path: str, device: str = "cpu", dtype: str = "float32", model=None):
        import torch
        from ..models.fuxi_s2s import FuXiS2SNoise
        from ..models.fuxi_s2s.convert import load_official
        self.torch, self.Noise = torch, FuXiS2SNoise
        self.dt = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
        self.device = device
        self.model = model if model is not None else load_official(onnx_path, device=device, dtype=self.dt)
        c = self.model.cfg
        self.noise_shapes = [(c.latent, c.rank), (c.latent, c.n_tokens)]

    def start_member(self, seed, member):
        pass

    def step(self, x, k, seed, member):
        t = self.torch
        e1, e2 = noise_for(seed, member, k, self.noise_shapes)
        nz = self.Noise(t.from_numpy(e1)[None].to(self.device), t.from_numpy(e2)[None].to(self.device))
        with t.no_grad():
            y = self.model(t.from_numpy(x).to(self.device), t.tensor([float(k)], device=self.device), noise=nz)
        return y.float().cpu().numpy()


# ---------------------------------------------------------------------------------------------------------- forecaster
@dataclass
class RunIndex:
    """Chunked run on disk: one netCDF (+ manifest) per member."""
    files: List[str]
    manifest: str
    members: List[int]

    def open(self):
        import xarray as xr
        return xr.concat([xr.open_dataset(f) for f in self.files], "member")


class FuXiS2SForecaster:
    card = CARD

    def __init__(self, runner, checkpoint: str, weight_files: Sequence[str] = (), hash_files: bool = True, device: str = "cpu",
                 dtype: str = "float32"):
        self.runner, self.checkpoint = runner, checkpoint
        self.weight_files, self.hash_files = list(weight_files), hash_files
        self.device, self.dtype = device, dtype

    @staticmethod
    def _locate(path: str) -> str:
        p = os.path.join(path, "model-1.0", "fuxi_s2s.onnx") if os.path.isdir(path) and not path.endswith(".onnx") else path
        if os.path.isdir(p):
            p = os.path.join(p, "fuxi_s2s.onnx")
        if not os.path.exists(p):
            raise WeightsUnavailableError(
                f"FuXi-S2S ONNX not found at {p}. Download model-1.0.tar (2.08 GB, CC-BY-NC-ND-4.0) from Zenodo 10.5281/zenodo.15718402 "
                "yourself (`from weatherai.models.fuxi_s2s.convert import download; download(root)`); it is never redistributed.")
        return p

    @classmethod
    def from_onnx(cls, path: str, device: str = "cpu", threads: Optional[int] = None, noise: str = "seeded",
                  cache_dir: Optional[str] = None, hash_files: bool = True, noise_shapes=None) -> "FuXiS2SForecaster":
        p = cls._locate(path)
        r = _OnnxRunner(p, device, threads, noise, cache_dir, noise_shapes)
        blobs = sorted({os.path.join(os.path.dirname(p), f) for f in os.listdir(os.path.dirname(p)) if f != os.path.basename(p)
                        and not f.endswith(".onnx") and os.path.isfile(os.path.join(os.path.dirname(p), f))})
        return cls(r, p, [p] + blobs, hash_files, device)

    @classmethod
    def from_native(cls, path: str, device: str = "cpu", dtype: str = "float32", hash_files: bool = True, model=None):
        p = cls._locate(path) if model is None else path
        r = _NativeRunner(p, device, dtype, model)
        files = [] if model is not None else [p] + [os.path.join(os.path.dirname(p), "fuxi_s2s")]
        return cls(r, p, [f for f in files if os.path.exists(f)], hash_files and model is None, device, dtype)

    # ---------------------------------------------------------------------------------------------------------- run
    def _manifest(self, inp: FuXiS2SInputs, lead_days, members, seed, keep):
        man = RunManifest("fuxi-s2s", self.checkpoint, self.runner.backend,
                          config=dict(lead_days=lead_days, members=list(members), seed=seed, device=self.device, dtype=self.dtype,
                                      noise=getattr(self.runner, "noise", "seeded"), init=str(inp.init), source=inp.source,
                                      channels=keep), card=CARD.key)
        if self.hash_files:
            man.weights = [file_sha256(f) for f in self.weight_files]
        man.inputs = {"x": array_sha256(inp.x)}
        if inp.land_mask is not None:
            man.inputs["land_mask"] = array_sha256(inp.land_mask)
        if getattr(self.runner, "noise", "seeded") == "internal":
            man.notes.append("noise sampled inside the ONNX graph; seeded only through onnxruntime.set_seed")
        if self.runner.backend == "torch-native":
            man.notes.append("native backend: verified vs onnxruntime for 1 step + 3-step chain only (not bitwise: fp16 graph)")
        return man

    def run_member(self, inp: FuXiS2SInputs, lead_days: int, member: int, seed: int = 0, keep: Optional[Sequence[str]] = None,
                   _diag=None) -> ForecastResult:
        lead_days = self._check_lead(lead_days)
        diag = _diag or check_inputs(inp)
        keep = list(keep) if keep else list(CHANNELS)
        idx = [CHANNELS.index(c) for c in keep]
        ocean = inp.land_mask if inp.land_mask is not None else np.isfinite(inp.x[-1, CHANNELS.index("sst")])
        man = self._manifest(inp, lead_days, [member], seed, keep)
        x = inp.x[None].astype(np.float32)
        out = np.empty((1, lead_days, len(idx), 121, 240), np.float32)
        t0 = time.time()
        self.runner.start_member(seed, member)
        steps = []
        for k in range(lead_days):
            ts = time.time()
            x = self.runner.step(x, k, seed, member)          # official: the whole output is the next input (unmodified)
            steps.append(time.time() - ts)
            out[0, k] = x[0, -1, idx]
        isst = keep.index("sst") if "sst" in keep else None
        if isst is not None:
            out[:, :, isst] = np.where(ocean, out[:, :, isst], np.nan)
        man.timings_s = {"total_s": time.time() - t0, "mean_step_s": float(np.mean(steps)), "max_step_s": float(np.max(steps))}
        man.diagnostics["inputs"] = diag
        man.finish()
        init = np.datetime64(inp.init, "ns")
        day = np.timedelta64(1, "D")
        coords = {"member": np.array([member]), "lead": np.arange(1, lead_days + 1), "channel": np.array(keep), "lat": LAT, "lon": LON,
                  "init_time": init, "valid_start": (("lead",), init + np.arange(1, lead_days + 1) * day),
                  "valid_end": (("lead",), init + np.arange(2, lead_days + 2) * day),
                  "units": (("channel",), np.array([UNITS[c] for c in keep]))}
        attrs = dict(model="fuxi-s2s", product=CARD.product, lead_units="days", time_semantics=CARD.time_semantics,
                     member_kind="stochastic_latent_draw", seed=seed, tp_units="mm h-1 (daily-mean rate; x24 = mm/day)",
                     tp_note=FUXI_TP_NOTE, sst_mask="NaN over land (input land mask); land values are not fed back altered",
                     max_lead_days=MAX_LEAD_DAYS, source=inp.source, licence="weights CC-BY-NC-ND-4.0: non-commercial use only")
        # the land mask applies to the sst channel only, so it is stored as its own variable (not as a mask of "fields")
        data = {"fields": out, "ocean_mask": ocean.astype(np.int8)}
        dims = {"fields": ("member", "lead", "channel", "lat", "lon"), "ocean_mask": ("lat", "lon")}
        vattrs = {"fields": {"long_name": "FuXi-S2S daily-mean fields (see units coordinate)"},
                  "ocean_mask": {"long_name": "1 = ocean (sst defined), applies to the sst channel"}}
        return ForecastResult("fuxi-s2s", data, dims, coords, {}, vattrs, attrs, man).hash_outputs()

    @staticmethod
    def _check_lead(lead_days) -> int:
        n = int(lead_days)
        if n < 1:
            raise ValueError("lead_days must be >= 1")
        if n > MAX_LEAD_DAYS:
            raise HorizonError(f"FuXi-S2S runs at most {MAX_LEAD_DAYS} daily steps (lead encoding trained for 0..41); "
                               f"{n} days requested. Use a monthly/seasonal model for longer targets.")
        return n

    def run(self, inp: FuXiS2SInputs, lead_days: int, members=11, seed: int = 0, keep: Optional[Sequence[str]] = None,
            out_dir: Optional[str] = None, resume: bool = True):
        """``members``: count or explicit member ids. Without ``out_dir`` returns one in-memory ``ForecastResult``; with
        ``out_dir`` writes ``member_XX.nc`` (+ manifest) one member at a time, skips members already written with the same
        configuration (``resume``) and returns a ``RunIndex``."""
        lead_days = self._check_lead(lead_days)
        ids = list(range(int(members))) if np.isscalar(members) else [int(m) for m in members]
        diag = check_inputs(inp)
        if out_dir is None:
            est = len(ids) * lead_days * len(keep or CHANNELS) * 121 * 240 * 4 / 2 ** 30
            if est > 4:
                raise MemoryError(f"~{est:.1f} GB in memory; pass out_dir= to write members one by one, or keep= fewer channels")
            parts = [self.run_member(inp, lead_days, m, seed, keep, diag) for m in ids]
            first = parts[0]
            first.data["fields"] = np.concatenate([p.data["fields"] for p in parts], 0)
            first.coords["member"] = np.array(ids)
            first.manifest.config["members"] = ids
            first.manifest.timings_s = {f"member{m}_s": p.manifest.timings_s["total_s"] for m, p in zip(ids, parts)}
            return first.hash_outputs()
        os.makedirs(out_dir, exist_ok=True)
        files = []
        for m in ids:
            f = os.path.join(out_dir, f"member_{m:02d}.nc")
            ident = self._manifest(inp, lead_days, [m], seed, list(keep) if keep else list(CHANNELS)).config_hash()
            if resume and os.path.exists(f) and os.path.exists(f + ".manifest.json"):
                if RunManifest.read(f + ".manifest.json").config_hash() == ident:
                    files.append(f)
                    continue
            r = self.run_member(inp, lead_days, m, seed, keep, diag)
            r.save(f)
            files.append(f)
        idx = os.path.join(out_dir, "run.json")
        import json
        with open(idx, "w") as fh:
            json.dump({"model": "fuxi-s2s", "members": ids, "files": [os.path.basename(f) for f in files], "seed": seed,
                       "lead_days": lead_days, "init": str(inp.init)}, fh, indent=1)
        return RunIndex(files, idx, ids)
