"""Machine-readable model / checkpoint cards for the forecast-capable models of the zoo.

A card states *what a checkpoint needs and produces* (variables, units, history, grid, vertical coordinate, normalisation,
forcings, masks, time step, sources, licences) and *what has actually been verified*, as six independent evidence levels:

    implemented              code exists in this repository
    weights_obtainable       the trained weights can be downloaded by a user (says nothing about redistribution)
    real_data_inference      the model was run from real (reanalysis / analysis) initial conditions
    official_parity          outputs were compared with the official implementation on identical inputs and weights
    multi_init_long_rollout  several initial dates and rollouts over the full intended horizon were run and checked
    independent_hindcast     skill was measured on a hindcast set not used for training, by us

Each level is ``yes`` / ``partial`` / ``no`` / ``external`` (reported by another project, not reproducible from this repo)
with a short detail and a pointer to the evidence. A successful run is never a skill claim: only ``independent_hindcast``
speaks to skill, and it is ``no`` for every model here.

Two horizons are kept apart: ``technical_max_lead`` (what the code / checkpoint can run, with the reason) and
``validated_lead`` (the longest lead whose skill *we* verified on independent hindcasts; ``None`` everywhere today).
The paper's own claims are kept in ``paper_claims`` and are not treated as our validation.

This module imports nothing heavy (no torch / onnxruntime / jax) and never downloads anything.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

LEVELS = ("implemented", "weights_obtainable", "real_data_inference", "official_parity",
          "multi_init_long_rollout", "independent_hindcast")
STATES = ("yes", "partial", "no", "external", "unknown")
CHECKED = "2026-10-08"


@dataclass(frozen=True)
class Evidence:
    state: str
    detail: str = ""
    ref: str = ""                      # file in this repo, DOI or URL
    date: str = ""

    def __post_init__(self):
        if self.state not in STATES:
            raise ValueError(f"evidence state {self.state!r} not in {STATES}")


@dataclass(frozen=True)
class Var:
    name: str                          # model-side short name
    units: str
    long_name: str = ""
    levels: Tuple = ()                 # vertical levels (empty = single level)
    level_units: str = ""
    source_name: str = ""              # name in the usual source dataset (GODAS / ERA5)
    notes: str = ""


@dataclass(frozen=True)
class Forcing:
    name: str
    units: str
    when: str                          # "every step" | "init only" | "static"
    notes: str = ""


@dataclass(frozen=True)
class Lead:
    value: Optional[int]
    unit: str                          # "day" | "month" | "6h-step"
    reason: str = ""


@dataclass(frozen=True)
class CheckpointCard:
    key: str                           # "<model>/<checkpoint>"
    model: str
    checkpoint: str
    backend: str                       # torch-native | onnx-official | jax-official | none
    product: str                       # seasonal-monthly | subseasonal-daily | forced-atmosphere | weather-to-climate | climate-modes
    code_source: str
    code_licence: str
    weights_source: str
    weights_licence: str
    weights_redistributable: str       # "yes" | "no" | "unclear" -- obtainable != redistributable
    files: Tuple[str, ...]
    parameters: Optional[int]
    inputs: Tuple[Var, ...]
    history: str
    grid: str
    vertical: str
    normalisation: str
    outputs: Tuple[Var, ...]
    time_step: str
    time_semantics: str
    forcings: Tuple[Forcing, ...]
    spatial_mask: str
    ensemble: str
    technical_max_lead: Lead
    validated_lead: Lead
    status: Dict[str, Evidence]
    paper: str = ""
    paper_claims: str = ""
    resources: str = ""
    limitations: Tuple[str, ...] = ()
    api: str = ""                      # entry point in weatherai.inference

    def __post_init__(self):
        missing = [k for k in LEVELS if k not in self.status]
        if missing:
            raise ValueError(f"{self.key}: status levels missing {missing}")

    @property
    def runnable(self) -> bool:
        """Implemented, weights obtainable and offered as a forecaster backend (not a statement about skill)."""
        return (self.backend != "none" and self.status["implemented"].state in ("yes", "partial")
                and self.status["weights_obtainable"].state == "yes")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["runnable"] = self.runnable
        return d


# ----------------------------------------------------------------------------------------------------------- ORCA-DL
ORCA_DEPTHS = (10, 15, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, 800, 1000)
_ORCA_IN = (
    Var("so", "g/kg", "sea water salinity", ORCA_DEPTHS, "m", "salt (GODAS, kg/kg -> x1000)"),
    Var("thetao", "degC", "sea water potential temperature", ORCA_DEPTHS, "m", "pottmp (GODAS, K -> -273.15)"),
    Var("tos", "degC", "sea surface temperature of the model's top layer (CMIP6 tos during training)", (), "",
        "sst in the official GODAS example = GODAS 5 m potential temperature",
        "With GODAS initial conditions this is a near-surface water temperature PROXY (5 m), not a satellite/bulk SST."),
    Var("uo", "m/s", "eastward sea water velocity", ORCA_DEPTHS, "m", "ucur (GODAS)"),
    Var("vo", "m/s", "northward sea water velocity", ORCA_DEPTHS, "m", "vcur (GODAS)"),
    Var("zos", "m", "sea surface height above geoid", (), "", "sshg (GODAS)"),
    Var("tauu", "N/m^2", "eastward surface wind stress on the ocean (positive eastward)", (), "", "uflx (GODAS)",
        "Official example: equatorial-Pacific January mean -0.047 N/m^2 (easterly trades -> negative)."),
    Var("tauv", "N/m^2", "northward surface wind stress on the ocean", (), "", "vflx (GODAS)"),
)
_ORCA_OUT = tuple(v for v in _ORCA_IN if v.name not in ("tauu", "tauv"))

ORCA_DL = CheckpointCard(
    key="orca-dl/seed_N", model="orca-dl", checkpoint="seed_1 .. seed_8 (one checkpoint per training seed)",
    backend="torch-native", product="seasonal-monthly",
    code_source="https://github.com/OpenEarthLab/ORCA-DL (official PyTorch; native dependency-light port in weatherai/models/orca_dl)",
    code_licence="official repo: no LICENSE file (all rights reserved by default); this port: MIT",
    weights_source="https://huggingface.co/datasets/JayKuo/ORCA-DL-data (model_weights/seed_{1..8}.bin, stat/{mean,std}/*.npy); "
                   "mirror: OneDrive link in the official README",
    weights_licence="not stated (HF dataset card has no licence); Zenodo 15383228 (CC-BY-4.0) covers data only",
    weights_redistributable="no",
    files=("model_weights/seed_N.bin (2.16 GB each)", "stat/mean/<godas_var>.npy", "stat/std/<godas_var>.npy"),
    parameters=539_942_562,
    inputs=_ORCA_IN,
    history="1 monthly-mean state (the initial month); wind stress of the initial month only",
    grid="regular 1 deg, lat -63.5..63.5 (128, south->north), lon 0.5..359.5 (360); official preprocessing: cdo remapbil",
    vertical="16 depth levels " + ",".join(map(str, ORCA_DEPTHS)) + " m; official preprocessing: cdo intlevel (linear, no extrapolation)",
    normalisation="x_n = (x - mean[m]) / std[m] per grid point and level, m = calendar month of the field (0 = January); "
                  "input uses the initial month, lead L is de-normalised with month (m0 + L) % 12 (official demo.ipynb); "
                  "NaN -> 0 after normalisation (official)",
    outputs=_ORCA_OUT,
    time_step="1 month (monthly means)",
    time_semantics="lead L = monthly mean of calendar month (init month + L); the init month is the monthly mean used as input",
    forcings=(Forcing("tauu, tauv", "N/m^2", "init only",
                      "the model has no future-forcing input: wind stress is encoded once from the initial month"),),
    spatial_mask="ocean points where the GODAS input is defined (per variable; ~31-34 % of the 128x360 domain is NaN); "
                 "domain stops at 63.5S/63.5N -> no Arctic, no Southern Ocean poleward of 63.5S; outputs are masked to NaN outside "
                 "the input-valid mask, never zero-filled",
    ensemble="no stochastic sampling; members = different training seeds (seed_1..8) of the same model. Official predict.py "
             "averages the seeds' normalised outputs. Seeds are NOT independent models.",
    technical_max_lead=Lead(None, "month", "autoregressive: lead-time experts for 1..6, the lead-6 prediction is fed back; "
                                           "official predict.sh uses 24 months; no hard limit in code"),
    validated_lead=Lead(None, "month", "no independent multi-init hindcast run by WeatherAI"),
    status={
        "implemented": Evidence("yes", "native PyTorch port; parameter names follow the official module", "weatherai/models/orca_dl"),
        "weights_obtainable": Evidence("yes", "HF dataset (seeds 1-8) + OneDrive; download with weatherai.models.orca_dl.download",
                                       "https://huggingface.co/datasets/JayKuo/ORCA-DL-data", CHECKED),
        "real_data_inference": Evidence("partial", "one GODAS initial state (Jan 1980, official example), seed_1, 11 leads, CPU and HF GPU; "
                                        "SST_S2S reported a monthly inference with the official checkpoint (external)",
                                        "docs/results/orca_dl_real_case.json", "2026-10-02"),
        "official_parity": Evidence("yes", "7 monthly steps vs official modules, seed_1, rel. L2 <= 7.3e-7",
                                    "docs/results/orca_dl_parity.json", "2026-10-02"),
        "multi_init_long_rollout": Evidence("no", "seeds 2-8, other initial months/years and >11-month rollouts not run"),
        "independent_hindcast": Evidence("no", "no hindcast skill computed by WeatherAI"),
    },
    paper="Guo et al., Sci. Adv. 11, eadu2488 (2025)",
    paper_claims="seasonal-to-decadal ocean prediction skill reported in the paper (not reproduced here)",
    resources="GPU: 2.4 GB peak, 7 steps 0.82 s (HF ZeroGPU); CPU fp32 feasible (~2.2 GB weights per seed)",
    limitations=("GODAS 'sst' is 5 m potential temperature: SST proxy, keep separate from OISST/ERA5 SST",
                 "domain limited to 63.5S-63.5N", "wind stress held at the initial month",
                 "GODAS raw uflx/vflx sign convention not re-checked against PSL files here (PSL unreachable); "
                 "the official example is eastward-positive"),
    api="weatherai.inference.orca_dl.ORCADLForecaster",
)

# ---------------------------------------------------------------------------------------------------------- FuXi-S2S
FUXI_LEVELS = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)
_FUXI_IN = (
    Var("z", "m^2/s^2", "geopotential", FUXI_LEVELS, "hPa", "geopotential"),
    Var("t", "K", "temperature", FUXI_LEVELS, "hPa", "temperature"),
    Var("u", "m/s", "u wind", FUXI_LEVELS, "hPa", "u_component_of_wind"),
    Var("v", "m/s", "v wind", FUXI_LEVELS, "hPa", "v_component_of_wind"),
    Var("q", "kg/kg", "specific humidity", FUXI_LEVELS, "hPa", "specific_humidity"),
    Var("t2m", "K", "2 m temperature", (), "", "2m_temperature"),
    Var("d2m", "K", "2 m dewpoint", (), "", "2m_dewpoint_temperature"),
    Var("sst", "K", "ERA5 sea surface temperature (NaN over land, official mask.nc)", (), "", "sea_surface_temperature"),
    Var("ttr", "W/m^2", "top net thermal radiation as a mean flux", (), "", "top_net_thermal_radiation",
        "official input builder: ERA5 hourly accumulation [J/m^2] / 3600"),
    Var("10u", "m/s", "", (), "", "10m_u_component_of_wind"), Var("10v", "m/s", "", (), "", "10m_v_component_of_wind"),
    Var("100u", "m/s", "", (), "", "100m_u_component_of_wind"), Var("100v", "m/s", "", (), "", "100m_v_component_of_wind"),
    Var("msl", "Pa", "mean sea-level pressure", (), "", "mean_sea_level_pressure"),
    Var("tcwv", "kg/m^2", "total column water vapour", (), "", "total_column_water_vapour"),
    Var("tp", "mm/h", "precipitation, daily mean of ERA5 hourly accumulations, x1000, clipped to [0, 1000]", (), "",
        "total_precipitation", "see FUXI_TP_NOTE"),
)
FUXI_TP_NOTE = ("official make_input: tp = clip(ERA5 total_precipitation [m] * 1000, 0, 1000). The official sample "
                "(2020-06-01/02) matches the daily mean of ERA5 HOURLY accumulations -> the channel is a daily-mean rate in mm/h; "
                "multiply by 24 for mm/day. Verified against WeatherBench2 hourly ERA5 (docs/results/fuxi_s2s_input_check.json).")

FUXI_S2S = CheckpointCard(
    key="fuxi-s2s/model-1.0", model="fuxi-s2s", checkpoint="model-1.0 (fuxi_s2s.onnx + fp16 external data)",
    backend="onnx-official", product="subseasonal-daily",
    code_source="https://github.com/tpys/FuXi-S2S (official ONNX driver inference.py / data_util.py); native PyTorch port in "
                "weatherai/models/fuxi_s2s (secondary backend)",
    code_licence="official driver: see upstream repo; this port: MIT",
    weights_source="Zenodo 10.5281/zenodo.15718402, model-1.0.tar (2.08 GB)",
    weights_licence="CC-BY-NC-ND-4.0",
    weights_redistributable="no",
    files=("model-1.0/fuxi_s2s.onnx (1.6 MB graph)", "model-1.0/fuxi_s2s (2.08 GB fp16 external data)", "data/mask.nc (GitHub)"),
    parameters=1_038_652_304,
    inputs=_FUXI_IN,
    history="2 consecutive daily means (init day - 1, init day), 76 channels, order = weatherai.models.fuxi_s2s.CHANNELS",
    grid="regular 1.5 deg incl. poles, lat 90..-90 (121, north->south), lon 0..358.5 (240)",
    vertical="13 pressure levels 1000..50 hPa (channel order 1000 first)",
    normalisation="inside the ONNX graph (mean/std initialisers); NaN -> 0; precipitation input channel zeroed by the graph",
    outputs=_FUXI_IN,
    time_step="1 day",
    time_semantics="ERA5 daily means; input times = [init-1d, init]; output of step k (0-based) = daily mean of init + (k+1) days",
    forcings=(),
    spatial_mask="sst NaN over land in the input (official land_to_nan with data/mask.nc); WeatherAI re-applies the input land mask "
                 "to the sst output (the network writes values everywhere)",
    ensemble="stochastic latent: two Gaussian draws per step (RandomNormalLike in the ONNX). Official README example: 11 members. "
             "WeatherAI default replaces the two draws by graph inputs with seeded noise rng([seed, member, step]) "
             "(identical to the internal sampling in distribution, reproducible, shared with the native backend).",
    technical_max_lead=Lead(42, "day", "lead-time encoding trained for step 0..41; official total_step = 42; refused beyond"),
    validated_lead=Lead(None, "day", "no independent hindcast by WeatherAI"),
    status={
        "implemented": Evidence("yes", "ONNX runner (official graph) + native PyTorch port", "weatherai/inference/fuxi_s2s.py"),
        "weights_obtainable": Evidence("yes", "Zenodo, user download; never redistributed", "https://zenodo.org/records/15718402", CHECKED),
        "real_data_inference": Evidence("partial", "official ERA5 sample (2020-06-01/02): native fp32 1 step + 3-step chain on CPU, 1 step on "
                                        "HF GPU; SST_S2S reported 11 members x 42 days with the official ONNX (external)",
                                        "docs/results/fuxi_s2s_parity.json", "2026-10-02"),
        "official_parity": Evidence("yes", "native fp32 vs onnxruntime fp16 graph, same injected noise: step-0 rel. L2 6.3e-6, "
                                    "3-step chain <= 1.4e-5 (worst channel 1.2e-3); not bitwise (fp16 vs fp32)",
                                    "docs/results/fuxi_s2s_parity.json", "2026-10-02"),
        "multi_init_long_rollout": Evidence("external", "42-day x 11-member ONNX run reported by SST_S2S; not run from this repo"),
        "independent_hindcast": Evidence("no", "no S2S skill (RPSS/CRPS/ACC) computed by WeatherAI"),
    },
    paper="Chen et al., Nat. Commun. 15, 6425 (2024)",
    paper_claims="paper reports S2S skill over 42 days (weeks 3-6) against ECMWF S2S; not reproduced here",
    resources="native fp32: 5.4 GB GPU peak, 0.67 s/step (HF ZeroGPU); ~25 s/step on 8 CPU cores; ONNX fp16 graph on CPU similar",
    limitations=("CC-BY-NC-ND: non-commercial, no derivatives of the weights; outputs must not be presented as an official product",
                 "the native port is a secondary backend verified for 1 step + 3-step chain only",
                 "official ONNX samples noise without a seed; reproducibility needs the WeatherAI noise injection"),
    api="weatherai.inference.fuxi_s2s.FuXiS2SForecaster",
)

# -------------------------------------------------------------------------------------------------------------- ACE2
ACE2_ERA5 = CheckpointCard(
    key="ace2/ACE2-ERA5", model="ace2", checkpoint="ace2_era5_ckpt.tar", backend="torch-native", product="forced-atmosphere",
    code_source="https://github.com/ai2cm/ace (fme, Apache-2.0); native SFNO + stepper in weatherai/models/ace2",
    code_licence="Apache-2.0", weights_source="https://huggingface.co/allenai/ACE2-ERA5", weights_licence="Apache-2.0",
    weights_redistributable="yes",
    files=("ace2_era5_ckpt.tar (1.8 GB)", "inference_config.yaml", "initial_conditions/ic_2020.nc", "forcing_data/forcing_2020.nc"),
    parameters=455_831_040,
    inputs=(Var("<44 in_names from the checkpoint>", "physical", "prognostic state (8 hybrid-sigma layers + surface) and forcings; "
                "names read from stepper.config.in_names at load time"),),
    history="1 state (6-hourly instantaneous)",
    grid="1 deg Gauss-Legendre 180x360", vertical="8 hybrid sigma-pressure layers (ak, bk in the checkpoint)",
    normalisation="per-variable scalar mean/std from the checkpoint (stepper.normalizer)",
    outputs=(Var("<50 out_names from the checkpoint>", "physical", "includes PRATEsfc [kg m-2 s-1], TMP2m, surface fluxes"),),
    time_step="6 h", time_semantics="instantaneous state every 6 h; fluxes/precipitation are 6-h means",
    forcings=(Forcing("surface_temperature", "K", "every step (t+1)", "SST over ocean (round(ocean_fraction)==1) is PRESCRIBED"),
              Forcing("ocean_fraction", "1", "every step"), Forcing("sea_ice_fraction", "1", "every step"),
              Forcing("DSWRFtoa", "W/m^2", "every step (t+1)"), Forcing("global_mean_co2", "mol/mol", "every step"),
              Forcing("HGTsfc, land_fraction", "m, 1", "static")),
    spatial_mask="global; SST taken from the forcing wherever round(ocean_fraction) == 1",
    ensemble="deterministic network; ensembles need IC/forcing perturbations (not implemented)",
    technical_max_lead=Lead(None, "6h-step", "no code limit; multi-year runs are the design target of the paper"),
    validated_lead=Lead(None, "6h-step", "WeatherAI: one 8-day case only"),
    status={
        "implemented": Evidence("yes", "native SFNO + corrector stepper", "weatherai/models/ace2"),
        "weights_obtainable": Evidence("yes", "HF allenai/ACE2-ERA5", "https://huggingface.co/allenai/ACE2-ERA5", CHECKED),
        "real_data_inference": Evidence("partial", "2020-06-01 IC, 32 steps (8 days), CPU + HF GPU", "tests/models/ace2/data/real_case_report.json", "2026-10-02"),
        "official_parity": Evidence("yes", "rollout vs fme, 12 steps, max rel. error 7.9e-5", "tests/models/ace2/data/fme_rollout_parity_12steps.json", "2026-10-02"),
        "multi_init_long_rollout": Evidence("no", "multi-year / multi-init runs not done"),
        "independent_hindcast": Evidence("no", ""),
    },
    paper="Watt-Meyer et al., npj Clim. Atmos. Sci. 8, 205 (2025)",
    resources="3.1 GB GPU peak, 32 steps 3.5 s (HF ZeroGPU)",
    limitations=("atmosphere-only: SST and sea ice are inputs, so a run is an atmospheric response to the given SST, NOT an SST forecast",
                 "with observed future SST the run is a conditional hindcast, not a forecast"),
    api="weatherai.inference.ace2.ACE2Forecaster",
)

# --------------------------------------------------------------------------------------------------------- NeuralGCM
NEURALGCM = CheckpointCard(
    key="neuralgcm/v1", model="neuralgcm", checkpoint="deterministic_{2_8,1_4,0_7}_deg, stochastic_1_4_deg, stochastic_{precip,evap}_2_8_deg",
    backend="jax-official", product="weather-to-climate",
    code_source="https://github.com/neuralgcm/neuralgcm (official JAX) + dinosaur dycore; weatherai wraps it unchanged",
    code_licence="Apache-2.0",
    weights_source="gs://neuralgcm/models/v1/*.pkl (public); precip checkpoints Zenodo 10.5281/zenodo.17109230",
    weights_licence="v1: CC-BY-SA-4.0; precip/evap: CC-BY-4.0", weights_redistributable="yes",
    files=("<name>.pkl (58 MB for 2.8 deg ... ~1 GB for 0.7 deg)",), parameters=None,
    inputs=(Var("<model.input_variables>", "ERA5 units", "geopotential, temperature, u, v, specific humidity, cloud water/ice on 37 pressure levels"),),
    history="1 instantaneous state", grid="Gaussian grid of the checkpoint (2.8 deg: 128x64)", vertical="37 pressure levels in, sigma levels inside",
    normalisation="inside the official model", outputs=(Var("<model.output variables>", "ERA5 units", "same as inputs + diagnostics"),),
    time_step="internal 1 h (2.8 deg) ... ; output every requested timedelta", time_semantics="instantaneous",
    forcings=(Forcing("sea_surface_temperature", "K", "every step", "WeatherAI policy must be chosen: persist_initial or provided"),
              Forcing("sea_ice_cover", "1", "every step", "same")),
    spatial_mask="global", ensemble="stochastic checkpoints: seed -> member; deterministic checkpoints: one member",
    technical_max_lead=Lead(None, "6h-step", "no code limit; forcings beyond the given series are held per policy"),
    validated_lead=Lead(None, "6h-step", "WeatherAI: demo snapshot + one 24-h precipitation case"),
    status={
        "implemented": Evidence("yes", "official JAX path wrapped (forecast backend); the native PyTorch port is PARTIAL (learned "
                                "components + spherical harmonics, no dycore) and is NOT offered as a forecaster",
                                "weatherai/models/neuralgcm"),
        "weights_obtainable": Evidence("yes", "public GCS bucket", "https://storage.googleapis.com/neuralgcm/models/v1/", CHECKED),
        "real_data_inference": Evidence("partial", "precip checkpoints 24 h from 2020-01-01 vs ERA5 (2026-10-02); NeuralGCMForecaster "
                                        "(official JAX, deterministic_2_8_deg, CPU) on the official demo ERA5 snapshot 1959-01-02, "
                                        "4 x 6 h, 185 s incl. compile, peak RSS 2.6 GB (2026-10-08). Lead 0 (decoded initial state) "
                                        "is not the analysis: 157 K at 10 hPa vs input minimum 183 K",
                                        "tests/inference/test_ace2_neuralgcm_unicm.py", CHECKED),
        "official_parity": Evidence("yes", "forecast path IS the official code (wrapper only converts I/O)", "weatherai/models/neuralgcm/neuralgcm.py"),
        "multi_init_long_rollout": Evidence("no", ""),
        "independent_hindcast": Evidence("no", ""),
    },
    paper="Kochkov et al., Nature 632, 1060 (2024)",
    limitations=("atmosphere-only: SST/sea ice prescribed (persisted or given) -> not an SST forecast",
                 "partial PyTorch port must not be used for forecasts"),
    api="weatherai.inference.neuralgcm.NeuralGCMForecaster",
)

# ------------------------------------------------------------------------------------------------------------- UniCM
UNICM = CheckpointCard(
    key="unicm/none", model="unicm", checkpoint="none published", backend="none", product="climate-modes",
    code_source="https://github.com/tsinghua-fib-lab/UniCM-Global-Climate-Modes (MIT; HEAD 67fe4c1, 2026-02-06); native port in weatherai/models/unicm",
    code_licence="MIT", weights_source="none: README expects a locally trained runs/SaveModel (20 seeds trained on CMIP6 CESM2-FV2 + reanalyses)",
    weights_licence="n/a", weights_redistributable="no", files=(), parameters=12_726_037,
    inputs=(Var("5 channels", "normalised", "upper-ocean / atmosphere anomaly fields on a 12x72 grid (see official LoadData.py)"),),
    history="12 months", grid="5 deg 12x72 (official ORAS5 regions)", vertical="n/a", normalisation="official LoadData.py (not ported)",
    outputs=(Var("fields + climate-mode indices", "normalised"),), time_step="1 month", time_semantics="monthly means",
    forcings=(), spatial_mask="official region masks", ensemble="20 seed models bagged (official)",
    technical_max_lead=Lead(24, "month", "pred 24 in the official config"),
    validated_lead=Lead(None, "month", "no trained weights"),
    status={
        "implemented": Evidence("yes", "architecture bit-exact vs official code with RANDOM weights", "docs/results/unicm_parity.json"),
        "weights_obtainable": Evidence("no", "official GitHub HEAD 67fe4c1 contains no checkpoint and no link (re-checked " + CHECKED +
                                       "); Zenodo 10.5281/zenodo.19173780 is code-only and HF had no checkpoint (checked 2026-10-01; "
                                       "Zenodo/HF unreachable from this environment on " + CHECKED + ")",
                                       "https://github.com/tsinghua-fib-lab/UniCM-Global-Climate-Modes", CHECKED),
        "real_data_inference": Evidence("no", "impossible without weights; random-weight outputs are never forecasts"),
        "official_parity": Evidence("partial", "random weights only (structure parity)", "docs/results/unicm_parity.json"),
        "multi_init_long_rollout": Evidence("no", ""),
        "independent_hindcast": Evidence("no", ""),
    },
    paper="Yuan et al., Nat. Mach. Intell. 8, 930-941 (2026)",
    limitations=("weights not obtainable -> blocked", "retraining = separate project: CMIP6 + ORAS5/ERA5/GODAS/SODA data, ~20 seeds, "
                 "explicit train/validation/test years"),
    api="weatherai.inference.unicm.UniCMForecaster (refuses without a user-trained checkpoint)",
)

CARDS: Dict[str, CheckpointCard] = {c.model: c for c in (ORCA_DL, FUXI_S2S, ACE2_ERA5, NEURALGCM, UNICM)}


def get(model: str) -> CheckpointCard:
    k = model.lower().replace("_", "-")
    if k not in CARDS:
        raise KeyError(f"no card for {model!r}; have {sorted(CARDS)}")
    return CARDS[k]


def to_json(indent: int = 1) -> str:
    return json.dumps({k: c.to_dict() for k, c in CARDS.items()}, indent=indent, default=list)


def summary_rows() -> List[dict]:
    rows = []
    for c in CARDS.values():
        r = dict(model=c.model, backend=c.backend, product=c.product, runnable=c.runnable,
                 technical_max_lead=f"{c.technical_max_lead.value or 'unbounded'} {c.technical_max_lead.unit}",
                 validated_lead=c.validated_lead.value, weights_licence=c.weights_licence,
                 redistributable=c.weights_redistributable)
        r.update({k: c.status[k].state for k in LEVELS})
        rows.append(r)
    return rows
