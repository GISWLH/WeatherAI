"""Load the official ArchesWeather / ArchesWeatherGen checkpoints (HF ``gcouairon/ArchesWeather``, BSD-3; never committed) and the
small statistics files shipped inside the ``geoarches`` repo (BSD-3: constant masks, normalisation stats, prediction-residual stats)."""
from __future__ import annotations

import os
import urllib.request

import torch

from .model import ArchesConfig, ArchesWeather, ArchesWeatherGen, State

HF_REPO = "gcouairon/ArchesWeather"
GEOARCHES_RAW = "https://raw.githubusercontent.com/INRIA/geoarches/main/geoarches/stats/"
STAT_FILES = ["archesweather_constant_masks.pt", "pangu_norm_stats.nc", "deltapred24_aws_denorm.nc"]
SURFACE = ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "mean_sea_level_pressure"]
LEVEL = ["geopotential", "u_component_of_wind", "v_component_of_wind", "temperature", "specific_humidity", "vertical_velocity"]
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


def download(local_dir: str, which=("archesweathergen_checkpoint.ckpt", "archesweather-m-seed0_checkpoint.ckpt"), token=None) -> None:
    from huggingface_hub import hf_hub_download
    os.makedirs(os.path.join(local_dir, "stats"), exist_ok=True)
    for f in which:
        hf_hub_download(HF_REPO, f, local_dir=local_dir, token=token or os.environ.get("HF_TOKEN"))
    for f in STAT_FILES:
        p = os.path.join(local_dir, "stats", f)
        if not os.path.exists(p):
            urllib.request.urlretrieve(GEOARCHES_RAW + f, p)


def load_stats(stats_dir: str) -> dict:
    """mean/std (normalisation), and the state scaler used by the generative model (data std / prediction-residual std, w/3)."""
    import xarray as xr
    nd = xr.open_dataset(os.path.join(stats_dir, "pangu_norm_stats.nc"))
    rd = xr.open_dataset(os.path.join(stats_dir, "deltapred24_aws_denorm.nc"))
    f = lambda a: torch.from_numpy(a).float()
    mean_s = f(nd[SURFACE].sel(statistic="mean").to_array().values)[..., None, None, None]
    std_s = f(nd[SURFACE].sel(statistic="std").to_array().values)[..., None, None, None]
    mean_l = f(nd[LEVEL].sel(statistic="mean", level=LEVELS).to_array().values)[..., None, None]
    std_l = f(nd[LEVEL].sel(statistic="std", level=LEVELS).to_array().values)[..., None, None]
    dsr = f(rd[SURFACE].sel(statistic="diff_std").to_array().values)[..., None, None, None]
    dlv = f(rd[LEVEL].sel(statistic="diff_std", level=LEVELS).to_array().values)[..., None, None]
    scaler_l = std_l / dlv
    scaler_l[LEVEL.index("vertical_velocity")] /= 3.0  # official: downweight_vertical_velocity
    return {"mean": {"surface": mean_s, "level": mean_l}, "std": {"surface": std_s, "level": std_l},
            "scaler": {"surface": std_s / dsr, "level": scaler_l}}


def normalize(state: State, stats: dict) -> State:
    return {k: (state[k] - stats["mean"][k].to(state[k].device)) / stats["std"][k].to(state[k].device) for k in state}


def denormalize(state: State, stats: dict) -> State:
    return {k: state[k] * stats["std"][k].to(state[k].device) + stats["mean"][k].to(state[k].device) for k in state}


def _masks(stats_dir):
    return torch.load(os.path.join(stats_dir, "archesweather_constant_masks.pt"), weights_only=True)


def _sd(path):  # the Lightning checkpoint pickles omegaconf objects (hyper-parameters) -> `pip install omegaconf`
    return torch.load(path, map_location="cpu", weights_only=False)["state_dict"]


def load_official_det(ckpt_dir: str, name: str = "archesweather-m-seed0", device="cpu") -> ArchesWeather:
    m = ArchesWeather(ArchesConfig(depth_multiplier=2, n_concatenated_states=1), _masks(os.path.join(ckpt_dir, "stats")))
    m.load_state_dict(_sd(os.path.join(ckpt_dir, f"{name}_checkpoint.ckpt")), strict=True)
    return m.to(device).eval()


def load_official_gen(ckpt_dir: str, device="cpu") -> tuple[ArchesWeatherGen, dict]:
    """ArchesWeatherGen incl. its 4 embedded deterministic members; returns (model, normalisation stats)."""
    sd_dir = os.path.join(ckpt_dir, "stats")
    m = ArchesWeatherGen(constant_masks=_masks(sd_dir))
    m.load_state_dict(_sd(os.path.join(ckpt_dir, "archesweathergen_checkpoint.ckpt")), strict=True)
    st = load_stats(sd_dir)
    m.state_scaler_surface.copy_(st["scaler"]["surface"])
    m.state_scaler_level.copy_(st["scaler"]["level"])
    return m.to(device).eval(), st


def wb2_to_state(ds, time) -> State:
    """WeatherBench2 ERA5 1.5-degree (240x121) -> raw (un-normalised) model state: latitude north->south, longitude starting at 180 deg
    (the official ``Era5Dataset.convert_to_tensordict`` flips latitude and rolls longitude by half the grid)."""
    d = ds.sel(time=time)
    surf = torch.stack([torch.from_numpy(d[v].transpose("latitude", "longitude").values) for v in SURFACE])[:, None]  # (4,1,H,W)
    lev = torch.stack([torch.from_numpy(d[v].sel(level=LEVELS).transpose("level", "latitude", "longitude").values) for v in LEVEL])
    out = {"surface": surf.float(), "level": lev.float()}
    return {k: v.flip(-2).roll(v.shape[-1] // 2, -1)[None] for k, v in out.items()}


def state_to_wb2_grid(state: State):
    """Inverse layout change (for plotting / comparison with WeatherBench2): (lat south->north, lon 0..358.5)."""
    return {k: v.roll(-(v.shape[-1] // 2), -1).flip(-2) for k, v in state.items()}
