"""Aardvark Weather end-to-end system: observation **encoder**, forecast **processor**, station
**decoder** (explicit PyTorch port of the official ``models.py`` / ``misc_downscaling_functionality.py`` /
``e2e_model.py``; CC0, https://github.com/anna-allen/aardvark-weather-public).

* :class:`AardvarkEncoder` – ``ConvCNPWeather(mode="assimilation", decoder="vit_assimilation")``:
  per-instrument set-convolutions (IASI/ASCAT are interpolated, HadISD/ICOADS/IGRA scattered ->
  grid, GridSat/AMSU-A/AMSU-B/HIRS gridded -> grid) + elevation / climatology / time channels
  (277 channels) -> pointwise MLP -> patch-3 ViT (8 blocks) -> normalised 24-channel initial state.
* :class:`AardvarkProcessor` (in ``aardvark.py``) – one 24 h step (ViT).
* :class:`AardvarkDecoder` – ``ConvCNPWeatherOnToOff``: U-Net on the forecast grid -> Gaussian
  set-conv to station locations -> station MLP (+ altitude, lon/lat, time).
* :class:`AardvarkE2E` – encoder -> ``lead_time`` processors -> decoder (official
  ``ConvCNPWeatherE2E.forward`` incl. (un)normalisation), without mutating the input task.

Parameter names equal the official ones, so the released encoder / decoder / processor checkpoints
load with ``strict=True`` (see ``load_official_*``). Layout quirks of the official code (lon/lat
transposes, cylindrical padding applied along the last axis) are reproduced on purpose.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import norm_factors as nf
from .aardvark import MLP, AardvarkProcessor, ViT
from .setconv import SetConv
from .unet import Unet

# number of channels of every encoder input (official ``ConvCNPWeather.forward``, one time frame)
ENCODER_CHANNELS = dict(iasi=52, ascat=17, hadisd=8, icoads=10, sat=4, amsua=26, amsub=24, igra=48, hirs=52, elev=7, clim=24, time=5)
assert sum(ENCODER_CHANNELS.values()) == 277


def _t(name, v):
    return torch.tensor(v, dtype=torch.float32)


class AardvarkEncoder(nn.Module):
    """Observations -> normalised 24-channel gridded initial state, output ``(B, n_lat, n_lon, 24)``.

    ``task`` keys (all ``*_current``): ``x_context_hadisd`` (list of 4 ``(B, 2, N)`` lon/lat tensors) and
    ``y_context_hadisd`` (list of 4 ``(B, N)``) for HadISD; ``sat_x`` (2 coords) / ``sat`` ``(B, 2, nx, ny)``;
    ``icoads_x`` / ``icoads`` ``(B, 5, N)``; ``igra_x`` / ``igra`` ``(B, 24, N)``; ``amsua_x`` / ``amsua``
    ``(B, 180, 360, 13)`` (official sample sizes; ``x`` = two coordinate vectors of lengths 360 and 180);
    ``amsub_x`` / ``amsub`` ``(B, 360, 181, 12)``; ``hirs_x`` / ``hirs`` ``(B, 360, 181, 26)``;
    ``iasi`` ``(B, w, h, 52)``; ``ascat`` ``(B, w, h, 17)``; ``era5_elev`` ``(B, 7, n_lat, n_lon)``;
    ``climatology`` ``(B, 24, n_lon, n_lat)``; ``aux_time`` ``(B, 5)``. Coordinates are degrees / 360.
    NaN = missing. Use :func:`task_from_official` to convert an official sample task.
    """

    def __init__(
        self,
        n_lon: int = 240,
        n_lat: int = 121,
        vit_size: Tuple[int, int] = (256, 128),
        embed_dim: int = 512,
        depth: int = 8,
        patch_size: int = 3,
        num_heads: int = 16,
        decoder_depth: int = 4,
        out_channels: int = 24,
        learnable_setconvs: bool = False,
    ):
        super().__init__()
        self.n_lon, self.n_lat, self.vit_size, self.out_channels = n_lon, n_lat, tuple(vit_size), out_channels
        mk = lambda mode, n: nn.ModuleList(  # noqa: E731
            [SetConv(0.001, mode, True, learnable=learnable_setconvs) for _ in range(n)]
        )
        # parameters present in official checkpoints (``ascat_setconvs``/``sc_out`` are unused in forward)
        self.ascat_setconvs = SetConv(0.001, "OnToOn", True)
        self.sc_out = SetConv(0.001, "OnToOff", False)
        self.amsua_setconvs, self.amsub_setconvs, self.hirs_setconvs = mk("OnToOn", 13), mk("OnToOn", 12), mk("OnToOn", 26)
        self.sat_setconvs = mk("OnToOn", 2)
        self.hadisd_setconvs, self.icoads_setconvs, self.igra_setconvs = mk("OffToOn", 5), mk("OffToOn", 5), mk("OffToOn", 24)
        self.decoder_lr = ViT(256, out_channels, embed_dim, self.vit_size, patch_size, depth, decoder_depth, num_heads,
                              4.0, per_var_embedding=False, mlp_in=sum(ENCODER_CHANNELS.values()))
        self.mlp = MLP(out_channels, out_channels, 128, 4)  # unused in forward; kept for checkpoint parity
        self.register_buffer("grid_lon", torch.linspace(0, 360, n_lon)[None] / 360, persistent=False)
        self.register_buffer("grid_lat", torch.linspace(-90, 90, n_lat)[None] / 360, persistent=False)

    # -- per-instrument encoders ----------------------------------------------------------------
    @property
    def int_grid(self) -> List[torch.Tensor]:
        return [self.grid_lon, self.grid_lat]

    def _grid_sc(self, convs, x_in, y, n):
        return torch.cat(
            [convs[i](x_in=list(x_in), wt=y[:, i : i + 1], x_out=self.int_grid) for i in range(n)], dim=1
        )

    def _regrid(self, y: torch.Tensor) -> torch.Tensor:
        """IASI / ASCAT: NaN->0, nearest-resize to the internal grid, flip latitude axis."""
        y = torch.nan_to_num(y, nan=0.0)
        e = F.interpolate(y.permute(0, 3, 1, 2), size=(self.n_lon, self.n_lat))
        return torch.flip(e, dims=[-1])

    def encode_observations(self, task: Dict) -> torch.Tensor:
        """The 277-channel input stack ``(B, 277, n_lon, n_lat)`` (before the ViT)."""
        t = task
        had = torch.cat(
            [
                self.hadisd_setconvs[c](
                    x_in=[t["x_context_hadisd"][c][:, 0, :], t["x_context_hadisd"][c][:, 1, :]],
                    wt=t["y_context_hadisd"][c].unsqueeze(1),
                    x_out=self.int_grid,
                )
                for c in range(4)
            ],
            dim=1,
        )
        ico = torch.cat(
            [self.icoads_setconvs[c](x_in=t["icoads_x"], wt=t["icoads"][:, c].unsqueeze(1), x_out=self.int_grid) for c in range(5)], dim=1
        )
        sat = self._grid_sc(self.sat_setconvs, t["sat_x"], t["sat"], t["sat"].shape[1])
        amsua = t["amsua"].clone()
        amsua[..., -1] = float("nan")
        amsua[amsua == 0] = float("nan")
        amsua = self._grid_sc(self.amsua_setconvs, t["amsua_x"], amsua.permute(0, 3, 2, 1), 13)
        amsub = t["amsub"].clone()
        amsub[amsub == 0] = float("nan")
        amsub = self._grid_sc(self.amsua_setconvs, t["amsub_x"], amsub.permute(0, 3, 1, 2), 12)  # official reuses amsua convs
        igra = torch.cat(
            [self.igra_setconvs[c](x_in=t["igra_x"], wt=t["igra"][:, c].unsqueeze(1), x_out=self.int_grid) for c in range(24)], dim=1
        )
        hirs = t["hirs"].clone()
        hirs[hirs == 0] = float("nan")
        hirs = self._grid_sc(self.hirs_setconvs, t["hirs_x"], hirs.permute(0, 3, 1, 2), 26)
        elev = torch.flip(t["era5_elev"].permute(0, 1, 3, 2), dims=[2])  # (B, 7, n_lon, n_lat), lon axis flipped (as official)
        time_ch = torch.ones_like(elev[:, :5]) * t["aux_time"].unsqueeze(-1).unsqueeze(-1)
        x = torch.cat(
            [self._regrid(t["iasi"]), self._regrid(t["ascat"]), had, ico, sat, amsua, amsub, igra, hirs, elev, t["climatology"], time_ch],
            dim=1,
        )
        return x

    def forward(self, task: Dict) -> torch.Tensor:
        x = self.encode_observations(task)
        if x.shape[-1] > x.shape[-2]:
            x = x.permute(0, 1, 3, 2)
        x = F.interpolate(x, size=self.vit_size)  # (B, 277, 256, 128), nearest
        x = self.decoder_lr(x)  # lead time = 1 (as official), (B, H', W', 24)
        x = F.interpolate(x.permute(0, 3, 1, 2), size=(self.n_lon, self.n_lat))  # (B, 24, n_lon, n_lat)
        return x.permute(0, 3, 2, 1)  # (B, n_lat, n_lon, 24)


class _ResBlock(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(n, n), nn.ReLU())

    def forward(self, x):
        return self.block(x) + x


class DownscalingMLP(nn.Module):
    """Station MLP: Linear -> h_layers x residual(Linear-ReLU) -> Linear."""

    def __init__(self, in_channels: int, out_channels: int, h_channels: int = 64, h_layers: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_channels, h_channels), *[_ResBlock(h_channels) for _ in range(h_layers)],
                                 nn.Linear(h_channels, out_channels))

    def forward(self, x):
        return self.mlp(x)


class AardvarkDecoder(nn.Module):
    """Forecast grid state -> station predictions ``(B, N)`` (one variable, e.g. 2 m temperature).

    ``task`` keys: ``y_context`` ``(B, 24 + 12, n_lon, n_lat)`` (state + 12 aux channels, official 36),
    ``x_context`` (lon, lat grid coords ``(1, n_lon)``, ``(1, n_lat)``), ``x_target`` ``(B, 2, N)``,
    ``alt_target`` ``(B, 2, N)``, ``aux_time`` ``(B, 5, 1, 1)``.
    """

    def __init__(self, in_channels: int = 36, int_channels: int = 24, div_factor: int = 1, h_channels: int = 64, h_layers: int = 2):
        super().__init__()
        self.sc_out = SetConv(0.001, "OnToOff", density_channel=False)
        self.decoder_lr = Unet(in_channels, int_channels, div_factor)
        self.mlp = DownscalingMLP(int_channels + 9, 1, h_channels, h_layers)

    def forward(self, task: Dict) -> torch.Tensor:
        x = self.decoder_lr(task["y_context"])  # (B, n_lon, n_lat, C)
        x = x.permute(0, 3, 1, 2)
        x_target = task["x_target"]
        N = x_target.shape[2]
        x = self.sc_out(x_in=list(task["x_context"]), wt=x, x_out=[x_target[:, 0, :], x_target[:, 1, :]])  # (B, C, N)
        aux_time = task["aux_time"].squeeze(-1).repeat(1, 1, N)
        x = torch.cat([x, task["alt_target"], x_target, aux_time], dim=1).permute(0, 2, 1)
        return self.mlp(x).squeeze(-1)


class AardvarkE2E(nn.Module):
    """Encoder -> ``lead_time`` x processor (24 h each) -> station decoder (official E2E forward)."""

    def __init__(self, encoder: AardvarkEncoder, processors: Sequence[AardvarkProcessor], decoder: AardvarkDecoder,
                 input_mean=None, input_std=None, diff_mean=None, diff_std=None):
        super().__init__()
        self.encoder = encoder
        self.processors = nn.ModuleList(processors)
        self.decoder = decoder
        for n, v, d in (("input_mean", input_mean, nf.MEAN_4U_1), ("input_std", input_std, nf.STD_4U_1),
                        ("diff_mean", diff_mean, nf.MEAN_DIFF_4U_1), ("diff_std", diff_std, nf.STD_DIFF_4U_1)):
            self.register_buffer(n, (torch.as_tensor(d, dtype=torch.float32) if v is None else torch.as_tensor(v, dtype=torch.float32)), persistent=False)

    def forward(self, task: Dict, return_gridded: bool = False):
        """``task = {"assimilation": ..., "forecast": {"y_context", "lt"}, "downscaling": {...}}`` (official layout).

        Returns station forecast ``(B, N)`` (normalised; un-normalise with ``norm_factors``), or
        ``(station, forecast (B, n_lat, n_lon, 24) physical units, initial_state)`` if ``return_gridded``.
        """
        fc_task = dict(task["forecast"])
        ds_task = dict(task["downscaling"])
        x0 = self.encoder(task["assimilation"])  # (B, n_lat, n_lon, 24) normalised
        y_ctx = torch.cat([x0.permute(0, 3, 2, 1), fc_task["y_context"][:, 24:]], dim=1)
        ds_ctx = torch.cat([x0.permute(0, 3, 2, 1), ds_task["y_context"][:, 24:]], dim=1)
        forecast = None
        for proc in self.processors:
            d = proc(y_ctx, fc_task["lt"])  # (B, n_lat, n_lon, 24) normalised tendency
            base = (y_ctx[:, :-11].permute(0, 2, 3, 1) * self.input_std + self.input_mean).permute(0, 3, 2, 1)
            forecast = (self.diff_mean + d * self.diff_std) + base.permute(0, 2, 3, 1)  # physical units
            nxt = ((forecast - self.input_mean) / self.input_std).permute(0, 3, 2, 1)
            ds_ctx = torch.cat([nxt, ds_ctx[:, 24:]], dim=1)
            y_ctx = torch.cat([nxt, y_ctx[:, 24:]], dim=1)
        ds_task["y_context"] = ds_ctx
        station = self.decoder(ds_task)
        if return_gridded:
            init = x0 * self.input_std + self.input_mean  # (B, n_lat, n_lon, 24)
            return station, forecast, init
        return station


# ------------------------------------------------------------------------------------ utilities
def task_from_official(sample: Dict) -> Dict:
    """Convert an official ``sample_data_final.pkl`` task into the (non-mutated, cleaned) layout used here."""
    a, f, d = sample["assimilation"], sample["forecast"], sample["downscaling"]
    asm = {
        "x_context_hadisd": a["x_context_hadisd_current"], "y_context_hadisd": a["y_context_hadisd_current"],
        "sat_x": a["sat_x_current"], "sat": a["sat_current"], "icoads_x": a["icoads_x_current"], "icoads": a["icoads_current"],
        "igra_x": a["igra_x_current"], "igra": a["igra_current"], "amsua_x": a["amsua_x_current"], "amsua": a["amsua_current"],
        "amsub_x": a["amsub_x_current"], "amsub": a["amsub_current"], "hirs_x": a["hirs_x_current"], "hirs": a["hirs_current"],
        "iasi": a["iasi_current"], "ascat": a["ascat_current"], "era5_elev": a["era5_elev_current"],
        "climatology": a["climatology_current"], "aux_time": a["aux_time_current"],
    }
    return {"assimilation": asm, "forecast": dict(f), "downscaling": dict(d)}


def load_official_sample(path: str) -> Dict:
    """Load the official ``data/sample_data_final.pkl`` (pickled CUDA tensors) on CPU and convert it with
    :func:`task_from_official`. Only load files from the official repository (pickle)."""
    import io
    import pickle

    import torch.storage as ts

    orig = ts._load_from_bytes
    ts._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
    try:
        with open(path, "rb") as f:
            sample = pickle.load(f)
    finally:
        ts._load_from_bytes = orig
    return task_from_official(sample)


def _strip(sd):
    return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}


def _load(path, device):
    # official files are pickled training checkpoints (weights_only=False): only load files from the official source
    return _strip(torch.load(path, map_location=device, weights_only=False)["model_state_dict"])


def load_official_encoder(path: str, device="cpu") -> AardvarkEncoder:
    """``trained_model/encoder/epoch_96`` (HF dataset ``av555/aardvark-weather``), strict load."""
    m = AardvarkEncoder()
    m.load_state_dict(_load(path, device), strict=True)
    return m.to(device)


def load_official_decoder(path: str, device="cpu") -> AardvarkDecoder:
    """``trained_model/decoder/<var>/lt_<k>/epoch_*`` (station decoder), strict load."""
    m = AardvarkDecoder()
    m.load_state_dict(_load(path, device), strict=True)
    return m.to(device)
