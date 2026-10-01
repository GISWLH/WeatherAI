"""Native PyTorch re-implementation of the *learned* parts of NeuralGCM (deterministic 2.8 deg checkpoint).

Scope (what is ported, all numerically checked against the official JAX/Haiku modules, see tests):

* ``ColumnMLP`` / ``EpdTower``: NeuralGCM's per-column encode-process-decode MLP towers (``towers.EpdTower``
  with ``ColumnTower`` / ``MlpUniform``: Linear -> 5 residual blocks of a gelu-MLP -> bias-free Linear).
* ``VerticalConvTower``: the 1-D vertical CNN embedding (5 x Conv1D(k=5, SAME) over pressure levels).
* ``SurfaceEmbedding``: land / sea / sea-ice surface towers blended by land-sea mask and sea-ice cover.
* ``LearnedPositionalFeatures`` and ``LearnedOrography`` (modal correction on top of a supplied base orography).
* ``NeuralGCMLearnedComponents``: all of the above for the 2.8 deg model (decoder, 2 encoders, physics
  parameterisation) with ``convert_official_params`` strict-loading the official Haiku pickle
  (14,518,180 weights, all 121 haiku modules consumed).

NOT ported (stuck points, see docs/model_status.md): the *feature builders* that assemble the tower inputs
(velocity/prognostic features in dimensional units, pressure/radiation/latitude features, memory features,
shift-and-normalise constants, input clipping/filters) and, above all, the dinosaur dynamical core
(sigma-coordinate moist primitive equations, IMEX-RK stepper, filters, corrector). Hence this module does
not forecast by itself; the official JAX wrapper (``NeuralGCMWrapper``) stays the oracle for forecasts.
The spherical-harmonic layer is in ``spectral.py``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F


def gelu(x: torch.Tensor) -> torch.Tensor:
    """``jax.nn.gelu(approximate=True)`` (gin ``gelu.approximate = True``)."""
    return F.gelu(x, approximate="tanh")


class ColumnMLP(nn.Module):
    """Haiku ``MlpUniform``: ``num_hidden`` x [Linear + gelu] then a final Linear (no final activation)."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_hidden: int, bias_hidden: bool = True, bias_out: bool = True):
        super().__init__()
        dims = [in_dim] + [hidden] * num_hidden
        self.hidden = nn.ModuleList(nn.Linear(a, b, bias=bias_hidden) for a, b in zip(dims[:-1], dims[1:]))
        self.final = nn.Linear(dims[-1], out_dim, bias=bias_out)

    def forward(self, x):
        for lin in self.hidden:
            x = gelu(lin(x))
        return self.final(x)


class EpdTower(nn.Module):
    """Encode (Linear, bias) -> ``num_blocks`` residual process MLPs -> decode MLP (no bias).

    Input / output are channel-first nodal arrays ``(C, lon, lat)`` or batched ``(B, C, lon, lat)``; the MLPs
    act independently on every (lon, lat) column, exactly as NeuralGCM's ``ColumnTower``.
    """

    def __init__(self, in_dim: int, out_dim: int, latent: int = 384, num_blocks: int = 5, process_hidden_layers: int = 3, decode_hidden_layers: int = 0):
        super().__init__()
        self.encode = ColumnMLP(in_dim, 0, latent, 0, bias_out=True)
        self.process = nn.ModuleList(
            ColumnMLP(latent, latent, latent, process_hidden_layers) for _ in range(num_blocks)
        )
        self.decode = ColumnMLP(latent, latent, out_dim, decode_hidden_layers, bias_hidden=False, bias_out=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batched = x.dim() == 4
        if not batched:
            x = x[None]
        b, c, h, w = x.shape
        z = x.permute(0, 2, 3, 1)  # (B, lon, lat, C): columns on the leading dims
        z = self.encode(z)
        for blk in self.process:
            z = z + blk(z)
        z = self.decode(z).permute(0, 3, 1, 2)
        return z if batched else z[0]


class VerticalConvTower(nn.Module):
    """Stack of Conv1D over the level axis; input ``(Cin, L, lon, lat)`` -> ``(Cout, L, lon, lat)``.

    gelu after every conv except the last (``activate_final=False``); haiku ``padding='SAME'`` (k=5 -> pad 2).
    """

    def __init__(self, in_ch: int, out_ch: int, channels: Sequence[int] = (64, 64, 64, 64), kernel: int = 5):
        super().__init__()
        chs = [in_ch] + list(channels) + [out_ch]
        self.convs = nn.ModuleList(nn.Conv1d(a, b, kernel, padding=kernel // 2) for a, b in zip(chs[:-1], chs[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c, l, h, w = x.shape
        z = x.permute(2, 3, 0, 1).reshape(h * w, c, l)  # columns -> batch
        for i, conv in enumerate(self.convs):
            z = conv(z)
            if i < len(self.convs) - 1:
                z = gelu(z)
        return z.reshape(h, w, -1, l).permute(2, 3, 0, 1)


class SurfaceEmbedding(nn.Module):
    """``NodalLandSeaIceEmbedding``: three small EPD surface towers, blended per grid column.

    out = land_frac * land(x_land) + (1-ice) * (1-land_frac) * sea(x_sea) + ice * (1-land_frac) * ice(x_ice)
    """

    def __init__(self, land_in: int = 4, sea_in: int = 1, ice_in: int = 5, out: int = 8):
        super().__init__()

        def tower(i):
            return EpdTower(i, out, latent=8, num_blocks=1, process_hidden_layers=3, decode_hidden_layers=1)

        self.land, self.sea, self.sea_ice = tower(land_in), tower(sea_in), tower(ice_in)

    def forward(self, x_land, x_sea, x_ice, land_frac, ice_cover):
        sea_frac = 1 - land_frac
        return (
            land_frac * self.land(x_land)
            + (1 - ice_cover) * sea_frac * self.sea(x_sea)
            + ice_cover * sea_frac * self.sea_ice(x_ice)
        )


class LearnedPositionalFeatures(nn.Module):
    """Learned ``(latent, lon, lat)`` map (zero-initialised in the official code), times a constant scale."""

    def __init__(self, latent: int = 8, nodal_shape=(128, 64), scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.positional_features = nn.Parameter(torch.zeros(latent, *nodal_shape))

    def forward(self):
        return self.scale * self.positional_features


class LearnedOrography(nn.Module):
    """Modal orography = base + scale * (correction scattered into the triangular-truncation mask)."""

    def __init__(self, mask: torch.Tensor, scale: float = 1e-5):
        super().__init__()
        self.register_buffer("mask", mask.bool(), persistent=False)
        self.scale = scale
        self.correction = nn.Parameter(torch.zeros(int(mask.sum())))

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        corr = torch.zeros(self.mask.shape, dtype=self.correction.dtype, device=self.correction.device)
        corr[self.mask] = self.correction
        return base + corr * self.scale


@dataclass
class NeuralGCMLearnedConfig:
    """Hyper-parameters of the 2.8 deg deterministic model (from the checkpoint's gin config)."""

    latent: int = 384
    num_blocks: int = 5
    nodal_shape: tuple = (128, 64)
    positional_latent: int = 8
    decoder_in: int = 784
    decoder_out: int = 259  # 37 levels x (u, v, T, z?, q, ice, liq) + ...
    encoder_in: int = 1052
    encoder_out: int = 193
    physics_in: int = 2365
    physics_out: int = 192
    surface_out: int = 8
    cnn_features: int = 32
    cnn_in: int = 9
    n_levels_model: int = 32
    # sizes of the triangular modal masks (coords.horizontal.mask.sum()) for the learned orographies
    encoder_orography: int = 4223
    corrector_orography: int = 4094
    orography_scale: float = 1e-5


class NeuralGCMLearnedComponents(nn.Module):
    """All trained neural pieces of the 2.8 deg deterministic NeuralGCM, as explicit modules.

    Each sub-module is the stand-alone equivalent of one Haiku module of the official model:

    ========================  =====================================================================
    ``decoder``               ``dimensional_learned_primitive_to_weatherbench_decoder`` EPD tower
    ``encoder`` / ``encoder_1``  the two ``learned_weatherbench_to_primitive_encoder`` EPD towers
    ``physics``               ``div_curl_neural_parameterization`` EPD tower (tendency network)
    ``surface``               ``nodal_land_sea_ice_embedding`` (physics-side surface features)
    ``volume_cnn``            ``embedding_volume_features`` vertical CNN (physics-side)
    ``pos_*``                 learned positional features of decoder / encoders / physics
    ``orog_*``                learned orography corrections of the encoders and the dycore corrector
    ========================  =====================================================================
    """

    def __init__(self, cfg: Optional[NeuralGCMLearnedConfig] = None, mask_enc: Optional[torch.Tensor] = None, mask_corr: Optional[torch.Tensor] = None):
        super().__init__()
        c = cfg or NeuralGCMLearnedConfig()
        self.cfg = c
        T = lambda i, o: EpdTower(i, o, c.latent, c.num_blocks)
        self.decoder = T(c.decoder_in, c.decoder_out)
        self.encoder = T(c.encoder_in, c.encoder_out)
        self.encoder_1 = T(c.encoder_in, c.encoder_out)
        self.physics = T(c.physics_in, c.physics_out)
        self.surface = SurfaceEmbedding(out=c.surface_out)
        self.volume_cnn = VerticalConvTower(c.cnn_in, c.cnn_features)
        P = lambda: LearnedPositionalFeatures(c.positional_latent, c.nodal_shape)
        self.pos_decoder, self.pos_encoder, self.pos_encoder_1, self.pos_physics = P(), P(), P(), P()
        if mask_enc is None or mask_corr is None:  # triangular-truncation masks of the data (TL63) / dycore grids
            from .spectral import SphericalHarmonicsGrid, SpectralGridConfig

            if mask_enc is None:
                mask_enc = SphericalHarmonicsGrid(SpectralGridConfig(64, 65, 128, 64)).mask
            if mask_corr is None:  # dycore grid: 63 longitude wavenumbers, 64 total (modal shape 125 x 64)
                mask_corr = SphericalHarmonicsGrid(SpectralGridConfig(63, 64, 128, 64)).mask
        assert int(mask_enc.sum()) == c.encoder_orography and int(mask_corr.sum()) == c.corrector_orography
        self.orog_encoder = LearnedOrography(mask_enc, c.orography_scale)
        self.orog_encoder_1 = LearnedOrography(mask_enc, c.orography_scale)
        self.orog_corrector = LearnedOrography(mask_corr, c.orography_scale)

    def forward(self, which: str, x: torch.Tensor) -> torch.Tensor:
        """Apply one named EPD tower to packed nodal features ``(C, lon, lat)`` (e.g. ``'physics'``)."""
        return getattr(self, which)(x)


# --------------------------------------------------------------------------- official parameter conversion
_PRE = "stochastic_modular_step_model/~/"


def _t(a) -> torch.Tensor:
    import numpy as np

    return torch.from_numpy(np.asarray(a, dtype=np.float32).copy())


def _load_mlp(mlp: ColumnMLP, params: Dict[str, dict], prefix: str, used: set):
    """prefix points at '.../mlp_uniform/~/linear_' ; hidden linears 0..n-1, final linear n."""
    lins = list(mlp.hidden) + [mlp.final]
    for i, lin in enumerate(lins):
        key = f"{prefix}{i}"
        p = params[key]
        lin.weight.data.copy_(_t(p["w"]).T)
        if lin.bias is not None:
            lin.bias.data.copy_(_t(p["b"]))
        elif "b" in p:
            raise ValueError(f"unexpected bias for {key}")
        used.add(key)


def _load_epd(tower: EpdTower, params, base: str, used: set, name_enc="encode_tower", name_dec="decode_tower", name_proc="process_tower"):
    """base: '.../epd_tower' (haiku module path up to and including 'epd_tower')."""
    _load_mlp(tower.encode, params, f"{base}/{name_enc}/~/mlp_uniform/~/linear_", used)
    for i, blk in enumerate(tower.process):
        suffix = "" if i == 0 else f"_{i}"
        _load_mlp(blk, params, f"{base}/{name_proc}{suffix}/~/mlp_uniform/~/linear_", used)
    _load_mlp(tower.decode, params, f"{base}/{name_dec}/~/mlp_uniform/~/linear_", used)


def convert_official_params(params: Dict[str, dict], model: Optional[NeuralGCMLearnedComponents] = None, mask_enc=None, mask_corr=None) -> NeuralGCMLearnedComponents:
    """Strict-load the official Haiku ``params`` dict (checkpoint['params']) into the native modules.

    Raises if any official parameter is left unconsumed or any shape mismatches."""
    m = model or NeuralGCMLearnedComponents(mask_enc=mask_enc, mask_corr=mask_corr)
    used: set = set()
    P = _PRE
    dec = P + "dimensional_learned_primitive_to_weatherbench_decoder"
    _load_epd(m.decoder, params, dec + "/nodal_mapping/~/epd_tower", used)
    m.pos_decoder.positional_features.data.copy_(_t(params[dec + "/~/combined_features/~/learned_positional_features"]["learned_positional_features"]))
    used.add(dec + "/~/combined_features/~/learned_positional_features")

    enc = P + "dimensional_learned_weatherbench_to_primitive_with_memory_encoder/~/learned_weatherbench_to_primitive_encoder"
    for suffix, tower, pos, orog in (("", m.encoder, m.pos_encoder, m.orog_encoder), ("_1", m.encoder_1, m.pos_encoder_1, m.orog_encoder_1)):
        b = enc + suffix
        _load_epd(tower, params, b + "/nodal_mapping/~/epd_tower", used)
        k = b + "/~/combined_features/~/learned_positional_features"
        pos.positional_features.data.copy_(_t(params[k]["learned_positional_features"]))
        used.add(k)
        k = b + "/~/learned_orography"
        orog.correction.data.copy_(_t(params[k]["orography"]))
        used.add(k)

    phys = P + "stochastic_physics_parameterization_step"
    dc = phys + "/~/div_curl_neural_parameterization"
    _load_epd(m.physics, params, dc + "/nodal_mapping/~/epd_tower", used)
    k = dc + "/~/combined_features/~/learned_positional_features"
    m.pos_physics.positional_features.data.copy_(_t(params[k]["learned_positional_features"]))
    used.add(k)
    k = phys + "/~/custom_coords_corrector/~/dycore_with_physics_corrector/~/learned_orography"
    m.orog_corrector.correction.data.copy_(_t(params[k]["orography"]))
    used.add(k)

    surf = dc + "/~/combined_features/~/embedding_surface_features/~/nodal_land_sea_ice_embedding"
    for suffix, tower in (("", m.surface.land), ("_1", m.surface.sea), ("_2", m.surface.sea_ice)):
        _load_epd(tower, params, f"{surf}/~/modal_to_nodal_embedding{suffix}/nodal_mapping/~/epd_tower", used,
                  "surface_model_encode_tower", "surface_model_decode_tower", "surface_model_process_tower")

    vol = dc + "/~/combined_features/~/embedding_volume_features/~/modal_to_nodal_embedding/nodal_volume_mapping/~/vertical_conv_tower/~/conv_level"
    for i, conv in enumerate(m.volume_cnn.convs):
        k = vol + ("" if i == 0 else f"_{i}")
        p = params[k]
        conv.weight.data.copy_(_t(p["w"]).permute(2, 1, 0))  # haiku (W, I, O) -> torch (O, I, W)
        conv.bias.data.copy_(_t(p["b"]).reshape(-1))
        used.add(k)

    left = sorted(set(params) - used)
    if left:
        raise KeyError(f"{len(left)} official parameter modules not consumed, e.g. {left[:3]}")
    return m


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
