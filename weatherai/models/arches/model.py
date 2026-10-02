"""ArchesWeather (deterministic) and ArchesWeatherGen (flow-matching ensemble on top of 4 deterministic models), native PyTorch.

State layout (same as the official code): a dict ``{"surface": (B,4,1,H,W), "level": (B,6,Pl,H,W)}`` of *normalised* fields on the
1.5 degree grid (H=121 lat from north to south, W=240 lon starting at 180 degrees). Papers: Couairon et al. (ArchesWeather, 2024
arXiv:2412.12971; ArchesWeatherGen, Sci. Adv. 12, eadx2372, 2026). Official code (BSD-3): https://github.com/INRIA/geoarches.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .layers import CondBasicLayer, DownSample, LinVert, UpSample

State = dict  # {"surface": Tensor, "level": Tensor}


def cat_states(a: State, b: State) -> State:
    return {k: torch.cat([a[k], b[k]], dim=1) for k in ("surface", "level")}


@dataclass
class ArchesConfig:
    img_size: tuple = (13, 121, 240)       # (levels, lat, lon)
    surface_ch: int = 4
    level_ch: int = 6
    emb_dim: int = 192
    cond_dim: int = 256
    num_heads: tuple = (6, 12, 12, 6)
    window_size: tuple = (1, 6, 10)
    depth_multiplier: int = 2              # 2 = deterministic ArchesWeather-M, 1 = generative network
    droppath_coeff: float = 0.2
    mlp_ratio: float = 4.0
    n_concatenated_states: int = 1         # prev (det) | pred + prev + noisy (gen)
    axis_attn: bool = True
    swiglu: bool = True
    patch_size: tuple = (2, 2, 2)
    n_const: int = 3                       # land-sea mask, soil type, orography (constant fields)
    add_input_state: bool = False          # "skip" deterministic members predict a residual: out = net(x) + state

    @property
    def tensor_size(self):
        pl = (self.img_size[0] + (self.patch_size[0] - self.img_size[0] % self.patch_size[0]) % self.patch_size[0]) // self.patch_size[0] + 1
        lat = (self.img_size[1] - self.img_size[1] % 2) // self.patch_size[1]  # south pole row removed when lat is odd
        return pl, lat, self.img_size[2] // self.patch_size[2]


class TimestepEmbedder(nn.Module):
    """Sinusoidal embedding of a scalar (month, hour, or flow-matching timestep) followed by a 2-layer MLP."""

    def __init__(self, hidden, freq_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(freq_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.freq_dim = freq_dim

    def forward(self, t, max_period=10000):
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half).to(t.device)
        a = t[:, None].float() * freqs[None]
        return self.mlp(torch.cat([a.cos(), a.sin()], dim=-1))


class ArchesEmbedder(nn.Module):
    """3-D patch embedding of the pressure-level fields and 2-D patch embedding of surface + constant masks; PixelShuffle decoder."""

    def __init__(self, cfg: ArchesConfig, constant_masks: torch.Tensor | None = None):
        super().__init__()
        self.cfg = cfg
        pz, ph, pw = cfg.patch_size
        masks = constant_masks if constant_masks is not None else torch.zeros(cfg.n_const, 1, cfg.img_size[1] - cfg.img_size[1] % 2, cfg.img_size[2])
        self.register_buffer("constant_masks", masks.float(), persistent=False)
        n = cfg.n_concatenated_states
        self.level_proj = nn.Conv3d(cfg.level_ch * (1 + n), cfg.emb_dim, (pz, ph, pw), (pz, ph, pw))
        self.surface_proj = nn.Conv2d(cfg.n_const + cfg.surface_ch * (1 + n), cfg.emb_dim, (ph, pw), (ph, pw))
        l_pad = pz - cfg.img_size[0] % pz
        self.level_pads = (l_pad // 2, l_pad - l_pad // 2)
        self.surface_deconv = nn.Conv2d(2 * cfg.emb_dim, cfg.surface_ch * pw ** 2, 3, padding=1, bias=False)
        self.level_deconv = nn.Conv2d(cfg.emb_dim, cfg.level_ch * pw ** 2, 3, padding=1, bias=False)
        self.pixelshuffle = nn.PixelShuffle(pw)

    def encode(self, state: State, cond_state: State | None) -> torch.Tensor:
        B = state["surface"].shape[0]
        level, surface = state["level"], state["surface"].squeeze(-3)
        if surface.shape[-2] % 2:  # drop the south-pole row
            surface, level = surface[..., :-1, :], level[..., :-1, :]
        surface = torch.cat([surface, self.constant_masks[None, :, 0].to(surface.dtype).expand(B, -1, -1, -1)], dim=1)
        if cond_state is not None:
            cs, cl = cond_state["surface"].squeeze(-3), cond_state["level"]
            if cs.shape[-2] % 2:
                cs, cl = cs[..., :-1, :], cl[..., :-1, :]
            surface, level = torch.cat([surface, cs], dim=1), torch.cat([level, cl], dim=1)
        surface = self.surface_proj(surface)
        level = self.level_proj(F.pad(level, (0, 0, 0, 0, *self.level_pads)))
        return torch.cat([surface.unsqueeze(2), level], dim=2)

    def decode(self, x: torch.Tensor) -> State:
        surface, level = x[:, :, 0], x[:, :, 1:]
        out_s = self.pixelshuffle(self.surface_deconv(surface)).unsqueeze(-3)
        level = level.reshape(level.shape[0], level.shape[1] // 2, 2, *level.shape[2:]).flatten(2, 3)[:, :, 1:]  # undo 2x level patching
        level = level.movedim(-3, 1).flatten(0, 1)
        out_l = self.pixelshuffle(self.level_deconv(level))
        out_l = out_l.reshape(-1, self.cfg.img_size[0], *out_l.shape[1:]).movedim(1, -3)
        return {"surface": torch.cat([out_s, out_s[..., -1:, :]], dim=-2), "level": torch.cat([out_l, out_l[..., -1:, :]], dim=-2)}  # fake south pole


class ArchesBackbone(nn.Module):
    """adaLN-conditioned 3-D U-Transformer: 2 + 6 + 6 + 2 (x depth_multiplier) earth-specific blocks with one down/up-sample and a skip."""

    def __init__(self, cfg: ArchesConfig):
        super().__init__()
        self.cfg = cfg
        zdim, lat, lon = cfg.tensor_size
        self.zdim, r1 = zdim, (lat, lon)
        r2 = (lat // 2, lon // 2)
        d, dm, ws = cfg.emb_dim, cfg.depth_multiplier, cfg.window_size
        dp = np.linspace(0, cfg.droppath_coeff / dm, 8 * dm).tolist()
        kw = dict(cond_dim=cfg.cond_dim, window_size=ws, mlp_ratio=cfg.mlp_ratio, axis_attn=cfg.axis_attn, swiglu=cfg.swiglu)
        self.interaction_layer = LinVert(d, zdim)
        self.layer1 = CondBasicLayer(d, (zdim, *r1), 2 * dm, cfg.num_heads[0], drop_path=dp[:2 * dm], **kw)
        self.downsample = DownSample(d, (zdim, *r1), (zdim, *r2))
        self.layer2 = CondBasicLayer(2 * d, (zdim, *r2), 6 * dm, cfg.num_heads[1], drop_path=dp[2 * dm:], **kw)
        self.layer3 = CondBasicLayer(2 * d, (zdim, *r2), 6 * dm, cfg.num_heads[2], drop_path=dp[2 * dm:], **kw)
        self.upsample = UpSample(2 * d, d, (zdim, *r2), (zdim, *r1))
        self.layer4 = CondBasicLayer(2 * d, (zdim, *r1), 2 * dm, cfg.num_heads[3], drop_path=dp[:2 * dm], **kw)
        self.r1 = r1

    def forward(self, x, cond_emb):
        B, C, Pl, Lat, Lon = x.shape
        x = self.interaction_layer(x.reshape(B, C, -1).transpose(1, 2))
        x = self.layer1(x, cond_emb)
        skip = x
        x = self.layer3(self.layer2(self.downsample(x), cond_emb), cond_emb)
        x = self.layer4(torch.cat([self.upsample(x), skip], dim=-1), cond_emb)
        return x.transpose(1, 2).reshape(B, -1, self.zdim, *self.r1)


class ArchesWeather(nn.Module):
    """Deterministic 24 h model: (state, state 24 h earlier, month, hour) -> next normalised state."""

    def __init__(self, cfg: ArchesConfig | None = None, constant_masks: torch.Tensor | None = None):
        super().__init__()
        self.cfg = cfg or ArchesConfig()
        self.backbone = ArchesBackbone(self.cfg)
        self.embedder = ArchesEmbedder(self.cfg, constant_masks)
        self.month_embedder = TimestepEmbedder(self.cfg.cond_dim)
        self.hour_embedder = TimestepEmbedder(self.cfg.cond_dim)

    def forward(self, state: State, prev_state: State, month: torch.Tensor, hour: torch.Tensor) -> State:
        cond = self.month_embedder(month) + self.hour_embedder(hour)
        out = self.embedder.decode(self.backbone(self.embedder.encode(state, prev_state), cond))
        if self.cfg.add_input_state:
            out = {k: out[k] + state[k] for k in out}
        return out


class _DetEnsemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.core = nn.ModuleList(models)

    def forward(self, state, prev_state, month, hour) -> State:
        outs = [m(state, prev_state, month, hour) for m in self.core]
        return {k: torch.stack([o[k] for o in outs], 0).mean(0) for k in ("surface", "level")}


def legacy_overflow_time_features(timestamp_seconds) -> tuple[torch.Tensor, torch.Tensor]:
    """(month, hour) fed to the *generative* network's embedders by the official checkpoint.

    The released ArchesWeatherGen was trained with ``pd.to_datetime(timestamp.to(int32).numpy() * 10**9)`` (config flag
    ``cond_times_backward_compatible``): the int32 product wraps around, so the calendar features are those of a timestamp within
    +-2.1 s of 1970-01-01 (month in {1, 12}, hour in {0, 23}) and carry essentially no information. Reproduced here because the weights
    expect it. The deterministic members use the true month / hour."""
    ts = np.asarray(timestamp_seconds).astype(np.int64).astype(np.int32).astype(np.int64)
    prod = ((ts * 10 ** 9 + 2 ** 31) % 2 ** 32) - 2 ** 31
    dt = prod.astype("datetime64[ns]")
    month = dt.astype("datetime64[M]").astype(np.int64) % 12 + 1
    hour = dt.astype("datetime64[h]").astype(np.int64) % 24
    return torch.as_tensor(month), torch.as_tensor(hour)


class ArchesWeatherGen(nn.Module):
    """Probabilistic model: flow-matching ("sample"-prediction) network conditioned on [mean of the deterministic members, state(t-24h)].

    ``det_model.core`` holds the deterministic members (4 in the release); the checkpoint contains them. Forecast residuals are modelled in
    the ``state_scaler``-normalised space: ``x(t+24h) = det + noise_free_sample / state_scaler``."""

    def __init__(self, gen_cfg: ArchesConfig | None = None, det_cfg: ArchesConfig | None = None, n_det: int = 4,
                 constant_masks: torch.Tensor | None = None, num_train_timesteps: int = 1000):
        super().__init__()
        self.cfg = gen_cfg or ArchesConfig(depth_multiplier=1, n_concatenated_states=3)
        self.backbone = ArchesBackbone(self.cfg)
        self.embedder = ArchesEmbedder(self.cfg, constant_masks)
        self.month_embedder = TimestepEmbedder(self.cfg.cond_dim)
        self.hour_embedder = TimestepEmbedder(self.cfg.cond_dim)
        self.timestep_embedder = TimestepEmbedder(self.cfg.cond_dim)
        # release order: archesweather-m-seed0, -seed1, -m-skip-seed0, -m-skip-seed1 (the two "skip" members add the input state)
        base = det_cfg or ArchesConfig(depth_multiplier=2, n_concatenated_states=1)
        dcfgs = [dataclasses.replace(base, add_input_state=(i >= n_det // 2)) for i in range(n_det)] if det_cfg is None or n_det > 1 else [base]
        self.det_model = _DetEnsemble([ArchesWeather(c, constant_masks) for c in dcfgs])
        self.num_train_timesteps = num_train_timesteps
        c = self.cfg
        self.register_buffer("state_scaler_surface", torch.ones(c.surface_ch, 1, 1, 1), persistent=False)
        self.register_buffer("state_scaler_level", torch.ones(c.level_ch, c.img_size[0], 1, 1), persistent=False)

    # --- network -------------------------------------------------------------------------------------------------
    def predict(self, state, noisy, timesteps, pred_state, prev_state, month, hour, is_sampling=False) -> State:
        """One call of the generative network. ``timesteps`` in [1, 1000]. With ``is_sampling`` the x0-prediction is converted to the
        flow-matching velocity ``(noisy - x0) / sigma`` as the official sampler does."""
        inp = cat_states(prev_state, noisy)
        inp = cat_states(pred_state, inp)
        gm, gh = legacy_overflow_time_features(self._timestamp) if getattr(self, "_timestamp", None) is not None else (month, hour)
        cond = self.month_embedder(gm.to(state["surface"].device)) + self.hour_embedder(gh.to(state["surface"].device)) \
            + self.timestep_embedder(timesteps)
        out = self.embedder.decode(self.backbone(self.embedder.encode(state, inp), cond))
        if is_sampling:
            sig = (timesteps / self.num_train_timesteps)[:, None, None, None, None]
            out = {k: (noisy[k] - out[k]) / sig for k in out}
        return out

    # --- sampler -------------------------------------------------------------------------------------------------
    def flow_timesteps(self, num_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        """FlowMatchEulerDiscrete (shift 1): timesteps linspace(1000, 1, N); sigmas = t/1000 plus a final 0."""
        t = torch.linspace(float(self.num_train_timesteps), 1.0, num_steps)
        return t, torch.cat([t / self.num_train_timesteps, torch.zeros(1)])

    @torch.no_grad()
    def sample(self, state: State, prev_state: State, month, hour, timestamp=None, num_steps: int = 25, seed: int | None = None,
               noise: State | None = None, scale_input_noise: float | None = 1.05, pred_state: State | None = None) -> State:
        """One 24 h ensemble member (normalised units). ``timestamp`` (unix seconds, int tensor) drives the legacy generative time features."""
        dev = state["surface"].device
        self._timestamp = timestamp
        try:
            if pred_state is None:
                pred_state = self.det_model(state, prev_state, month, hour)
            if noise is None:
                g = torch.Generator(device=dev)
                if seed is not None:
                    g.manual_seed(seed)
                noise = {k: torch.empty_like(state[k]).normal_(generator=g) for k in ("surface", "level")}
            x = {k: v * scale_input_noise if scale_input_noise is not None else v for k, v in noise.items()}
            ts, sigmas = self.flow_timesteps(num_steps)
            for i in range(num_steps):
                v = self.predict(state, x, ts[i:i + 1].to(dev), pred_state, prev_state, month, hour, is_sampling=True)
                x = {k: x[k] + (sigmas[i + 1] - sigmas[i]).to(dev) * v[k] for k in x}
        finally:
            self._timestamp = None
        scale = {"surface": self.state_scaler_surface, "level": self.state_scaler_level}
        return {k: pred_state[k] + x[k] / scale[k] for k in x}


def ArchesWeather_lite() -> ArchesWeather:
    """Tiny trainable config (24x48 grid, 5 levels, dim 48): same code path as the 192-dim model."""
    cfg = ArchesConfig(img_size=(5, 25, 48), emb_dim=48, cond_dim=32, num_heads=(2, 4, 4, 2), window_size=(1, 3, 4), depth_multiplier=1,
                       n_concatenated_states=1)
    return ArchesWeather(cfg)


def ArchesWeatherGen_lite() -> ArchesWeatherGen:
    g = ArchesConfig(img_size=(5, 25, 48), emb_dim=48, cond_dim=32, num_heads=(2, 4, 4, 2), window_size=(1, 3, 4), depth_multiplier=1,
                     n_concatenated_states=3)
    d = ArchesConfig(img_size=(5, 25, 48), emb_dim=48, cond_dim=32, num_heads=(2, 4, 4, 2), window_size=(1, 3, 4), depth_multiplier=1,
                     n_concatenated_states=1)
    return ArchesWeatherGen(g, d, n_det=2)
