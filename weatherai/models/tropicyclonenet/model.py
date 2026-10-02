"""TropiCycloneNet TCN_M (Huang et al., Nat. Commun. 16, 5923, 2025): multimodal tropical-cyclone track + intensity forecaster.

Native re-implementation of the official ``TrajectoryGenerator`` (``TCNM/models_prior_unet.py``) for inference/fine-tuning:
  * ``Unet3D`` rolls the 8 observed 500 hPa geopotential patches (64x64) forward to 12 frames;
  * ``EnvNet`` (x2: one is only used by the *chooser*) embeds the tabular environment data (wind, intensity class, velocity, month,
    lon/lat one-hots, past 12/24 h direction, 24 h intensity change) + pooled geopotential, and a 2-layer Transformer summarises the 8 steps;
  * two LSTM encoders (6 h steps; inputs lon, lat, pressure, wind increments + image embedding), six LSTM decoders ("generators",
    one per ``net_chooser`` class) that roll out 4 steps (6..24 h) of (dlon, dlat, dpressure, dwind) increments;
  * ``net_chooser`` gives class logits; ``sample`` draws a generator per member and per storm (multimodal ensemble).
Tracks are in the dataset's normalised units: lon = (lon-180)/50*10... see ``units``; best-of-N evaluation as in the official script.
Everything is device-agnostic (the official code calls ``.cuda()`` internally).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .unet3d import Unet3D

ENV_KEYS = ["wind", "intensity_class", "move_velocity", "month", "location_long", "location_lat",
            "history_direction12", "history_direction24", "history_inte_change24"]
ENV_DIMS = [1, 6, 1, 12, 36, 12, 8, 8, 4]
# month -> generator class used to organise the training data (informational; the model learns its own chooser)
MONTH_GROUP = {"01": 0, "02": 0, "03": 0, "04": 0, "05": 0, "06": 1, "07": 2, "08": 2, "09": 2, "10": 1, "11": 0, "12": 0}


@dataclass
class TCNMConfig:
    obs_len: int = 8
    pred_len: int = 4
    embedding_dim: int = 32
    encoder_h_dim: int = 64
    decoder_h_dim: int = 64
    mlp_dim: int = 128
    noise_dim: int = 16
    num_gs: int = 6
    img_size: int = 64


class EnvNet(nn.Module):
    def __init__(self, obs_len=8):
        super().__init__()
        self.data_embed = nn.ModuleDict({k: nn.Linear(d, 16) for k, d in zip(ENV_KEYS, ENV_DIMS)})
        self.GPH_embed = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1), nn.BatchNorm2d(1), nn.LeakyReLU(), nn.AvgPool2d(8, 8))
        f = len(ENV_KEYS) * 16 + 8 * 8
        self.evn_extract = nn.Sequential(nn.Linear(f, f // 2), nn.ReLU(), nn.Linear(f // 2, f // 2), nn.ReLU(), nn.Linear(f // 2, 64))
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=64, nhead=4), num_layers=2)

    def forward(self, env, gph):
        """env: dict of (B, T, d); gph: (B, 1, T, H, W) normalised geopotential -> (B, 64) summary of the last time step."""
        gph = gph.permute(0, 2, 1, 3, 4)
        B, T = gph.shape[:2]
        feats = [self.data_embed[k](env[k]) for k in ENV_KEYS]
        feats.append(torch.stack([self.GPH_embed(gph[:, t]).reshape(B, -1) for t in range(T)], 1))
        x = self.evn_extract(torch.cat(feats, 2)).permute(1, 0, 2)          # (T, B, 64)
        return self.encoder(x)[-1]


class TrackEncoder(nn.Module):
    def __init__(self, emb, h):
        super().__init__()
        self.encoder = nn.LSTM(emb, h, 1)
        self.spatial_embedding = nn.Linear(4, emb)
        self.time_embedding = nn.Linear(4, emb)          # unused by the forward pass, present in the checkpoint

    def forward(self, traj_rel, img_embed):             # (T,B,4), (T,B,emb)
        x = self.spatial_embedding(traj_rel) + img_embed
        return self.encoder(x)[1][0]                    # final h: (1, B, h)


class Generator(nn.Module):
    """One LSTM decoder: rolls out ``pred_len`` (dlon, dlat, dp, dwind) increments, feeding each prediction back with the predicted image embedding."""

    def __init__(self, emb, h, pred_len):
        super().__init__()
        self.pred_len = pred_len
        self.decoder = nn.LSTM(emb, h, 1)
        self.spatial_embedding = nn.Linear(4, emb)
        self.time_embedding = nn.Linear(4, emb)          # unused, present in the checkpoint
        self.hidden2pos = nn.Linear(h, 4)

    def forward(self, last_rel, state, decoder_img, last_img):
        """last_rel: (B,4) last observed increment; state=(h,c) each (1,B,h); decoder_img: (pred_len,B,emb); last_img: (B,emb)."""
        x = (self.spatial_embedding(last_rel) + last_img).unsqueeze(0)
        outs = []
        for i in range(self.pred_len):
            out, state = self.decoder(x, state)
            rel = self.hidden2pos(out[0])
            outs.append(rel)
            x = (self.spatial_embedding(rel) + decoder_img[i]).unsqueeze(0)
        return torch.stack(outs, 0)                      # (pred_len, B, 4)


class TCNM(nn.Module):
    def __init__(self, cfg: TCNMConfig | None = None):
        super().__init__()
        self.cfg = c = cfg or TCNMConfig()
        self.Unet = Unet3D(1, 1)
        npix = c.img_size * c.img_size
        self.img_embedding = nn.Linear(npix, c.embedding_dim)
        self.img_embedding_real = nn.Linear(npix, c.embedding_dim)
        self.env_net = EnvNet(c.obs_len)                 # present in the checkpoint, unused at inference
        self.env_net_chooser = EnvNet(c.obs_len)
        self.feature2dech_env = nn.Linear(2 * c.encoder_h_dim, c.encoder_h_dim)
        self.feature2dech = nn.Linear(2 * c.encoder_h_dim, c.encoder_h_dim)
        self.encoder = TrackEncoder(c.embedding_dim, c.encoder_h_dim)
        self.encoder_env = TrackEncoder(c.embedding_dim, c.encoder_h_dim)
        self.gs = nn.ModuleList([Generator(c.embedding_dim, c.decoder_h_dim, c.pred_len) for _ in range(c.num_gs)])
        self.net_chooser = nn.Sequential(nn.Linear(c.encoder_h_dim, c.encoder_h_dim // 2), nn.ReLU(),
                                         nn.Linear(c.encoder_h_dim // 2, c.encoder_h_dim // 2), nn.ReLU(),
                                         nn.Linear(c.encoder_h_dim // 2, c.num_gs))
        # context MLP (no batch norm in the released checkpoint): h -> mlp_dim -> decoder_h - noise_dim
        self.mlp_decoder_context = nn.Sequential(nn.Linear(c.encoder_h_dim, c.mlp_dim), nn.ReLU(),
                                                 nn.Linear(c.mlp_dim, c.decoder_h_dim - c.noise_dim), nn.ReLU())

    def prepare(self, obs_traj_rel, image_obs, env):
        """Everything deterministic: returns (chooser logits (B,6), decoder initial hidden (1,B,h), img_embed (T+P,B,emb))."""
        B = obs_traj_rel.shape[1]
        T = self.cfg.obs_len
        chooser_feat = self.env_net_chooser(env, image_obs)
        real = self.img_embedding_real(image_obs.reshape(B, T, -1)).permute(1, 0, 2)
        h_env = self.encoder_env(obs_traj_rel, real)
        dec_h_env = self.feature2dech_env(torch.cat([h_env.squeeze(0), chooser_feat], 1))
        logits = self.net_chooser(dec_h_env)
        pred = self.Unet(image_obs)
        all_img = torch.cat([image_obs[:, :, :1], pred], 2)
        img_embed = self.img_embedding(all_img.reshape(B, T + self.cfg.pred_len, -1)).permute(1, 0, 2)
        h = self.encoder(obs_traj_rel, img_embed[:T])
        dec_h = self.feature2dech(torch.cat([h.squeeze(0), chooser_feat], 1))
        return logits, dec_h, img_embed, all_img

    def _decode(self, g, dec_h, noise, last_rel, img_embed):
        T = self.cfg.obs_len
        ctx = self.mlp_decoder_context(dec_h)
        h0 = torch.cat([ctx, noise], 1).unsqueeze(0)
        state = (h0, torch.zeros_like(h0))
        return self.gs[g](last_rel, state, img_embed[T:], img_embed[T - 1])

    def all_generators(self, obs_traj_rel, image_obs, env, noise):
        """Deterministic given ``noise`` (B, noise_dim): the six generators' increments (pred_len, 6, B, 4) and the chooser logits."""
        logits, dec_h, img_embed, _ = self.prepare(obs_traj_rel, image_obs, env)
        outs = [self._decode(g, dec_h, noise, obs_traj_rel[-1], img_embed) for g in range(self.cfg.num_gs)]
        return torch.stack(outs, 1), logits

    def sample(self, obs_traj_rel, image_obs, env, num_samples=6, generator: torch.Generator | None = None):
        """Multimodal ensemble as in the official script: per member draw a generator class per storm from softmax(logits) and fresh N(0,1) noise.
        Returns increments (pred_len, num_samples, B, 4) and the chosen classes (B, num_samples).  (The official sampling loop iterates
        ``range(n_unique)`` instead of the unique class ids, which leaves members unfilled when a batch does not use classes 0..k-1; this
        port decodes every class that was drawn.)"""
        logits, dec_h, img_embed, _ = self.prepare(obs_traj_rel, image_obs, env)
        B = obs_traj_rel.shape[1]
        classes = torch.distributions.Categorical(logits=logits).sample((num_samples,), ).transpose(0, 1) if generator is None else \
            torch.multinomial(logits.softmax(-1), num_samples, replacement=True, generator=generator)
        out = torch.zeros(self.cfg.pred_len, num_samples, B, 4, device=logits.device)
        for s in range(num_samples):
            for g in classes[:, s].unique().tolist():
                idx = (classes[:, s] == g).nonzero(as_tuple=True)[0]
                noise = torch.randn(len(idx), self.cfg.noise_dim, device=logits.device, generator=generator)
                out[:, s, idx] = self._decode(g, dec_h[idx], noise, obs_traj_rel[-1, idx], img_embed[:, idx])
        return out, classes


def relative_to_abs(rel, start):
    """rel (pred_len, S, B, d) increments -> absolute (pred_len, S, B, d) from start (B, d)."""
    return rel.cumsum(0) + start[None, None]


def to_physical(traj, me):
    """Dataset units -> physical: traj (...,2)=(lon, lat) -> (lon in 0.1 deg, lat in 0.1 deg); me (...,2)=(pressure hPa, wind m/s) (official ``toNE``)."""
    lon = traj[..., 0] / 10 * 500 + 1800
    lat = traj[..., 1] / 6 * 300
    return torch.stack([lon, lat], -1), torch.stack([me[..., 0] * 50 + 960, me[..., 1] * 25 + 40], -1)


def track_error_km(pred_ll, gt_ll):
    """Official ``trajectory_displacement_error``: lon/lat in 0.1 deg, 111 km per degree, cos(lat of ground truth)."""
    d = pred_ll - gt_ll
    dx = d[..., 0] / 10 * 111 * (gt_ll[..., 1] / 10 * torch.pi / 180).cos()
    dy = d[..., 1] / 10 * 111
    return (dx ** 2 + dy ** 2).sqrt()
