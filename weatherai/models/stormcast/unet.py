from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .layers import GroupNorm, Linear, PositionalEmbedding, ResampleConv2d, UNetBlock


@dataclass
class SongUNetConfig:
    img_resolution: tuple = (512, 640)  # (H, W)
    in_channels: int = 127
    out_channels: int = 99
    model_channels: int = 128
    channel_mult: tuple = (1, 2, 2, 2, 2)
    channel_mult_emb: int = 4
    num_blocks: int = 4
    attn_resolutions: tuple = ()
    bottleneck_attention: bool = True
    dropout: float = 0.0
    embedding_type: str = "zero"  # "zero" (regression) | "positional" (diffusion)
    additive_pos_embed: bool = False
    resample_filter: tuple = (1, 1)


class SongUNet(nn.Module):
    """U-Net with residual blocks per level, bottleneck attention and optional additive positional embedding."""

    def __init__(self, cfg: SongUNetConfig):
        super().__init__()
        self.cfg = cfg
        H, W = cfg.img_resolution
        mc, emb_ch = cfg.model_channels, cfg.model_channels * cfg.channel_mult_emb
        bk = dict(emb_channels=emb_ch, num_heads=1, dropout=cfg.dropout, resample_filter=cfg.resample_filter)
        if cfg.additive_pos_embed:
            self.spatial_emb = nn.Parameter(torch.zeros(1, mc, H, W))
            nn.init.trunc_normal_(self.spatial_emb, std=0.02)
        if cfg.embedding_type == "positional":
            self.map_noise = PositionalEmbedding(mc, endpoint=True)
            self.map_layer0 = Linear(mc, emb_ch)
            self.map_layer1 = Linear(emb_ch, emb_ch)
        else:
            self.register_buffer("zero_emb", torch.zeros(1, emb_ch), persistent=False)
        self.enc = nn.ModuleDict()
        skips, cout = [], cfg.in_channels
        for level, mult in enumerate(cfg.channel_mult):
            res = H >> level
            if level == 0:
                cin, cout = cout, mc
                self.enc[f"{res}x{res}_conv"] = ResampleConv2d(cin, cout, 3)
                skips.append(cout)
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock(cout, cout, down=True, **bk)
                skips.append(cout)
            for idx in range(cfg.num_blocks):
                cin, cout = cout, mc * mult
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(cin, cout, attention=res in cfg.attn_resolutions, **bk)
                skips.append(cout)
        self.dec = nn.ModuleDict()
        for level, mult in reversed(list(enumerate(cfg.channel_mult))):
            res = H >> level
            if level == len(cfg.channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = UNetBlock(cout, cout, attention=cfg.bottleneck_attention, **bk)
                self.dec[f"{res}x{res}_in1"] = UNetBlock(cout, cout, **bk)
            else:
                self.dec[f"{res}x{res}_up"] = UNetBlock(cout, cout, up=True, **bk)
            for idx in range(cfg.num_blocks + 1):
                cin, cout = cout + skips.pop(), mc * mult
                attn = idx == cfg.num_blocks and res in cfg.attn_resolutions
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(cin, cout, attention=attn, **bk)
            if level == 0:
                self.dec[f"{res}x{res}_aux_norm"] = GroupNorm(cout, eps=1e-6)
                self.dec[f"{res}x{res}_aux_conv"] = ResampleConv2d(cout, cfg.out_channels, 3)
        self._mult = 2 ** (len(cfg.channel_mult) - 1)

    def embed(self, noise_labels: torch.Tensor) -> torch.Tensor:
        if self.cfg.embedding_type != "positional":
            return self.zero_emb.repeat(noise_labels.shape[0], 1)
        e = self.map_noise(noise_labels)
        e = e.reshape(e.shape[0], 2, -1)
        e = torch.cat([e[:, 1:], e[:, :1]], dim=1).reshape(e.shape[0], -1)  # official quirk: [cos|sin] -> [sin|cos]
        return F.silu(self.map_layer1(F.silu(self.map_layer0(e))))

    def forward(self, x: torch.Tensor, noise_labels: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4 and x.shape[-1] % self._mult == 0 and x.shape[-2] % self._mult == 0, \
            f"spatial dims must be multiples of {self._mult}, got {tuple(x.shape[-2:])}"
        emb = self.embed(noise_labels).to(x.dtype)
        skips = []
        for name, block in self.enc.items():
            if name.endswith("_conv"):
                x = block(x)
                if self.cfg.additive_pos_embed:
                    x = x + self.spatial_emb.to(x.dtype)
            else:
                x = block(x, emb)
            skips.append(x)
        tmp = None
        for name, block in self.dec.items():
            if name.endswith("aux_norm"):
                tmp = block(x)
            elif name.endswith("aux_conv"):
                return block(F.silu(tmp))
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        raise RuntimeError("decoder has no output head")


class StormCastUNet(nn.Module):
    """Deterministic regression net: SongUNet(x, noise_labels=0) (state-dict prefix model.)."""

    def __init__(self, cfg: SongUNetConfig):
        super().__init__()
        assert cfg.embedding_type == "zero"
        self.model = SongUNet(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, torch.zeros(x.shape[0], dtype=x.dtype, device=x.device))


class EDMPrecond(nn.Module):
    """EDM preconditioned denoiser D(x; sigma | condition) (state-dict prefix model.)."""

    def __init__(self, cfg: SongUNetConfig, sigma_data: float = 0.5):
        super().__init__()
        assert cfg.embedding_type == "positional"
        self.model = SongUNet(cfg)
        self.sigma_data = sigma_data

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        x = x.float()
        s = sigma.float().reshape(-1, 1, 1, 1)
        sd = self.sigma_data
        c_skip = sd ** 2 / (s ** 2 + sd ** 2)
        c_out = s * sd / (s ** 2 + sd ** 2).sqrt()
        c_in = 1 / (sd ** 2 + s ** 2).sqrt()
        c_noise = s.log() / 4
        arg = c_in * x
        if condition is not None:
            arg = torch.cat([arg, condition.to(arg.dtype)], dim=1)
        F_x = self.model(arg, c_noise.flatten())
        return c_skip * x + c_out * F_x.float()
