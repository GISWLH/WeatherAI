"""Native PyTorch Aurora (Bodnar et al., *Nature* 641, 2025) with an explicit forward().

Architecture (all in this package, no dependency on the ``aurora`` pip package):

    encoder.Perceiver3DEncoder  patch-embed -> Perceiver level aggregation -> + pos/scale/time
    swin3d.Swin3DBackbone       3-stage Swin-3D U-Net (patch merging / splitting, AdaLN on lead time)
    decoder.Perceiver3DDecoder  Perceiver level de-aggregation -> per-variable linear unpatchify

Parameter names and tensor layouts follow microsoft/aurora (MIT), so the official checkpoints
(e.g. ``aurora-0.25-small-pretrained.ckpt``) load with ``strict=True``
(:meth:`Aurora.load_official_checkpoint`). The official package is only used as a *reference
oracle* in tests / parity checks (:mod:`weatherai.models.aurora.official`).

Not included (upstream features): LoRA fine-tuning adapters, stochastic (ensemble) mode,
level-conditioned patch embeddings, dynamic / atmos static variables, separate Perceiver
heads, modulation heads, activation-checkpoint helpers, roll-out utilities. Hence the
fine-tuned 1.3 B / HRES / air-pollution / wave variants are *not* loadable here; the
pretrained 0.25° variants (``aurora-0.25-pretrained``, ``-small-pretrained``) are.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .decoder import Perceiver3DDecoder
from .encoder import Perceiver3DEncoder
from .stats import ATMOS_LOC, ATMOS_SCALE, SURF_LOC, SURF_SCALE
from .swin3d import Swin3DBackbone

SURF_VARS = ("2t", "10u", "10v", "msl")
STATIC_VARS = ("lsm", "z", "slt")
ATMOS_VARS = ("z", "u", "v", "t", "q")


def _lvl(l: float) -> str:
    l = round(float(l), 3)
    return str(int(l) if l % 1 == 0 else l).replace(".", "_")


class Aurora(nn.Module):
    """Aurora with plain-tensor I/O. Defaults = the 1.3 B-parameter configuration.

    Shapes (B batch, T history = 2, L pressure levels, H x W lat/lon grid):

    * ``surf``   ``(B, T, 4, H, W)``  order ``2t, 10u, 10v, msl``
    * ``atmos``  ``(B, T, 5, L, H, W)`` order ``z, u, v, t, q``
    * ``static`` ``(3, H, W)`` order ``lsm, z, slt``

    ``forward`` returns ``(surf_pred (B, 4, H', W'), atmos_pred (B, 5, L, H', W'))`` for
    ``timestep`` ahead, in physical units. Inputs are normalised internally with the official
    per-variable statistics and predictions un-normalised. ``H'`` is ``H`` with one surplus
    latitude row cropped (needs ``H % patch_size in {0, 1}``; ``W % patch_size == 0``).
    Latitudes must decrease, longitudes in ``[0, 360)`` increase.
    """

    def __init__(
        self,
        *,
        window_size: Tuple[int, int, int] = (2, 6, 12),
        encoder_depths: Sequence[int] = (6, 10, 8),
        encoder_num_heads: Sequence[int] = (8, 16, 32),
        decoder_depths: Sequence[int] = (8, 10, 6),
        decoder_num_heads: Sequence[int] = (32, 16, 8),
        latent_levels: int = 4,
        patch_size: int = 4,
        embed_dim: int = 512,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        drop_rate: float = 0.0,
        enc_depth: int = 1,
        dec_depth: int = 1,
        dec_mlp_ratio: float = 2.0,
        perceiver_ln_eps: float = 1e-5,
        max_history_size: int = 2,
        timestep: timedelta = timedelta(hours=6),
        stabilise_level_agg: bool = False,
        use_updated_lead_time_embedding: bool = False,
        surf_stats: Optional[Dict[str, Tuple[float, float]]] = None,
        atmos_levels: Sequence[float] = (100, 250, 500, 850),
    ):
        super().__init__()
        self.patch_size = patch_size
        self.max_history_size = max_history_size
        self.timestep = timestep
        self.surf_stats = dict(surf_stats or {})
        self.atmos_levels = tuple(atmos_levels)

        self.encoder = Perceiver3DEncoder(
            SURF_VARS, STATIC_VARS, ATMOS_VARS, patch_size, latent_levels, embed_dim, num_heads,
            embed_dim // num_heads, drop_rate, enc_depth, mlp_ratio, max_history_size,
            perceiver_ln_eps, stabilise_level_agg, use_updated_lead_time_embedding,
        )
        self.backbone = Swin3DBackbone(
            embed_dim, encoder_depths, encoder_num_heads, decoder_depths, decoder_num_heads,
            window_size, mlp_ratio, True, drop_rate, drop_rate, drop_path, use_updated_lead_time_embedding,
        )
        # the backbone output has 2*embed_dim channels (skip concatenation)
        self.decoder = Perceiver3DDecoder(
            SURF_VARS, ATMOS_VARS, patch_size, embed_dim * 2, dec_depth, embed_dim * 2 // num_heads,
            num_heads, dec_mlp_ratio, 0.0, perceiver_ln_eps,
        )

    # -- normalisation (official statistics) -------------------------------------------------
    def _surf_stat(self, name: str):
        if name in self.surf_stats:
            return self.surf_stats[name]
        return SURF_LOC[name], SURF_SCALE[name]

    def _atmos_stats(self, name: str, levels: Sequence[float], like: torch.Tensor):
        try:
            loc = [ATMOS_LOC[f"{name}_{_lvl(l)}"] for l in levels]
            sc = [ATMOS_SCALE[f"{name}_{_lvl(l)}"] for l in levels]
        except KeyError as e:
            raise ValueError(f"no built-in normalisation statistics for {e}; supported levels: "
                             "50,100,150,200,250,300,400,500,600,700,850,925,1000 hPa") from None
        return (torch.tensor(loc, dtype=like.dtype, device=like.device)[:, None, None],
                torch.tensor(sc, dtype=like.dtype, device=like.device)[:, None, None])

    def normalise(self, surf, static, atmos, levels):
        s = torch.stack([(surf[:, :, i] - self._surf_stat(n)[0]) / self._surf_stat(n)[1]
                         for i, n in enumerate(SURF_VARS)], dim=2)
        st = torch.stack([(static[i] - self._surf_stat(n)[0]) / self._surf_stat(n)[1]
                          for i, n in enumerate(STATIC_VARS)], dim=0)
        a = []
        for i, n in enumerate(ATMOS_VARS):
            loc, sc = self._atmos_stats(n, levels, atmos)
            a.append((atmos[:, :, i] - loc) / sc)
        return s, st, torch.stack(a, dim=2)

    def unnormalise(self, surf, atmos, levels):
        s = torch.stack([surf[:, i] * self._surf_stat(n)[1] + self._surf_stat(n)[0]
                         for i, n in enumerate(SURF_VARS)], dim=1)
        a = []
        for i, n in enumerate(ATMOS_VARS):
            loc, sc = self._atmos_stats(n, levels, atmos)
            a.append(atmos[:, i] * sc + loc)
        return s, torch.stack(a, dim=1)

    # -- forward ------------------------------------------------------------------------------
    def forward(
        self,
        surf: torch.Tensor,
        static: torch.Tensor,
        atmos: torch.Tensor,
        lat: Optional[torch.Tensor] = None,
        lon: Optional[torch.Tensor] = None,
        time: Optional[Sequence[datetime]] = None,
        atmos_levels: Optional[Sequence[float]] = None,
        normalise: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        levels = tuple(atmos_levels) if atmos_levels is not None else self.atmos_levels
        B, T, _, H, W = surf.shape
        if atmos.shape[3] != len(levels):
            raise ValueError(f"atmos has {atmos.shape[3]} levels but atmos_levels has {len(levels)}")
        P = self.patch_size
        if W % P:
            raise ValueError("Width of the data must be a multiple of the patch size.")
        if H % P not in (0, 1):
            raise ValueError("There can at most be one latitude too many.")
        if lat is None:
            lat = torch.linspace(90, -90, H)
        if lon is None:
            lon = torch.linspace(0, 360, W + 1)[:-1]
        if time is None:
            time = (datetime(2020, 6, 1, 12, 0),) * B
        p = next(self.parameters())
        surf, static, atmos = surf.to(p.dtype), static.to(p.dtype), atmos.to(p.dtype)
        if normalise:
            surf, static, atmos = self.normalise(surf, static, atmos, levels)

        if H % P == 1:  # crop the surplus latitude row (upstream behaviour)
            surf, atmos, static, lat = surf[..., :-1, :], atmos[..., :-1, :], static[..., :-1, :], lat[:-1]
            H -= 1
        device = surf.device
        lat, lon = lat.to(device), lon.to(device)
        patch_res = (self.encoder.latent_levels, H // P, W // P)
        static = static[None, None].expand(B, T, -1, -1, -1)
        lead = torch.full((B,), self.timestep.total_seconds() / 3600, device=device, dtype=p.dtype)

        x = self.encoder(surf, static, atmos, lat, lon, levels, time, lead)
        x = self.backbone(x, lead, patch_res)
        s, a = self.decoder(x, lat, lon, levels, patch_res)
        if normalise:
            s, a = self.unnormalise(s.float(), a.float(), levels)
        return s, a

    # -- checkpoints --------------------------------------------------------------------------
    def load_official_checkpoint(self, path: str, strict: bool = True) -> None:
        """Load an official pretrained ``.ckpt`` (``microsoft/aurora`` on Hugging Face)."""
        d = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(adapt_official_state_dict(d, self.patch_size, self.max_history_size), strict=strict)


def adapt_official_state_dict(d: Dict[str, torch.Tensor], patch_size: int = 4, max_history_size: int = 2):
    """Convert the pretrained-checkpoint layout (``net.`` prefix, fused ``weight`` / ``surf_head``
    tensors) to this package's per-variable layout (same renaming as upstream ``compat.py``)."""
    d = {(k[4:] if k.startswith("net.") else k): v for k, v in d.items()}
    if "encoder.surf_token_embeds.weight" in d:
        w = d.pop("encoder.surf_token_embeds.weight")
        assert w.shape[1] == 4 + 3
        for i, n in enumerate(SURF_VARS + STATIC_VARS):
            d[f"encoder.surf_token_embeds.weights.{n}"] = w[:, [i]]
    if "encoder.atmos_token_embeds.weight" in d:
        w = d.pop("encoder.atmos_token_embeds.weight")
        assert w.shape[1] == 5
        for i, n in enumerate(ATMOS_VARS):
            d[f"encoder.atmos_token_embeds.weights.{n}"] = w[:, [i]]
    for head, names in (("surf", SURF_VARS), ("atmos", ATMOS_VARS)):
        if f"decoder.{head}_head.weight" in d:
            w, b = d.pop(f"decoder.{head}_head.weight"), d.pop(f"decoder.{head}_head.bias")
            V = len(names)
            assert w.shape[0] == V * patch_size**2
            w, b = w.reshape(patch_size**2, V, -1), b.reshape(patch_size**2, V)
            for i, n in enumerate(names):
                d[f"decoder.{head}_heads.{n}.weight"] = w[:, i]
                d[f"decoder.{head}_heads.{n}.bias"] = b[:, i]
    cur = d["encoder.surf_token_embeds.weights.2t"].shape[2]
    if max_history_size < cur:
        raise AssertionError(f"checkpoint history {cur} > model max_history_size {max_history_size}")
    if max_history_size > cur:  # zero-pad extra history slices (as upstream)
        for k in list(d):
            if k.startswith(("encoder.surf_token_embeds.weights.", "encoder.atmos_token_embeds.weights.")):
                w = d[k]
                nw = torch.zeros(w.shape[0], 1, max_history_size, *w.shape[3:], dtype=w.dtype)
                nw[:, :, :cur] = w
                d[k] = nw
    return d


def Aurora_lite(
    embed_dim: int = 64,
    depth: int = 1,
    num_heads: int = 2,
    window_size: Tuple[int, int, int] = (2, 2, 2),
    patch_size: int = 4,
    latent_levels: int = 4,
    atmos_levels: Sequence[float] = (100, 250, 500, 850),
) -> Aurora:
    """Tiny random-init native Aurora (~3 M params) for CPU smoke tests, training demos and
    architecture experiments: 1 Swin block per stage, 64-d embeddings. Every field of
    :class:`Aurora` can be overridden by constructing ``Aurora(...)`` directly."""
    return Aurora(
        embed_dim=embed_dim, encoder_depths=(depth,) * 3, decoder_depths=(depth,) * 3,
        encoder_num_heads=(num_heads,) * 3, decoder_num_heads=(num_heads,) * 3, num_heads=num_heads,
        window_size=window_size, patch_size=patch_size, latent_levels=latent_levels,
        atmos_levels=atmos_levels,
    )


def Aurora_small(
    pretrained: bool = False,
    checkpoint_path: Optional[str] = None,
    atmos_levels: Sequence[float] = (100, 250, 500, 850),
) -> Aurora:
    """Official ``AuroraSmallPretrained`` configuration (~113 M params, embed_dim 256).

    ``checkpoint_path`` loads a local official ckpt; ``pretrained=True`` downloads
    ``aurora-0.25-small-pretrained.ckpt`` (~450 MB) from HF ``microsoft/aurora``. Both are
    strict loads. Upstream labels this model "should only be used for debugging".
    """
    m = Aurora(
        encoder_depths=(2, 6, 2), encoder_num_heads=(4, 8, 16), decoder_depths=(2, 6, 2),
        decoder_num_heads=(16, 8, 4), embed_dim=256, num_heads=8, atmos_levels=atmos_levels,
    )
    if checkpoint_path is None and pretrained:
        from huggingface_hub import hf_hub_download

        checkpoint_path = hf_hub_download(
            "microsoft/aurora", "aurora-0.25-small-pretrained.ckpt",
            revision="0be7e57c685dac86b78c4a19a3ab149d13c6a3dd",
        )
    if checkpoint_path:
        m.load_official_checkpoint(checkpoint_path)
    return m
