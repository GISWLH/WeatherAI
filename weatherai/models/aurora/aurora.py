"""Thin tensor-in / tensor-out wrapper around Microsoft's official Aurora (PyTorch).

Aurora (Bodnar et al., *Nature* 641, 2025) is already a PyTorch model released under
the MIT license (https://github.com/microsoft/aurora). WeatherAI therefore does **not**
re-implement its Swin-3D / Perceiver backbone; this module only

* builds the official ``aurora.Aurora`` with a *lite* (tiny, random-init) configuration or
  the official ``AuroraSmallPretrained`` configuration (+ optional official checkpoint),
* exposes a plain-tensor ``forward`` so Aurora can be called like the other zoo models.

Install the upstream package first: ``pip install microsoft-aurora``.
All numerics are those of upstream ``aurora``; this wrapper adds no model code.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

SURF_VARS = ("2t", "10u", "10v", "msl")
STATIC_VARS = ("lsm", "z", "slt")
ATMOS_VARS = ("z", "u", "v", "t", "q")


def aurora_available() -> bool:
    try:
        import aurora  # noqa: F401

        return True
    except Exception:
        return False


def _import_aurora():
    try:
        import aurora
    except ImportError as e:  # pragma: no cover - exercised only without the dep
        raise ImportError(
            "weatherai.models.aurora needs the official Aurora package: "
            "`pip install microsoft-aurora` (https://github.com/microsoft/aurora)."
        ) from e
    return aurora


class AuroraWrapper(nn.Module):
    """Plain-tensor interface to an upstream ``aurora.Aurora`` instance.

    Shapes (B batch, T history = 2, L pressure levels, H x W lat/lon grid):

    * ``surf``   ``(B, T, 4, H, W)``  order ``2t, 10u, 10v, msl``
    * ``atmos``  ``(B, T, 5, L, H, W)`` order ``z, u, v, t, q``
    * ``static`` ``(3, H, W)`` order ``lsm, z, slt``

    ``forward`` returns ``(surf_pred (B, 4, H', W'), atmos_pred (B, 5, L, H', W'))`` for the
    next ``timestep`` (6 h). ``H', W'`` are H, W cropped to multiples of the patch size
    (upstream behaviour). Latitudes must decrease and longitudes in ``[0, 360)`` increase.
    """

    def __init__(self, core: nn.Module, atmos_levels: Sequence[float] = (100, 250, 500, 850)):
        super().__init__()
        self.core = core
        self.atmos_levels = tuple(atmos_levels)

    # -- helpers ---------------------------------------------------------------------
    def make_batch(
        self,
        surf: torch.Tensor,
        static: torch.Tensor,
        atmos: torch.Tensor,
        lat: Optional[torch.Tensor] = None,
        lon: Optional[torch.Tensor] = None,
        time: Optional[Sequence[datetime]] = None,
        atmos_levels: Optional[Sequence[float]] = None,
    ):
        aur = _import_aurora()
        B, _, _, H, W = surf.shape
        if lat is None:
            lat = torch.linspace(90, -90, H)
        if lon is None:
            lon = torch.linspace(0, 360, W + 1)[:-1]
        if time is None:
            time = (datetime(2020, 6, 1, 12, 0),) * B
        levels = tuple(atmos_levels) if atmos_levels is not None else self.atmos_levels
        if atmos.shape[3] != len(levels):
            raise ValueError(f"atmos has {atmos.shape[3]} levels but atmos_levels has {len(levels)}")
        return aur.Batch(
            surf_vars={k: surf[:, :, i] for i, k in enumerate(SURF_VARS)},
            static_vars={k: static[i] for i, k in enumerate(STATIC_VARS)},
            atmos_vars={k: atmos[:, :, i] for i, k in enumerate(ATMOS_VARS)},
            metadata=aur.Metadata(lat=lat, lon=lon, time=tuple(time), atmos_levels=levels),
        )

    def forward(
        self,
        surf: torch.Tensor,
        static: torch.Tensor,
        atmos: torch.Tensor,
        lat: Optional[torch.Tensor] = None,
        lon: Optional[torch.Tensor] = None,
        time: Optional[Sequence[datetime]] = None,
        atmos_levels: Optional[Sequence[float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = self.make_batch(surf, static, atmos, lat, lon, time, atmos_levels)
        pred = self.core(batch)
        s = torch.stack([pred.surf_vars[k][:, 0] for k in SURF_VARS], dim=1)
        a = torch.stack([pred.atmos_vars[k][:, 0] for k in ATMOS_VARS], dim=1)
        return s, a

    def load_official_checkpoint(self, path: Optional[str] = None, strict: bool = True) -> None:
        """Load an official checkpoint (HF ``microsoft/aurora``) into the wrapped model."""
        if path is None:
            self.core.load_checkpoint(strict=strict)
        else:
            self.core.load_checkpoint_local(path, strict=strict)


def Aurora_lite(
    embed_dim: int = 64,
    depth: int = 1,
    num_heads: int = 2,
    window_size: Tuple[int, int, int] = (2, 2, 2),
    patch_size: int = 4,
    latent_levels: int = 4,
    atmos_levels: Sequence[float] = (100, 250, 500, 850),
) -> AuroraWrapper:
    """Tiny random-init Aurora (~3 M params with defaults) for CPU smoke tests / prototyping.

    Same upstream architecture (3-D Perceiver encoder, 3-stage Swin-3D U-Net backbone,
    Perceiver decoder) with 1 block per stage and 64-d embeddings. No pretrained weights.
    """
    aur = _import_aurora()
    core = aur.Aurora(
        embed_dim=embed_dim,
        encoder_depths=(depth,) * 3,
        decoder_depths=(depth,) * 3,
        encoder_num_heads=(num_heads,) * 3,
        decoder_num_heads=(num_heads,) * 3,
        num_heads=num_heads,
        window_size=window_size,
        patch_size=patch_size,
        latent_levels=latent_levels,
        use_lora=False,
    )
    return AuroraWrapper(core, atmos_levels)


def Aurora_small(
    pretrained: bool = False,
    checkpoint_path: Optional[str] = None,
    atmos_levels: Sequence[float] = (100, 250, 500, 850),
) -> AuroraWrapper:
    """Upstream ``AuroraSmallPretrained`` architecture (~113 M params, embed_dim 256).

    ``pretrained=True`` downloads ``aurora-0.25-small-pretrained.ckpt`` (~450 MB) from the
    Hugging Face repo ``microsoft/aurora`` (or loads ``checkpoint_path``). Upstream labels
    this model "should only be used for debugging".
    """
    aur = _import_aurora()
    core = aur.AuroraSmallPretrained()
    w = AuroraWrapper(core, atmos_levels)
    if pretrained or checkpoint_path:
        w.load_official_checkpoint(checkpoint_path)
    return w
