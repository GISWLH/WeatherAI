"""UniCM: blocked for forecasting -- no trained weights are published (see ``cards.UNICM``).

``UniCMForecaster.from_checkpoint()`` refuses without a checkpoint path. A user-trained ``model_best.pkl`` can be loaded
(strict) for research, but the official input pipeline (``LoadData.py`` normalisation, region masks) is not ported, so
``run`` still refuses: outputs of random or unvalidated weights must never enter a forecast product.

Retraining is a separate project: CMIP6 (official: CESM2-FV2) + ORAS5/ERA5/GODAS/SODA, 20 seeds, explicit
train / validation / test years with no overlap, then an independent hindcast before any report use.
"""
from __future__ import annotations

import os
from typing import Optional

from .cards import UNICM as CARD
from .errors import BackendUnavailableError, WeightsUnavailableError


class UniCMForecaster:
    card = CARD

    def __init__(self, model, path: str):
        self.model, self.path = model, path

    @classmethod
    def from_checkpoint(cls, path: Optional[str] = None) -> "UniCMForecaster":
        if not path or not os.path.exists(path):
            raise WeightsUnavailableError("UniCM: no published checkpoint (" + CARD.status["weights_obtainable"].detail + "). "
                                          "Random-weight models are not forecasts.")
        from ..models.unicm import load_official
        return cls(load_official(path), path)

    def run(self, *a, **k):
        raise BackendUnavailableError("UniCM: the official input pipeline (LoadData.py normalisation and region masks) is not ported; "
                                      "a user-trained checkpoint can be inspected via .model but is not run as a forecast")
