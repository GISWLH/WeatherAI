"""Load the official FuXi-ENS ONNX initialisers into :class:`FuXiENS` (strict) without ever
materialising a second copy of the 10 GB of weights.

The Zenodo release ``model/fuxi_ens.onnx`` (108 MB graph) keeps 492 of its 498 tensors in the
external-data file ``model/fuxi_ens`` (9.9 GB, flat fp32 blob; offset/length stored per tensor).
Tensors are exposed as ``np.memmap`` views (copy-on-write) so the model parameters are backed by
the page cache and RAM usage stays low; pass ``materialize=True`` to copy into RAM / move to GPU.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import torch

from .model import FuXiENS, FuXiENSConfig

__all__ = ["read_onnx_initializers", "load_official", "ATTN_MASK_SUFFIX"]

ATTN_MASK_SUFFIX = "attn_mask"


def read_onnx_initializers(onnx_path: str, data_dir: Optional[str] = None, include_masks: bool = False
                           ) -> Dict[str, np.ndarray]:
    """Returns {initialiser name: np.ndarray (memmap for external tensors)}."""
    import onnx
    from onnx import numpy_helper

    data_dir = data_dir or os.path.dirname(os.path.abspath(onnx_path))
    model = onnx.load(onnx_path, load_external_data=False)
    out: Dict[str, np.ndarray] = {}
    mmaps: Dict[str, np.memmap] = {}
    for init in model.graph.initializer:
        if init.name.endswith(ATTN_MASK_SUFFIX) and not include_masks:
            continue
        if init.external_data:
            ext = {e.key: e.value for e in init.external_data}
            loc = ext["location"]
            if loc not in mmaps:
                mmaps[loc] = np.memmap(os.path.join(data_dir, loc), dtype=np.uint8, mode="c")
            off, ln = int(ext.get("offset", 0)), int(ext["length"])
            assert init.data_type == 1, "expected fp32 initialisers"
            arr = np.frombuffer(mmaps[loc][off:off + ln], dtype=np.float32).reshape(tuple(init.dims))
        else:
            arr = numpy_helper.to_array(init)
        out[init.name] = arr
    return out


def load_official(onnx_path: str, data_dir: Optional[str] = None, device="cpu", dtype=torch.float32,
                  materialize: bool = False) -> FuXiENS:
    """Build the official-size model and load every ONNX initialiser with ``strict=True``.

    The attention masks are *derived* (``make_shift_mask``) rather than stored; the tests assert
    they equal the ONNX ``attn_mask`` initialisers.
    """
    tensors = read_onnx_initializers(onnx_path, data_dir)
    with torch.device("meta"):
        model = FuXiENS(FuXiENSConfig.official())
    sd = {k: torch.from_numpy(v) for k, v in tensors.items()}
    if materialize or device != "cpu" or dtype != torch.float32:
        sd = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype) for k, v in sd.items()}
        if materialize:   # `.to` is a no-op for same-device/dtype memmap views -> force private RAM copies
            sd = {k: v.clone() for k, v in sd.items()}
    model.load_state_dict(sd, strict=True, assign=True)
    model.reset_derived_buffers()
    if device != "cpu":
        model.to(device)
    return model.eval()
