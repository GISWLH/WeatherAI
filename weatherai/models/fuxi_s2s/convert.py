"""Load the official FuXi-S2S ONNX initialisers (fp16, external-data file) into :class:`FuXiS2S`, strict.

``model-1.0/fuxi_s2s.onnx`` (1.6 MB graph) + ``model-1.0/fuxi_s2s`` (2.08 GB flat fp16 blob).  Weights are
licensed CC-BY-NC-ND-4.0: download them yourself (:func:`download`), they are never redistributed here.
Tensors are memory-mapped; ``load_official`` upcasts to ``dtype`` (fp32 by default).
"""
from __future__ import annotations

import os
import tarfile
import urllib.request
from typing import Dict, Optional

import numpy as np
import torch

from .model import FREQS, FuXiS2S, FuXiS2SConfig, make_relative_tables, make_shift_mask

__all__ = ["download", "read_onnx_initializers", "load_official", "ZENODO_URL"]

ZENODO_URL = "https://zenodo.org/records/15718402/files/{}"


def download(root: str, with_data: bool = False) -> str:
    """Fetch ``model-1.0.tar`` (2.08 GB) from Zenodo 15718402 into ``root``; returns the ``fuxi_s2s.onnx`` path."""
    onnx_path = os.path.join(root, "model-1.0", "fuxi_s2s.onnx")
    if not os.path.exists(onnx_path):
        os.makedirs(root, exist_ok=True)
        tar = os.path.join(root, "model-1.0.tar")
        urllib.request.urlretrieve(ZENODO_URL.format("model-1.0.tar"), tar)
        with tarfile.open(tar) as t:
            t.extractall(root)
        os.remove(tar)
    if with_data and not os.path.exists(os.path.join(root, "data", "input.nc")):
        import zipfile
        z = os.path.join(root, "data.zip")
        urllib.request.urlretrieve(ZENODO_URL.format("data.zip"), z)
        zipfile.ZipFile(z).extractall(root)
    return onnx_path


def read_onnx_initializers(onnx_path: str):
    """({name: ndarray (memmap for external tensors)}, {alias name: source name} from Identity nodes)."""
    import onnx
    from onnx import numpy_helper
    d = os.path.dirname(os.path.abspath(onnx_path))
    m = onnx.load(onnx_path, load_external_data=False)
    out, mm = {}, {}
    for init in m.graph.initializer:
        if init.external_data:
            ext = {e.key: e.value for e in init.external_data}
            loc = ext["location"]
            if loc not in mm:
                mm[loc] = np.memmap(os.path.join(d, loc), dtype=np.uint8, mode="r")
            off, ln = int(ext.get("offset", 0)), int(ext["length"])
            dt = {10: np.float16, 1: np.float32, 7: np.int64}[init.data_type]
            out[init.name] = np.frombuffer(mm[loc][off:off + ln], dtype=dt).reshape(tuple(init.dims))
        else:
            out[init.name] = numpy_helper.to_array(init)
    alias = {n.output[0]: n.input[0] for n in m.graph.node if n.op_type == "Identity" and n.input[0] in out}
    return out, alias


def load_official(onnx_path: str, device="cpu", dtype=torch.float32, check_derived: bool = True) -> FuXiS2S:
    """Official-size model with every ONNX tensor loaded (``strict=True``); derived tables are verified."""
    t, alias = read_onnx_initializers(onnx_path)
    for k, v in alias.items():
        t.setdefault(k, t[v])
    with torch.device("meta"):
        model = FuXiS2S(FuXiS2SConfig.official())
    sd = {}
    for k in model.state_dict():
        if k not in t:
            raise KeyError(f"{k} not in ONNX initialisers")
        sd[k] = torch.from_numpy(np.array(t[k])).to(device=device, dtype=dtype)
    model.load_state_dict(sd, strict=True, assign=True)
    # materialise the non-persistent buffers (created on the meta device)
    cfg = model.cfg
    for m in model.modules():
        if hasattr(m, "relative_coords_table"):
            tab, idx = make_relative_tables(cfg.window)
            m.relative_coords_table = tab.to(device=device, dtype=dtype)
            m.relative_position_index = idx.to(device)
        if hasattr(m, "attn_mask"):
            m.attn_mask = make_shift_mask(cfg.grid, cfg.window, cfg.shift).to(device)
        if hasattr(m, "freqs"):
            m.freqs = torch.tensor(FREQS, device=device)
    if check_derived:
        k0 = "decoder.0.layers.0.blocks.0.attn.k_bias"
        assert not np.any(t[k0]), "k_bias expected to be all zeros"
        tab, idx = make_relative_tables(cfg.window)
        assert np.allclose(t["decoder.0.layers.0.blocks.0.attn.relative_coords_table"].astype(np.float32), tab.numpy(), atol=2e-3)
        assert np.array_equal(t["decoder.0.layers.0.blocks.0.attn.relative_position_index"], idx.numpy())
        assert np.allclose(t["decoder.0.layers.0.blocks.1.attn_mask"].astype(np.float32), make_shift_mask(cfg.grid, cfg.window, cfg.shift).numpy())
        assert not np.any(t["dist_p.cov.bias"])
    return model.eval()
