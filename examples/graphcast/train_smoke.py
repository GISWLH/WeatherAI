#!/usr/bin/env python3
"""GraphCast_lite synthetic Adam train smoke (CPU / GPU / HF Spaces).

Usage:
  python examples/graphcast/train_smoke.py
  python examples/graphcast/train_smoke.py --steps 3 --device cuda
  python examples/graphcast/train_smoke.py --in-channels 69 --cache /tmp/gc.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Optional, Tuple

import torch
from torch import nn


def synthetic_pair(
    in_channels: int = 4,
    img_size: Tuple[int, int] = (32, 64),
) -> Tuple[torch.Tensor, torch.Tensor]:
    h, w = img_size
    x = torch.randn(1, in_channels, h, w)
    y = x + 0.01 * torch.randn_like(x)
    return x, y


def run_train(
    steps: int = 2,
    device: str = "cpu",
    in_channels: int = 4,
    img_size: Tuple[int, int] = (32, 64),
    mesh_level: int = 1,
    processor_layers: int = 2,
    hidden_dim: int = 32,
    lr: float = 1e-3,
    cache_path: Optional[str] = None,
    load_cache_only: bool = False,
) -> int:
    try:
        from weatherai.models import GraphCast_lite
    except ImportError:
        # HF Space layout: package under WeatherAI/
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if root not in sys.path:
            sys.path.insert(0, root)
        from weatherai.models import GraphCast_lite

    device_t = torch.device(device)
    print(f"torch={torch.__version__}")
    print(f"device={device_t}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if device_t.type == "cuda" and torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")

    if load_cache_only:
        if not cache_path or not os.path.isfile(cache_path):
            print("STATUS=FAIL")
            print(f"missing_cache={cache_path}")
            return 1
        blob = torch.load(cache_path, map_location="cpu", weights_only=False)
        x, y = blob["x"], blob["y"]
        in_channels = int(x.shape[1])
        img_size = (int(x.shape[2]), int(x.shape[3]))
        print(f"loaded_cache={cache_path} shape={tuple(x.shape)}")
    else:
        x, y = synthetic_pair(in_channels, img_size)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            torch.save({"x": x, "y": y}, cache_path)
            print(f"wrote_cache={cache_path}")

    model = GraphCast_lite(
        img_size=img_size,
        in_channels=in_channels,
        mesh_level=mesh_level,
        processor_layers=processor_layers,
        hidden_dim=hidden_dim,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params:,}")
    print(
        f"mesh_level={model.mesh_level} mesh_nodes={model.num_mesh_nodes} "
        f"processor_layers={model.processor_layers} hidden_dim={model.hidden_dim}"
    )

    model = model.to(device_t)
    x = x.to(device_t)
    y = y.to(device_t)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    losses = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
        print(f"step={step} loss={losses[-1]:.6f} finite={torch.isfinite(pred).all().item()}")

    if device_t.type == "cuda" and torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"peak_cuda_mem_gb={peak:.3f}")

    ok = steps == 0 or (len(losses) == steps and all(np_isfinite(v) for v in losses))
    print("STATUS=OK" if ok else "STATUS=FAIL")
    return 0 if ok else 1


def np_isfinite(v: float) -> bool:
    return v == v and v not in (float("inf"), float("-inf"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--in-channels", type=int, default=4)
    p.add_argument("--img-h", type=int, default=32)
    p.add_argument("--img-w", type=int, default=64)
    p.add_argument("--mesh-level", type=int, default=1)
    p.add_argument("--processor-layers", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cache", default=None, help="Optional .pt cache path for x/y")
    p.add_argument("--load-cache-only", action="store_true")
    args = p.parse_args(argv)
    try:
        return run_train(
            steps=args.steps,
            device=args.device,
            in_channels=args.in_channels,
            img_size=(args.img_h, args.img_w),
            mesh_level=args.mesh_level,
            processor_layers=args.processor_layers,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            cache_path=args.cache,
            load_cache_only=args.load_cache_only,
        )
    except Exception:
        print("STATUS=FAIL")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
