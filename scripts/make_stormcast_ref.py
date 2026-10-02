"""Generate the tiny CPU reference (strided slices) used by the GPU smoke test. Needs the official checkpoint (not committed)."""
import sys
import numpy as np
import torch
from weatherai.models.stormcast import load_official

ck = sys.argv[1] if len(sys.argv) > 1 else "/workspace/ckpt/stormcast"
m = load_official(ck, with_metadata=False)
g = torch.Generator().manual_seed(0)
x = torch.randn(1, 127, 512, 640, generator=g)
xn = 3 * torch.randn(1, 99, 512, 640, generator=g)
cond = torch.randn(1, 200, 512, 640, generator=g)
sl = (slice(None), slice(None, None, 4), slice(None, None, 31), slice(None, None, 37))
with torch.no_grad():
    yr = m.regression(x)
    yd = m.diffusion(xn, torch.tensor([5.0]), cond)
np.savez_compressed("tests/models/stormcast/data/ref_cpu_fp32.npz", reg=yr[sl].numpy(), edm_sigma5=yd[sl].numpy(),
                    reg_std=yr.std().item(), edm_std=yd.std().item())
print(yr[sl].shape, yr.std().item(), yd.std().item())
