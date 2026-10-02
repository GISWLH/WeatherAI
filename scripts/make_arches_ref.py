"""CPU reference slices for the GPU smoke test (outputs of the native port, which is parity-checked vs geoarches at full size)."""
import sys

import numpy as np
import torch

from weatherai.models.arches import load_official_gen

ck = sys.argv[1] if len(sys.argv) > 1 else "/workspace/ckpt/arches"
m, st = load_official_gen(ck)
g = torch.Generator().manual_seed(0)


def rs():
    return {"surface": torch.randn(1, 4, 1, 121, 240, generator=g) * 0.7, "level": torch.randn(1, 6, 13, 121, 240, generator=g) * 0.7}


s, p, noise = rs(), rs(), rs()
month, hour, ts = torch.tensor([6]), torch.tensor([12]), torch.tensor([1591012800])
with torch.no_grad():
    det = m.det_model.core[2](s, p, month, hour)   # a "skip" member (add_input_state)
    avg = m.det_model(s, p, month, hour)
    out = m.sample(s, p, month, hour, timestamp=ts, num_steps=5, noise=noise, pred_state=avg)
np.savez_compressed("tests/models/arches/data/ref_cpu_fp32.npz", det_skip_surface=det["surface"][..., ::7, ::11].numpy(),
                    det_skip_level=det["level"][..., ::7, ::11].numpy(), sample5_surface=out["surface"][..., ::7, ::11].numpy(),
                    sample5_level=out["level"][..., ::7, ::11].numpy())
print("ok")
