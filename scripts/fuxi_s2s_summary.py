"""CPU fingerprint of one official-weights step on the official sample input -> docs/results/fuxi_s2s_cpu_summary.json."""
import json, sys
import torch
from weatherai.models.fuxi_s2s import make_input
from weatherai.models.fuxi_s2s.convert import load_official
from weatherai.models.fuxi_s2s.summary import fixed_noise, summarise

root = "/workspace/ckpt/fuxi_s2s"
m = load_official(f"{root}/model-1.0/fuxi_s2s.onnx")
x, init = make_input("/workspace/scratch/FuXi-S2S/data")
with torch.no_grad():
    out = m(torch.from_numpy(x)[None], torch.zeros(1), noise=fixed_noise(m))
res = {"init_dates": init, "seed": 0, "step": 0, "mean_std": summarise(out)}
json.dump(res, open("docs/results/fuxi_s2s_cpu_summary.json", "w"), indent=1)
print(res)
