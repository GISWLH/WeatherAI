# FuXi-ENS — native PyTorch (reverse-engineered from the official ONNX; numerically checked vs onnxruntime)

[← Model zoo](../../README.md) · [All model pages](README.md) · **Paper:** *Science Advances* 2025, [doi:10.1126/sciadv.adu2854](https://doi.org/10.1126/sciadv.adu2854)

FuXi-ENS (*Science Advances* 2025) is a latent-noise ensemble forecaster. The only official artefact is `fuxi_ens.onnx`
(Zenodo [10.5281/zenodo.15124541](https://zenodo.org/records/15124541), **CC-BY-NC-4.0 — non-commercial; weights are never committed**:
download with `python scripts/fetch_fuxi_ens.py --out /workspace/ckpt/fuxi_ens`, 9.9 GB). `weatherai/models/fuxi_ens/model.py` is an
explicit PyTorch re-implementation whose structure, constants and parameter names were read out of the 26,018-node graph; **the official
code has no PyTorch definition, so the ONNX graph itself is the oracle**.

```
input [B,2,78,721,1440] (2 steps x (5 vars x 13 levels + 13 surface))  --(x-mean)/std, NaN->0, accumulated channels 73-77 := 0-->  xn
 dist_p  : PatchEmbed (Conv 8x8 on [state;static const] -> LN) -> dropout "drop0" -> layers.0 (7 AdaLN blocks) -> dropout "drop1"
           -> layers.1 (7 blocks) -> final AdaLN (+patch-embed residual) -> ConvTranspose 8x8 heads: mean, logvar     (14 blocks)
 sample  = mean + exp(logvar/2) * eps                                  <- the ensemble noise ("RandomNormalLike")
 decoder.0: z = xn + sample (channels 73-77 := 0) -> PatchEmbed -> 6 layers x 7 AdaLN blocks -> final AdaLN (+residual)
           -> pl_head (65 ch) || sf_head (13 ch)  -> x*std+mean ; tp = exp(clip(.,0,7)) - 1                          (42 blocks)
output [B,2,78,721,1440] = concat(input[:, -1], new state)           (== the ONNX output; inference.py feeds it back as next input)
```

AdaLN block = `adaln = Linear(SiLU(cond))` -> `(shift, scale, gate)` x 2; `x += g1*Attn(LN(x)(1+s1)+b1)`, `x += g2*MLP(LN(x)(1+s2)+b2)`;
`cond = MLP(sin/cos(step)) + MLP(hour/24) + MLP(doy/365)`. Attention: 24 heads x 64, 18x18 windows over the 90x180 token grid,
odd blocks cyclic-shifted by 9 with the stored −100 mask, **1-D RoPE on the window-flattened index** (not 2-D), q and k scaled by d^-1/4;
MLP is a gated GELU (`a*gelu(b)`, `fc1` 1536→8192 no bias); LayerNorm is the *unbiased-variance* form (eps 1e-6). All of this was recovered
from the graph and then verified numerically (below). 2,474,861,958 parameters + buffers `mean/std [78,1,1]` and `const [6,721,1440]`
(parameter count equals the ONNX weight elements exactly; the 22 `attn_mask` initialisers are derived by `make_shift_mask` and
tested equal to the ONNX tensors).

```python
from weatherai.models.fuxi_ens import load_official, FuXiENSNoise, FuXiENS_lite, FuXiENSConfig
model = load_official("/workspace/ckpt/fuxi_ens/fuxi_ens.onnx")          # strict state-dict load, weights memory-mapped (low RAM)
g = torch.Generator().manual_seed(0)
noise = FuXiENSNoise.sample(model.cfg, 1, g)                             # 2 dropout masks + latent normal noise, all explicit
y = model(x, step, hour, doy, noise)                                     # x: [1,2,78,721,1440] physical units; step=t, hour=h/24, doy=min(365,doy)/365
y2 = model(x, step, hour, doy, generator=torch.Generator().manual_seed(1))  # another ensemble member
lite = FuXiENS_lite()                                                    # ~2.1 M params, same code, trainable (see tests)
```

**Random ops in the exported graph (handled):** the ONNX has no eval switch — `RandomUniformLike` (×2: dropout p=0.2 on the patch embedding
`drop0` and on the output of `dist_p.layers.0` `drop1`, keep-prob 0.8, scale 1/0.8) and `RandomNormalLike` (latent noise) are always active, so
two ONNX runs differ. For a deterministic comparison `scripts/fuxi_ens_fixed_noise_onnx.py` rewrites the 108 MB graph so these three tensors
are graph *inputs* (no op is changed), and both runtimes receive identical tensors (torch generator seed 1234). `FuXiENSNoise` is the
corresponding explicit input of the PyTorch model; `FuXiENS.forward` without `noise` draws all three from a `torch.Generator`.

**Verified (CPU, fp32; `scripts/fuxi_ens_ort_segments.py` + `scripts/fuxi_ens_verify.py`; official `input.nc` 2018-09-13 18Z/2018-09-14 00Z,
step=0, fixed noise):** one full forward (all 14 + 42 blocks), strict load of all 498 initialisers. Stage-by-stage vs onnxruntime 1.30 (max |Δ| / max |ref|):
normalised input 0 (exact), patch embed 5e-7, dist_p layer 0 1.8e-6, dist_p out 1.4e-5, mean/logvar 4.7e-7 / 1.2e-6, sample 1.2e-7, decoder input `z` 6e-8,
decoder layers 0–3 ≤ 1.5e-6, layer 4–5 1.7e-4–2.3e-4 (a few outlier tokens, rel-RMSE 1e-5), decoder out 8e-5. **Final output** (physical units, all 78 channels × 2 steps):
max |Δ| = 1.66 on |values| up to 2.06e5 (geopotential; max |Δ|/max|ref| = 8.1e-6); per-channel max |Δ| / channel-std ≤ 3.3e-3 (precipitation `tp`, whose
`exp(clip)-1` amplifies logit errors) and ≤ 1.8e-3 for every other channel; relative RMSE ≤ 1.1e-5 for every channel; NaN (SST land) positions identical; the pass-through
step is bit-identical. That is fp32-rounding level for a 2.5 B-parameter network, **not** bit-exactness.

**HF GPU (ZeroGPU Space, NVIDIA RTX PRO 6000 Blackwell MIG 2g.48gb, torch 2.13+cu130; `scripts/gpu_smoke.py fuxi_ens`): PASS.** The 9.9 GB weights were fetched *on the Space*
from Zenodo with `scripts/fetch_fuxi_ens.py` (parallel HTTP range requests + local inflate: ≈ 430–480 s versus ≈ 3 MB/s ≈ 55 min for a single stream; `/tmp` has ≈ 3 TB free) and strict-loaded into CPU RAM
(0.8 s, memory-mapped); then, inside the 120 s ZeroGPU window, moved to the GPU in **fp32 (no reduced precision was needed)**: 3 s, 11.5 GB resident, 16.0 GB peak, one forward 2.5 s.
Same official `input.nc` and the same fixed noise (torch CPU generator, seed 1234) as the CPU run; compared on the committed stride-12 slice of the onnxruntime output (all 78 channels × both steps):
max |Δ| = 0.078 (max |Δ|/channel-std = 5.9e-4, tolerance 1e-2·std); step-0 pass-through equal; a second member with different noise differs. TF32 disabled. The Space also passes the 10 synthetic unit tests (2 file-dependent tests skipped there). Only this single forward step was run on the GPU (no 15-day rollout, no bf16/fp16).
Caveat: the Space weights are preloaded in a background thread (`fetch_fuxi_ens` endpoint) because a redeploy wipes `/tmp` and downloading does not fit into one 120 s GPU call.
