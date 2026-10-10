# Aurora — native PyTorch implementation

[← Model zoo](../../README.md) · [All model pages](README.md) · **Paper:** *Nature* 2025, [doi:10.1038/s41586-025-09005-y](https://doi.org/10.1038/s41586-025-09005-y)

`weatherai.models.aurora` re-implements Aurora (*Nature* 2025; MIT,
[microsoft/aurora](https://github.com/microsoft/aurora)) as explicit `nn.Module`s with readable `forward()`s,
in the style of the Pangu/FuXi code (no dependency on the `aurora` pip package):

| file | contents |
|---|---|
| `patch_embed.py` | `LevelPatchEmbed` – per-variable 3-D patch conv over (history × P × P) |
| `perceiver.py` | `PerceiverAttention`, `PerceiverResampler` (post-norm cross-attention + MLP) |
| `encoder.py` | `Perceiver3DEncoder` – patch-embed surface(+static) and every level → level aggregation → + position / patch-scale / lead-time / absolute-time Fourier embeddings |
| `swin3d.py` | `Swin3DBackbone`: `SwinStage`s of `Swin3DBlock` (shifted 3-D windows with periodic-longitude mask, AdaLN on lead time), `PatchMerging3D`, `PatchSplitting3D` (linear1 → reshape/permute → crop padding → norm → linear2) |
| `decoder.py` | `Perceiver3DDecoder` – level de-aggregation + per-variable linear unpatchify |
| `aurora.py` | `Aurora` (plain-tensor I/O, official normalisation statistics, `Aurora_lite`, `Aurora_small`) |
| `official.py` | the old thin wrapper over `microsoft-aurora` – **reference oracle only** (tests / parity) |

```python
import torch
from weatherai.models import Aurora, Aurora_lite, Aurora_small

m = Aurora_lite().eval()                      # ~3 M params, random init, CPU friendly
surf   = torch.randn(1, 2, 4, 16, 32) + torch.tensor([278., 0, 0, 1e5]).view(1, 1, 4, 1, 1)  # (B,T=2,[2t,10u,10v,msl],H,W)
static = torch.randn(3, 16, 32)               # [lsm, z, slt]
atmos  = torch.randn(1, 2, 5, 4, 16, 32)      # (B,T,[z,u,v,t,q],L=4,H,W)
with torch.no_grad():
    surf_next, atmos_next = m(surf, static, atmos)   # physical units, +6 h

# fully configurable / trainable: any depths, heads, window, patch size, embed dim
my = Aurora(embed_dim=128, encoder_depths=(2, 4, 2), decoder_depths=(2, 4, 2),
            encoder_num_heads=(4, 8, 16), decoder_num_heads=(16, 8, 4), num_heads=4, window_size=(2, 6, 12))
loss = (my(surf, static, atmos)[0] - surf[:, 1]).pow(2).mean(); loss.backward()   # see tests for a full training step

# Official small pretrained checkpoint (~450 MB from HF microsoft/aurora), strict=True load into the native model
small = Aurora_small(pretrained=True).eval()  # or Aurora_small(checkpoint_path="aurora-0.25-small-pretrained.ckpt")
```

**Verified (numerically, vs. the official `microsoft-aurora` package at b628d7c, same weights, same inputs; fp32,
`atol=1e-5, rtol=1e-4` – observed max abs difference was exactly 0.0 on CPU):** random-weight state-dict transfer
`strict=True` for 3 configs (window (2,2,2)/(2,3,5)/(2,6,12), uneven depths, v3 lead-time embedding), grids
16×32 … 32×64 incl. the odd-latitude crop (H%4==1) and patch grids that need merge-padding / split-cropping
(25×44); the official `aurora-0.25-small-pretrained.ckpt` strict-loads into the native `Aurora_small` and
reproduces the official output on 33×64 inputs. HF ZeroGPU results: see [docs/model_status.md](../model_status.md).
Unit tests: `tests/models/aurora` (shapes, odd grids, merge/split + window helpers, param-name layout,
25-step training demo, grads, parity).
**Not verified / not included:** skill on real ERA5/HRES data (random inputs only), multi-step rollout;
upstream features that are *not* ported: LoRA, stochastic/ensemble mode, level-conditioned embeddings,
dynamic/atmos-static variables, separate/modulation heads (so the fine-tuned 1.3 B, HRES-0.1°, air-pollution,
wave and v1.5 checkpoints are **not** loadable; `aurora-0.25-pretrained` and `-small-pretrained` are); the 1.3 B
checkpoint was not run; built-in normalisation stats only for the 13 standard ERA5 levels.
