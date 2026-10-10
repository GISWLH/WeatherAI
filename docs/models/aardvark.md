# Aardvark Weather — native PyTorch encoder + processor + decoder

[← Model zoo](../../README.md) · [All model pages](README.md) · **Paper:** *Nature* 2025, [doi:10.1038/s41586-025-08897-0](https://doi.org/10.1038/s41586-025-08897-0)

Aardvark Weather (*Nature* 2025; official code CC0) = observation **encoder** → **processor** → station **decoder**.
`weatherai.models.aardvark` now contains explicit ports of all three (parameter names identical to the official ones):

| file | contents |
|---|---|
| `setconv.py` | `SetConv` (Gaussian ConvDeepSet: off-grid→grid, grid→grid, grid→off-grid, density channel) |
| `aardvark.py` | `ViT` (per-variable patch embed + variable aggregation, or MLP + single embed), `AardvarkProcessor` (24 h step) |
| `unet.py` | cylindrical-padding conv / transposed conv, `Down`, `Up`, `Unet` |
| `system.py` | `AardvarkEncoder` (instrument set-convs + 277-channel stack + patch-3 ViT), `AardvarkDecoder` (U-Net → station set-conv → MLP), `AardvarkE2E`, official-checkpoint / sample loaders |

```python
import torch
from weatherai.models.aardvark import (AardvarkProcessor_lite, AardvarkE2E, load_official_encoder, load_official_decoder,
                                       load_official_processor, load_official_sample)

m = AardvarkProcessor_lite().eval()                    # 0.23 M params, random init, 60x31 grid; trainable
y = m(torch.randn(1, 35, 60, 31))                      # (1, 31, 60, 24): normalised 24 h tendency

# Official weights (HF dataset av555/aardvark-weather, trained_model/...) - all strict=True loads:
enc = load_official_encoder("trained_model/encoder/epoch_96")
dec = load_official_decoder("trained_model/decoder/tas/lt_1/epoch_18")
proc = load_official_processor("trained_model/processor/forecast_1/epoch_0")
task = load_official_sample("aardvark-weather-public/data/sample_data_final.pkl")     # official sample (CUDA pickle -> CPU)
station, forecast, init_state = AardvarkE2E(enc, [proc], dec).eval()(task, return_gridded=True)  # 24 h forecast; station = normalised tas
# every piece can be built with other sizes / depths and trained (see tests: tiny encoder+processor+decoder training step)
```

**Verified (CPU; official code run on the official sample as reference, `scripts/aardvark_system_reference.py`):**
encoder initial state max |Δ| = 0.0 (atol 1e-5, rtol 1e-4); station decoder (tas, lt_1) max |Δ| = 0; E2E (encoder → 1 processor → decoder)
station output max |Δ| = 2.9e-6, gridded forecast max relative error 8e-5 (values up to 1.2e5); processor on random input max |Δ| ≈ 2e-4
(unchanged earlier check). Unit tests: set-conv vs naive loop, NaN handling, cylindrical conv shapes, parameter names, a training step on a tiny
whole system. HF ZeroGPU: see [docs/model_status.md](../model_status.md).
**Not implemented / not verified:** the official data loaders and training scripts (they depend on multi-TB local memmaps; the repo itself says they
cannot be executed), the FiLM / attention options of the official U-Net (unused by the released weights), `e2e_finetuned` checkpoints (not loaded/tested),
decoder checkpoints other than `tas/lt_1`, processors `forecast_2..10`, multi-step E2E, other station variables (`ws`), skill vs. observations.
The official sample has one timestep only, so the comparison covers that single case.
