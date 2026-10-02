"""UniCM: native model vs the official ``src/models.py`` (random weights; no public checkpoint exists).

Official modules are imported unchanged from a clone of tsinghua-fib-lab/UniCM-Global-Climate-Modes;
only ``Trainer`` (training loop, needs pynvml/sklearn) is stubbed.  The official state_dict is loaded
into the native model with strict=True, then eval rollouts and teacher-forced passes (dropout off)
and gradients are compared.
"""
import json, sys, types, argparse
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="/workspace/scratch/UniCM-Global-Climate-Modes/src")
ap.add_argument("--out", default="docs/results/unicm_parity.json")
ap.add_argument("--lite", action="store_true")
ap.add_argument("--save-ref", default="")
a = ap.parse_args()
sys.path.insert(0, a.repo)
sys.modules["Trainer"] = types.SimpleNamespace(TrainLoop=None)
from models import UniCM as OfficialUniCM          # noqa: E402
from weatherai.models.unicm import UniCM, UniCMConfig   # noqa: E402

cfg = UniCMConfig.lite() if a.lite else UniCMConfig.official()
cfg.dropout = 0.0
ns = types.SimpleNamespace(
    d_size=cfg.d_size, device=torch.device("cpu"), input_channal=cfg.in_channels, patch_size=list(cfg.patch_size),
    emb_spatial_size=cfg.n_patches, nheads=cfg.nheads, dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout,
    num_encoder_layers=cfg.num_encoder_layers, num_decoder_layers=cfg.num_decoder_layers,
    val_relative=cfg.val_relative, t20d_mode=cfg.t20d_mode, mode_interaction=cfg.mode_interaction,
    his_len=cfg.his_len, pred_len=cfg.pred_len, autoregressive=cfg.autoregressive)
torch.manual_seed(0)
off = OfficialUniCM(ns).eval()
nat = UniCM(cfg).eval()
res = {"config": "lite" if a.lite else "official", "params": sum(p.numel() for p in nat.parameters())}
res["strict_load"] = str(nat.load_state_dict(off.state_dict(), strict=True))
res["params_official"] = sum(p.numel() for p in off.parameters())

B, (H, W) = 2, cfg.grid
T = cfg.his_len + cfg.pred_len
g = torch.Generator().manual_seed(1)
x = torch.randn(B, T, cfg.in_channels, H, W, generator=g)
xm = torch.randn(B, cfg.n_mode_series, T, generator=g)
months = (torch.arange(T)[None] + torch.tensor([[3], [7]])) % 12


def rel(u, v):
    return float((u - v).norm() / v.norm())


with torch.no_grad():
    o1, o2 = off(x, xm, months, train=False)
    n1, n2 = nat(x, xm, months, train=False)
res["eval_field_rel_l2"], res["eval_field_max_abs"] = rel(n1, o1), float((n1 - o1).abs().max())
res["eval_mode_rel_l2"], res["eval_mode_max_abs"] = rel(n2, o2), float((n2 - o2).abs().max())
res["eval_out_shapes"] = [list(o1.shape), list(o2.shape)]
res["eval_field_std"] = float(o1.std())
e1, e2 = o1, o2
with torch.no_grad():
    o1, o2 = off(x, xm, months, train=True, sv_ratio=0.0)
    n1, n2 = nat(x, xm, months, train=True, sv_ratio=0.0)
res["train_pass_field_rel_l2"], res["train_pass_mode_rel_l2"] = rel(n1, o1), rel(n2, o2)
if a.save_ref:
    import numpy as np
    np.savez_compressed(a.save_ref, x=x.numpy(), xm=xm.numpy(), months=months.numpy(), field=e1.numpy(), mode=e2.numpy(),
                        **{"sd/" + k: v.numpy() for k, v in off.state_dict().items()})
off.train(); nat.train()
sum(t.square().mean() for t in off(x, xm, months, train=True)).backward()
sum(t.square().mean() for t in nat(x, xm, months, train=True)).backward()
gnorm = float(torch.sqrt(sum(p.grad.square().sum() for p in off.parameters() if p.grad is not None)))
gs = []
npn = dict(nat.named_parameters())
for k, po in off.named_parameters():
    pn = npn[k]
    if po.grad is None:
        gs.append((k, pn.grad is None)); continue
    gs.append((k, float((pn.grad - po.grad).norm() / (po.grad.norm() + 1e-5 * gnorm))))
res["grad_note"] = "per-parameter ||dnative-dofficial|| / (||dofficial|| + 1e-5*global norm); key-projection biases have zero gradient analytically (softmax shift invariance)"
res["grad_worst"] = sorted([(v, k) for k, v in gs if isinstance(v, float)])[-5:]
res["grad_max_rel_l2"] = max(v for k, v in gs if isinstance(v, float))
res["params_without_grad_official"] = [k for k, v in gs if v is True]
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps(res, indent=1))
