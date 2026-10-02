"""Rollout parity of the native ACE2 port vs the official ``fme`` Stepper (run in an env with `pip install fme`)."""
import sys, warnings, json
warnings.filterwarnings("ignore")
import torch
from weatherai.models.ace2 import load_official
from weatherai.models.ace2.data import load_case

CK = "/workspace/ckpt/ace2"
n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
ic_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
torch.set_num_threads(4)
ic, forcing, times = load_case(CK, ic_idx, n)
m = load_official(CK)

from fme.ace.stepper import load_stepper
from fme.core.optimization import NullOptimization
st = load_stepper(CK + "/ace2_era5_ckpt.tar")
ic_d = {k: v[:, None] for k, v in ic.items()}
need = set(st._input_only_names) | set(st._step_obj.next_step_input_names)
f_d = {k: forcing[k] for k in need}
with torch.no_grad():
    ref = [r.output for r in st.predict_generator(ic_d, f_d, n, NullOptimization(), None)]
    mine = m.rollout(ic, forcing, n)
rep = []
for s, (a, b) in enumerate(zip(ref, mine)):
    worst = 0.0; worst_n = None; rel = {}
    for k in m.cfg.out_names:
        d = (a[k] - b[k]).abs().max().item(); sc = a[k].abs().max().item() + 1e-30
        rel[k] = d / sc
        if rel[k] > worst: worst, worst_n = rel[k], k
    rep.append({"step": s + 1, "max_rel_err": worst, "worst_var": worst_n,
                "ps_abs_diff": (a["PRESsfc"] - b["PRESsfc"]).abs().max().item(),
                "t2m_abs_diff": (a["TMP2m"] - b["TMP2m"]).abs().max().item()})
    print(rep[-1], flush=True)
json.dump(rep, open("/tmp/ace2_parity.json", "w"))
