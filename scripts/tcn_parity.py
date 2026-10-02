"""TropiCycloneNet parity: native `weatherai.models.tropicyclonenet` vs the official modules (/workspace/scratch/TropiCycloneNet) on real NI test samples
built by the *official* dataset class (needs opencv, installed in /workspace/venv/tcn_extra) from the layout made by scripts/tcn_prepare_ni.py.
The official generator calls `.cuda()` internally; this script makes `.cuda()` a no-op so it runs on CPU (official code untouched)."""
import json, os, sys
import numpy as np
import torch

OFF = "/workspace/scratch/TropiCycloneNet"
ROOT = "/workspace/ckpt/tcn/ni_root"
CK = "/workspace/ckpt/tcn/checkpoint_with_model_16000.pt"
N = int(os.environ.get("N", "24"))
sys.path[:0] = ["/workspace/venv/tcn_extra", OFF]
torch.Tensor.cuda = lambda self, *a, **k: self
torch.nn.Module.cuda = lambda self, *a, **k: self

from TCNM.data.trajectoriesWithMe_unet import TrajectoryDataset, seq_collate        # official
from TCNM.models_prior_unet import TrajectoryGenerator                                # official
from weatherai.models.tropicyclonenet import load_official, make_sample, sequences
from weatherai.models.tropicyclonenet.data import rel
from weatherai.models.tropicyclonenet.model import ENV_KEYS

ck = torch.load(CK, map_location="cpu", weights_only=False)
a = ck["args"]
off = TrajectoryGenerator(obs_len=a["obs_len"], pred_len=a["pred_len"], embedding_dim=a["embedding_dim"], encoder_h_dim=a["encoder_h_dim_g"],
                          decoder_h_dim=a["decoder_h_dim_g"], mlp_dim=a["mlp_dim"], num_layers=a["num_layers"], noise_dim=a["noise_dim"],
                          noise_type=a["noise_type"], noise_mix_type=a["noise_mix_type"], pooling_type=a["pooling_type"],
                          pool_every_timestep=a["pool_every_timestep"], dropout=a["dropout"], bottleneck_dim=a["bottleneck_dim"],
                          neighborhood_size=a["neighborhood_size"], grid_size=a["grid_size"], batch_norm=a["batch_norm"])
off.load_state_dict(ck["g_state"])
off.eval()
nat = load_official(CK)

ds = TrajectoryDataset({"root": ROOT, "type": "test"}, obs_len=8, pred_len=4, skip=1, delim="\t", areas=["NI"])
print("official dataset sequences:", len(ds), flush=True)
idx = np.linspace(0, len(ds) - 1, N).astype(int)
res = {"n_samples": int(N), "dataset_len": len(ds)}

# (a) data pipeline: my loader vs the official dataset item
items = [ds[i] for i in idx]
batch = seq_collate(items)
(obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, _, _, seq_se, obs_Me, pred_Me, obs_rel_Me, pred_rel_Me, _, _, image_obs, image_pre, env, tyID) = batch
obs_traj = torch.cat([obs_traj, obs_Me], 2)
obs_traj_rel = torch.cat([obs_traj_rel, obs_rel_Me], 2)
gt = torch.cat([pred_traj, pred_Me], 2)
dmax = {"image": 0.0, "traj": 0.0, "rel": 0.0, "env": 0.0, "gt": 0.0}
ours = []
for j, i in enumerate(idx):
    info = ds.tyID[ds.seq_start_end[i][0]]
    tyname = info["old"][0]
    f = f"{ROOT}/BST_data/NI/test/{tyname}.txt"
    seqs = [s for s in sequences(f) if s["dates"] == info["tydate"]]
    assert len(seqs) == 1, (tyname, info["tydate"][:2])
    s = make_sample(ROOT, f, seqs[0])
    ours.append(s)
    dmax["image"] = max(dmax["image"], float((s["image_obs"][0] - image_obs[j]).abs().max()))
    dmax["traj"] = max(dmax["traj"], float((s["obs_traj"][:, 0] - obs_traj[:, j]).abs().max()))
    dmax["rel"] = max(dmax["rel"], float((s["obs_traj_rel"][:, 0] - obs_traj_rel[:, j]).abs().max()))
    dmax["gt"] = max(dmax["gt"], float((s["gt"][:, 0] - gt[:, j]).abs().max()))
    for k in ENV_KEYS:
        dmax["env"] = max(dmax["env"], float((s["env"][k][0] - env[k][j]).abs().max()))
res["data_pipeline_max_abs_diff"] = dmax
print(dmax, flush=True)

B = len(idx)
cat = lambda key: torch.cat([o[key] for o in ours], 1)
n_obs, n_rel, n_img = cat("obs_traj"), cat("obs_traj_rel"), torch.cat([o["image_obs"] for o in ours], 0)
n_env = {k: torch.cat([o["env"][k] for o in ours], 0) for k in ENV_KEYS}

# (b) deterministic: all six generators with shared noise + chooser logits
# NB: the official all_g_out branch ignores `user_noise` (mix_noise is called without it), so the noise is matched through the RNG stream instead.
torch.manual_seed(0)
noise = torch.randn(B, 16)
torch.manual_seed(0)
with torch.no_grad():
    o_pred, _, o_logits, _ = off(obs_traj, obs_traj_rel, seq_se, image_obs, env, all_g_out=True)
    n_pred, n_logits = nat.all_generators(n_rel := obs_traj_rel, image_obs, env, noise)
    n_pred2, n_logits2 = nat.all_generators(n_rel_ := n_rel, image_obs, env, noise)
    # native on *its own* loaded data (checks loader + model together)
    p3, l3 = nat.all_generators(n_rel_, n_img, n_env, noise)
o_pred = o_pred.permute(0, 1, 2, 3)                      # (pred_len, 6, B, 4)
rl2 = lambda x, y: float((x - y).norm() / y.norm())
res["generators_rel_l2_native_vs_official_same_inputs"] = rl2(n_pred, o_pred)
res["generators_max_abs"] = float((n_pred - o_pred).abs().max())
res["chooser_logits_max_abs"] = float((n_logits - o_logits).abs().max())
res["generators_rel_l2_native_data_vs_official_data"] = rl2(p3, o_pred)
res["pred_increment_std"] = float(o_pred.std())

# (c) sampling with identical RNG streams (official quirk: loop over range(n_unique))
torch.manual_seed(123)
with torch.no_grad():
    o_s, _, _, o_idx = off(obs_traj, obs_traj_rel, seq_se, image_obs, env, num_samples=6, all_g_out=False)
torch.manual_seed(123)
with torch.no_grad():
    n_s, n_idx = nat.sample(obs_traj_rel, image_obs, env, num_samples=6)
same_idx = bool((torch.as_tensor(o_idx) == n_idx).all())
filled = []
for s in range(6):
    u = np.unique(o_idx[:, s])
    filled.append(bool((u == np.arange(len(u))).all()))
res["sampled_classes_identical"] = same_idx
res["members_where_official_loop_covers_all_drawn_classes"] = int(sum(filled))
res["members_total"] = 6
if all(filled):
    res["sampled_rel_l2_native_vs_official"] = rl2(n_s, o_s)
else:
    ok = [s for s in range(6) if filled[s]]
    res["sampled_rel_l2_native_vs_official_on_members_where_loop_is_complete"] = rl2(n_s[:, ok], o_s[:, ok]) if ok else None
    cols = [s for s in range(6) if not filled[s]]
    res["official_unfilled_values_are_ones"] = bool(((o_s[:, cols] == 1).any()))
print(json.dumps(res, indent=1))
json.dump(res, open("docs/results/tcn_parity.json", "w"), indent=1)

# (d) the official sampling quirk on a single storm: classes are looped as range(n_unique), not over the drawn ids
bad_off, bad_nat = 0, 0
for j in range(8):
    sl = slice(j, j + 1)
    one = lambda t: t[:, sl] if t.dim() >= 2 else t
    env1 = {k: v[sl] for k, v in env.items()}
    torch.manual_seed(7 + j)
    with torch.no_grad():
        o1, _, _, oi = off(obs_traj[:, sl], obs_traj_rel[:, sl], torch.tensor([[0, 1]]), image_obs[sl], env1, num_samples=6, all_g_out=False)
    torch.manual_seed(7 + j)
    with torch.no_grad():
        n1, ni = nat.sample(obs_traj_rel[:, sl], image_obs[sl], env1, num_samples=6)
    bad_off += int(((o1 == 1).all(-1).all(0)).sum())             # members left at the 'ones' initialisation
    bad_nat += int(((n1 == 0).all(-1).all(0)).sum())
res["single_storm_members_unfilled_official"] = bad_off
res["single_storm_members_unfilled_native"] = bad_nat
res["single_storm_members_total"] = 8 * 6
print({k: res[k] for k in ("single_storm_members_unfilled_official", "single_storm_members_unfilled_native", "single_storm_members_total")})
json.dump(res, open("docs/results/tcn_parity.json", "w"), indent=1)
