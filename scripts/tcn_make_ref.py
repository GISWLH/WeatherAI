"""Store 4 real NI windows (TCND, CC-BY-4.0, Huang et al. 2025; storm FANI 2019) and the CPU outputs of the native model for the GPU smoke test / unit tests."""
import numpy as np, torch
from weatherai.models.tropicyclonenet import load_official, make_sample, sequences

ROOT = "/workspace/ckpt/tcn/ni_root"
f = f"{ROOT}/BST_data/NI/test/NI2019BSTFANI.txt"
seqs = sequences(f)[1:21:5][:4]
S = [make_sample(ROOT, f, s) for s in seqs]
cat = lambda k: torch.cat([s[k] for s in S], 1)
rel, obs, gt = cat("obs_traj_rel"), cat("obs_traj"), cat("gt")
img = torch.cat([s["image_obs"] for s in S], 0)
env = {k: torch.cat([s["env"][k] for s in S], 0) for k in S[0]["env"]}
m = load_official("/workspace/ckpt/tcn/checkpoint_with_model_16000.pt")
torch.manual_seed(0)
noise = torch.randn(4, 16)
with torch.no_grad():
    gens, logits = m.all_generators(rel, img, env, noise)
np.savez_compressed("tests/models/tropicyclonenet/data/ref_ni_fani.npz", obs_traj=obs.numpy(), obs_traj_rel=rel.numpy(), gt=gt.numpy(), image_obs=img.numpy(),
                    noise=noise.numpy(), gens=gens.numpy(), logits=logits.numpy(), dates=np.array([s["start_date"] for s in S]), **{f"env_{k}": v.numpy() for k, v in env.items()})
print(gens.shape, logits.softmax(-1).numpy().round(2))
