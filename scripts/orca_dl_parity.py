"""ORCA-DL parity: native `weatherai.models.orca_dl` vs the official `OpenEarthLab/ORCA-DL` modules (clone at /workspace/scratch/ORCA-DL),
same seed_1 checkpoint, official demo input (GODAS, Jan, normalised). Official code needs `transformers`, `timm`, `global_land_mask`
(installed under /workspace/venv/orca_extra); a one-line shim restores `ModelOutput` for a newer transformers. Runs each model in turn
to keep memory low and writes docs/results/orca_dl_parity.json.  usage: python scripts/orca_dl_parity.py {official|native|compare} [steps]"""
import json, os, sys, time
import numpy as np
import torch

CK = os.environ.get("ORCA_CKPT", "/workspace/ckpt/orca_dl")
OFF = os.environ.get("ORCA_OFFICIAL", "/workspace/scratch/ORCA-DL")
TMP = "/tmp/orca_parity"
os.makedirs(TMP, exist_ok=True)
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 7
torch.set_num_threads(int(os.environ.get("NT", "4")))


def inputs():
    from weatherai.models.orca_dl.data import download_example, load_example
    ex = download_example("/workspace/scratch/orca_example")
    o, a = load_example(ex, CK, month=0)
    return torch.from_numpy(o), torch.from_numpy(a)


def run_official():
    sys.path[:0] = ["/workspace/venv/orca_extra", OFF]
    import transformers.modeling_utils as mu
    from transformers.utils import ModelOutput
    mu.ModelOutput = ModelOutput
    from model import ORCADLConfig, ORCADLModel
    m = ORCADLModel(ORCADLConfig.from_json_file(f"{OFF}/model_config.json"))
    m.load_state_dict(torch.load(f"{CK}/model_weights/seed_1.bin", map_location="cpu"))
    m.eval()
    o, a = inputs()
    t = time.time()
    with torch.no_grad():
        out = m(ocean_vars=o, atmo_vars=a, predict_time_steps=STEPS, return_dict=True).preds
    print("official seconds", time.time() - t, out.shape, flush=True)
    np.save(f"{TMP}/official.npy", out.numpy())


def run_native():
    from weatherai.models.orca_dl import load_official
    m = load_official(CK, seed=1)
    o, a = inputs()
    t = time.time()
    with torch.no_grad():
        out = m(o, a, predict_time_steps=STEPS)
    print("native seconds", time.time() - t, out.shape, flush=True)
    np.save(f"{TMP}/native.npy", out.numpy())


def compare():
    a, b = np.load(f"{TMP}/official.npy"), np.load(f"{TMP}/native.npy")
    res = {"steps": STEPS, "max_abs": [], "rel_l2": [], "official_std": float(a.std())}
    for t in range(a.shape[1]):
        d = np.abs(a[:, t] - b[:, t])
        res["max_abs"].append(float(d.max()))
        res["rel_l2"].append(float(np.linalg.norm(a[:, t] - b[:, t]) / np.linalg.norm(a[:, t])))
    print(json.dumps(res, indent=1))
    json.dump(res, open("docs/results/orca_dl_parity.json", "w"), indent=1)


if __name__ == "__main__":
    {"official": run_official, "native": run_native, "compare": compare}[sys.argv[1]]()
