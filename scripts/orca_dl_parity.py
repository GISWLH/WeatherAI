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


def lite():
    """Random-weight small config: official modules vs native with the official state dict copied over (CPU, seconds)."""
    sys.path[:0] = ["/workspace/venv/orca_extra", OFF]
    import transformers.modeling_utils as mu
    from transformers.utils import ModelOutput
    mu.ModelOutput = ModelOutput
    from model import ORCADLConfig as OC, ORCADLModel as OM
    from weatherai.models.orca_dl import ORCADL
    from weatherai.models.orca_dl.convert import convert_state_dict
    kw = dict(lat_space=(-31.5, 31.5, 32), lon_space=(0.5, 359.5, 60), in_chans=[2, 2, 1, 2, 2, 1], out_chans=[2, 2, 1, 2, 2, 1], embed_dim=24,
              lg_hidden_dim=192, enc_heads=(2, 4, 8), lg_heads=(8, 8), window_size=(4, 5), max_t=3, atmo_dims=2, is_moe_atmo=False,
              var_list=["so", "thetao", "tos", "uo", "vo", "zos"], var_index=[0, 1, 0, 2, 3, 1])
    torch.manual_seed(0)
    off = OM(OC(**kw)).eval()
    for p_ in off.parameters():
        if p_.dim() > 1:
            torch.nn.init.normal_(p_, std=0.05)
    from weatherai.models.orca_dl.model import ORCADLConfig as NC
    nat = ORCADL(NC(**{k: v for k, v in kw.items() if k in NC.__dataclass_fields__}))
    nat.load_state_dict(convert_state_dict(off.state_dict(), nat), strict=True)
    nat.eval()
    o, a = torch.randn(2, 10, 32, 60), torch.randn(2, 2, 32, 60)
    with torch.no_grad():
        ya = off(ocean_vars=o, atmo_vars=a, predict_time_steps=4, return_dict=True).preds
        yb = nat(o, a, predict_time_steps=4)
    res = {"lite_max_abs": float((ya - yb).abs().max()), "lite_rel_l2": float((ya - yb).norm() / ya.norm()), "steps": 4}
    print(res)
    json.dump(res, open("docs/results/orca_dl_parity_lite.json", "w"), indent=1)


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
    {"lite": lite, "official": run_official, "native": run_native, "compare": compare}[sys.argv[1]]()
