"""WeatherPre tab for the existing ZeroGPU Space (LonghaoWang/weatherai-graphcast-smoke).
Clones GISWLH/WeatherPre at run time, builds real initial conditions on CPU (ERA5 from WeatherBench 2, or IFS open-data
analysis), then runs the *upstream* microsoft-aurora package on the GPU in chunks of CHUNK steps (each GPU call < 120 s)
and returns z500/t850/t2m/msl at the requested leads. No token is needed or written."""
from __future__ import annotations
import os, pathlib, subprocess, sys, time, traceback
import gradio as gr
import spaces

REPO = "/tmp/WeatherPre"
CHUNK = 8           # 6-hour steps per GPU call

def _ensure_repo():
    if not os.path.isdir(REPO):
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/GISWLH/WeatherPre", REPO], check=True)
    else:
        subprocess.run(["git", "-C", REPO, "pull", "-q"], check=False)
    if REPO not in sys.path:
        sys.path.insert(0, REPO)

@spaces.GPU(duration=120)
def _aurora_gpu(batch, steps, variant, keep, lead0, time0):
    try:
        import torch
        from weatherpre.adapters import aurora_run
        t = time.time()
        ds, state = aurora_run.forecast(batch, int(steps), variant, "cuda", keep=keep, lead0=lead0, time0=time0,
                                        return_state=True, user_schema=True)
        return ds, state, f"peak_mem={torch.cuda.max_memory_allocated()/2**30:.1f}GiB chunk_s={time.time()-t:.0f}"
    except Exception:
        return None, None, "GPU-WORKER " + traceback.format_exc()[-2500:]

def run_aurora(init: str, source: str, leads: str, variant: str, restart: bool = True):
    """ONE chunk per call (a ZeroGPU proxy token only lives for one request): the rollout state is kept in /tmp between calls.
    init like 2020-10-01T00 ; source era5|ifs ; leads '24,48,...' (hours, multiples of 6) ; variant small|pretrained|finetuned ;
    restart=True starts from the initial conditions, False continues the saved state. Status starts with 'MORE' or 'OK'."""
    try:
        _ensure_repo()
        import torch, xarray as xr
        from weatherpre.adapters import aurora_run
        from weatherpre.common import parse_time
        from huggingface_hub import hf_hub_download
        t0 = time.time(); t = parse_time(init)
        keep = sorted({int(x) for x in leads.replace(" ", "").split(",") if x})
        assert all(h % 6 == 0 and h > 0 for h in keep), "leads must be positive multiples of 6 h"
        total = max(keep) // 6
        sf = pathlib.Path(f"/tmp/aurora_state_{t:%Y%m%dT%H}_{source}_{variant}.pt")
        if restart or not sf.exists():
            batch, done = (aurora_run.batch_from_era5(t) if source == "era5" else aurora_run.batch_from_ifs(t, pathlib.Path("/tmp/ifs"))), 0
            hf_hub_download("microsoft/aurora", {"small": "aurora-0.25-small-pretrained.ckpt", "pretrained": "aurora-0.25-pretrained.ckpt", "finetuned": "aurora-0.25-finetuned.ckpt"}[variant])
        else:
            st = torch.load(sf, weights_only=False); batch, done = st["batch"], st["done"]
        prep = time.time() - t0
        k = min(CHUNK, total - done)
        ds, nxt, m = _aurora_gpu(batch, k, variant, keep, done * 6, str(t))
        if ds is None: return None, m
        done += k
        path = f"/tmp/aurora_{variant}_{t:%Y%m%dT%H}_{source}_{done*6}.nc"
        ds.to_netcdf(path)
        if done >= total:
            sf.unlink(missing_ok=True); head = "OK"
        else:
            torch.save({"batch": nxt, "done": done}, sf); head = "MORE"
        return path, f"{head} init={init} source={source} variant={variant} chunk_to={done*6}h/{total*6}h kept={ds.sizes['lead']} prep_s={prep:.0f} {m}"
    except Exception:
        return None, traceback.format_exc()[-3000:]

def tab():
    with gr.Accordion("WeatherPre: Aurora forecast from real initial conditions (ZeroGPU)", open=False):
        i = gr.Textbox("2020-10-01T00", label="init (UTC, YYYY-MM-DDTHH)")
        s = gr.Dropdown(["era5", "ifs"], value="era5", label="initial conditions: era5 (WB2, historic) / ifs (open-data analysis, last ~4 days)")
        n = gr.Textbox("24,48,72", label="leads [h], comma separated (multiples of 6)")
        v = gr.Dropdown(["small", "pretrained", "finetuned"], value="pretrained", label="Aurora checkpoint")
        r = gr.Checkbox(True, label="restart (False = continue saved rollout state)")
        b = gr.Button("Run Aurora")
        f = gr.File(label="forecast.nc"); t = gr.Textbox(label="status", lines=6)
        b.click(run_aurora, [i, s, n, v, r], [f, t], api_name="weatherpre_aurora")
