"""Official-model runtime. GPU entry points never substitute random weights."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("WEATHERAI_CACHE", "/tmp/weatherai-official")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import json
import sys
import threading
import traceback
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "WeatherAI"))
import gradio as gr
import spaces

STATE = {}


def status():
    import importlib.metadata as md
    return json.dumps({"jobs": STATE, "disk_free_GiB": shutil.disk_usage('/tmp').free / 2**30,
                       "versions": {k: md.version(k) for k in ['jax', 'flax', 'orbax-checkpoint', 'tensorstore', 'xarray', 'pandas', 'zarr']}}, indent=2)


def prepare(stage):
    if STATE.get(stage, {}).get("status") == "running":
        return status()
    STATE[stage] = {"status": "running", "log": []}
    def work():
        try:
            from scripts.official_validation import prepare_assets
            result = prepare_assets(stage, log=lambda s: STATE[stage]["log"].append(str(s)))
            STATE[stage].update(status="ready", result=result)
        except Exception:
            STATE[stage].update(status="error", error=traceback.format_exc()[-6000:])
    threading.Thread(target=work, daemon=True).start()
    return status()


@spaces.GPU(duration=120)
def run_official(stage):
    from scripts.official_validation import run
    try:
        return json.dumps(run(stage), indent=2)
    except Exception:
        return json.dumps({'status':'ERROR','model':stage,'error':traceback.format_exc()}, indent=2)


with gr.Blocks(title="WeatherAI official models") as demo:
    gr.Markdown("# Official author models and pretrained weights\nPreparation downloads the authors' artifacts. No lite fallback.")
    stage = gr.Dropdown(['genfocal_sr', 'genfocal_debias', 'graphcast', 'graphcast_small', 'graphcast_small_erf'], value='genfocal_sr', label='Official model')
    output = gr.Textbox(lines=24)
    gr.Button('Prepare official assets (CPU)').click(prepare, stage, output, api_name='prepare')
    gr.Button('Status').click(status, [], output, api_name='status')
    gr.Button('Run pretrained inference (ZeroGPU)').click(run_official, stage, output, api_name='run_official')

if __name__ == '__main__':
    demo.launch()
