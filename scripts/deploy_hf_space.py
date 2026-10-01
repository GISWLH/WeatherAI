"""Stage the repo + hf_space/ app and upload to a Hugging Face Space.

    HF_TOKEN=... python scripts/deploy_hf_space.py [--space LonghaoWang/weatherai-graphcast-smoke]

The token is only read from the environment (never stored in the repo).
"""
import argparse
import os
import shutil
import tempfile

from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEEP = ["weatherai", "tests", "scripts", "LICENSE", "pyproject.toml"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="LonghaoWang/weatherai-graphcast-smoke")
    ap.add_argument("--hardware", default="zero-a10g")
    a = ap.parse_args()
    api = HfApi()
    api.create_repo(a.space, repo_type="space", space_sdk="gradio", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for f in os.listdir(os.path.join(ROOT, "hf_space")):
            shutil.copy(os.path.join(ROOT, "hf_space", f), tmp)
        dst = os.path.join(tmp, "WeatherAI")
        for k in KEEP:
            s = os.path.join(ROOT, k)
            if os.path.isdir(s):
                shutil.copytree(s, os.path.join(dst, k), ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                os.makedirs(dst, exist_ok=True)
                shutil.copy(s, dst)
        api.upload_folder(folder_path=tmp, repo_id=a.space, repo_type="space", commit_message="Deploy WeatherAI smoke")
    try:
        api.request_space_hardware(a.space, a.hardware)
    except Exception as e:  # already set / not permitted
        print("hardware request:", e)
    print(f"https://huggingface.co/spaces/{a.space}")


if __name__ == "__main__":
    main()
