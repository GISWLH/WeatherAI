"""Run the deployed three-paper smoke, preserving real remote evidence."""
import argparse
import datetime
import json
from pathlib import Path

from gradio_client import Client
from huggingface_hub import HfApi, get_token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default="LonghaoWang/weatherai-graphcast-smoke")
    parser.add_argument("--out", default="docs/results/research_20261019_zerogpu.json")
    args = parser.parse_args()
    api = HfApi()
    info = api.space_info(args.space)
    client = Client(args.space, token=get_token(), verbose=False)
    result = json.loads(client.predict(api_name="/research_gpu"))
    result.update(space=args.space, space_revision=info.sha,
                  timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if result.get("device") != "cuda" or not result.get("gpu"):
        raise RuntimeError("Remote result did not establish a CUDA run")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result.get("status") != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
