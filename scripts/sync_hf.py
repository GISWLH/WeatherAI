"""Publish the WeatherAI model card and model pages to the Hugging Face Hub (a model repo; no weights are uploaded).

    HF_TOKEN=... python scripts/sync_hf.py [--repo LonghaoWang/WeatherAI]
    python scripts/sync_hf.py --dry-run /tmp/hf_card        # write the staged files only

The card is README.md with Hub front matter; links are rewritten for the Hub: ``docs/models/*`` -> ``models/*``,
``docs/assets/*`` -> ``assets/*``, everything else -> absolute GitHub URLs. The token is read from the
environment only (``HF_TOKEN``), never from or into the repository. Run by .github/workflows/sync_hf.yml.
"""
import argparse
import os
import re
import shutil
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GITHUB = "https://github.com/GISWLH/WeatherAI"
FRONT = """---
license: mit
library_name: pytorch
tags:
- weather
- weather-forecasting
- climate
- earth-science
- sea-surface-temperature
- pytorch
- jax
- model-zoo
---

> Synced from [GISWLH/WeatherAI]({gh}) — code, tests and issues live on GitHub. No model weights are stored here:
> official weights stay with their authors and are downloaded on demand under their own licences.

"""
LINK = re.compile(r'(\]\(|src=")([^)"\s]+)')


def _hub_path(path):
    if path == "README.md":
        return "README.md"
    for src, dst in (("docs/models/", "models/"), ("docs/assets/", "assets/")):
        if path.startswith(src):
            return dst + path[len(src):]
    return None


def rewrite(text, src_file, hub_file):
    """Rewrite relative links of ``src_file`` (repo path) for its Hub location ``hub_file``."""
    def sub(m):
        target = m.group(2)
        if re.match(r"^[a-z]+:|^#", target):
            return m.group(0)
        path, frag = (target.split("#", 1) + [""])[:2]
        repo_path = os.path.normpath(os.path.join(os.path.dirname(src_file), path))
        hub = _hub_path(repo_path)
        if hub is not None:
            new = os.path.relpath(hub, os.path.dirname(hub_file) or ".")
        else:
            kind = "tree" if os.path.isdir(os.path.join(ROOT, repo_path)) else "blob"
            new = f"{GITHUB}/{kind}/main/{repo_path}"
        return m.group(1) + new + ("#" + frag if frag else "")
    return LINK.sub(sub, text)


def stage(out):
    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    os.makedirs(os.path.join(out, "assets"), exist_ok=True)
    card = rewrite(open(os.path.join(ROOT, "README.md"), encoding="utf-8").read(), "README.md", "README.md")
    open(os.path.join(out, "README.md"), "w", encoding="utf-8").write(FRONT.format(gh=GITHUB) + card)
    pages = sorted(f for f in os.listdir(os.path.join(ROOT, "docs", "models")) if f.endswith(".md"))
    for f in pages:
        text = open(os.path.join(ROOT, "docs", "models", f), encoding="utf-8").read()
        open(os.path.join(out, "models", f), "w", encoding="utf-8").write(rewrite(text, f"docs/models/{f}", f"models/{f}"))
    for f in ("banner.png", "logo.png", "icon.png", "logo.svg"):
        shutil.copy(os.path.join(ROOT, "docs", "assets", f), os.path.join(out, "assets", f))
    return pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="LonghaoWang/WeatherAI")
    ap.add_argument("--dry-run", default="", help="write the staged card to this directory and stop")
    a = ap.parse_args()
    if a.dry_run:
        print("staged", len(stage(a.dry_run)), "model pages in", a.dry_run)
        return
    from huggingface_hub import HfApi

    try:
        sha = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        sha = "local"
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(a.repo, repo_type="model", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage(tmp)
        api.upload_folder(folder_path=tmp, repo_id=a.repo, repo_type="model", commit_message=f"Sync from GitHub {sha}",
                          delete_patterns=["models/*.md"])
    print(f"https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
