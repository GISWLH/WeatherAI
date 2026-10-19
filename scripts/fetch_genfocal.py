"""Inspect/download one official GCS checkpoint subtree, never the 39 GB tar.

Default is inspect-only. --download writes files; verifies sizes and MD5 when
GCS supplies it, refuses disk overcommit, never deletes other cached models.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import urllib.parse
import urllib.request

DEFAULT_PREFIX = "super_resolution/checkpoints/conus_7d_t2m_w10m_q1000_msl/"


def inventory(prefix):
    if not prefix or not prefix.endswith("/") or ".." in prefix.split("/"):
        raise ValueError("use an explicit checkpoint directory prefix ending in /")
    files, token = [], None
    while True:
        query = {"prefix": prefix, "maxResults": 1000}
        if token:
            query["pageToken"] = token
        url = "https://storage.googleapis.com/storage/v1/b/genfocal/o?" + urllib.parse.urlencode(query)
        with urllib.request.urlopen(url, timeout=60) as response:
            page = json.load(response)
        files += [item for item in page.get("items", []) if not item["name"].endswith(("/", "_$folder$"))]
        token = page.get("nextPageToken")
        if not token:
            break
    if not files:
        raise ValueError("No checkpoint objects found for this prefix")
    return files


def download(files, prefix, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    total = sum(int(item["size"]) for item in files)
    if shutil.disk_usage(output).free < total + 1024**3:
        raise RuntimeError(f"Need {total / 1024**3:.2f} GiB plus 1 GiB spare; no files were deleted")
    for item in files:
        relative = item["name"][len(prefix):]
        target = (output / relative).resolve()
        if output not in target.parents:
            raise ValueError("Object path escapes output directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        digest = hashlib.md5()
        url = "https://storage.googleapis.com/genfocal/" + urllib.parse.quote(item["name"], safe="/")
        with urllib.request.urlopen(url, timeout=120) as response, part.open("wb") as stream:
            while block := response.read(1024**2):
                digest.update(block)
                stream.write(block)
        if part.stat().st_size != int(item["size"]):
            raise RuntimeError(f"Incomplete download: {relative}")
        if item.get("md5Hash") and base64.b64encode(digest.digest()).decode() != item["md5Hash"]:
            raise RuntimeError(f"Checksum mismatch: {relative}")
        part.replace(target)
        print(relative)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--out", default="/tmp/genfocal-conus")
    parser.add_argument("--manifest")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    files = inventory(args.prefix)
    manifest = {"bucket": "genfocal", "prefix": args.prefix, "bytes": sum(int(f["size"]) for f in files),
                "files": [{k: f[k] for k in ("name", "size", "md5Hash") if k in f} for f in files]}
    if args.manifest:
        Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f'{len(files)} files; {manifest["bytes"] / 1024**3:.3f} GiB; prefix gs://genfocal/{args.prefix}')
    if args.download:
        download(files, args.prefix, args.out)
