"""Reproducible retrieval of original source and inference-only checkpoint trees."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.parse
import urllib.request

SOURCES = {
    "genfocal": {"repository": "https://github.com/google-research/swirl-dynamics.git",
                 "revision": "0beb5d20a5b31969e12069ba7d86fc0537e08528", "license": "Apache-2.0"},
    "graphcast": {"repository": "https://github.com/google-deepmind/graphcast.git",
                  "revision": "08cf73625c9d12bd9aaa038868bcb2fe488f2a22", "license": "Apache-2.0"},
    "pbc": {"repository": "https://github.com/mouatadid/pbc.git",
            "revision": "d59e18f720bac712e18563bd469d4679c120f9e5", "license": "MIT"},
}
GENFOCAL = {
    "debiasing": ("debiasing/checkpoints/reflow_uvit/300000/default/", "params."),
    "super_resolution": ("super_resolution/checkpoints/conus_7d_t2m_w10m_q1000_msl/300000/default/", "ema_state.ema."),
}


def cache_root():
    return Path(os.environ.get("WEATHERAI_CACHE", Path.home() / ".cache" / "weatherai"))


def checkout(name, root=None):
    """Fetch exact author revision into a dedicated cache (no upstream edits)."""
    spec = SOURCES[name]
    path = (Path(root) if root else cache_root()) / "sources" / (name + "-" + spec["revision"][:12])
    if not (path / ".git").exists():
        path.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "remote", "add", "origin", spec["repository"]], check=True)
    head = subprocess.run(["git", "-C", str(path), "rev-parse", "--verify", "HEAD"], capture_output=True)
    if head.returncode:
        subprocess.run(["git", "-C", str(path), "fetch", "--depth", "1", "origin", spec["revision"]], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", "FETCH_HEAD"], check=True)
    revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(path), "diff", "--name-only", "HEAD"], text=True).strip()
    if revision != spec["revision"] or dirty:
        raise RuntimeError(f"Official source cache has changed: {path}")
    return path


def download(url, destination, *, sha256=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_name(destination.name + ".part")
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            while block := response.read(8 * 1024**2):
                output.write(block)
        temporary.replace(destination)
    digest = hashlib.sha256()
    with destination.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            digest.update(block)
    if sha256 and digest.hexdigest() != sha256:
        raise ValueError(f"Checksum mismatch: {destination}")
    return destination, digest.hexdigest()


def genfocal_parameters(stage, root=None, *, workers=8, log=print, load=True, device=None):
    """Read exact selected tensors from the public Orbax OCDBT checkpoint.

    HTTP range reads avoid downloading optimizer states (39 GB debias archive).
    Debiasing uses raw params, matching upstream sampling_from_batch; SR uses
    EMA, matching upstream DenoisingTrainer(use_ema=True). No tensor conversion,
    model surgery or random initialization. Each local tensor has a SHA-256.
    """
    import numpy as np
    import tensorstore as ts
    prefix, tree_prefix = GENFOCAL[stage]
    output = (Path(root) if root else cache_root()) / "genfocal" / stage
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    store_spec = {"driver": "ocdbt", "base": {"driver": "http", "base_url": "https://storage.googleapis.com/genfocal/" + prefix}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["source"] != prefix or manifest["tree_prefix"] != tree_prefix:
            raise ValueError("Checkpoint manifest does not match the selected official stage")
        records = manifest["tensors"]
    else:
        kv = ts.KvStore.open(store_spec).result()
        names = sorted(key.decode()[:-len("/.zarray")] for key in kv.list().result()
                       if key.decode().startswith(tree_prefix) and key.endswith(b"/.zarray"))
        if not names:
            raise RuntimeError("No official parameter tensors found")

        def fetch(item):
            index, name = item
            target = output / f"{index:05d}.npy"
            # Re-fetch incomplete runs; manifest is written only after all tensors finish.
            array = ts.open({"driver": "zarr", "kvstore": {**store_spec, "path": name + "/"}}, open=True).result().read().result()
            with target.open("wb") as stream:
                np.save(stream, array, allow_pickle=False)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            return {"name": name[len(tree_prefix):], "file": target.name, "shape": list(array.shape),
                    "dtype": str(array.dtype), "bytes": array.nbytes, "sha256": digest}

        records = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, record in enumerate(pool.map(fetch, enumerate(names))):
                records.append(record)
                if (index + 1) % 100 == 0:
                    log(f"{stage}: {index + 1}/{len(names)} tensors")
        manifest = {"source": prefix, "tree_prefix": tree_prefix, "tensors": records,
                    "parameter_count": sum(int(np.prod(r["shape"])) for r in records),
                    "bytes": sum(r["bytes"] for r in records)}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not load:
        return manifest
    # Orbax flattened names contain literal dots in Flax module names.
    # Reconstruct from key_metadata, never by splitting the storage key.
    metadata_path, _ = download("https://storage.googleapis.com/genfocal/" + prefix + "_METADATA",
                                output / "_METADATA.json")
    metadata = json.loads(metadata_path.read_text())
    paths = {}
    prefix_parts = tree_prefix.rstrip(".").split(".")
    for entry in metadata["tree_metadata"].values():
        keys = [part["key"] for part in entry["key_metadata"]]
        if keys[:len(prefix_parts)] == prefix_parts:
            paths[".".join(keys[len(prefix_parts):])] = keys[len(prefix_parts):]
    tree = {}
    for record in records:
        target = output / record["file"]
        if hashlib.sha256(target.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Cached parameter checksum mismatch: {target}")
        leaf = tree
        parts = paths[record["name"]]
        for part in parts[:-1]:
            leaf = leaf.setdefault(part, {})
        array = np.load(target, allow_pickle=False)
        if device is not None:
            import jax
            array = jax.device_put(array, device)
            array.block_until_ready()
        leaf[parts[-1]] = array
    return tree, manifest


def gcs_url(bucket, name):
    return f"https://storage.googleapis.com/{bucket}/" + urllib.parse.quote(name, safe="/")
