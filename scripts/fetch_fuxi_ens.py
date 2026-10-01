"""Download the FuXi-ENS weights (+ sample input) from Zenodo without fetching the whole zip.

Zenodo record 10.5281/zenodo.15124541 (CC-BY-NC-4.0, non-commercial use only) ships one 10 GB zip,
``FuXi-ENS-main.zip``.  The zip's central directory is read over HTTP (remotezip); for each wanted member
(``model/fuxi_ens`` 9.9 GB weights, ``model/fuxi_ens.onnx`` 108 MB graph, ``data/input.nc`` 648 MB sample)
the *raw deflate bytes* are fetched with several parallel HTTP range requests (a single stream is only ~3 MB/s
on the HF Space vs ~25 MB/s on the box) and inflated locally in one pass.  Weights must NOT be committed to git.
"""
import argparse
import concurrent.futures as cf
import os
import struct
import time
import zlib

URL = "https://zenodo.org/api/records/15124541/files/FuXi-ENS-main.zip/content"
FILES = {"model/fuxi_ens.onnx": 108565174, "model/fuxi_ens": 9945359232, "data/input.nc": 647880197}


def _range(start, end, tries=8):
    import requests

    for k in range(tries):
        try:
            r = requests.get(URL, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=180)
            r.raise_for_status()
            assert len(r.content) == end - start
            return r.content
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def _data_offset(info):
    h = _range(info.header_offset, info.header_offset + 30)
    sig, = struct.unpack("<I", h[:4])
    assert sig == 0x04034B50, "bad zip local header"
    n, m = struct.unpack("<HH", h[26:30])
    return info.header_offset + 30 + n + m


def fetch(out, files=tuple(FILES), log=print, workers=16, part=1 << 26):
    from remotezip import RemoteZip

    os.makedirs(out, exist_ok=True)
    for rel in files:
        dst = os.path.join(out, os.path.basename(rel))
        if os.path.exists(dst) and os.path.getsize(dst) == FILES[rel]:
            log(f"have {dst}")
            continue
        t0 = time.time()
        with RemoteZip(URL) as z:
            info = z.getinfo("FuXi-ENS-main/" + rel)
        assert info.file_size == FILES[rel], (rel, info.file_size)
        base, csize = _data_offset(info), info.compress_size
        raw = dst + ".raw"
        with open(raw, "wb") as f:
            f.truncate(csize)
        spans = [(a, min(a + part, csize)) for a in range(0, csize, part)]

        def job(sp):
            a, b = sp
            data = _range(base + a, base + b)
            with open(raw, "r+b") as f:
                f.seek(a)
                f.write(data)

        with cf.ThreadPoolExecutor(workers) as ex:
            list(ex.map(job, spans))
        log(f"{rel}: raw {csize} B in {time.time() - t0:.0f}s, inflating")
        d = zlib.decompressobj(-15) if info.compress_type == 8 else None
        n = 0
        with open(raw, "rb") as src, open(dst + ".part", "wb") as f:
            while True:
                buf = src.read(1 << 24)
                if not buf:
                    break
                out_b = d.decompress(buf) if d else buf
                f.write(out_b)
                n += len(out_b)
            if d:
                tail = d.flush()
                f.write(tail)
                n += len(tail)
        assert n == FILES[rel], (n, FILES[rel])
        os.remove(raw)
        os.replace(dst + ".part", dst)
        log(f"done {dst} {n} B in {time.time() - t0:.0f}s")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/ckpt/fuxi_ens")
    ap.add_argument("--only-graph", action="store_true", help="only fuxi_ens.onnx (108 MB)")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    fetch(a.out, ("model/fuxi_ens.onnx",) if a.only_graph else tuple(FILES), workers=a.workers)
