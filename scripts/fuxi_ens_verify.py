"""Numerical verification of the native PyTorch FuXi-ENS against the official ONNX (onnxruntime, CPU).

Step 1 (``fuxi_ens_ort_segments.py``): run the official ``fuxi_ens.onnx`` with its 3 random ops
(2 dropout masks + latent normal noise) turned into graph inputs, in segments, saving every cut tensor
to ``<ckpt>/ref``.
Step 2 (this script): load the same checkpoint into :class:`FuXiENS` (strict), feed the *same* noise and
compare stage by stage and the final output.  Prints/writes a JSON report.
"""
import argparse, gc, json, os, time
import numpy as np
import torch

CK = os.environ.get("FUXI_ENS_CKPT", "/workspace/ckpt/fuxi_ens")


def noise_feeds(seed=1234):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(1, 16200, 1536, generator=g), torch.rand(1, 16200, 1536, generator=g),
            torch.randn(1, 156, 721, 1440, generator=g))


def load_input():
    import pandas as pd
    import xarray as xr
    ds = xr.open_dataset(f"{CK}/input.nc")
    x = ds["fuxi_ens"].values[None].astype(np.float32)
    t = pd.to_datetime(ds.time.values[-1])
    return x, np.array([0], np.float32), np.array([t.hour / 24], np.float32), np.array([min(365, t.day_of_year) / 365], np.float32)


def err(a, b, chunk=1 << 22):
    """Error statistics in float64, computed chunk-wise (tensors are up to 650 MB). NaNs (the input's SST land mask is
    NaN and is passed through by the model) must sit at identical positions and are excluded from the statistics."""
    a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    mx = sa = sb = sq_ref = mref = 0.0
    n = nan_mismatch = n_nan = n_big = 0
    argmax = 0
    for i in range(0, a.size, chunk):
        x, y = a[i:i + chunk].astype(np.float64), b[i:i + chunk].astype(np.float64)
        nx, ny = np.isnan(x), np.isnan(y)
        nan_mismatch += int((nx != ny).sum()); n_nan += int(ny.sum())
        ok = ~(nx | ny)
        x, y = x[ok], y[ok]
        if x.size == 0:
            continue
        d = np.abs(x - y)
        if d.max() > mx:
            mx, argmax = float(d.max()), i + int(np.flatnonzero(ok)[int(d.argmax())])
        sa += float(d.sum()); sb += float((d ** 2).sum()); sq_ref += float((y ** 2).sum()); n += x.size
        mref = max(mref, float(np.abs(y).max()))
    rms_ref = (sq_ref / n) ** 0.5
    return dict(max_abs=mx, mean_abs=sa / n, max_ref=mref, max_abs_over_max_ref=mx / (mref + 1e-30),
                rmse_over_rms_ref=(sb / n) ** 0.5 / (rms_ref + 1e-30), argmax_flat=argmax, nan_positions_mismatch=nan_mismatch,
                n_nan=n_nan)


class Recorder(dict):
    """Compares each stage with the stored ONNX tensor as soon as it is produced and keeps only a few tensors
    (the box has ~10 GB RAM usable and the weights are memory-mapped)."""
    KEEP = {"xn", "sample", "z"}

    def __init__(self, ref_dir):
        super().__init__()
        self.ref_dir, self.report = ref_dir, {}

    def __setitem__(self, k, v):
        p = f"{self.ref_dir}/{k}.npy"
        if os.path.exists(p):
            r = np.load(p, mmap_mode="r")
            self.report[k] = err(v.detach().float().cpu().numpy(), r)
            print(f"{k:8s} {self.report[k]}", flush=True)
        if k in ("final",) or k in self.KEEP:
            super().__setitem__(k, v)


@torch.no_grad()
def run_stages(model, x, step, hour, doy, noise, dtype=None, device="cpu", out=None):
    """Same math as ``FuXiENS.forward`` but returning the intermediate tensors that the ONNX segments expose."""
    from weatherai.models.fuxi_ens.model import _dropout_from_uniform
    cfg = model.cfg
    d0, d1, eps = (n.to(device) for n in noise)
    x, step, hour, doy = (torch.as_tensor(t).to(device) for t in (x, step, hour, doy))
    out = out if out is not None else {}
    cast = (lambda t: t.to(dtype)) if dtype else (lambda t: t)
    xn = model.normalise(x); out["xn"] = xn
    dp = model.dist_p
    cond = dp.condition(step, hour, doy); out["dcond"] = cond
    emb = dp.patch_embed(cast(xn.flatten(1, 2)), model.const[None]); out["demb"] = emb
    h = _dropout_from_uniform(emb, d0, cfg.dropout_p); out["d0"] = h
    for i, layer in enumerate(dp.layers):
        h = layer(h, cond, dp.rope_cos, dp.rope_sin)
        if i == 0:
            out["L0"] = h
            h = _dropout_from_uniform(h, d1, cfg.dropout_p); out["d1"] = h
    h = dp.norm_layer(h, cond) + emb; out["dout"] = h
    m = dp.tokens_to_map(h)
    mean, logvar = dp.crop(dp.mean(m)), dp.crop(dp.logvar(m))
    out["mean"], out["logvar"] = mean, logvar
    sample = mean.float() + torch.exp(0.5 * logvar.float()) * eps
    out["sample"] = sample
    z = xn + sample.view_as(xn)
    z = torch.cat([z[:, :, :-cfg.n_accum], torch.zeros_like(z[:, :, -cfg.n_accum:])], 2); out["z"] = z
    dec = model.decoder[0]
    cond = dec.condition(step, hour, doy); out["deccond"] = cond
    emb = dec.patch_embed(cast(z.flatten(1, 2)), model.const[None]); out["dec_emb"] = emb
    h = emb
    for li, layer in enumerate(dec.layers):
        h = layer(h, cond, dec.rope_cos, dec.rope_sin); out[f"dl{li}"] = h
    h = dec.norm_layer(h, cond) + emb; out["dec_out"] = h
    m = dec.tokens_to_map(h)
    y = dec.crop(torch.cat([dec.pred_layer.pl_head(m), dec.pred_layer.sf_head(m)], 1))
    y = model.denormalise(y)
    out["final"] = torch.cat([x[:, -1:].float(), y[:, None]], 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--report", default=f"{CK}/ref/report.json")
    a = ap.parse_args()
    from weatherai.models.fuxi_ens import load_official
    t0 = time.time()
    model = load_official(f"{CK}/fuxi_ens.onnx", device=a.device)       # strict state-dict load
    print(f"strict load OK in {time.time()-t0:.0f}s; params={model.num_parameters():,}", flush=True)
    x, step, hour, doy = load_input()
    noise = noise_feeds()
    t0 = time.time()
    rec = Recorder(f"{CK}/ref")
    out = run_stages(model, x, step, hour, doy, noise, device=a.device, out=rec)
    print(f"torch forward {time.time()-t0:.0f}s", flush=True)
    fin = out["final"].cpu().numpy()[0]
    ref = np.load(f"{CK}/ref/final.npy", mmap_mode="r")[0]
    import xarray as xr
    chans = [str(c) for c in xr.open_dataset(f"{CK}/input.nc").channel.values]
    std = model.std.view(-1).cpu().numpy()
    per = {}
    for t in (0, 1):
        for c, name in enumerate(chans):
            e = err(fin[t, c], ref[t, c])
            per[f"t{t}_{name}"] = dict(max_abs=e["max_abs"], max_abs_over_channel_std=e["max_abs"] / float(std[c]),
                                       rmse_over_rms_ref=e["rmse_over_rms_ref"], nan_mismatch=e["nan_positions_mismatch"])
    rec.report["final_per_channel"] = per
    worst = max(per.items(), key=lambda kv: kv[1]["max_abs_over_channel_std"])
    print("final per-channel worst (max_abs / channel std):", worst, flush=True)
    print("final t0 (passthrough) exact:", bool(np.array_equal(np.nan_to_num(fin[0]), np.nan_to_num(np.asarray(ref[0])))), flush=True)
    json.dump(rec.report, open(a.report, "w"), indent=1)


if __name__ == "__main__":
    main()
