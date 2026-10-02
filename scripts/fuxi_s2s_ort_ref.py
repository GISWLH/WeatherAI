"""Run the official FuXi-S2S ONNX (onnxruntime, CPU) for one step with *fixed* latent noise and save
intermediate tensors as the numerical reference for the native port.

The two ``RandomNormalLike`` nodes are replaced by graph inputs ``eps1`` (128, 12) and ``eps2`` (128, 7200);
nothing else in the graph is changed.  The weights (CC-BY-NC-ND-4.0) are read from a local download and are
never copied anywhere.
"""
import argparse, os, time
import numpy as np, onnx, xarray as xr
from onnx import TensorProto, helper
import onnxruntime as ort

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/workspace/ckpt/fuxi_s2s")
ap.add_argument("--repo", default="/workspace/scratch/FuXi-S2S")
ap.add_argument("--mask", default="/workspace/scratch/FuXi-S2S/data/mask.nc")
ap.add_argument("--n_steps", type=int, default=3)
ap.add_argument("--step", type=float, default=0.0)
ap.add_argument("--out", default="/workspace/ckpt/fuxi_s2s/ref/step0.npz")
a = ap.parse_args()
mdir = os.path.join(a.root, "model-1.0")
m = onnx.load(os.path.join(mdir, "fuxi_s2s.onnx"), load_external_data=False)
g = m.graph
rn = {n.output[0]: n for n in g.node if n.op_type == "RandomNormalLike"}
assert len(rn) == 2
names = sorted(rn, key=lambda k: list(g.node).index(rn[k]))
for k, (nm, shp) in zip(names, [("eps1", [128, 12]), ("eps2", [128, 7200])]):
    g.node.remove(rn[k])
    g.input.append(helper.make_tensor_value_info(k, TensorProto.DOUBLE, shp))
T = lambda n: f"/{n}_output_0"
TAPS = dict(xn=T("Cast_9"), emb=T("dist_p/Add"), L0=T("dist_p/layers.0/blocks.5/Add_1"), L1=T("dist_p/layers.1/blocks.5/Add_1"),
            fpn=T("dist_p/fpn/fpn.1/Mul_1"), mean=T("dist_p/Reshape_3"), cov=T("dist_p/Reshape_6"), diag=T("dist_p/Reshape_4"),
            z=T("Add_5"), dec_in=T("decoder.0/Add"), D0=T("decoder.0/layers.0/blocks.11/Add_1"),
            D1=T("decoder.0/layers.1/blocks.11/Add_1"), head=T("decoder.0/head/head.2/Add"), out="10001")
for k, v in TAPS.items():
    if k != "out":
        g.output.append(helper.make_empty_tensor_value_info(v))
tmp = os.path.join(mdir, "fuxi_s2s_fixed_noise.onnx")        # only the 1.6 MB graph; weights stay external
onnx.save(m, tmp)
import sys
sys.path.insert(0, a.repo)
from data_util import make_input          # official input builder (raw physical units; data.zip's input.nc is pre-normalised)
x = make_input(os.path.join(a.repo, "data/sample"))
mask = xr.open_dataarray(a.mask)
dim = "channel"
ch = x[dim].values.tolist()
i = ch.index("sst")
x.data[:, i] = x.sel({dim: "sst"}).where(mask).data
x = x.values.astype(np.float32)[None]
rng = np.random.default_rng(0)
eps1 = rng.standard_normal((128, 12)).astype(np.float32)
eps2 = rng.standard_normal((128, 7200)).astype(np.float32)
so = ort.SessionOptions(); so.intra_op_num_threads = 8; so.enable_cpu_mem_arena = False; so.enable_mem_pattern = False; so.enable_mem_reuse = False
t = time.time()
sess = ort.InferenceSession(tmp, so, providers=["CPUExecutionProvider"])
print("load", time.time() - t, flush=True)
feeds = {"input": x, "step": np.array([a.step], np.float32), names[0]: eps1.astype(np.float64), names[1]: eps2.astype(np.float64)}
t = time.time()
outs = sess.run(None, feeds)
print("run", time.time() - t, flush=True)
names_out = [o.name for o in sess.get_outputs()]
res = {k: outs[names_out.index(v)].astype(np.float32) for k, v in TAPS.items()}
chain = {"out_0": res["out"]}
cur = res["out"]
for k in range(1, a.n_steps):            # autoregressive chain: feed the (fp16-graph) output back with step k
    e1 = rng.standard_normal((128, 12)).astype(np.float32); e2 = rng.standard_normal((128, 7200)).astype(np.float32)
    t = time.time()
    o = sess.run([TAPS["out"]], {"input": cur, "step": np.array([a.step + k], np.float32),
                                 names[0]: e1.astype(np.float64), names[1]: e2.astype(np.float64)})[0]
    print("step", k, time.time() - t, flush=True)
    chain.update({f"eps1_{k}": e1, f"eps2_{k}": e2, f"out_{k}": o.astype(np.float32)})
    cur = o.astype(np.float32)
np.savez(a.out, input=x, eps1=eps1, eps2=eps2, step=np.array([a.step], np.float32), **res, **chain)
print({k: v.shape for k, v in res.items()})
