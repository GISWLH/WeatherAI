"""Run the official FuXi-ENS ONNX through onnxruntime *in segments* (fixed noise) and save every cut tensor.

A single onnxruntime session needs >15 GB RAM for the 2.49 B-parameter graph (it copies the 10 GB
external weights into anonymous memory), so we cut the graph at well-defined tensors with
``onnx.utils.extract_model`` and run the segments one after another, feeding each segment the
*onnxruntime* outputs of the previous one.  Chaining the segments is mathematically identical to the
full graph (no op is changed).  Outputs go to <ck>/ref/*.npy (float32).
"""
import argparse, gc, os, sys, time
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(__file__))
from fuxi_ens_verify import CK, load_input, noise_feeds  # noqa: E402

T = lambda n: f"/{n}_output_0"
N = dict(
    xn=T("Cast_8"), dcond=T("dist_p/Add_1"), demb=T("dist_p/patch_embed/norm/Mul_2"), deccond=T("decoder.0/Add_1"),
    d0=T("dist_p/drop0/Mul"), L0=T("dist_p/layers.0/blocks.6/Add_6"), d1=T("dist_p/drop1/Mul"), dout=T("dist_p/Add_2"),
    mean=T("dist_p/Slice_2"), logvar=T("dist_p/Slice_4"), sample=T("Add_2"), z=T("ScatterND_1"), dec_emb=T("decoder.0/patch_embed/norm/Mul_2"),
    dec_out=T("decoder.0/Add_2"), final="26775",
    u0=T("dist_p/drop0/RandomUniformLike"), u1=T("dist_p/drop1/RandomUniformLike"), eps=T("RandomNormalLike"),
)
N.update(input="input", step="step", hour="hour", doy="doy")
for i in range(6):
    N[f"dl{i}"] = T(f"decoder.0/layers.{i}/blocks.6/Add_6")
    
SEGMENTS = [
    ("S1_pre", ["input", "step", "hour", "doy"], ["xn", "dcond", "demb", "deccond"]),
    ("S2_dist_layer0", ["demb", "u0", "dcond"], ["d0", "L0"]),
    ("S3_dist_layer1", ["L0", "u1", "dcond", "demb"], ["d1", "dout"]),
    ("S4_dist_heads_sample", ["dout", "eps", "xn"], ["mean", "logvar", "sample", "z"]),
    ("S5_dec_l0", ["z", "deccond"], ["dec_emb", "dl0"]),
    ("S6_dec_l1", ["dl0", "deccond", "dec_emb"], ["dl1"]),
    ("S7_dec_l2", ["dl1", "deccond", "dec_emb"], ["dl2"]),
    ("S8_dec_l3", ["dl2", "deccond", "dec_emb"], ["dl3"]),
    ("S9_dec_l4", ["dl3", "deccond", "dec_emb"], ["dl4"]),
    ("S10_dec_l5", ["dl4", "deccond", "dec_emb"], ["dl5"]),
    ("S11_dec_post", ["dl5", "deccond", "dec_emb", "input", "z"], ["dec_out", "final"]),
]


_FULL = {}


def extract(src, dst, in_names, out_names):
    """Sub-graph between tensors ``in_names`` -> ``out_names``; weights stay external (no weight loading)."""
    if "m" not in _FULL:
        _FULL["m"] = onnx.load(src, load_external_data=False)
        _FULL["prod"] = {o: n for n in _FULL["m"].graph.node for o in n.output}
    m, prod = _FULL["m"], _FULL["prod"]
    need, stack, seen = [], list(out_names), set(in_names)
    nodes = {}
    while stack:
        t = stack.pop()
        if t in seen or t == "":
            continue
        seen.add(t)
        n = prod.get(t)
        if n is None:
            continue                       # initialiser or graph input
        if id(n) not in nodes:
            nodes[id(n)] = n
            stack.extend(n.input)
    order = [n for n in m.graph.node if id(n) in nodes]
    used = {i for n in order for i in n.input}
    inits = [i for i in m.graph.initializer if i.name in used]
    new = onnx.ModelProto(); new.CopyFrom(m)
    del new.graph.node[:]; del new.graph.initializer[:]; del new.graph.input[:]; del new.graph.output[:]; del new.graph.value_info[:]
    new.graph.node.extend(order); new.graph.initializer.extend(inits)
    for t in in_names:
        new.graph.input.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
    for t in out_names:
        new.graph.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
    onnx.save(new, dst)


def main():
    out = f"{CK}/ref"; os.makedirs(out, exist_ok=True)
    src = f"{CK}/fuxi_ens_fixednoise.onnx"
    x, step, hour, doy = load_input()
    d0, d1, eps = noise_feeds()
    # keep every tensor on disk (np.load mmap) -- the box has ~10 GB usable RAM and ORT itself needs several GB
    for k, v in {"input": x, "step": step, "hour": hour, "doy": doy, "u0": d0.numpy(), "u1": d1.numpy(), "eps": eps.numpy()}.items():
        np.save(f"{out}/{k}.npy", v)
    del x, d0, d1, eps; gc.collect()
    ld = lambda k: np.load(f"{out}/{k}.npy", mmap_mode="r")
    times = {}
    only = os.environ.get('ONLY')
    for name, ins, outs in SEGMENTS:
        if only and not name.startswith(only):
            continue
        path = f"{CK}/seg_{name}.onnx"
        extract(src, path, [N[i] for i in ins], [N[o] for o in outs])
        so = ort.SessionOptions(); so.intra_op_num_threads = os.cpu_count(); so.enable_cpu_mem_arena = False; so.enable_mem_pattern = False
        t0 = time.time()
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        names = [i.name for i in sess.get_inputs()]
        inv = {N[k]: k for k in N}
        feeds = {n: np.ascontiguousarray(ld(inv[n])) for n in names}
        res = sess.run([N[o] for o in outs], feeds)
        times[name] = time.time() - t0
        for o, r in zip(outs, res):
            np.save(f"{out}/{o}.npy", r)
        print(f"{name}: {times[name]:.0f}s  inputs={names}  outputs={[(o, r.shape) for o, r in zip(outs, res)]}", flush=True)
        del sess, res, feeds; gc.collect(); os.remove(path)
    for k in ("u0", "u1", "eps"):
        pass
    print("ORT total", sum(times.values()))


if __name__ == "__main__":
    main()
