"""Make the official FuXi-ENS ONNX deterministic: replace its 3 random ops by graph inputs.

The exported graph contains ``RandomUniformLike`` (drop0, drop1: dropout masks, keep prob 0.8) and
``RandomNormalLike`` (latent noise).  We delete those nodes and expose their outputs as inputs so
the same noise tensors can be fed to onnxruntime and to the PyTorch port.  Optionally expose
intermediate tensors as extra outputs for stage-by-stage comparison.  Only the 108 MB graph file is
rewritten; weights stay in the external-data file next to it.
"""
import argparse
import os

import onnx
from onnx import TensorProto, helper

NOISE = {  # node output name -> (input name, shape)
    "/dist_p/drop0/RandomUniformLike_output_0": ("noise_drop0", [1, 16200, 1536]),
    "/dist_p/drop1/RandomUniformLike_output_0": ("noise_drop1", [1, 16200, 1536]),
    "/RandomNormalLike_output_0": ("noise_eps", [1, 156, 721, 1440]),
}
STAGES = {  # name in our comparison -> onnx tensor
    "dist_emb": "/dist_p/patch_embed/norm/Mul_2_output_0",
    "dist_drop0": "/dist_p/drop0/Mul_output_0",
    "dist_L0": "/dist_p/layers.0/blocks.6/Add_6_output_0",
    "dist_drop1": "/dist_p/drop1/Mul_output_0",
    "dist_out": "/dist_p/Add_2_output_0",         # tokens after final AdaLN + residual
    "dist_mean": "/dist_p/Slice_2_output_0",
    "dec_emb": "/decoder.0/patch_embed/norm/Mul_2_output_0",
    "dec_b0": "/decoder.0/layers.0/blocks.0/Add_6_output_0",
    "dec_b1": "/decoder.0/layers.0/blocks.1/Add_6_output_0",
    "dec_L0": "/decoder.0/layers.0/blocks.6/Add_6_output_0",
    "dec_L3": "/decoder.0/layers.3/blocks.6/Add_6_output_0",
    "dec_out": "/decoder.0/Add_2_output_0",
}


def build(src, dst, stages=True):
    m = onnx.load(src, load_external_data=False)
    g = m.graph
    keep, removed = [], []
    for n in g.node:
        if n.output and n.output[0] in NOISE:
            removed.append(n.output[0])
        else:
            keep.append(n)
    assert len(removed) == 3, removed
    del g.node[:]
    g.node.extend(keep)
    for out, (name, shape) in NOISE.items():
        # rename: consumers read `out`; make `out` itself a graph input
        g.input.append(helper.make_tensor_value_info(out, TensorProto.FLOAT, shape))
    if stages:
        for k, t in STAGES.items():
            g.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
    onnx.save(m, dst)
    return removed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/workspace/ckpt/fuxi_ens/fuxi_ens.onnx")
    ap.add_argument("--dst", default="/workspace/ckpt/fuxi_ens/fuxi_ens_fixednoise.onnx")
    ap.add_argument("--no-stages", action="store_true")
    a = ap.parse_args()
    print(build(a.src, a.dst, not a.no_stages))
