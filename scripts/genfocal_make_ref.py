"""Reference generator for the GenFocal flow core: runs the OFFICIAL JAX code (swirl_dynamics.projects.genfocal.debiasing) on a
tiny deterministic Flax flow network and stores weights, inputs and official outputs in ``tests/models/genfocal/data/ref_flow.npz``.

Needs jax + flax + clu and a swirl-dynamics checkout (``SWIRL=<path>``).  tensorflow / grain / gin / orbax are not used by the
code paths exercised here and are replaced by import stubs.  Usage: ``SWIRL=/path/swirl-dynamics python scripts/genfocal_make_ref.py``.
"""
import importlib.abc
import importlib.machinery
import os
import sys
import types


class _M(types.ModuleType):
    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return type(k, (), {"__init__": lambda s, *a, **kw: None, "__call__": lambda s, *a, **kw: None})


class _F(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    roots = ("gin", "tensorflow", "tensorboard", "grain", "orbax", "tensorstore", "apache_beam", "xarray_beam", "gcsfs", "zarr")

    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in self.roots:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        m = _M(spec.name)
        m.__path__ = []
        return m

    def exec_module(self, m):
        pass


sys.meta_path.append(_F())
sys.path.insert(0, os.environ["SWIRL"])

import flax.linen as nn  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from swirl_dynamics.projects.genfocal.debiasing import inference_utils as iu  # noqa: E402
from swirl_dynamics.projects.genfocal.debiasing import models as M  # noqa: E402

B, T, X, Y, C = 2, 3, 8, 6, 4


class TinyFlow(nn.Module):
    """v = tanh(x A + cm B1 + cs B2 + sigma e) D + bias; accepts scalar or batched sigma like RescaledUnet3d."""

    @nn.compact
    def __call__(self, x, sigma, cond=None, *, is_training):
        if sigma.ndim < 1:
            sigma = jnp.broadcast_to(sigma, (x.shape[0],))
        h = nn.Dense(16, name="a")(x) + nn.Dense(16, name="b1")(cond["channel:mean"]) + nn.Dense(16, name="b2")(cond["channel:std"])
        h = h + sigma[:, None, None, None, None] * self.param("e", nn.initializers.normal(1.0), (16,)) * 3.0
        return nn.Dense(x.shape[-1], name="d")(jnp.tanh(h))


rs = np.random.RandomState(0)
x0, x1 = rs.randn(B, T, X, Y, C).astype("f4"), rs.randn(B, T, X, Y, C).astype("f4")
cm, cs = rs.randn(B, T, X, Y, C).astype("f4"), np.abs(rs.randn(B, T, X, Y, C)).astype("f4") + 0.5
om, os_ = rs.randn(B, T, X, Y, C).astype("f4"), np.abs(rs.randn(B, T, X, Y, C)).astype("f4") + 0.5
u = rs.rand(B).astype("f4")
wn = (np.abs(rs.randn(1, T, X, Y, C)) + 0.1).astype("f4")

net = TinyFlow()
cond = {"channel:mean": cm, "channel:std": cs}
variables = net.init(jax.random.PRNGKey(0), x=x0, sigma=jnp.ones((B,)), cond=cond, is_training=False)
shape = (T, X, Y, C)
out = {}
for tag, kind in (("u", "uniform"), ("logi", "logistic")):
    ts = (lambda rng, sh: jnp.asarray(u)) if kind == "uniform" else (lambda rng, sh: jax.lax.logistic(jnp.asarray(u)))
    for wtag, w in (("", None), ("_w", jnp.asarray(wn))):
        model = M.ConditionalReFlowModel(input_shape=shape, flow_model=net, time_sampling=ts, weighted_norm=w,
                                         cond_shape={"channel:mean": shape, "channel:std": shape})
        batch = {"x_0": x0, "x_1": x1, **cond}
        loss, _ = model.loss_fn(variables["params"], batch, jax.random.PRNGKey(1), {})
        out[f"loss_{tag}{wtag}"] = np.float32(loss)

state = types.SimpleNamespace(model_variables=variables)
model = M.ConditionalReFlowModel(input_shape=shape, flow_model=net, cond_shape={"channel:mean": shape, "channel:std": shape})
fl = lambda a: a.reshape(B * T, X, Y, C)   # the official sampler takes (n_time, lon, lat, channels) and chunks time itself
batch = {"x_0": fl(x0), "channel:mean": fl(cm), "channel:std": fl(cs), "output_mean": fl(om), "output_std": fl(os_)}
for n in (4, 8):
    for rev in (False, True):
        o = iu.sampling_from_batch(batch, model, state, num_sampling_steps=n, time_chunk_size=T, time_to_channel=False, reverse_flow=rev)
        out[f"sample_n{n}_{'rev' if rev else 'fwd'}"] = np.asarray(o).reshape(B, T, X, Y, C)

flat = {("p/" + "/".join(str(k.key) for k in kp)): np.asarray(v) for kp, v in jax.tree_util.tree_flatten_with_path(variables["params"])[0]}
dst = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "models", "genfocal", "data", "ref_flow.npz")
np.savez_compressed(dst, x0=x0, x1=x1, cm=cm, cs=cs, om=om, os=os_, u=u, wn=wn, **flat, **out)
print("saved", dst, {k: float(v) for k, v in out.items() if k.startswith("loss")})
