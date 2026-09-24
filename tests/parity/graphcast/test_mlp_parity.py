"""MLP + LayerNorm parity: identical Haiku weights → Torch output match."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from weatherai.models.graphcast.layers import MLP

from ._helpers import (
    ATOL,
    RTOL,
    flatten_haiku_params,
    load_haiku_mlp_into_torch,
    max_abs_rel,
    requires_jax_parity,
)


@requires_jax_parity
@pytest.mark.parametrize("batch,in_dim,hidden,out_dim", [(4, 10, 16, 8), (2, 24, 8, 8)])
def test_mlp_relu_layernorm_matches_haiku(batch, in_dim, hidden, out_dim):
    import jax
    import jax.numpy as jnp
    import haiku as hk

    def make_fn(x):
        y = hk.nets.MLP(
            output_sizes=[hidden, out_dim],
            activation=jax.nn.relu,
            name="parity_mlp",
        )(x)
        return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="parity_ln")(y)

    net = hk.without_apply_rng(hk.transform(make_fn))
    key = jax.random.PRNGKey(0)
    x_np = np.random.default_rng(0).standard_normal((batch, in_dim)).astype(np.float32)
    params = net.init(key, jnp.asarray(x_np))
    y_jax = np.asarray(net.apply(params, jnp.asarray(x_np)))

    flat = flatten_haiku_params(params)
    torch_mlp = MLP(in_dim, out_dim, hidden, n_hidden=1, layer_norm=True, activation="relu")
    load_haiku_mlp_into_torch(torch_mlp, flat, "parity_mlp", "parity_ln")
    y_t = torch_mlp(torch.from_numpy(x_np)).detach().numpy()

    abs_e, rel_e = max_abs_rel(y_t, y_jax)
    assert abs_e < ATOL and rel_e < RTOL * 10, (
        f"MLP parity failed: max_abs={abs_e:.3e} max_rel={rel_e:.3e}"
    )
    assert np.allclose(y_t, y_jax, atol=ATOL, rtol=RTOL)
