"""One mesh GraphNet / InteractionNetwork layer parity (tiny synthetic graph)."""

from __future__ import annotations

import numpy as np
import torch

from weatherai.models.graphcast.layers import GraphNetBlock

from ._helpers import (
    ATOL,
    RTOL,
    flatten_haiku_params,
    load_haiku_mlp_into_torch,
    max_abs_rel,
    requires_jax_parity,
)


@requires_jax_parity
def test_one_graphnet_layer_matches_jraph_interaction_network():
    import jax
    import jax.numpy as jnp
    import haiku as hk
    import jraph

    latent = 8
    n_nodes = 12
    rng = np.random.default_rng(0)
    nodes_np = rng.standard_normal((n_nodes, latent)).astype(np.float32)
    senders = np.concatenate([np.arange(n_nodes), np.arange(n_nodes)]).astype(np.int32)
    receivers = np.concatenate(
        [(np.arange(n_nodes) + 1) % n_nodes, (np.arange(n_nodes) - 1) % n_nodes]
    ).astype(np.int32)
    n_edges = len(senders)
    edges_np = rng.standard_normal((n_edges, latent)).astype(np.float32)

    def forward(nodes, edges):
        def edge_update(e, s, r):
            x = jnp.concatenate([e, s, r], axis=-1)
            y = hk.nets.MLP([latent, latent], activation=jax.nn.relu, name="edge_mlp")(x)
            return hk.LayerNorm(-1, True, True, name="edge_ln")(y)

        def node_update(n, rec_msg):
            x = jnp.concatenate([n, rec_msg], axis=-1)
            y = hk.nets.MLP([latent, latent], activation=jax.nn.relu, name="node_mlp")(x)
            return hk.LayerNorm(-1, True, True, name="node_ln")(y)

        net = jraph.InteractionNetwork(
            update_edge_fn=edge_update,
            update_node_fn=node_update,
            aggregate_edges_for_nodes_fn=jraph.segment_sum,
            include_sent_messages_in_node_update=False,
        )
        g = jraph.GraphsTuple(
            nodes=nodes,
            edges=edges,
            senders=senders,
            receivers=receivers,
            n_node=jnp.array([n_nodes]),
            n_edge=jnp.array([n_edges]),
            globals=None,
        )
        out = net(g)
        # Residuals outside MLPs (DeepTypedGraphNet._process_step)
        return g.nodes + out.nodes, g.edges + out.edges

    hk_net = hk.without_apply_rng(hk.transform(forward))
    key = jax.random.PRNGKey(1)
    params = hk_net.init(key, jnp.asarray(nodes_np), jnp.asarray(edges_np))
    yn, ye = [np.asarray(t) for t in hk_net.apply(params, jnp.asarray(nodes_np), jnp.asarray(edges_np))]

    flat = flatten_haiku_params(params)
    block = GraphNetBlock(latent, hidden_dim=latent, n_hidden=1, activation="relu")
    load_haiku_mlp_into_torch(block.edge_mlp, flat, "edge_mlp", "edge_ln")
    load_haiku_mlp_into_torch(block.node_mlp, flat, "node_mlp", "node_ln")

    tn, te = block(
        torch.from_numpy(nodes_np),
        torch.from_numpy(edges_np),
        torch.from_numpy(senders.astype(np.int64)),
        torch.from_numpy(receivers.astype(np.int64)),
    )
    tn_np, te_np = tn.detach().numpy(), te.detach().numpy()

    n_abs, n_rel = max_abs_rel(tn_np, yn)
    e_abs, e_rel = max_abs_rel(te_np, ye)
    assert n_abs < ATOL and e_abs < ATOL, (
        f"GraphNet parity failed: nodes max_abs={n_abs:.3e} edges max_abs={e_abs:.3e}"
    )
    assert np.allclose(tn_np, yn, atol=ATOL, rtol=RTOL)
    assert np.allclose(te_np, ye, atol=ATOL, rtol=RTOL)
    # Store observed errors for humans reading pytest -q -s
    print(f"[parity] GraphNet nodes max_abs={n_abs:.3e} max_rel={n_rel:.3e}")
    print(f"[parity] GraphNet edges max_abs={e_abs:.3e} max_rel={e_rel:.3e}")
