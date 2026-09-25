"""Pure-torch GraphNet building blocks for GraphCast-style message passing.

Original WeatherAI code (InteractionNetwork-style). No torch_geometric / DGL.

Numerics follow DeepMind GraphCast (``DeepTypedGraphNet`` built by
``graphcast.GraphCast``) and are covered by ``tests/parity/graphcast``:

* MLPs are ``Linear → act → … → Linear → LayerNorm``; activation **SiLU**
  ("swish", what GraphCast passes for grid2mesh / mesh / mesh2grid).
* Edge/node MLPs emit *deltas* that are added as residuals outside the MLP.
* Node updates aggregate (sum) the **edge deltas** of incoming edges and
  concatenate ``[node, aggregated]``.

See ``docs/graphcast_parity.md`` for the verified stage-by-stage matrix.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn

Activation = Union[str, Callable[[torch.Tensor], torch.Tensor]]

#: Default MLP activation (GraphCast uses ``activation="swish"``).
DEFAULT_ACTIVATION = "silu"


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Sum ``src`` rows into ``dim_size`` bins specified by ``index`` (1-D)."""
    out = src.new_zeros((dim_size,) + src.shape[1:])
    return out.index_add_(0, index, src)


def _segment_sum_batched(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """(B, E, D) → (B, dim_size, D) sum over edges grouped by ``index``."""
    out = src.new_zeros((src.shape[0], dim_size) + src.shape[2:])
    return out.index_add_(1, index, src)


def _resolve_activation(activation: Activation) -> Callable[[torch.Tensor], torch.Tensor]:
    if callable(activation) and not isinstance(activation, str):
        return activation
    name = str(activation).lower()
    if name in ("relu", "relu_"):
        return torch.relu
    if name in ("silu", "swish"):
        return torch.nn.functional.silu
    if name in ("gelu",):
        return torch.nn.functional.gelu
    raise ValueError(f"Unsupported activation {activation!r}")


class _Activation(nn.Module):
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


class MLP(nn.Module):
    """Haiku-style MLP + optional LayerNorm (GraphCast MLPs).

    Layout matches ``hk.nets.MLP(output_sizes=[hidden]*n_hidden + [out])``
    followed by ``hk.LayerNorm``: Linear → Act → … → Linear → (LayerNorm).
    Default activation is **SiLU** (GraphCast ``activation="swish"``).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        n_hidden: int = 1,
        layer_norm: bool = True,
        activation: Activation = DEFAULT_ACTIVATION,
    ):
        super().__init__()
        if n_hidden < 0:
            raise ValueError(f"n_hidden must be >= 0, got {n_hidden}")
        act_fn = _resolve_activation(activation)
        dims = [in_dim] + [hidden_dim] * n_hidden + [out_dim]
        seq: list[nn.Module] = []
        for i in range(len(dims) - 1):
            seq.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                seq.append(_Activation(act_fn))
        self.net = nn.Sequential(*seq)
        self.norm = nn.LayerNorm(out_dim) if layer_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))


def _as_batched(*xs: torch.Tensor):
    batched = xs[0].ndim == 3
    if batched:
        return True, xs
    return False, tuple(x.unsqueeze(0) for x in xs)


class GraphNetBlock(nn.Module):
    """One InteractionNetwork step on a homogeneous directed edge set.

    Edge update:  Δe = MLP([e, v_src, v_dst]);   e ← e + Δe
    Node update:  Δv = MLP([v, Σ_in m]);         v ← v + Δv

    ``aggregate`` selects the per-edge message ``m`` that is summed:

    * ``"delta"`` (default, DeepMind GraphCast / jraph): ``m = Δe``.
    * ``"updated"``: ``m = e + Δe`` (the residual-summed edge latent). This is
      what NVIDIA PhysicsNeMo's ``MeshNodeBlock`` does inside the GraphCast
      processor; it exists only to pin down that known difference in the
      parity tests and is not the canonical path.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: Optional[int] = None,
        n_hidden: int = 1,
        activation: Activation = DEFAULT_ACTIVATION,
        aggregate: str = "delta",
    ):
        super().__init__()
        if aggregate not in ("delta", "updated"):
            raise ValueError(f"aggregate must be 'delta' or 'updated', got {aggregate!r}")
        h = hidden_dim or latent_dim
        self.aggregate = aggregate
        self.edge_mlp = MLP(3 * latent_dim, latent_dim, h, n_hidden=n_hidden, activation=activation)
        self.node_mlp = MLP(2 * latent_dim, latent_dim, h, n_hidden=n_hidden, activation=activation)

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """nodes (N, D) | (B, N, D); edges (E, D) | (B, E, D); senders/receivers (E,)."""
        batched, (nodes, edges) = _as_batched(nodes, edges)
        N = nodes.shape[1]
        edge_delta = self.edge_mlp(torch.cat([edges, nodes[:, senders], nodes[:, receivers]], dim=-1))
        new_edges = edges + edge_delta
        msg = edge_delta if self.aggregate == "delta" else new_edges
        agg = _segment_sum_batched(msg, receivers, N)
        nodes = nodes + self.node_mlp(torch.cat([nodes, agg], dim=-1))
        if not batched:
            return nodes.squeeze(0), new_edges.squeeze(0)
        return nodes, new_edges


class BipartiteGraphNetBlock(nn.Module):
    """One message-passing step on a bipartite src → dst edge set.

    Matches one GraphCast ``DeepTypedGraphNet`` step on the grid2mesh /
    mesh2grid graphs::

        Δe   = MLP_e([e, src[s], dst[r]]);   e   ← e + Δe
        dst  ← dst + MLP_dst([dst, Σ_r Δe])
        src  ← src + MLP_src(src)            # only if update_src

    Source nodes receive no edges, so their update has no aggregation term
    (GraphCast Grid2Mesh updates grid nodes this way). In Mesh2Grid GraphCast
    also evaluates a mesh-node MLP but discards its output, so
    ``update_src=False`` is exactly equivalent there.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: Optional[int] = None,
        n_hidden: int = 1,
        update_src: bool = True,
        activation: Activation = DEFAULT_ACTIVATION,
    ):
        super().__init__()
        h = hidden_dim or latent_dim
        self.update_src = update_src
        self.edge_mlp = MLP(3 * latent_dim, latent_dim, h, n_hidden=n_hidden, activation=activation)
        self.dst_mlp = MLP(2 * latent_dim, latent_dim, h, n_hidden=n_hidden, activation=activation)
        self.src_mlp = (
            MLP(latent_dim, latent_dim, h, n_hidden=n_hidden, activation=activation)
            if update_src
            else None
        )

    def forward(
        self,
        src_nodes: torch.Tensor,
        dst_nodes: torch.Tensor,
        edges: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ):
        batched, (src_nodes, dst_nodes, edges) = _as_batched(src_nodes, dst_nodes, edges)
        Nd = dst_nodes.shape[1]
        edge_delta = self.edge_mlp(
            torch.cat([edges, src_nodes[:, senders], dst_nodes[:, receivers]], dim=-1)
        )
        agg = _segment_sum_batched(edge_delta, receivers, Nd)
        dst_nodes = dst_nodes + self.dst_mlp(torch.cat([dst_nodes, agg], dim=-1))
        if self.src_mlp is not None:
            src_nodes = src_nodes + self.src_mlp(src_nodes)
        edges = edges + edge_delta
        if not batched:
            return src_nodes.squeeze(0), dst_nodes.squeeze(0), edges.squeeze(0)
        return src_nodes, dst_nodes, edges
