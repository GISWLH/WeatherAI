"""Pure-torch GraphNet building blocks for GraphCast-style message passing.

Original WeatherAI code (InteractionNetwork-style). No torch_geometric / DGL.

MLP + GraphNetBlock numerics are aligned with DeepMind GraphCast / jraph
InteractionNetwork + Haiku MLP→LayerNorm (see ``docs/graphcast_parity.md``):
edge/node MLPs emit *deltas*; node aggregation uses edge deltas **before**
residual add (matching JAX ``DeepTypedGraphNet._process_step``).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn

Activation = Union[str, Callable[[torch.Tensor], torch.Tensor]]


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Sum ``src`` rows into ``dim_size`` bins specified by ``index`` (1-D)."""
    out = src.new_zeros((dim_size,) + src.shape[1:])
    if src.ndim == 1:
        return out.scatter_add_(0, index, src)
    idx = index.view(-1, *([1] * (src.ndim - 1))).expand_as(src)
    return out.scatter_add_(0, idx, src)


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


class MLP(nn.Module):
    """Haiku-style MLP + optional LayerNorm (GraphCast processor MLPs).

    Layout matches ``hk.nets.MLP(output_sizes=[hidden]*n_hidden + [out])`` then
    ``hk.LayerNorm``: Linear → Act → … → Linear → (LayerNorm). Default
    activation is **ReLU** (GraphCast JAX default), not SiLU.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        n_hidden: int = 1,
        layer_norm: bool = True,
        activation: Activation = "relu",
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


class _Activation(nn.Module):
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


class GraphNetBlock(nn.Module):
    """One InteractionNetwork step on a homogeneous directed edge set.

    Edge update:  Δe = MLP([e, v_src, v_dst]);   e ← e + Δe
    Node update:  Δv = MLP([v, agg(Δe_in)]);     v ← v + Δv

    Aggregation uses **edge deltas** (not residual-summed edges), matching
    jraph InteractionNetwork + DeepMind residual outside the MLP.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: Optional[int] = None,
        n_hidden: int = 1,
        activation: Activation = "relu",
    ):
        super().__init__()
        h = hidden_dim or latent_dim
        self.edge_mlp = MLP(
            3 * latent_dim, latent_dim, h, n_hidden=n_hidden, activation=activation
        )
        self.node_mlp = MLP(
            2 * latent_dim, latent_dim, h, n_hidden=n_hidden, activation=activation
        )

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        nodes : (N, D) or (B, N, D)
        edges : (E, D) or (B, E, D)
        senders / receivers : (E,) long
        """
        batched = nodes.ndim == 3
        if not batched:
            nodes = nodes.unsqueeze(0)
            edges = edges.unsqueeze(0)
        B, N, D = nodes.shape
        E = edges.shape[1]

        v_src = nodes[:, senders]  # (B, E, D)
        v_dst = nodes[:, receivers]
        e_in = torch.cat([edges, v_src, v_dst], dim=-1)
        edge_delta = self.edge_mlp(e_in.reshape(B * E, -1)).reshape(B, E, D)

        # Aggregate incoming *deltas* per node (JAX InteractionNetwork order)
        agg = edge_delta.new_zeros(B, N, D)
        for b in range(B):
            agg[b] = scatter_sum(edge_delta[b], receivers, N)
        n_in = torch.cat([nodes, agg], dim=-1)
        node_delta = self.node_mlp(n_in.reshape(B * N, -1)).reshape(B, N, D)

        nodes = nodes + node_delta
        edges = edges + edge_delta

        if not batched:
            return nodes.squeeze(0), edges.squeeze(0)
        return nodes, edges


class BipartiteGraphNetBlock(nn.Module):
    """One message-passing step from src nodes → dst nodes (bipartite).

    Updates edges and **destination** nodes; optionally also updates source
    nodes (Grid2Mesh keeps both latents in the paper). Aggregation uses edge
    deltas before residual, same as ``GraphNetBlock``.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: Optional[int] = None,
        n_hidden: int = 1,
        update_src: bool = True,
        activation: Activation = "relu",
    ):
        super().__init__()
        h = hidden_dim or latent_dim
        self.update_src = update_src
        self.edge_mlp = MLP(
            3 * latent_dim, latent_dim, h, n_hidden=n_hidden, activation=activation
        )
        self.dst_mlp = MLP(
            2 * latent_dim, latent_dim, h, n_hidden=n_hidden, activation=activation
        )
        self.src_mlp = (
            MLP(2 * latent_dim, latent_dim, h, n_hidden=n_hidden, activation=activation)
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
        batched = src_nodes.ndim == 3
        if not batched:
            src_nodes = src_nodes.unsqueeze(0)
            dst_nodes = dst_nodes.unsqueeze(0)
            edges = edges.unsqueeze(0)
        B = src_nodes.shape[0]
        Ns, D = src_nodes.shape[1], src_nodes.shape[2]
        Nd = dst_nodes.shape[1]
        E = edges.shape[1]

        v_src = src_nodes[:, senders]
        v_dst = dst_nodes[:, receivers]
        edge_delta = self.edge_mlp(
            torch.cat([edges, v_src, v_dst], dim=-1).reshape(B * E, -1)
        ).reshape(B, E, D)

        agg_dst = edge_delta.new_zeros(B, Nd, D)
        for b in range(B):
            agg_dst[b] = scatter_sum(edge_delta[b], receivers, Nd)
        dst_nodes = dst_nodes + self.dst_mlp(
            torch.cat([dst_nodes, agg_dst], dim=-1).reshape(B * Nd, -1)
        ).reshape(B, Nd, D)

        if self.src_mlp is not None:
            agg_src = edge_delta.new_zeros(B, Ns, D)
            for b in range(B):
                agg_src[b] = scatter_sum(edge_delta[b], senders, Ns)
            src_nodes = src_nodes + self.src_mlp(
                torch.cat([src_nodes, agg_src], dim=-1).reshape(B * Ns, -1)
            ).reshape(B, Ns, D)

        edges = edges + edge_delta

        if not batched:
            return src_nodes.squeeze(0), dst_nodes.squeeze(0), edges.squeeze(0)
        return src_nodes, dst_nodes, edges
