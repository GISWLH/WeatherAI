"""Pure-torch GraphNet building blocks for GraphCast-style message passing.

Original WeatherAI code (InteractionNetwork-style). No torch_geometric / DGL.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Sum ``src`` rows into ``dim_size`` bins specified by ``index`` (1-D)."""
    out = src.new_zeros((dim_size,) + src.shape[1:])
    # index: (E,) → expand for broadcast over feature dims
    if src.ndim == 1:
        return out.scatter_add_(0, index, src)
    idx = index.view(-1, *([1] * (src.ndim - 1))).expand_as(src)
    return out.scatter_add_(0, idx, src)


class MLP(nn.Module):
    """LayerNorm + SiLU MLP used for node/edge updates."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        n_hidden: int = 1,
        layer_norm: bool = True,
    ):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * n_hidden + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(out_dim) if layer_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))


class GraphNetBlock(nn.Module):
    """One InteractionNetwork step on a homogeneous directed edge set.

    Edge update:  e ← e + MLP([e, v_src, v_dst])
    Node update:  v ← v + MLP([v, agg(e_in)])
    """

    def __init__(self, latent_dim: int, hidden_dim: Optional[int] = None, n_hidden: int = 1):
        super().__init__()
        h = hidden_dim or latent_dim
        self.edge_mlp = MLP(3 * latent_dim, latent_dim, h, n_hidden=n_hidden)
        self.node_mlp = MLP(2 * latent_dim, latent_dim, h, n_hidden=n_hidden)

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
        edges = edges + self.edge_mlp(e_in.reshape(B * E, -1)).reshape(B, E, D)

        # Aggregate incoming messages per node
        agg = edges.new_zeros(B, N, D)
        for b in range(B):
            agg[b] = scatter_sum(edges[b], receivers, N)
        n_in = torch.cat([nodes, agg], dim=-1)
        nodes = nodes + self.node_mlp(n_in.reshape(B * N, -1)).reshape(B, N, D)

        if not batched:
            return nodes.squeeze(0), edges.squeeze(0)
        return nodes, edges


class BipartiteGraphNetBlock(nn.Module):
    """One message-passing step from src nodes → dst nodes (bipartite).

    Updates edges and **destination** nodes; optionally also updates source
    nodes (Grid2Mesh keeps both latents in the paper).
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: Optional[int] = None,
        n_hidden: int = 1,
        update_src: bool = True,
    ):
        super().__init__()
        h = hidden_dim or latent_dim
        self.update_src = update_src
        self.edge_mlp = MLP(3 * latent_dim, latent_dim, h, n_hidden=n_hidden)
        self.dst_mlp = MLP(2 * latent_dim, latent_dim, h, n_hidden=n_hidden)
        self.src_mlp = (
            MLP(2 * latent_dim, latent_dim, h, n_hidden=n_hidden) if update_src else None
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
        edges = edges + self.edge_mlp(
            torch.cat([edges, v_src, v_dst], dim=-1).reshape(B * E, -1)
        ).reshape(B, E, D)

        agg_dst = edges.new_zeros(B, Nd, D)
        for b in range(B):
            agg_dst[b] = scatter_sum(edges[b], receivers, Nd)
        dst_nodes = dst_nodes + self.dst_mlp(
            torch.cat([dst_nodes, agg_dst], dim=-1).reshape(B * Nd, -1)
        ).reshape(B, Nd, D)

        if self.src_mlp is not None:
            agg_src = edges.new_zeros(B, Ns, D)
            for b in range(B):
                agg_src[b] = scatter_sum(edges[b], senders, Ns)
            src_nodes = src_nodes + self.src_mlp(
                torch.cat([src_nodes, agg_src], dim=-1).reshape(B * Ns, -1)
            ).reshape(B, Ns, D)

        if not batched:
            return src_nodes.squeeze(0), dst_nodes.squeeze(0), edges.squeeze(0)
        return src_nodes, dst_nodes, edges
