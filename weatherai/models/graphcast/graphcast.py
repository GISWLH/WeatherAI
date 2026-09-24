"""GraphCast encode–process–decode weather forecast backbone (PyTorch).

Clean WeatherAI-style skeleton inspired by Lam et al., Science 2023 /
arXiv:2212.12794. Independent reimplementation — does **not** copy DeepMind
JAX GraphCast or NVIDIA PhysicsNeMo sources (Apache-2.0).

Tensor API (zoo-consistent)::

    x: (B, C, H, W)  →  y: (B, C, H, W)   # residual next-step prediction

Architecture
------------
1. Embed lat-lon grid channels → latent grid nodes
2. Grid2Mesh encoder (bipartite GNN, 1 step)
3. Mesh processor (multi-layer message passing on multi-mesh)
4. Mesh2Grid decoder (bipartite GNN, 1 step)
5. Grid MLP → ``out_channels`` residual, added to input (channel-truncated if needed)

``mesh_level`` default is 2 (162 nodes) for manageability; the paper uses ~6
(40,962 nodes). See ``docs/graphcast_specs.md``.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import nn

from .layers import MLP, BipartiteGraphNetBlock, GraphNetBlock
from .mesh import build_graphs, mesh_num_nodes


def _to_hw(img_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(img_size, int):
        return (img_size, img_size)
    return (int(img_size[0]), int(img_size[1]))


class GraphCast(nn.Module):
    """Lat-lon grid ↔ icosahedral mesh encode–process–decode forecast model.

    Parameters
    ----------
    img_size :
        ``(H, W)`` lat × lon. Default ``(64, 128)`` (practical smoke grid).
    in_channels / out_channels :
        Channel counts. If ``out_channels`` is None, equals ``in_channels``.
        Residual is applied on ``min(in, out)`` leading channels.
    mesh_level :
        Icosahedral refinement depth. Paper ≈ 6; default 2 for size.
    processor_layers :
        Unshared mesh GNN layers. Paper = 16; default 4.
    hidden_dim :
        Latent / MLP width. Paper latent ≈ 512; default 64.
    g2m_k / m2g_k :
        k-NN bipartite connectivity (paper uses radius queries).
    use_multi_mesh :
        If True, processor edges = union over levels 0..mesh_level.
        If False, only finest-level edges (lite-friendly).
    residual :
        If True, output = input (+ channel align) + predicted delta.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = (64, 128),
        in_channels: int = 69,
        out_channels: Optional[int] = None,
        mesh_level: int = 2,
        processor_layers: int = 4,
        hidden_dim: int = 64,
        g2m_k: int = 4,
        m2g_k: int = 4,
        use_multi_mesh: bool = True,
        residual: bool = True,
        mlp_hidden_layers: int = 1,
    ):
        super().__init__()
        self.img_size = _to_hw(img_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels) if out_channels is not None else self.in_channels
        self.mesh_level = int(mesh_level)
        self.processor_layers = int(processor_layers)
        self.hidden_dim = int(hidden_dim)
        self.residual = bool(residual)
        self.use_multi_mesh = bool(use_multi_mesh)

        graphs = build_graphs(
            img_size=self.img_size,
            mesh_level=self.mesh_level,
            g2m_k=g2m_k,
            m2g_k=m2g_k,
            use_multi_mesh=use_multi_mesh,
        )
        self.num_mesh_nodes = graphs.mesh_nodes.shape[0]
        self.num_grid_nodes = graphs.grid_nodes.shape[0]
        assert self.num_mesh_nodes == mesh_num_nodes(self.mesh_level)
        assert self.num_grid_nodes == self.img_size[0] * self.img_size[1]

        # Static graph buffers (moved with .to(device))
        self.register_buffer("mesh_xyz", torch.from_numpy(graphs.mesh_nodes))
        self.register_buffer("grid_xyz", torch.from_numpy(graphs.grid_nodes))
        self.register_buffer("mesh_senders", torch.from_numpy(graphs.mesh_senders).long())
        self.register_buffer("mesh_receivers", torch.from_numpy(graphs.mesh_receivers).long())
        self.register_buffer("mesh_edge_attr", torch.from_numpy(graphs.mesh_edge_attr))
        self.register_buffer("g2m_senders", torch.from_numpy(graphs.g2m_senders).long())
        self.register_buffer("g2m_receivers", torch.from_numpy(graphs.g2m_receivers).long())
        self.register_buffer("g2m_edge_attr", torch.from_numpy(graphs.g2m_edge_attr))
        self.register_buffer("m2g_senders", torch.from_numpy(graphs.m2g_senders).long())
        self.register_buffer("m2g_receivers", torch.from_numpy(graphs.m2g_receivers).long())
        self.register_buffer("m2g_edge_attr", torch.from_numpy(graphs.m2g_edge_attr))

        D = hidden_dim
        # Node / edge embedders
        # Grid nodes: channels + xyz; mesh nodes: xyz only at encode time
        self.grid_node_embed = MLP(
            self.in_channels + 3, D, D, n_hidden=mlp_hidden_layers, layer_norm=True
        )
        self.mesh_node_embed = MLP(3, D, D, n_hidden=mlp_hidden_layers, layer_norm=True)
        self.g2m_edge_embed = MLP(4, D, D, n_hidden=mlp_hidden_layers, layer_norm=True)
        self.mesh_edge_embed = MLP(4, D, D, n_hidden=mlp_hidden_layers, layer_norm=True)
        self.m2g_edge_embed = MLP(4, D, D, n_hidden=mlp_hidden_layers, layer_norm=True)

        self.grid2mesh = BipartiteGraphNetBlock(
            D, hidden_dim=D, n_hidden=mlp_hidden_layers, update_src=True
        )
        self.processor = nn.ModuleList(
            [GraphNetBlock(D, hidden_dim=D, n_hidden=mlp_hidden_layers) for _ in range(processor_layers)]
        )
        self.mesh2grid = BipartiteGraphNetBlock(
            D, hidden_dim=D, n_hidden=mlp_hidden_layers, update_src=False
        )
        # Final grid head (no LayerNorm on last projection)
        self.grid_out = nn.Sequential(
            nn.Linear(D, D),
            nn.SiLU(),
            nn.Linear(D, self.out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
        B, C, H, W = x.shape
        if (H, W) != self.img_size:
            raise ValueError(f"Spatial {(H, W)} != img_size {self.img_size}")
        if C != self.in_channels:
            raise ValueError(f"Channels C={C} != in_channels={self.in_channels}")

        # (B, H*W, C)
        grid_feat = x.reshape(B, C, H * W).permute(0, 2, 1).contiguous()
        grid_xyz = self.grid_xyz.unsqueeze(0).expand(B, -1, -1)
        mesh_xyz = self.mesh_xyz.unsqueeze(0).expand(B, -1, -1)

        grid_nodes = self.grid_node_embed(torch.cat([grid_feat, grid_xyz], dim=-1))
        mesh_nodes = self.mesh_node_embed(mesh_xyz)

        g2m_e = self.g2m_edge_embed(self.g2m_edge_attr).unsqueeze(0).expand(B, -1, -1)
        mesh_e = self.mesh_edge_embed(self.mesh_edge_attr).unsqueeze(0).expand(B, -1, -1)
        m2g_e = self.m2g_edge_embed(self.m2g_edge_attr).unsqueeze(0).expand(B, -1, -1)

        # Encode: grid → mesh
        grid_nodes, mesh_nodes, g2m_e = self.grid2mesh(
            grid_nodes, mesh_nodes, g2m_e, self.g2m_senders, self.g2m_receivers
        )

        # Process on multi-mesh
        for block in self.processor:
            mesh_nodes, mesh_e = block(
                mesh_nodes, mesh_e, self.mesh_senders, self.mesh_receivers
            )

        # Decode: mesh → grid
        _, grid_nodes, m2g_e = self.mesh2grid(
            mesh_nodes, grid_nodes, m2g_e, self.m2g_senders, self.m2g_receivers
        )

        delta = self.grid_out(grid_nodes)  # (B, Ng, Cout)
        delta = delta.permute(0, 2, 1).reshape(B, self.out_channels, H, W)

        if not self.residual:
            return delta

        if self.out_channels == self.in_channels:
            return x + delta
        # Channel mismatch: residual on shared leading channels
        out = delta.clone()
        n = min(self.in_channels, self.out_channels)
        out[:, :n] = out[:, :n] + x[:, :n]
        return out


def GraphCast_lite(
    img_size: Union[int, Tuple[int, int]] = (32, 64),
    in_channels: int = 4,
    out_channels: Optional[int] = None,
    mesh_level: int = 1,
    processor_layers: int = 2,
    hidden_dim: int = 32,
    g2m_k: int = 3,
    m2g_k: int = 3,
    use_multi_mesh: bool = False,
    residual: bool = True,
    **kwargs,
) -> GraphCast:
    """CPU / ZeroGPU-friendly GraphCast defaults (single mesh, small latent).

    Default ``in_channels=4`` for fast smoke; pass ``in_channels=69`` to match
    FengWu-style 13-level stacks on a tiny grid.
    """
    return GraphCast(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        mesh_level=mesh_level,
        processor_layers=processor_layers,
        hidden_dim=hidden_dim,
        g2m_k=g2m_k,
        m2g_k=m2g_k,
        use_multi_mesh=use_multi_mesh,
        residual=residual,
        **kwargs,
    )
