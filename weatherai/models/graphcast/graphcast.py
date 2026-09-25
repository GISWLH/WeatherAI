"""GraphCast encode–process–decode weather forecast backbone (PyTorch).

WeatherAI reimplementation of Lam et al., Science 2023 / arXiv:2212.12794.
Independent code — does **not** vendor DeepMind JAX GraphCast or NVIDIA
PhysicsNeMo sources (both Apache-2.0; used only as test references).

Tensor API (zoo-consistent)::

    x: (B, C, H, W)  →  y: (B, C, H, W)   # residual next-step prediction

Architecture (stage-wise parity with DeepMind GraphCast, see
``docs/graphcast_parity.md``)
-------------------------------------------------------------------------
1. Embed: grid nodes ``MLP([x, sinlat, coslon, sinlon])``, mesh nodes
   ``MLP([0…0, sinlat, coslon, sinlon])``, each edge set ``MLP([|d|, d])``
   (receiver-local relative positions, normalised by the max edge length).
2. Grid2Mesh: one bipartite InteractionNetwork step over radius-query edges
   (``0.6 ×`` longest finest-mesh edge); grid nodes get ``grid += MLP(grid)``.
3. Processor: ``processor_layers`` unshared InteractionNetwork steps on the
   multi-mesh (edges of all levels 0..mesh_level).
4. Mesh2Grid: one bipartite step over containing-triangle edges, then a grid
   MLP (no LayerNorm) → ``out_channels``; residual-added to the input.

``mesh_level`` default is 2 (162 nodes) for manageability; the paper uses 6
(40,962 nodes).
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

from .layers import DEFAULT_ACTIVATION, MLP, Activation, BipartiteGraphNetBlock, GraphNetBlock
from .mesh import GraphCastGraphs, build_graphs, mesh_num_nodes


def _to_hw(img_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(img_size, int):
        return (img_size, img_size)
    return (int(img_size[0]), int(img_size[1]))


def _long(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(a, dtype=np.int64))


def _f32(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(a, dtype=np.float32))


class GraphCast(nn.Module):
    """Lat-lon grid ↔ icosahedral multi-mesh encode–process–decode model.

    Parameters
    ----------
    img_size :
        ``(H, W)`` lat × lon. Default ``(64, 128)``. Ignored when
        ``grid_lat``/``grid_lon`` or ``graphs`` are given.
    in_channels / out_channels :
        Channel counts. If ``out_channels`` is None, equals ``in_channels``.
        Residual is applied on ``min(in, out)`` leading channels.
    mesh_level :
        Icosahedral refinement depth (paper: 6; default 2).
    processor_layers :
        Unshared mesh GNN layers (paper: 16; default 4).
    hidden_dim :
        Latent / MLP width (paper: 512; default 64).
    use_multi_mesh :
        If True (GraphCast), processor edges = union of levels 0..mesh_level.
        If False, finest-level edges only.
    residual :
        If True, output = input (+ channel align) + predicted delta.
    mlp_hidden_layers :
        Hidden layers per MLP (paper: 1).
    radius_query_fraction_edge_length :
        Grid2Mesh radius as a fraction of the longest finest-mesh edge (0.6).
    grid_lat / grid_lon :
        Optional explicit 1-D grid coordinates in degrees (lat-major nodes);
        default is a cell-centred grid (``mesh.default_grid_lat_lon``).
    m2g_backend :
        Containing-triangle search for Mesh2Grid: ``"numpy"`` (default,
        dependency-free) or ``"trimesh"`` (bit-identical to the DeepMind
        builder incl. tie-breaking on shared triangle edges).
    activation :
        MLP activation, default SiLU (GraphCast ``"swish"``).
    graphs :
        Optional pre-built ``GraphCastGraphs`` (overrides graph arguments).
    g2m_k / m2g_k :
        Deprecated and ignored (former k-NN connectivity).
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = (64, 128),
        in_channels: int = 69,
        out_channels: Optional[int] = None,
        mesh_level: int = 2,
        processor_layers: int = 4,
        hidden_dim: int = 64,
        g2m_k: Optional[int] = None,
        m2g_k: Optional[int] = None,
        use_multi_mesh: bool = True,
        residual: bool = True,
        mlp_hidden_layers: int = 1,
        radius_query_fraction_edge_length: float = 0.6,
        grid_lat: Optional[Sequence[float]] = None,
        grid_lon: Optional[Sequence[float]] = None,
        m2g_backend: str = "numpy",
        activation: Activation = DEFAULT_ACTIVATION,
        graphs: Optional[GraphCastGraphs] = None,
    ):
        super().__init__()
        if g2m_k is not None or m2g_k is not None:
            warnings.warn(
                "GraphCast(g2m_k=..., m2g_k=...) is deprecated and ignored: Grid2Mesh uses "
                "a radius query and Mesh2Grid containing-triangle connectivity.",
                DeprecationWarning,
                stacklevel=2,
            )
        if graphs is None:
            graphs = build_graphs(
                img_size=_to_hw(img_size),
                mesh_level=int(mesh_level),
                use_multi_mesh=use_multi_mesh,
                radius_query_fraction_edge_length=radius_query_fraction_edge_length,
                grid_lat=None if grid_lat is None else np.asarray(grid_lat),
                grid_lon=None if grid_lon is None else np.asarray(grid_lon),
                m2g_backend=m2g_backend,
            )
        self.graphs_meta = dict(query_radius=graphs.query_radius, m2g_backend=m2g_backend)
        self.img_size = tuple(graphs.img_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels) if out_channels is not None else self.in_channels
        self.mesh_level = int(graphs.mesh_level)
        self.processor_layers = int(processor_layers)
        self.hidden_dim = int(hidden_dim)
        self.residual = bool(residual)
        self.use_multi_mesh = bool(use_multi_mesh)

        self.num_mesh_nodes = graphs.mesh_nodes.shape[0]
        self.num_grid_nodes = graphs.grid_nodes.shape[0]
        assert self.num_mesh_nodes == mesh_num_nodes(self.mesh_level)
        assert self.num_grid_nodes == self.img_size[0] * self.img_size[1]

        # Static graph buffers (moved with .to(device)).
        self.register_buffer("mesh_xyz", _f32(graphs.mesh_nodes))
        self.register_buffer("grid_xyz", _f32(graphs.grid_nodes))
        self.register_buffer("mesh_node_features", _f32(graphs.mesh_node_features))
        self.register_buffer("grid_node_features", _f32(graphs.grid_node_features))
        for name in ("mesh", "g2m", "m2g"):
            self.register_buffer(f"{name}_senders", _long(getattr(graphs, f"{name}_senders")))
            self.register_buffer(f"{name}_receivers", _long(getattr(graphs, f"{name}_receivers")))
            self.register_buffer(f"{name}_edge_attr", _f32(getattr(graphs, f"{name}_edge_attr")))

        D, nh, act = self.hidden_dim, int(mlp_hidden_layers), activation
        nf = self.grid_node_features.shape[1]
        ef = self.g2m_edge_attr.shape[1]
        mk = lambda i, o, ln=True: MLP(i, o, D, n_hidden=nh, layer_norm=ln, activation=act)
        self.grid_node_embed = mk(self.in_channels + nf, D)
        # GraphCast feeds mesh nodes [zeros(in_channels), structural features].
        self.mesh_node_embed = mk(self.in_channels + nf, D)
        self.g2m_edge_embed = mk(ef, D)
        self.mesh_edge_embed = mk(ef, D)
        self.m2g_edge_embed = mk(ef, D)

        self.grid2mesh = BipartiteGraphNetBlock(D, D, nh, update_src=True, activation=act)
        self.processor = nn.ModuleList(
            [GraphNetBlock(D, D, nh, activation=act) for _ in range(self.processor_layers)]
        )
        self.mesh2grid = BipartiteGraphNetBlock(D, D, nh, update_src=False, activation=act)
        self.grid_out = mk(D, self.out_channels, ln=False)

    # ------------------------------------------------------------------ stages
    def run_grid2mesh(self, grid_inputs: torch.Tensor):
        """Embed + Grid2Mesh. ``grid_inputs`` (B, Ng, C) lat-major.

        Returns ``(grid_latent, mesh_latent, g2m_edge_latent)``.
        """
        B = grid_inputs.shape[0]
        gf = self.grid_node_features.to(grid_inputs.dtype).unsqueeze(0).expand(B, -1, -1)
        mf = self.mesh_node_features.to(grid_inputs.dtype).unsqueeze(0).expand(B, -1, -1)
        zeros = grid_inputs.new_zeros(B, self.num_mesh_nodes, grid_inputs.shape[-1])
        grid = self.grid_node_embed(torch.cat([grid_inputs, gf], dim=-1))
        mesh = self.mesh_node_embed(torch.cat([zeros, mf], dim=-1))
        e = self.g2m_edge_embed(self.g2m_edge_attr.to(grid_inputs.dtype)).unsqueeze(0).expand(B, -1, -1)
        return self.grid2mesh(grid, mesh, e, self.g2m_senders, self.g2m_receivers)

    def run_processor(self, mesh_latent: torch.Tensor):
        """Multi-mesh processor. Returns ``(mesh_latent, mesh_edge_latent)``."""
        B = mesh_latent.shape[0]
        e = self.mesh_edge_embed(self.mesh_edge_attr.to(mesh_latent.dtype)).unsqueeze(0).expand(B, -1, -1)
        for block in self.processor:
            mesh_latent, e = block(mesh_latent, e, self.mesh_senders, self.mesh_receivers)
        return mesh_latent, e

    def run_mesh2grid(self, mesh_latent: torch.Tensor, grid_latent: torch.Tensor):
        """Mesh2Grid + output MLP. Returns ``(grid_outputs (B,Ng,Cout), grid_latent, m2g_edge_latent)``."""
        B = mesh_latent.shape[0]
        e = self.m2g_edge_embed(self.m2g_edge_attr.to(mesh_latent.dtype)).unsqueeze(0).expand(B, -1, -1)
        _, grid_latent, e = self.mesh2grid(mesh_latent, grid_latent, e, self.m2g_senders, self.m2g_receivers)
        return self.grid_out(grid_latent), grid_latent, e

    def predict_grid(self, grid_inputs: torch.Tensor) -> torch.Tensor:
        """(B, Ng, C) → (B, Ng, Cout) network output (no residual)."""
        grid, mesh, _ = self.run_grid2mesh(grid_inputs)
        mesh, _ = self.run_processor(mesh)
        out, _, _ = self.run_mesh2grid(mesh, grid)
        return out

    # ----------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
        B, C, H, W = x.shape
        if (H, W) != self.img_size:
            raise ValueError(f"Spatial {(H, W)} != img_size {self.img_size}")
        if C != self.in_channels:
            raise ValueError(f"Channels C={C} != in_channels={self.in_channels}")

        grid_inputs = x.reshape(B, C, H * W).permute(0, 2, 1)  # lat-major nodes
        delta = self.predict_grid(grid_inputs)
        delta = delta.permute(0, 2, 1).reshape(B, self.out_channels, H, W)

        if not self.residual:
            return delta
        if self.out_channels == self.in_channels:
            return x + delta
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
        use_multi_mesh=use_multi_mesh,
        residual=residual,
        **kwargs,
    )
