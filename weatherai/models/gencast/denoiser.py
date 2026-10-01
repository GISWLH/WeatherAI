"""Native PyTorch port of the official GenCast denoiser network (``weathernext1_gen/denoiser.py``).

Architecture (explicit modules, readable ``forward``; parameter names mirror the Haiku names so the
official ``.npz`` weights convert 1:1, see :func:`convert_official_params`)::

    noise level σ ──log──▶ Fourier features (cos‖sin, 32 freq) ─▶ MLP(32,16) [gelu-tanh] ─▶ cond (B,16)
    grid features (B,Ng,C) ‖ grid structural feats ──▶ grid2mesh GNN ──▶ latent grid / latent mesh
        encoder MLPs for grid nodes, mesh nodes (zeros‖structural), grid→mesh edges; ONE message-passing step
    latent mesh ──▶ mesh transformer: 16 × [LN → cond-affine → k-hop MHA, LN → cond-affine → FFW(gelu-tanh)]
                                       → LN → cond-affine
    mesh ‖ latent grid ──▶ mesh2grid GNN (edge encoder, 1 step, decoder MLP on grid nodes) ──▶ (B,Ng,out)

Every MLP inside the GNNs is ``Linear → swish → Linear → LayerNorm(no affine) → noise-conditioned
affine`` (``scale = 1 + W·cond``) and the transformer blocks use the same conditioned LayerNorm.

Only the *network* is ported here (the EDM wrapper / sampler live in :mod:`.gencast`). Scope and
verification status: see ``docs/model_status.md`` and ``tests/models/gencast``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .graphs import GenCastGraphs, build_gencast_graphs, khop_attention_mask


# ----------------------------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------------------------
@dataclass
class SparseTransformerConfig:
    attention_k_hop: int = 16
    d_model: int = 512
    num_layers: int = 16
    num_heads: int = 4
    ffw_hidden: int = 2048
    attention_type: str = "dense"  # "dense" (masked SDPA) | "triblockdiag" (banded, memory-light)


@dataclass
class DenoiserConfig:
    """Mirror of the official ``DenoiserArchitectureConfig`` (+ ``NoiseEncoderConfig``)."""

    transformer: SparseTransformerConfig = field(default_factory=SparseTransformerConfig)
    mesh_size: int = 4
    latent_size: int = 512
    hidden_layers: int = 1
    radius_query_fraction_edge_length: float = 0.6
    node_output_size: int = 84
    in_channels: int = 264  # official: 176 input + 4 forcing + 84 noisy-target channels
    noise_base_period: float = 16.0
    noise_num_frequencies: int = 32
    noise_mlp_sizes: Tuple[int, int] = (32, 16)


# ----------------------------------------------------------------------------------------------
# building blocks
# ----------------------------------------------------------------------------------------------
def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """``jax.nn.gelu`` (the official default is the tanh approximation)."""
    return F.gelu(x, approximate="tanh")


def fourier_features(values: torch.Tensor, base_period: float, num_frequencies: int) -> torch.Tensor:
    """Official ``model_utils.fourier_features``: ``[cos(2π k x/T), sin(2π k x/T)]``, k = 1..K."""
    freqs = torch.arange(1, num_frequencies + 1, dtype=values.dtype, device=values.device) / base_period
    ang = values[..., None] * (2 * math.pi * freqs)
    return torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)


class FourierFeaturesMLP(nn.Module):
    """Noise-level encoder: ``log σ`` → Fourier features → ``Linear → gelu → Linear``."""

    def __init__(self, base_period=16.0, num_frequencies=32, sizes=(32, 16), apply_log_first=True):
        super().__init__()
        self.base_period, self.num_frequencies, self.apply_log_first = base_period, num_frequencies, apply_log_first
        self.linear_0 = nn.Linear(2 * num_frequencies, sizes[0])
        self.linear_1 = nn.Linear(sizes[0], sizes[1])

    def forward(self, noise_levels: torch.Tensor) -> torch.Tensor:  # (B,) -> (B, sizes[-1])
        v = torch.log(noise_levels) if self.apply_log_first else noise_levels
        h = fourier_features(v, self.base_period, self.num_frequencies)
        return self.linear_1(gelu_tanh(self.linear_0(h)))


class LinearNormConditioning(nn.Module):
    """``x * (1 + s) + o`` with ``(s, o) = Linear(cond)`` (official ``dense.LinearNormConditioning``)."""

    def __init__(self, features: int, cond_dim: int):
        super().__init__()
        self.linear = nn.Linear(cond_dim, 2 * features)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x (B, N, D), cond (B, cond_dim)
        scale_minus_one, offset = self.linear(cond).unsqueeze(1).chunk(2, dim=-1)
        return x * (scale_minus_one + 1.0) + offset


class ConditionedMLP(nn.Module):
    """``Linear → swish → … → Linear → LayerNorm(no affine) → noise-conditioned affine``.

    Attributes are named like the Haiku modules (``linear_0``, ``linear_1``, ``norm_conditioning``) so
    state-dict keys map directly. ``hidden_layers`` = number of hidden Linear layers (official: 1).
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int, cond_dim: Optional[int], hidden_layers: int = 1,
                 layer_norm: bool = True):
        super().__init__()
        dims = [in_dim] + [hidden] * hidden_layers + [out_dim]
        self.num_linear = len(dims) - 1
        for i in range(self.num_linear):
            setattr(self, f"linear_{i}", nn.Linear(dims[i], dims[i + 1]))
        self.layer_norm = layer_norm
        self.norm_conditioning = LinearNormConditioning(out_dim, cond_dim) if (layer_norm and cond_dim) else None

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        for i in range(self.num_linear):
            x = getattr(self, f"linear_{i}")(x)
            if i < self.num_linear - 1:
                x = F.silu(x)  # "swish"
        if self.layer_norm:
            x = F.layer_norm(x, x.shape[-1:])  # create_scale=False, create_offset=False
            if self.norm_conditioning is not None:
                x = self.norm_conditioning(x, cond)
        return x


def _segment_sum(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """(B,E,D) summed into (B,n,D) by receiver index (float32 accumulation like ``f32_aggregation``)."""
    out = torch.zeros(src.shape[0], n, src.shape[2], dtype=torch.float32, device=src.device)
    out.index_add_(1, index, src.float())
    return out.to(src.dtype)


class Grid2MeshGNN(nn.Module):
    """Official ``grid2mesh_gnn``: embed edges/nodes, one InteractionNetwork step with residuals.

    Edge update  Δe = MLP([e, v_grid[s], v_mesh[r]]);  e ← e + Δe
    Mesh update  Δv = MLP([v_mesh, Σ_r Δe]);            v_mesh ← v_mesh + Δv
    Grid update  Δv = MLP(v_grid)   (grid nodes receive no edges);   v_grid ← v_grid + Δv
    """

    def __init__(self, node_in: int, edge_in: int, d: int, cond_dim: int, hidden_layers: int):
        super().__init__()
        mk = lambda i, o: ConditionedMLP(i, d, o, cond_dim, hidden_layers)  # noqa: E731
        self.encoder_edges_grid2mesh = mk(edge_in, d)
        self.encoder_nodes_grid_nodes = mk(node_in, d)
        self.encoder_nodes_mesh_nodes = mk(node_in, d)
        self.processor_edges_0_grid2mesh = mk(3 * d, d)
        self.processor_nodes_0_grid_nodes = mk(d, d)
        self.processor_nodes_0_mesh_nodes = mk(2 * d, d)

    def forward(self, grid_in, mesh_in, edge_in, senders, receivers, cond):
        grid = self.encoder_nodes_grid_nodes(grid_in, cond)
        mesh = self.encoder_nodes_mesh_nodes(mesh_in, cond)
        edge = self.encoder_edges_grid2mesh(edge_in, cond)
        # --- one message-passing step
        d_edge = self.processor_edges_0_grid2mesh(torch.cat([edge, grid[:, senders], mesh[:, receivers]], -1), cond)
        agg = _segment_sum(d_edge, receivers, mesh.shape[1])
        d_mesh = self.processor_nodes_0_mesh_nodes(torch.cat([mesh, agg], -1), cond)
        d_grid = self.processor_nodes_0_grid_nodes(grid, cond)
        return mesh + d_mesh, grid + d_grid


class Mesh2GridGNN(nn.Module):
    """Official ``mesh2grid_gnn`` (nodes not re-embedded; edges embedded; 1 step; decoder on grid nodes).

    The official step also evaluates a mesh-node MLP whose result is discarded (only grid nodes are
    decoded); its weights are kept (``processor_nodes_0_mesh_nodes``) for checkpoint compatibility but
    it is not evaluated — the output is unchanged.
    """

    def __init__(self, edge_in: int, d: int, cond_dim: int, hidden_layers: int, out_dim: int):
        super().__init__()
        mk = lambda i, o: ConditionedMLP(i, d, o, cond_dim, hidden_layers)  # noqa: E731
        self.encoder_edges_mesh2grid = mk(edge_in, d)
        self.processor_edges_0_mesh2grid = mk(3 * d, d)
        self.processor_nodes_0_grid_nodes = mk(2 * d, d)
        self.processor_nodes_0_mesh_nodes = mk(d, d)  # unused in forward (output discarded upstream)
        self.decoder_nodes_grid_nodes = ConditionedMLP(d, d, out_dim, None, hidden_layers, layer_norm=False)

    def forward(self, mesh, grid, edge_in, senders, receivers, cond):
        edge = self.encoder_edges_mesh2grid(edge_in, cond)
        d_edge = self.processor_edges_0_mesh2grid(torch.cat([edge, mesh[:, senders], grid[:, receivers]], -1), cond)
        agg = _segment_sum(d_edge, receivers, grid.shape[1])
        grid = grid + self.processor_nodes_0_grid_nodes(torch.cat([grid, agg], -1), cond)
        return self.decoder_nodes_grid_nodes(grid)


class MeshTransformerBlock(nn.Module):
    """``x += MHA(cond-LN(x)); x += FFW(cond-LN(x))`` with k-hop masked attention (official ``Block``)."""

    def __init__(self, d: int, heads: int, ffw_hidden: int, cond_dim: int):
        super().__init__()
        self.heads, self.head_dim = heads, d // heads
        self.norm_conditioning = LinearNormConditioning(d, cond_dim)  # before attention
        self.norm_conditioning_1 = LinearNormConditioning(d, cond_dim)  # before FFW
        self.mha_proj_q = nn.Linear(d, d, bias=False)
        self.mha_proj_k = nn.Linear(d, d, bias=False)
        self.mha_proj_v = nn.Linear(d, d, bias=False)
        self.mha_final = nn.Linear(d, d)
        self.ffw_up = nn.Linear(d, ffw_hidden)
        self.ffw_down = nn.Linear(ffw_hidden, d)

    def attention(self, h: torch.Tensor, attn: "MeshAttention") -> torch.Tensor:
        B, N, D = h.shape
        split = lambda t: t.view(B, N, self.heads, self.head_dim).transpose(1, 2)  # noqa: E731  (B,H,N,hd)
        q, k, v = split(self.mha_proj_q(h)), split(self.mha_proj_k(h)), split(self.mha_proj_v(h))
        o = attn(q, k, v)  # (B,H,N,hd)
        return self.mha_final(o.transpose(1, 2).reshape(B, N, D))

    def forward(self, x, cond, attn):
        h = self.norm_conditioning(F.layer_norm(x, x.shape[-1:]), cond)
        x = x + self.attention(h, attn)
        h = self.norm_conditioning_1(F.layer_norm(x, x.shape[-1:]), cond)
        return x + self.ffw_down(gelu_tanh(self.ffw_up(h)))


class MeshAttention(nn.Module):
    """k-hop attention on the (banded) mesh. Holds the mask (dense) or its tri-block-diagonal blocks."""

    def __init__(self, mask: np.ndarray, kind: str = "dense"):
        super().__init__()
        self.kind = kind
        n = mask.shape[0]
        self.n = n
        if kind == "dense":
            self.register_buffer("mask", torch.from_numpy(mask), persistent=False)
        elif kind == "triblockdiag":
            # banded structure from the reverse Cuthill-McKee permutation: split the mask into
            # (diag, super-diag, sub-diag) blocks of size = bandwidth (official ``mask_block_diags``).
            rows, cols = np.nonzero(mask)
            bs = int(max((rows - cols).max(), (cols - rows).max()) + 1)
            nb = -(-n // bs)
            pad = nb * bs - n
            m = np.pad(mask, ((0, pad), (0, pad)))
            blocks = m.reshape(nb, bs, nb, bs)
            idx = np.arange(nb)
            diag = blocks[idx, :, idx, :]
            up = np.concatenate([blocks[idx[:-1], :, idx[:-1] + 1, :], np.zeros((1, bs, bs), bool)], 0)
            low = np.concatenate([np.zeros((1, bs, bs), bool), blocks[idx[1:], :, idx[1:] - 1, :]], 0)
            self.bs, self.nb, self.pad = bs, nb, pad
            self.register_buffer("blk_mask", torch.from_numpy(np.stack([diag, up, low], 0)), persistent=False)
        else:
            raise ValueError(kind)

    def forward(self, q, k, v):  # (B,H,N,hd)
        if self.kind == "dense":
            return F.scaled_dot_product_attention(q, k, v, attn_mask=self.mask)
        B, H, N, hd = q.shape
        bs, nb = self.bs, self.nb
        blk = lambda t: F.pad(t, (0, 0, 0, self.pad)).view(B, H, nb, bs, hd)  # noqa: E731
        q, k, v = blk(q), blk(k), blk(v)
        zero = torch.zeros_like(k[:, :, :1])
        k_up = torch.cat([k[:, :, 1:], zero], 2)  # keys of the next block
        k_lo = torch.cat([zero, k[:, :, :-1]], 2)  # keys of the previous block
        v_up = torch.cat([v[:, :, 1:], zero], 2)
        v_lo = torch.cat([zero, v[:, :, :-1]], 2)
        scale = hd ** -0.5
        neg = torch.finfo(q.dtype).min
        logits = [
            (torch.einsum("bhnqd,bhnkd->bhnqk", q, kk) * scale).masked_fill(~m[None, None], neg)
            for kk, m in zip((k, k_up, k_lo), self.blk_mask)
        ]
        mx = torch.stack([l.amax(-1, keepdim=True) for l in logits]).amax(0)
        ex = [torch.exp(l - mx) for l in logits]
        den = sum(e.sum(-1, keepdim=True) for e in ex)
        out = sum(torch.einsum("bhnqk,bhnkd->bhnqd", e / den, vv) for e, vv in zip(ex, (v, v_up, v_lo)))
        return out.reshape(B, H, nb * bs, hd)[:, :, :N]


class MeshTransformer(nn.Module):
    def __init__(self, cfg: SparseTransformerConfig, cond_dim: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            MeshTransformerBlock(cfg.d_model, cfg.num_heads, cfg.ffw_hidden, cond_dim) for _ in range(cfg.num_layers)
        )
        self.final_norm_conditioning = LinearNormConditioning(cfg.d_model, cond_dim)

    def forward(self, x, cond, attn):
        for blk in self.blocks:
            x = blk(x, cond, attn)
        return self.final_norm_conditioning(F.layer_norm(x, x.shape[-1:]), cond)


# ----------------------------------------------------------------------------------------------
# the denoiser
# ----------------------------------------------------------------------------------------------
class GenCastDenoiser(nn.Module):
    """Official GenCast denoiser network on a lat/lon grid (``(B, Ng, C) , σ(B) -> (B, Ng, out)``).

    ``Ng = H*W`` flattened lat-major. ``C = cfg.in_channels`` is the number of stacked channels
    (inputs ‖ forcings ‖ noisy targets, official: 264). Pass pre-built ``graphs`` or
    ``grid_lat/grid_lon`` (1-D degrees) to build them.
    """

    def __init__(self, cfg: DenoiserConfig, grid_lat: Optional[np.ndarray] = None, grid_lon: Optional[np.ndarray] = None,
                 graphs: Optional[GenCastGraphs] = None):
        super().__init__()
        self.cfg = cfg
        if graphs is None:
            graphs = build_gencast_graphs(cfg.mesh_size, grid_lat, grid_lon, cfg.radius_query_fraction_edge_length)
        self.grid_shape = (len(graphs.grid_lat), len(graphs.grid_lon))
        self.num_grid_nodes, self.num_mesh_nodes = graphs.num_grid_nodes, graphs.num_mesh_nodes
        f32 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
        i64 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.long)  # noqa: E731
        reg = lambda n, t: self.register_buffer(n, t, persistent=False)  # noqa: E731
        reg("grid_node_features", f32(graphs.grid_node_features))
        reg("mesh_node_features", f32(graphs.mesh_node_features))
        for name in ("g2m", "m2g"):
            reg(f"{name}_senders", i64(getattr(graphs, f"{name}_senders")))
            reg(f"{name}_receivers", i64(getattr(graphs, f"{name}_receivers")))
            reg(f"{name}_edge_attr", f32(getattr(graphs, f"{name}_edge_attr")))

        t = cfg.transformer
        self.noise_encoder = FourierFeaturesMLP(cfg.noise_base_period, cfg.noise_num_frequencies, cfg.noise_mlp_sizes)
        cond = cfg.noise_mlp_sizes[-1]
        nf = cfg.in_channels + self.grid_node_features.shape[1]
        ef = self.g2m_edge_attr.shape[1]
        self.grid2mesh_gnn = Grid2MeshGNN(nf, ef, cfg.latent_size, cond, cfg.hidden_layers)
        mask = khop_attention_mask(graphs.mesh_senders, graphs.mesh_receivers, graphs.num_mesh_nodes, t.attention_k_hop)
        self.mesh_attention = MeshAttention(mask, t.attention_type)
        self.mesh_transformer = MeshTransformer(t, cond)
        self.mesh2grid_gnn = Mesh2GridGNN(ef, cfg.latent_size, cond, cfg.hidden_layers, cfg.node_output_size)

    def forward(self, grid_features: torch.Tensor, noise_levels: torch.Tensor) -> torch.Tensor:
        B, Ng, C = grid_features.shape
        assert Ng == self.num_grid_nodes and C == self.cfg.in_channels, (grid_features.shape, self.cfg.in_channels)
        dt = grid_features.dtype
        cond = self.noise_encoder(noise_levels.to(dt))  # (B, 16)
        bcast = lambda a: a.to(dt).unsqueeze(0).expand(B, -1, -1)  # noqa: E731
        grid_in = torch.cat([grid_features, bcast(self.grid_node_features)], -1)
        mesh_in = torch.cat([grid_features.new_zeros(B, self.num_mesh_nodes, C), bcast(self.mesh_node_features)], -1)
        mesh, grid = self.grid2mesh_gnn(grid_in, mesh_in, bcast(self.g2m_edge_attr), self.g2m_senders, self.g2m_receivers, cond)
        mesh = self.mesh_transformer(mesh, cond, self.mesh_attention)
        return self.mesh2grid_gnn(mesh, grid, bcast(self.m2g_edge_attr), self.m2g_senders, self.m2g_receivers, cond)


# ----------------------------------------------------------------------------------------------
# official parameter conversion
# ----------------------------------------------------------------------------------------------
_PREFIX = "params:"


def haiku_key_to_torch(key: str) -> Optional[str]:
    """Map an official ``params:...:{w,b}`` key to the state-dict key of :class:`GenCastDenoiser`."""
    if not key.startswith(_PREFIX):
        return None
    path, leaf = key[len(_PREFIX):].rsplit(":", 1)
    leaf = {"w": "weight", "b": "bias"}[leaf]
    m = re.fullmatch(r"fourier_features_mlp/~/mlp/~/(linear_\d+)", path)
    if m:
        return f"noise_encoder.{m.group(1)}.{leaf}"
    m = re.fullmatch(r"(grid2mesh_gnn|mesh2grid_gnn)/~_networks_builder/(.+)", path)
    if m:
        gnn, rest = m.groups()
        r = re.fullmatch(r"(.+)_mlp/~/(linear_\d+)", rest)
        if r:
            return f"{gnn}.{r.group(1)}.{r.group(2)}.{leaf}"
        r = re.fullmatch(r"(.+)_norm_conditioning/linear", rest)
        if r:
            return f"{gnn}.{r.group(1)}.norm_conditioning.linear.{leaf}"
    m = re.fullmatch(r"mesh_transformer/~/transformer/block_(\d+)/(.+)", path)
    if m:
        i, rest = int(m.group(1)), m.group(2)
        r = re.fullmatch(r"block_\d+_(norm_conditioning(?:_1)?)/linear", rest)
        name = f"{r.group(1)}.linear" if r else rest
        return f"mesh_transformer.blocks.{i}.{name}.{leaf}"
    if path == "mesh_transformer/~/transformer/transformer_final_norm_conditioning/linear":
        return f"mesh_transformer.final_norm_conditioning.linear.{leaf}"
    raise KeyError(f"unmapped official parameter {key!r}")


def convert_official_params(npz: Mapping[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """Official GenCast ``.npz`` (haiku params; linear ``w`` is ``(in, out)``) → torch state dict."""
    sd = {}
    for k in npz.keys():
        tk = haiku_key_to_torch(k)
        if tk is None:
            continue
        a = np.asarray(npz[k])
        sd[tk] = torch.from_numpy(a.T.copy() if tk.endswith(".weight") else a.copy())
    return sd


def config_from_official_npz(npz: Mapping[str, np.ndarray], **overrides) -> DenoiserConfig:
    """Read the architecture config stored in an official checkpoint."""
    g = lambda k: npz["denoiser_architecture_config:" + k].item()  # noqa: E731
    t = lambda k: g("sparse_transformer_config:" + k)  # noqa: E731
    cfg = DenoiserConfig(
        transformer=SparseTransformerConfig(
            attention_k_hop=int(t("attention_k_hop")), d_model=int(t("d_model")), num_layers=int(t("num_layers")),
            num_heads=int(t("num_heads")), ffw_hidden=int(t("ffw_hidden")),
        ),
        mesh_size=int(g("mesh_size")), latent_size=int(g("latent_size")), hidden_layers=int(g("hidden_layers")),
        radius_query_fraction_edge_length=float(g("radius_query_fraction_edge_length")),
        noise_base_period=float(npz["noise_encoder_config:base_period"]),
        noise_num_frequencies=int(npz["noise_encoder_config:num_frequencies"]),
        noise_mlp_sizes=(int(npz["noise_encoder_config:output_sizes:0"]), int(npz["noise_encoder_config:output_sizes:1"])),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def GenCastDenoiser_from_official(npz_path: str, grid_lat, grid_lon, mesh_size: Optional[int] = None,
                                  attention_type: str = "dense", attention_k_hop: Optional[int] = None) -> GenCastDenoiser:
    """Build the denoiser with the official architecture config and load the official weights (strict).

    ``mesh_size`` / ``attention_k_hop`` may be reduced (weights are independent of both) — numerically valid
    for parity tests, but the model is then *not* the trained configuration.
    """
    npz = np.load(npz_path, allow_pickle=False)
    cfg = config_from_official_npz(npz)
    num_levels = len([k for k in npz.files if k.startswith("task_config:pressure_levels:")])
    n_out = npz["params:mesh2grid_gnn/~_networks_builder/decoder_nodes_grid_nodes_mlp/~/linear_1:b"].shape[0]
    n_in = npz["params:grid2mesh_gnn/~_networks_builder/encoder_nodes_grid_nodes_mlp/~/linear_0:w"].shape[0] - 3
    cfg.node_output_size, cfg.in_channels = int(n_out), int(n_in)
    if mesh_size is not None:
        cfg.mesh_size = mesh_size
    cfg.transformer.attention_type = attention_type
    if attention_k_hop is not None:
        cfg.transformer.attention_k_hop = attention_k_hop
    model = GenCastDenoiser(cfg, grid_lat, grid_lon)
    model.load_state_dict(convert_official_params(npz), strict=True)
    return model.eval()


# ----------------------------------------------------------------------------------------------
# official channel layout (xarray <-> stacked tensors), numpy/torch only
# ----------------------------------------------------------------------------------------------
def stack_variables(variables: Mapping[str, "np.ndarray | torch.Tensor"], batch: int, grid_shape: Tuple[int, int]) -> torch.Tensor:
    """Official ``dataset_to_stacked`` channel layout → ``(B, H*W, C)``.

    Variables are concatenated in **alphabetical order**; each contributes its non-(batch,lat,lon) dims
    flattened in the order they appear (``time`` then ``level``). Accepted shapes per variable:
    ``(B,T,L,H,W)`` atmospheric, ``(B,T,H,W)`` surface, ``(B,T)`` time forcing (broadcast over the grid),
    ``(H,W)`` static (broadcast over batch).
    """
    H, W = grid_shape
    chans = []
    for name in sorted(variables):
        v = torch.as_tensor(variables[name])
        if v.ndim == 5:
            v = v.permute(0, 3, 4, 1, 2).reshape(v.shape[0], H, W, -1)
        elif v.ndim == 4:
            v = v.permute(0, 2, 3, 1)
        elif v.ndim == 2 and v.shape == (batch, v.shape[1]) and v.shape != (H, W):
            v = v[:, None, None, :].expand(batch, H, W, v.shape[1])
        elif v.ndim == 2:
            v = v[None, :, :, None].expand(batch, H, W, 1)
        else:
            raise ValueError(f"{name}: unsupported shape {tuple(v.shape)}")
        chans.append(v)
    return torch.cat(chans, -1).reshape(batch, H * W, -1)


def unstack_variables(x: torch.Tensor, template: Mapping[str, Tuple[int, ...]], grid_shape: Tuple[int, int]) -> Dict[str, torch.Tensor]:
    """Inverse for outputs: ``(B,H*W,C)`` → ``{name: (B,T,[L,]H,W)}``; ``template[name]`` = non-grid dims, e.g. ``(1,13)``."""
    B = x.shape[0]
    H, W = grid_shape
    x = x.reshape(B, H, W, -1)
    out, i = {}, 0
    for name in sorted(template):
        dims = tuple(template[name])
        n = int(np.prod(dims))
        v = x[..., i:i + n].reshape(B, H, W, *dims)
        out[name] = v.permute(0, 3, 4, 1, 2) if len(dims) == 2 else v.permute(0, 3, 1, 2)
        i += n
    return out


def denoiser_inputs(inputs: Mapping, forcings: Mapping, noisy_targets: Mapping, grid_shape: Tuple[int, int]) -> torch.Tensor:
    """Official input assembly: ``stack(inputs) ‖ stack(forcings ∪ noisy_targets)`` → ``(B, H*W, C)``.

    ``inputs`` must not contain ``noise_level_encodings`` (that is produced by the noise encoder).
    """
    batch = int(next(iter(noisy_targets.values())).shape[0])
    f = dict(forcings)
    f.update(noisy_targets)
    return torch.cat([stack_variables(inputs, batch, grid_shape), stack_variables(f, batch, grid_shape)], -1)
