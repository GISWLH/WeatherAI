"""Native PyTorch port of the WeatherNext Cyclones (FGN) neural network (``weathernext2/architecture.py``).

Official layout (Haiku, ``architecture.ForwardPass``); the stochasticity of the Functional Generative Network
enters only as a 32-channel standard-normal ``noise`` vector per sample which conditions every normalisation::

    noise (B,32) ──Linear(32,32, no bias)──▶ cond
    grid vars (+ sin lat, sin lon, cos lon) ─▶ grid encoder  MLP ─┐                  ┌▶ mesh transformer (16 × cond-LN, 4-head k-hop MHA, cond-LN, FFW)
    global vars (year progress) + spatial   ─▶ mesh encoder  MLP ─┴▶ grid→mesh GNN ─┘          │
                                                                    (ball query)            mesh latents
    grid latents (updated by grid→mesh GNN) ──────────────────────▶ mesh→grid GNN ◀─────────┘ (closest triangle)
                                                                         ▼
                                                  grid decoder: Linear → swish → Linear(→ all target channels)
                                                  (+ sigmoid(x − 2) on ``cyclone_exists_gaussian_unit_mode``)

Compared with the GenCast denoiser (:mod:`weatherai.models.gencast.denoiser`, whose transformer / attention /
graph code is re-used verbatim) the differences are: noise (not a noise level) conditions the network; the
encoders / decoder are single MLPs over *all* variables (official "split input matmul" == one big Linear whose
rows are per-(variable, time) blocks); the GNN edge MLPs use the "pre-gather matmul" form
``swish(W_e·e + W_s·v_s[senders] (+ W_r·v_r[receivers]) + b) → Linear → LayerNorm → cond-affine``;
grid→mesh has no receiver term; edges are sorted by receiver; targets are predicted directly (no residual).

Parameter names mirror the Haiku structure so the official ``.npz`` converts 1:1 (:func:`convert_official_params`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..gencast.denoiser import (
    ConditionedMLP,
    LinearNormConditioning,
    MeshAttention,
    MeshTransformer,
    SparseTransformerConfig,
    _segment_sum,
    unstack_variables,
)
from ..gencast.graphs import khop_attention_mask
from .graphs import WNCGraphs, build_wnc_graphs

_PFX = "params:multimodality_forward/"


# ----------------------------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------------------------
@dataclass
class WNCConfig:
    """Mirror of the official ``ForwardPass`` kwargs (Mini config) + channel layout of the data."""

    transformer: SparseTransformerConfig = field(default_factory=lambda: SparseTransformerConfig(attention_k_hop=16, d_model=512, num_layers=16, num_heads=4, ffw_hidden=2048))
    mesh_splits: int = 5
    latent_size: int = 512
    hidden_layers: int = 1
    edge_latent_size: int = 32
    noise_channels: int = 32
    radius_fraction: float = 0.6
    grid_in_channels: int = 0  # dynamic grid channels (excluding the 3 spatial ones)
    mesh_in_channels: int = 0  # dynamic mesh channels (global vars; excluding the 3 spatial ones)
    target_layout: Tuple[Tuple[str, int], ...] = ()  # (name, #channels) in output order (alphabetical, as official)
    sigmoid_offsets: Dict[str, float] = field(default_factory=lambda: {"cyclone_exists_gaussian_unit_mode": -2.0})

    @property
    def out_channels(self) -> int:
        return sum(n for _, n in self.target_layout)


# ----------------------------------------------------------------------------------------------
# GNN pieces
# ----------------------------------------------------------------------------------------------
class PreGatherEdgeMLP(nn.Module):
    """Edge update with the official ``pre_gather_matmul`` form.

    ``m = swish(W_e e + (W_s v_s)[senders] (+ (W_r v_r)[receivers]) + b0)`` → ``Linear → LayerNorm → cond-affine``.
    (Equal to an MLP on ``concat(e, v_s[senders], v_r[receivers])`` but the node matmuls run before the gather.)
    """

    def __init__(self, edge_dim: int, node_dim: int, hidden: int, out: int, cond_dim: int, use_receiver: bool):
        super().__init__()
        self.edge_proj = nn.Linear(edge_dim, hidden, bias=False)
        self.sender_proj = nn.Linear(node_dim, hidden, bias=False)
        self.receiver_proj = nn.Linear(node_dim, hidden, bias=False) if use_receiver else None
        self.bias_0 = nn.Parameter(torch.zeros(hidden))
        self.linear_1 = nn.Linear(hidden, out)
        self.norm_conditioning = LinearNormConditioning(out, cond_dim)

    def forward(self, edges, sender_nodes, senders, receiver_nodes, receivers, cond):
        h = self.edge_proj(edges) + self.sender_proj(sender_nodes)[:, senders] + self.bias_0
        if self.receiver_proj is not None:
            h = h + self.receiver_proj(receiver_nodes)[:, receivers]
        h = self.linear_1(F.silu(h))
        return self.norm_conditioning(F.layer_norm(h, h.shape[-1:]), cond)


class GridToMeshGNN(nn.Module):
    """Official ``grid_to_mesh_gnn`` (PointsMeshTypedGraphGNN, points→mesh, ball query, 1 message-passing step).

    e = EdgeEncoder(spatial edge feats);  m = EdgeMLP(e, v_grid[s]);  mesh += NodeMLP([mesh, Σ_r m]);  grid += NodeMLP(grid)
    """

    def __init__(self, d: int, ed: int, cond_dim: int, hidden_layers: int):
        super().__init__()
        self.edge_encoder = ConditionedMLP(4, ed, ed, cond_dim, hidden_layers)
        self.edge_mlp = PreGatherEdgeMLP(ed, d, d, d, cond_dim, use_receiver=False)
        self.mesh_node_mlp = ConditionedMLP(2 * d, d, d, cond_dim, hidden_layers)
        self.point_node_mlp = ConditionedMLP(d, d, d, cond_dim, hidden_layers)

    def forward(self, grid, mesh, edge_attr, senders, receivers, cond):
        e = self.edge_encoder(edge_attr, cond)
        msg = self.edge_mlp(e, grid, senders, None, receivers, cond)
        agg = _segment_sum(msg, receivers, mesh.shape[1])
        mesh = mesh + self.mesh_node_mlp(torch.cat([mesh, agg], -1), cond)
        grid = grid + self.point_node_mlp(grid, cond)
        return mesh, grid


class MeshToGridGNN(nn.Module):
    """Official ``mesh_to_grid_gnn`` (mesh→points, closest triangle, 1 step). Edge MLP has sender *and* receiver terms.

    The official step also updates mesh nodes (``mesh_node_mlp``) but discards them; the weights are kept
    for checkpoint compatibility, the (unused) update is not evaluated.
    """

    def __init__(self, d: int, ed: int, cond_dim: int, hidden_layers: int):
        super().__init__()
        self.edge_encoder = ConditionedMLP(4, ed, ed, cond_dim, hidden_layers)
        self.edge_mlp = PreGatherEdgeMLP(ed, d, d, d, cond_dim, use_receiver=True)
        self.mesh_node_mlp = ConditionedMLP(d, d, d, cond_dim, hidden_layers)  # unused in forward
        self.point_node_mlp = ConditionedMLP(2 * d, d, d, cond_dim, hidden_layers)

    def forward(self, mesh, grid, edge_attr, senders, receivers, cond):
        e = self.edge_encoder(edge_attr, cond)
        msg = self.edge_mlp(e, mesh, senders, grid, receivers, cond)
        agg = _segment_sum(msg, receivers, grid.shape[1])
        return grid + self.point_node_mlp(torch.cat([grid, agg], -1), cond)


# ----------------------------------------------------------------------------------------------
# the network
# ----------------------------------------------------------------------------------------------
class WeatherNextCyclonesNet(nn.Module):
    """``(grid_x (B,Ng,Cg), mesh_x (B,Cm), noise (B,32)) → (B,Ng,C_out)`` on a lat/lon grid (flattened lat-major).

    ``grid_x`` / ``mesh_x`` are the dynamic channels in the official order (see :func:`stack_grid_inputs`,
    :func:`stack_mesh_inputs`); the sin/cos position features are appended internally.
    """

    def __init__(self, cfg: WNCConfig, grid_lat: Optional[np.ndarray] = None, grid_lon: Optional[np.ndarray] = None,
                 graphs: Optional[WNCGraphs] = None):
        super().__init__()
        self.cfg = cfg
        if graphs is None:
            graphs = build_wnc_graphs(cfg.mesh_splits, grid_lat, grid_lon, cfg.radius_fraction)
        self.grid_shape = (len(graphs.base.grid_lat), len(graphs.base.grid_lon))
        self.num_grid_nodes, self.num_mesh_nodes = graphs.num_grid_nodes, graphs.num_mesh_nodes
        f32 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
        i64 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.long)  # noqa: E731
        reg = lambda n, t: self.register_buffer(n, t, persistent=False)  # noqa: E731
        f64 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)  # noqa: E731  (cast to the input dtype in forward)
        reg("grid_spatial", f32(graphs.grid_spatial))
        reg("mesh_spatial", f32(graphs.mesh_spatial))
        for name in ("g2m", "m2g"):
            reg(f"{name}_senders", i64(getattr(graphs, f"{name}_senders")))
            reg(f"{name}_receivers", i64(getattr(graphs, f"{name}_receivers")))
            reg(f"{name}_edge_attr", f64(getattr(graphs, f"{name}_edge_attr")))

        d, c = cfg.latent_size, cfg.noise_channels
        self.noise_encoder = nn.Linear(c, c, bias=False)  # global_norm_conditioning_encoder
        self.grid_encoder = ConditionedMLP(cfg.grid_in_channels + 3, d, d, c, cfg.hidden_layers)
        self.mesh_encoder = ConditionedMLP(cfg.mesh_in_channels + 3, d, d, c, cfg.hidden_layers)
        self.grid_to_mesh_gnn = GridToMeshGNN(d, cfg.edge_latent_size, c, cfg.hidden_layers)
        mask = khop_attention_mask(graphs.base.mesh_senders, graphs.base.mesh_receivers, graphs.num_mesh_nodes, cfg.transformer.attention_k_hop)
        self.mesh_attention = MeshAttention(mask, cfg.transformer.attention_type)
        self.mesh_transformer = MeshTransformer(cfg.transformer, c)
        self.mesh_to_grid_gnn = MeshToGridGNN(d, cfg.edge_latent_size, c, cfg.hidden_layers)
        self.decoder_hidden = nn.Linear(d, d)  # grid_decoder/shared_dense/mlp/linear_0 (+ swish)
        self.decoder_out = nn.Linear(d, cfg.out_channels)  # grid_decoder/split_output_linear (all targets)

    def forward(self, grid_x: torch.Tensor, mesh_x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        B, Ng, _ = grid_x.shape
        assert Ng == self.num_grid_nodes, (grid_x.shape, self.num_grid_nodes)
        dt = grid_x.dtype
        cond = self.noise_encoder(noise.to(dt))  # (B, 32)
        bc = lambda a: a.to(dt).unsqueeze(0).expand(B, -1, -1)  # noqa: E731
        grid = self.grid_encoder(torch.cat([grid_x, bc(self.grid_spatial)], -1), cond)
        mesh_in = torch.cat([mesh_x.to(dt)[:, None, :].expand(-1, self.num_mesh_nodes, -1), bc(self.mesh_spatial)], -1)
        mesh = self.mesh_encoder(mesh_in, cond)
        mesh, grid = self.grid_to_mesh_gnn(grid, mesh, bc(self.g2m_edge_attr), self.g2m_senders, self.g2m_receivers, cond)
        mesh = self.mesh_transformer(mesh, cond, self.mesh_attention)
        grid = self.mesh_to_grid_gnn(mesh, grid, bc(self.m2g_edge_attr), self.m2g_senders, self.m2g_receivers, cond)
        out = self.decoder_out(F.silu(self.decoder_hidden(grid)))
        return self._activations(out)

    def _activations(self, out: torch.Tensor) -> torch.Tensor:
        """Per-variable output activations (official ``per_var_activation_fns``): sigmoid(x + offset)."""
        i = 0
        for name, n in self.cfg.target_layout:
            if name in self.cfg.sigmoid_offsets:
                out = out.clone()
                out[..., i:i + n] = torch.sigmoid(out[..., i:i + n] + self.cfg.sigmoid_offsets[name])
            i += n
        return out


# ----------------------------------------------------------------------------------------------
# official channel layout (names sorted alphabetically, `forcing_*` < `input_*` < `spatial`; per variable: time, then level)
# ----------------------------------------------------------------------------------------------
STATIC_VARS = ("geopotential_at_surface", "land_sea_mask")


def _to_channels(v: torch.Tensor, name: str, B: int, H: int, W: int, static_names: Sequence[str]) -> torch.Tensor:
    if name in static_names:  # (H,W)
        return v[None, :, :, None].expand(B, H, W, 1)
    if v.ndim == 5:  # (B,T,L,H,W) → time-major then level
        return v.permute(0, 3, 4, 1, 2).reshape(B, H, W, -1)
    if v.ndim == 4:  # (B,T,H,W)
        return v.permute(0, 2, 3, 1)
    if v.ndim == 3:  # (B,T,W): official day-progress layout
        return v.permute(0, 2, 1)[:, None, :, :].expand(B, H, W, v.shape[1])
    if v.ndim == 2:  # (B,T) global
        return v[:, None, None, :].expand(B, H, W, v.shape[1])
    raise ValueError(f"{name}: unsupported shape {tuple(v.shape)}")


def stack_grid_inputs(inputs: Mapping, forcings: Mapping, grid_shape: Tuple[int, int], static_names: Sequence[str] = STATIC_VARS) -> torch.Tensor:
    """Dynamic grid channels ``(B, H*W, C)``: forcings (sorted) ‖ inputs (sorted); ``noise`` is excluded."""
    H, W = grid_shape
    chans = []
    for group in (forcings, inputs):  # "forcing_" < "input_"
        for name in sorted(group):
            if name == "noise":
                continue
            v = torch.as_tensor(group[name])
            B = v.shape[0] if name not in static_names else None
            if B is None:
                B = int(next(torch.as_tensor(x).shape[0] for n, x in inputs.items() if n not in static_names))
            chans.append(_to_channels(v, name, B, H, W, static_names))
    B = chans[0].shape[0]
    return torch.cat(chans, -1).reshape(B, H * W, -1)


def stack_mesh_inputs(inputs: Mapping, forcings: Mapping, static_names: Sequence[str] = STATIC_VARS) -> torch.Tensor:
    """Dynamic mesh channels ``(B, C)``: only variables without lat/lon dims, i.e. ``(B,T)`` year-progress."""
    chans = []
    for group in (forcings, inputs):
        for name in sorted(group):
            v = torch.as_tensor(group[name])
            if name != "noise" and name not in static_names and v.ndim == 2:
                chans.append(v)
    return torch.cat(chans, -1)


# ----------------------------------------------------------------------------------------------
# official parameter conversion
# ----------------------------------------------------------------------------------------------
def _mlp_keys(sd, npz, base: str, tgt: str, consumed: set, first_weight: Optional[torch.Tensor] = None):
    """``base`` = Haiku path of a DenseLayer (…/mlp/linear_i, …/normalization/linear_norm_conditioning/linear)."""
    T = lambda a: torch.from_numpy(np.ascontiguousarray(np.asarray(a).T))  # noqa: E731
    for k in list(npz.keys()):
        if not k.startswith(_PFX + base + "/") and not k.startswith(_PFX + base + ":"):
            continue
        rest = k[len(_PFX + base):]
        m = re.fullmatch(r"/mlp/(linear_\d+):([wb])", rest)
        if m:
            sd[f"{tgt}.{m.group(1)}.{'weight' if m.group(2) == 'w' else 'bias'}"] = T(npz[k]) if m.group(2) == "w" else torch.from_numpy(npz[k].copy())
            consumed.add(k)
            continue
        m = re.fullmatch(r"/normalization/linear_norm_conditioning/linear:([wb])", rest)
        if m:
            sd[f"{tgt}.norm_conditioning.linear.{'weight' if m.group(1) == 'w' else 'bias'}"] = T(npz[k]) if m.group(1) == "w" else torch.from_numpy(npz[k].copy())
            consumed.add(k)
    if first_weight is not None:
        sd[f"{tgt}.linear_0.weight"] = first_weight


def _split_input_weight(npz, module: str, consumed: set) -> torch.Tensor:
    """Concatenate the official per-(variable, time) rows of ``{module}/split_input_matmul`` → ``(C_in, out)``.

    Order: variable names sorted (``forcing_*`` < ``input_*`` < ``spatial``; static vars have no ``time=``), within a
    variable the time blocks in increasing time, each block holding its levels. Returns the torch weight ``(out, C_in)``.
    """
    pre = _PFX + module + "/split_input_matmul:w_"
    entries: Dict[str, List[Tuple[int, str]]] = {}
    for k in npz.keys():
        if not k.startswith(pre):
            continue
        s = k[len(pre):]
        m = re.fullmatch(r"(.+?)(?:_level=[\d,]+)?(?:_time=(-?\d+))?", s)
        var = m.group(1)
        if var.startswith("spatial_feature="):
            var = "spatial"
        entries.setdefault(var, []).append((int(m.group(2)) if m.group(2) is not None else 0, k))
    rows = []
    for var in sorted(entries):
        for _, k in sorted(entries[var]):
            rows.append(np.asarray(npz[k]))
            consumed.add(k)
    return torch.from_numpy(np.ascontiguousarray(np.concatenate(rows, 0).T))


def convert_official_params(npz: Mapping[str, np.ndarray], target_layout: Optional[Sequence[Tuple[str, int]]] = None) -> Tuple[Dict[str, torch.Tensor], List[Tuple[str, int]]]:
    """Official WN-C ``.npz`` (Haiku; ``w`` is ``(in, out)``) → (torch state dict, decoder layout ``[(var, n_channels)]``).

    Every ``params:`` entry must be consumed exactly once (raises otherwise).
    """
    T = lambda a: torch.from_numpy(np.ascontiguousarray(np.asarray(a).T))  # noqa: E731
    C = lambda a: torch.from_numpy(np.asarray(a).copy())  # noqa: E731
    sd: Dict[str, torch.Tensor] = {}
    used: set = set()
    keys = {k for k in npz.keys() if k.startswith("params:")}

    k = _PFX + "global_norm_conditioning_encoder/split_input_matmul:w_input_noise_noise_channels=range(32)"
    sd["noise_encoder.weight"] = T(npz[k])
    used.add(k)
    for enc in ("grid_encoder", "mesh_encoder"):
        w = _split_input_weight(npz, enc, used)
        _mlp_keys(sd, npz, f"{enc}/shared_dense", enc, used, first_weight=w)
    for gnn, e_name, n_recv in (("grid_to_mesh_gnn", "points_to_mesh_nodes", False), ("mesh_to_grid_gnn", "mesh_to_points_nodes", True)):
        w = _split_input_weight(npz, f"{gnn}/edge_encoder", used)
        _mlp_keys(sd, npz, f"{gnn}/edge_encoder/shared_dense", f"{gnn}.edge_encoder", used, first_weight=w)
        dg = f"{gnn}/deep_gnn/"
        for part, tgt in (("edge", "edge_proj"), ("sender", "sender_proj"), ("receiver", "receiver_proj")):
            kk = _PFX + dg + f"processor_edges_0_{part}_{e_name}:w"
            if kk in npz:
                sd[f"{gnn}.edge_mlp.{tgt}.weight"] = T(npz[kk])
                used.add(kk)
        kk = _PFX + dg + f"processor_edges_0_{e_name}/mlp/linear_0:b"
        sd[f"{gnn}.edge_mlp.bias_0"] = C(npz[kk])
        used.add(kk)
        tmp: Dict[str, torch.Tensor] = {}
        _mlp_keys(tmp, npz, dg + f"processor_edges_0_{e_name}", f"{gnn}.edge_mlp", used)
        sd.update({a: b for a, b in tmp.items() if "linear_1" in a or "norm_conditioning" in a})
        for node, tgt in (("mesh_nodes", "mesh_node_mlp"), ("point_nodes", "point_node_mlp")):
            _mlp_keys(sd, npz, dg + f"processor_nodes_0_{node}", f"{gnn}.{tgt}", used)
    # transformer (same layout as GenCast)
    for k in sorted(keys):
        m = re.fullmatch(_PFX + r"mesh_transformer/transformer/block_(\d+)/(.+):([wb])", k)
        if m:
            i, rest, leaf = int(m.group(1)), m.group(2), m.group(3)
            r = re.fullmatch(r"block_\d+_(norm_conditioning(?:_1)?)/linear", rest)
            name = f"{r.group(1)}.linear" if r else rest
            a = np.asarray(npz[k])
            sd[f"mesh_transformer.blocks.{i}.{name}.{'weight' if leaf == 'w' else 'bias'}"] = T(a) if leaf == "w" else C(a)
            used.add(k)
        elif k.startswith(_PFX + "mesh_transformer/transformer/transformer_final_norm_conditioning/linear:"):
            leaf = k[-1]
            sd[f"mesh_transformer.final_norm_conditioning.linear.{'weight' if leaf == 'w' else 'bias'}"] = T(npz[k]) if leaf == "w" else C(npz[k])
            used.add(k)
    # decoder
    sd["decoder_hidden.weight"], sd["decoder_hidden.bias"] = T(npz[_PFX + "grid_decoder/shared_dense/mlp/linear_0:w"]), C(npz[_PFX + "grid_decoder/shared_dense/mlp/linear_0:b"])
    used |= {_PFX + "grid_decoder/shared_dense/mlp/linear_0:w", _PFX + "grid_decoder/shared_dense/mlp/linear_0:b"}
    outs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    pre = _PFX + "grid_decoder/split_output_linear:"
    for k in keys:
        if k.startswith(pre + "w_"):
            s = k[len(pre) + 2:]
            m = re.fullmatch(r"(.+?)(?:_level=[\d,]+)?_time=(-?\d+)", s)
            kb = pre + "b_" + s
            outs[m.group(1)] = (np.asarray(npz[k]), np.asarray(npz[kb]))
            used |= {k, kb}
    order = sorted(outs)
    if target_layout is not None:
        assert [n for n, _ in target_layout] == order, (order, target_layout)
    layout = [(n, outs[n][0].shape[1]) for n in order]
    sd["decoder_out.weight"] = torch.from_numpy(np.ascontiguousarray(np.concatenate([outs[n][0] for n in order], 1).T))
    sd["decoder_out.bias"] = torch.from_numpy(np.concatenate([outs[n][1] for n in order], 0).copy())
    missing = keys - used
    if missing:
        raise KeyError(f"unconverted official parameters: {sorted(missing)[:5]} (+{len(missing)})")
    return sd, layout


def config_from_official_npz(npz: Mapping[str, np.ndarray], mesh_splits: Optional[int] = None, attention_type: str = "dense",
                             attention_k_hop: Optional[int] = None) -> WNCConfig:
    """Config of the released WeatherNextCyclones_Mini network (the ``.npz`` carries no architecture config, so the
    constants follow ``weathernext2/configs/WeatherNextCyclones_Mini``; sizes that follow from shapes are read from the weights)."""
    sd_probe: set = set()
    grid_c = _split_input_weight(npz, "grid_encoder", sd_probe).shape[1] - 3
    mesh_c = _split_input_weight(npz, "mesh_encoder", sd_probe).shape[1] - 3
    d = int(npz[_PFX + "grid_encoder/shared_dense/mlp/linear_1:b"].shape[0])
    _, layout = convert_official_params(npz)
    tr = SparseTransformerConfig(attention_k_hop=16 if attention_k_hop is None else attention_k_hop, d_model=d, num_layers=16,
                                 num_heads=4, ffw_hidden=int(npz[_PFX + "mesh_transformer/transformer/block_00/ffw_up:b"].shape[0]),
                                 attention_type=attention_type)
    return WNCConfig(transformer=tr, mesh_splits=5 if mesh_splits is None else mesh_splits, latent_size=d, grid_in_channels=grid_c,
                     mesh_in_channels=mesh_c, target_layout=tuple(layout))


def WeatherNextCyclonesNet_from_official(npz_path: str, grid_lat, grid_lon, mesh_splits: Optional[int] = None,
                                         attention_type: str = "dense", attention_k_hop: Optional[int] = None) -> WeatherNextCyclonesNet:
    """Build the network with the official Mini architecture and load the official weights (strict).

    ``mesh_splits`` / ``attention_k_hop`` may be reduced (weights are independent of both): useful for fast parity tests, but
    then the model is *not* the trained configuration (trained: 5 splits = 10 242 mesh nodes, k-hop 16).
    """
    npz = np.load(npz_path, allow_pickle=False)
    cfg = config_from_official_npz(npz, mesh_splits, attention_type, attention_k_hop)
    sd, _ = convert_official_params(npz)
    model = WeatherNextCyclonesNet(cfg, grid_lat, grid_lon)
    model.load_state_dict(sd, strict=True)
    return model.eval()


def unstack_outputs(x: torch.Tensor, cfg: WNCConfig, grid_shape: Tuple[int, int], num_levels: int = 13) -> Dict[str, torch.Tensor]:
    """``(B,Ng,C)`` → ``{name: (B,1,[L,]H,W)}`` using the decoder layout."""
    tmpl = {n: ((1, c) if c > 1 else (1,)) for n, c in cfg.target_layout}
    return unstack_variables(x, tmpl, grid_shape)
