"""Reference runners for three-way GraphCast parity (tests only, not packaged).

* DeepMind JAX GraphCast (``/workspace/tmp/graphcast-jax``): the real
  ``weathernext1_graph.graphcast.GraphCast`` stage methods, tiny config,
  randomised parameters.
* NVIDIA PhysicsNeMo GraphCast modules (``physicsnemo`` + PyG backend on CPU):
  ``GraphCastEncoderEmbedder`` / ``MeshGraphEncoder`` / ``GraphCastProcessor`` /
  ``GraphCastDecoderEmbedder`` / ``MeshGraphDecoder`` / ``MeshGraphMLP``.
* WeatherAI ``GraphCast``.

The same (JAX) parameters are mapped into both torch implementations.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from ._helpers import ensure_jax_ref_on_path, flatten_haiku_params

SURFACE_TARGETS = ("2m_temperature", "mean_sea_level_pressure", "10m_u_component_of_wind")


@dataclass(frozen=True)
class Case:
    name: str
    grid: str  # "4x8" | "6x12" | "poles5x8"
    mesh_level: int
    latent: int
    n_steps: int
    n_in: int = 3
    n_out: int = 2
    seed: int = 0
    dtype: str = "float32"

    def lat_lon(self) -> Tuple[np.ndarray, np.ndarray]:
        spec = self.grid
        centred_lon = spec.endswith("c")  # "6x12c": cell-centred longitudes too
        spec = spec.rstrip("c")
        if spec.startswith("poles"):
            h, w = (int(v) for v in spec[5:].split("x"))
            lat = np.linspace(-90.0, 90.0, h)
        else:
            h, w = (int(v) for v in spec.split("x"))
            lat = -90.0 + (np.arange(h) + 0.5) * 180.0 / h
        lon = (np.arange(w) + (0.5 if centred_lon else 0.0)) * 360.0 / w
        return lat.astype(np.float32), lon.astype(np.float32)

    @property
    def num_grid(self) -> int:
        lat, lon = self.lat_lon()
        return lat.size * lon.size

    def inputs(self) -> np.ndarray:
        rng = np.random.default_rng(1000 + self.seed)
        return rng.standard_normal((self.num_grid, self.n_in)).astype(self.dtype)

    def __str__(self) -> str:  # pytest ids
        return self.name + ("-fp64" if self.dtype == "float64" else "")


def case(grid: str, level: int, latent: int, steps: int, **kw) -> Case:
    return Case(f"{grid}-L{level}-D{latent}-N{steps}", grid, level, latent, steps, **kw)


def _randomise(params, seed: int, dtype: str = "float32"):
    """Replace Haiku params by seeded random values (incl. biases / LN)."""
    rng = np.random.default_rng(seed)
    out = {}
    for mod, leaves in params.items():
        out[mod] = {}
        for k, v in leaves.items():
            shape = np.shape(v)
            if k == "w":
                val = rng.standard_normal(shape) / np.sqrt(shape[0])
            elif k == "scale":
                val = 1.0 + 0.2 * rng.standard_normal(shape)
            else:  # b / offset
                val = 0.2 * rng.standard_normal(shape)
            # Draw in float32 so fp32 / fp64 runs use the same parameter values.
            out[mod][k] = np.asarray(np.asarray(val, dtype=np.float32), dtype=dtype)
    return out


def _edge_arrays(tg, name):
    key = tg.edge_key_by_name(name)
    e = tg.edges[key]
    return (np.asarray(e.indices.senders), np.asarray(e.indices.receivers),
            np.asarray(e.features, dtype=np.float32))


@functools.lru_cache(maxsize=None)
def jax_reference(c: Case) -> Dict[str, np.ndarray]:
    """Run the real JAX GraphCast stages; return graphs, params and activations."""
    ensure_jax_ref_on_path()
    import haiku as hk
    import jax
    from weathernext.weathernext1_graph import graphcast as gcm

    mc = gcm.ModelConfig(resolution=0, mesh_size=c.mesh_level, latent_size=c.latent,
                         gnn_msg_steps=c.n_steps, hidden_layers=1,
                         radius_query_fraction_edge_length=0.6)
    tc = gcm.TaskConfig(input_variables=("2m_temperature",),
                        target_variables=SURFACE_TARGETS[: c.n_out], forcing_variables=(),
                        pressure_levels=(), input_duration="12h")
    lat, lon = c.lat_lon()
    graphs: Dict[str, np.ndarray] = {}

    def fwd(x):
        g = gcm.GraphCast(mc, tc)
        g._init_mesh_properties()
        g._init_grid_properties(grid_lat=lat, grid_lon=lon)
        g._grid2mesh_graph_structure = g._init_grid2mesh_graph()
        g._mesh_graph_structure = g._init_mesh_graph()
        g._mesh2grid_graph_structure = g._init_mesh2grid_graph()
        if not graphs:
            for nm, tg in (("g2m", g._grid2mesh_graph_structure), ("mesh", g._mesh_graph_structure),
                           ("m2g", g._mesh2grid_graph_structure)):
                key = {"g2m": "grid2mesh", "mesh": "mesh", "m2g": "mesh2grid"}[nm]
                s, r, f = _edge_arrays(tg, key)
                graphs[f"{nm}_senders"], graphs[f"{nm}_receivers"], graphs[f"{nm}_edge_attr"] = s, r, f
                graphs[f"{nm}_edge_attr_raw"] = np.asarray(tg.edges[tg.edge_key_by_name(key)].features)
            g2m = g._grid2mesh_graph_structure
            graphs["grid_node_features"] = np.asarray(g2m.nodes["grid_nodes"].features, np.float32)
            graphs["mesh_node_features"] = np.asarray(g2m.nodes["mesh_nodes"].features, np.float32)
            graphs["grid_node_features_raw"] = np.asarray(g2m.nodes["grid_nodes"].features)
            graphs["mesh_node_features_raw"] = np.asarray(g2m.nodes["mesh_nodes"].features)
        cap = {}

        def wrap(name, mod):
            def call(graph):
                out = mod(graph)
                cap[name] = out
                return out
            return call

        if c.dtype == "float64":
            # Debug aid only: GraphCast builds grid2mesh with f32_aggregation=True
            # (sums messages in float32 even under x64). Disable it so the fp64
            # comparison measures implementation error, not that cast.
            g._grid2mesh_gnn._f32_aggregation = False
        g._grid2mesh_gnn = wrap("g2m", g._grid2mesh_gnn)
        g._mesh_gnn = wrap("mesh", g._mesh_gnn)
        g._mesh2grid_gnn = wrap("m2g", g._mesh2grid_gnn)
        mesh_g2m, grid_g2m = g._run_grid2mesh_gnn(x[:, None])
        mesh_proc = g._run_mesh_gnn(mesh_g2m)
        out = g._run_mesh2grid_gnn(mesh_proc, grid_g2m)
        return dict(
            mesh_g2m=mesh_g2m[:, 0], grid_g2m=grid_g2m[:, 0],
            g2m_edges=cap["g2m"].edges[cap["g2m"].edge_key_by_name("grid2mesh")].features[:, 0],
            mesh_proc=mesh_proc[:, 0],
            mesh_edges=cap["mesh"].edges[cap["mesh"].edge_key_by_name("mesh")].features[:, 0],
            m2g_edges=cap["m2g"].edges[cap["m2g"].edge_key_by_name("mesh2grid")].features[:, 0],
            out=out[:, 0],
        )

    import contextlib

    x = c.inputs()
    t = hk.transform(fwd)
    ctx = contextlib.nullcontext()
    if c.dtype == "float64":  # debug aid: whole JAX network in float64
        if hasattr(jax, "enable_x64"):  # config-state context manager (new JAX)
            ctx = jax.enable_x64(True)
        else:  # older JAX
            from jax.experimental import enable_x64

            ctx = enable_x64()
    with ctx:
        params = _randomise(t.init(jax.random.PRNGKey(c.seed), x), c.seed, c.dtype)
        acts = jax.device_get(t.apply(params, None, x))
    res = {k: np.asarray(v, c.dtype) for k, v in acts.items()}
    res.update(graphs)
    res["x"] = x
    res["params"] = flatten_haiku_params(params)
    return res


# ----------------------------------------------------------------- weights
_G2M, _MESH, _M2G = (f"{n}/~_networks_builder" for n in ("grid2mesh_gnn", "mesh_gnn", "mesh2grid_gnn"))


def haiku_mlp(flat, prefix: str, ln: bool = True):
    """Return [(W_torch(out,in), b), ...], (scale, offset)|None for one Haiku MLP."""
    lins = []
    i = 0
    while f"{prefix}_mlp/~/linear_{i}/w" in flat:
        lins.append((flat[f"{prefix}_mlp/~/linear_{i}/w"].T.copy(), flat[f"{prefix}_mlp/~/linear_{i}/b"].copy()))
        i += 1
    norm = (flat[f"{prefix}_layer_norm/scale"], flat[f"{prefix}_layer_norm/offset"]) if ln else None
    return lins, norm


def _copy_into(linears, norm_mod, spec):
    import torch

    lins, norm = spec
    assert len(linears) == len(lins), (len(linears), len(lins))
    with torch.no_grad():
        for mod, (w, b) in zip(linears, lins):
            assert tuple(mod.weight.shape) == w.shape, (tuple(mod.weight.shape), w.shape)
            mod.weight.copy_(torch.from_numpy(w))
            mod.bias.copy_(torch.from_numpy(b))
        if norm is not None:
            norm_mod.weight.copy_(torch.from_numpy(np.asarray(norm[0])))
            norm_mod.bias.copy_(torch.from_numpy(np.asarray(norm[1])))


def _linears(seq):
    import torch

    return [m for m in seq if isinstance(m, torch.nn.Linear)]


def _norm(seq):
    import torch

    ns = [m for m in seq if isinstance(m, torch.nn.LayerNorm)]
    return ns[0] if ns else None


def load_ours(model, flat, n_steps: int) -> None:
    """Map JAX GraphCast params into ``weatherai...GraphCast``."""
    def put(mlp, prefix, ln=True):
        _copy_into(_linears(mlp.net), mlp.norm if ln else None, haiku_mlp(flat, prefix, ln))

    put(model.grid_node_embed, f"{_G2M}/encoder_nodes_grid_nodes")
    put(model.mesh_node_embed, f"{_G2M}/encoder_nodes_mesh_nodes")
    put(model.g2m_edge_embed, f"{_G2M}/encoder_edges_grid2mesh")
    put(model.grid2mesh.edge_mlp, f"{_G2M}/processor_edges_0_grid2mesh")
    put(model.grid2mesh.dst_mlp, f"{_G2M}/processor_nodes_0_mesh_nodes")
    put(model.grid2mesh.src_mlp, f"{_G2M}/processor_nodes_0_grid_nodes")
    put(model.mesh_edge_embed, f"{_MESH}/encoder_edges_mesh")
    assert len(model.processor) == n_steps
    for i, blk in enumerate(model.processor):
        put(blk.edge_mlp, f"{_MESH}/processor_edges_{i}_mesh")
        put(blk.node_mlp, f"{_MESH}/processor_nodes_{i}_mesh_nodes")
    put(model.m2g_edge_embed, f"{_M2G}/encoder_edges_mesh2grid")
    put(model.mesh2grid.edge_mlp, f"{_M2G}/processor_edges_0_mesh2grid")
    put(model.mesh2grid.dst_mlp, f"{_M2G}/processor_nodes_0_grid_nodes")
    put(model.grid_out, f"{_M2G}/decoder_nodes_grid_nodes", ln=False)


def ours_model(c: Case, ref, m2g_backend: str = "trimesh", aggregate: str = "delta"):
    import torch  # noqa: F401
    from weatherai.models.graphcast import GraphCast

    lat, lon = c.lat_lon()
    torch_seed(c.seed)
    model = GraphCast(in_channels=c.n_in, out_channels=c.n_out, mesh_level=c.mesh_level,
                      processor_layers=c.n_steps, hidden_dim=c.latent, grid_lat=lat,
                      grid_lon=lon, m2g_backend=m2g_backend, residual=False)
    for blk in model.processor:
        blk.aggregate = aggregate
    load_ours(model, ref["params"], c.n_steps)
    if c.dtype == "float64":
        model = model.double()
        # Debug aid: use the reference's full-precision structural features
        # (JAX keeps them in float64 and casts per run dtype).
        import torch

        for k in ("g2m_edge_attr", "mesh_edge_attr", "m2g_edge_attr", "grid_node_features", "mesh_node_features"):
            if f"{k}_raw" in ref:
                setattr(model, k, torch.from_numpy(np.asarray(ref[f"{k}_raw"], dtype=np.float64)))
    return model.eval()


def torch_seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)


# ------------------------------------------------------------ PhysicsNeMo
def _prepare_nv_env() -> None:
    # PhysicsNeMo wraps small helpers (e.g. ``sum_edge_node_feat``) in
    # ``torch.compile``; the box has no C++ toolchain, so run them eagerly.
    # This only disables compilation (same maths).
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    import torch._dynamo

    torch._dynamo.config.disable = True  # torch may already be imported


def nv_permute_node_mlp(spec, d_node: int):
    """JAX node MLP input is [node, agg]; PhysicsNeMo concatenates [agg, node]."""
    lins, norm = spec
    w0, b0 = lins[0]
    w0 = np.concatenate([w0[:, d_node:], w0[:, :d_node]], axis=1)
    return [(w0, b0)] + lins[1:], norm


def _put_nv_mlp(nv_mlp, spec):
    """PhysicsNeMo ``MeshGraphMLP`` / ``MeshGraphEdgeMLPConcat`` / ``...Sum``."""
    import torch

    if hasattr(nv_mlp, "lin_efeat"):  # concat trick: split first linear
        lins, norm = spec
        w0, b0 = lins[0]
        de, ds = nv_mlp.efeat_dim, nv_mlp.src_dim
        with torch.no_grad():
            nv_mlp.lin_efeat.copy_(torch.from_numpy(w0[:, :de].copy()))
            nv_mlp.lin_src.copy_(torch.from_numpy(w0[:, de:de + ds].copy()))
            nv_mlp.lin_dst.copy_(torch.from_numpy(w0[:, de + ds:].copy()))
            nv_mlp.bias.copy_(torch.from_numpy(b0))
        _copy_into(_linears(nv_mlp.model), _norm(nv_mlp.model), (lins[1:], norm))
    else:
        _copy_into(_linears(nv_mlp.model), _norm(nv_mlp.model), spec)


class NVStages:
    """PhysicsNeMo GraphCast building blocks configured to match JAX GraphCast."""

    def __init__(self, c: Case, ref, do_concat_trick: bool = False):
        _prepare_nv_env()
        import torch
        from physicsnemo.models.graphcast.graph_cast_processor import GraphCastProcessor
        from physicsnemo.nn.module.gnn_layers.embedder import (
            GraphCastDecoderEmbedder, GraphCastEncoderEmbedder)
        from physicsnemo.nn.module.gnn_layers.mesh_graph_decoder import MeshGraphDecoder
        from physicsnemo.nn.module.gnn_layers.mesh_graph_encoder import MeshGraphEncoder
        from physicsnemo.nn.module.gnn_layers.mesh_graph_mlp import MeshGraphMLP

        D, flat = c.latent, ref["params"]
        act = torch.nn.SiLU()
        common = dict(hidden_dim=D, hidden_layers=1, activation_fn=act, norm_type="LayerNorm")
        self.c = c
        self.enc_embed = GraphCastEncoderEmbedder(
            input_dim_grid_nodes=c.n_in + 3, input_dim_mesh_nodes=3, input_dim_edges=4,
            output_dim=D, **common)
        self.encoder = MeshGraphEncoder(
            aggregation="sum", input_dim_src_nodes=D, input_dim_dst_nodes=D, input_dim_edges=D,
            output_dim_src_nodes=D, output_dim_dst_nodes=D, output_dim_edges=D,
            do_concat_trick=do_concat_trick, **common)
        self.processor = GraphCastProcessor(
            aggregation="sum", processor_layers=c.n_steps, input_dim_nodes=D, input_dim_edges=D,
            do_concat_trick=do_concat_trick, **common)
        self.dec_embed = GraphCastDecoderEmbedder(input_dim_edges=4, output_dim=D, **common)
        self.decoder = MeshGraphDecoder(
            aggregation="sum", input_dim_src_nodes=D, input_dim_dst_nodes=D, input_dim_edges=D,
            output_dim_dst_nodes=D, output_dim_edges=D, do_concat_trick=do_concat_trick, **common)
        self.finale = MeshGraphMLP(input_dim=D, output_dim=c.n_out, hidden_dim=D, hidden_layers=1,
                                   activation_fn=act, norm_type=None)

        mesh_embed = haiku_mlp(flat, f"{_G2M}/encoder_nodes_mesh_nodes")
        # JAX feeds mesh nodes [zeros(C), feat]; the zero block contributes
        # nothing, so PhysicsNeMo's 3-input mesh embedder takes columns C: .
        (w0, b0), rest = mesh_embed[0][0], mesh_embed[0][1:]
        mesh_embed = ([(w0[:, c.n_in:].copy(), b0)] + rest, mesh_embed[1])
        _put_nv_mlp(self.enc_embed.grid_node_mlp, haiku_mlp(flat, f"{_G2M}/encoder_nodes_grid_nodes"))
        _put_nv_mlp(self.enc_embed.mesh_node_mlp, mesh_embed)
        _put_nv_mlp(self.enc_embed.grid2mesh_edge_mlp, haiku_mlp(flat, f"{_G2M}/encoder_edges_grid2mesh"))
        _put_nv_mlp(self.enc_embed.mesh_edge_mlp, haiku_mlp(flat, f"{_MESH}/encoder_edges_mesh"))
        _put_nv_mlp(self.encoder.edge_mlp, haiku_mlp(flat, f"{_G2M}/processor_edges_0_grid2mesh"))
        _put_nv_mlp(self.encoder.src_node_mlp, haiku_mlp(flat, f"{_G2M}/processor_nodes_0_grid_nodes"))
        _put_nv_mlp(self.encoder.dst_node_mlp,
                    nv_permute_node_mlp(haiku_mlp(flat, f"{_G2M}/processor_nodes_0_mesh_nodes"), D))
        for i in range(c.n_steps):
            eb, nb = self.processor.processor_layers[2 * i], self.processor.processor_layers[2 * i + 1]
            _put_nv_mlp(eb.edge_mlp, haiku_mlp(flat, f"{_MESH}/processor_edges_{i}_mesh"))
            _put_nv_mlp(nb.node_mlp,
                        nv_permute_node_mlp(haiku_mlp(flat, f"{_MESH}/processor_nodes_{i}_mesh_nodes"), D))
        _put_nv_mlp(self.dec_embed.mesh2grid_edge_mlp, haiku_mlp(flat, f"{_M2G}/encoder_edges_mesh2grid"))
        _put_nv_mlp(self.decoder.edge_mlp, haiku_mlp(flat, f"{_M2G}/processor_edges_0_mesh2grid"))
        _put_nv_mlp(self.decoder.node_mlp,
                    nv_permute_node_mlp(haiku_mlp(flat, f"{_M2G}/processor_nodes_0_grid_nodes"), D))
        _put_nv_mlp(self.finale, haiku_mlp(flat, f"{_M2G}/decoder_nodes_grid_nodes", ln=False))
        for m in (self.enc_embed, self.encoder, self.processor, self.dec_embed, self.decoder, self.finale):
            m.eval()

        # Graphs: identical (JAX) connectivity and features, as PyG objects.
        from torch_geometric.data import Data, HeteroData

        t = lambda a: torch.from_numpy(np.asarray(a))
        self.g2m = HeteroData()
        self.g2m["grid", "g2m", "mesh"].edge_index = torch.stack(
            [t(ref["g2m_senders"]).long(), t(ref["g2m_receivers"]).long()])
        self.mesh = Data(edge_index=torch.stack(
            [t(ref["mesh_senders"]).long(), t(ref["mesh_receivers"]).long()]))
        self.m2g = HeteroData()
        self.m2g["mesh", "m2g", "grid"].edge_index = torch.stack(
            [t(ref["m2g_senders"]).long(), t(ref["m2g_receivers"]).long()])
        self.feats = {k: t(ref[k]) for k in ("grid_node_features", "mesh_node_features",
                                             "g2m_edge_attr", "mesh_edge_attr", "m2g_edge_attr")}

    def grid2mesh(self, x):
        """Returns (grid_latent, mesh_latent, g2m_edge_latent)."""
        import torch

        with torch.no_grad():
            f = self.feats
            grid_in = torch.cat([x, f["grid_node_features"]], dim=-1)
            grid, mesh, g2m_e, _ = self.enc_embed(grid_in, f["mesh_node_features"],
                                                  f["g2m_edge_attr"], f["mesh_edge_attr"])
            # MeshGraphEncoder returns nodes only; recompute its (identical)
            # edge update with its own edge MLP to expose the edge latent.
            e_new = g2m_e + self.encoder.edge_mlp(g2m_e, (grid, mesh), self.g2m)
            grid, mesh = self.encoder(g2m_e, grid, mesh, self.g2m)
        return grid, mesh, e_new

    def process(self, mesh):
        import torch

        with torch.no_grad():
            # The mesh-edge embedder lives in GraphCastEncoderEmbedder.
            mesh_e = self.enc_embed.mesh_edge_mlp(self.feats["mesh_edge_attr"])
            mesh_e, mesh = self.processor(mesh_e, mesh, self.mesh)
        return mesh, mesh_e

    def mesh2grid(self, mesh, grid):
        """Returns (outputs, m2g_edge_latent)."""
        import torch

        with torch.no_grad():
            e = self.dec_embed(self.feats["m2g_edge_attr"])
            e_new = e + self.decoder.edge_mlp(e, (mesh, grid), self.m2g)
            grid = self.decoder(e, grid, mesh, self.m2g)
            return self.finale(grid), e_new


def physicsnemo_graph(c: Case):
    """PhysicsNeMo's own graph builder on the same lat/lon grid (multimesh)."""
    _prepare_nv_env()
    import torch
    from physicsnemo.models.graphcast.utils.graph import Graph

    lat, lon = c.lat_lon()
    grid = torch.stack(torch.meshgrid(torch.from_numpy(lat), torch.from_numpy(lon), indexing="ij"), dim=-1)
    g = Graph(grid, mesh_level=c.mesh_level, multimesh=True, backend="pyg")
    mesh, _ = g.create_mesh_graph(verbose=False)
    return g, mesh, g.create_g2m_graph(verbose=False), g.create_m2g_graph(verbose=False)


def _mlp_specs(c: Case):
    """Haiku module prefixes of JAX GraphCast with (in_dim, out_dim, layer_norm)."""
    D, C = c.latent, c.n_in
    specs = {
        f"{_G2M}/encoder_nodes_grid_nodes": (C + 3, D, True),
        f"{_G2M}/encoder_nodes_mesh_nodes": (C + 3, D, True),
        f"{_G2M}/encoder_edges_grid2mesh": (4, D, True),
        f"{_G2M}/processor_edges_0_grid2mesh": (3 * D, D, True),
        f"{_G2M}/processor_nodes_0_grid_nodes": (D, D, True),
        f"{_G2M}/processor_nodes_0_mesh_nodes": (2 * D, D, True),
        f"{_MESH}/encoder_edges_mesh": (4, D, True),
        f"{_M2G}/encoder_edges_mesh2grid": (4, D, True),
        f"{_M2G}/processor_edges_0_mesh2grid": (3 * D, D, True),
        f"{_M2G}/processor_nodes_0_grid_nodes": (2 * D, D, True),
        f"{_M2G}/decoder_nodes_grid_nodes": (D, c.n_out, False),
    }
    for i in range(c.n_steps):
        specs[f"{_MESH}/processor_edges_{i}_mesh"] = (3 * D, D, True)
        specs[f"{_MESH}/processor_nodes_{i}_mesh_nodes"] = (2 * D, D, True)
    return specs


@functools.lru_cache(maxsize=None)
def synthetic_reference(c: Case) -> Dict[str, np.ndarray]:
    """JAX-free stand-in: random Haiku-named params + WeatherAI graphs.

    Lets WeatherAI↔PhysicsNeMo tests run without the JAX stack.
    """
    from weatherai.models.graphcast.mesh import build_graphs

    rng = np.random.default_rng(10_000 + c.seed)
    flat: Dict[str, np.ndarray] = {}
    for prefix, (din, dout, ln) in _mlp_specs(c).items():
        dims = [din, c.latent, dout]
        for k in range(2):
            flat[f"{prefix}_mlp/~/linear_{k}/w"] = (rng.standard_normal((dims[k], dims[k + 1])) / np.sqrt(dims[k])).astype(np.float32)
            flat[f"{prefix}_mlp/~/linear_{k}/b"] = (0.2 * rng.standard_normal(dims[k + 1])).astype(np.float32)
        if ln:
            flat[f"{prefix}_layer_norm/scale"] = (1 + 0.2 * rng.standard_normal(dout)).astype(np.float32)
            flat[f"{prefix}_layer_norm/offset"] = (0.2 * rng.standard_normal(dout)).astype(np.float32)
    lat, lon = c.lat_lon()
    g = build_graphs(mesh_level=c.mesh_level, grid_lat=lat, grid_lon=lon, m2g_backend="numpy")
    ref: Dict[str, np.ndarray] = {"params": flat, "x": c.inputs()}
    for name in ("mesh", "g2m", "m2g"):
        for suffix in ("senders", "receivers", "edge_attr"):
            ref[f"{name}_{suffix}"] = getattr(g, f"{name}_{suffix}")
    ref["grid_node_features"] = g.grid_node_features
    ref["mesh_node_features"] = g.mesh_node_features
    return ref


def _t(a):
    import torch

    a = np.array(a, copy=True)
    return torch.from_numpy(a if a.dtype == np.float64 else a.astype(np.float32))[None]


def ours_grid2mesh(model, x):
    import torch

    with torch.no_grad():
        grid, mesh, e = model.run_grid2mesh(_t(x))
    return dict(grid_g2m=grid[0].numpy(), mesh_g2m=mesh[0].numpy(), g2m_edges=e[0].numpy())


def ours_processor(model, mesh_latent):
    import torch

    with torch.no_grad():
        mesh, e = model.run_processor(_t(mesh_latent))
    return dict(mesh_proc=mesh[0].numpy(), mesh_edges=e[0].numpy())


def ours_mesh2grid(model, mesh_latent, grid_latent):
    import torch

    with torch.no_grad():
        out, _, e = model.run_mesh2grid(_t(mesh_latent), _t(grid_latent))
    return dict(out=out[0].numpy(), m2g_edges=e[0].numpy())


def ours_chained(model, x):
    a = ours_grid2mesh(model, x)
    a.update(ours_processor(model, a["mesh_g2m"]))
    a.update(ours_mesh2grid(model, a["mesh_proc"], a["grid_g2m"]))
    return a


def nv_chained(nv: "NVStages", x):
    import torch

    grid, mesh, e = nv.grid2mesh(torch.from_numpy(np.asarray(x)))
    mp, me = nv.process(mesh)
    out, m2g_e = nv.mesh2grid(mp, grid)
    return {k: v.numpy() for k, v in dict(grid_g2m=grid, mesh_g2m=mesh, g2m_edges=e, mesh_proc=mp,
                                          mesh_edges=me, out=out, m2g_edges=m2g_e).items()}


def nv_isolated(nv: "NVStages", ref):
    """Each NV stage fed with the reference's own stage inputs."""
    import torch

    T = lambda k: torch.from_numpy(np.array(ref[k], dtype=np.float32, copy=True))
    grid, mesh, e = nv.grid2mesh(T("x"))
    mp, me = nv.process(T("mesh_g2m"))
    out, m2g_e = nv.mesh2grid(T("mesh_proc"), T("grid_g2m"))
    return {k: v.numpy() for k, v in dict(grid_g2m=grid, mesh_g2m=mesh, g2m_edges=e, mesh_proc=mp,
                                          mesh_edges=me, out=out, m2g_edges=m2g_e).items()}


STAGE_KEYS = {
    "grid2mesh": ("mesh_g2m", "grid_g2m", "g2m_edges"),
    "processor": ("mesh_proc", "mesh_edges"),
    "mesh2grid": ("out", "m2g_edges"),
}
