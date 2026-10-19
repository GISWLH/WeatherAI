"""Optional upstream JAX route; preserves full architectures and Orbax weights.

Requires the official swirl-dynamics runtime (Python >=3.11; Linux/3.12 is
recommended upstream). No dependencies are imported until a factory is called.
This route is separate from the native, approximate PyTorch lite networks.
"""
import json
from pathlib import Path


class GenFocal:
    """Author's Flax architecture with the published pretrained parameters.

    Construct with from_pretrained('super_resolution'|'debiasing'). Inputs
    follow the upstream normalized tensor/conditioning schema. No lite fallback.
    """
    implementation = "official"

    def __init__(self, stage, network, parameters, provenance):
        self.stage, self.network = stage, network
        self.parameters, self.provenance = parameters, provenance

    @classmethod
    def from_pretrained(cls, stage="super_resolution", *, cache_dir=None, log=print, device=None):
        from ...official.assets import checkout, genfocal_parameters, download, cache_root, SOURCES
        source = checkout("genfocal", cache_dir)
        # Installed package must correspond to the pinned source. Prefer the
        # verified checkout; no changed vendored model files.
        import sys
        if "swirl_dynamics" in sys.modules:
            import swirl_dynamics
            if Path(swirl_dynamics.__file__).resolve().parent.parent != source.resolve():
                raise RuntimeError("swirl_dynamics is already imported from another source; start a fresh process")
        sys.path.insert(0, str(source))
        root = Path(cache_dir) if cache_dir else cache_root()
        params, manifest = genfocal_parameters(stage, cache_dir, log=log, device=device)
        if stage == "super_resolution":
            cfg = source / "swirl_dynamics/projects/genfocal/super_resolution/configs/conus_sample.yaml"
            network, _ = GenFocalOfficial.super_resolution(cfg)
        elif stage == "debiasing":
            cfg, _ = download("https://storage.googleapis.com/genfocal/debiasing/checkpoints/reflow_uvit/config.json",
                              root / "genfocal/debiasing/config.json")
            model, _ = GenFocalOfficial.debiasing(cfg)
            network = model.flow_model
        else:
            raise ValueError("stage must be super_resolution or debiasing")
        return cls(stage, network, params, {"source_revision": SOURCES['genfocal']['revision'],
                    "checkpoint": manifest['source'], "parameter_tree": manifest['tree_prefix'],
                    "parameter_count": manifest['parameter_count'], "parameter_bytes": manifest['bytes'],
                    "pretrained": True})

    def __call__(self, x, sigma, cond):
        return self.network.apply({"params": self.parameters}, x=x, sigma=sigma, cond=cond, is_training=False)

    def to_device(self, device):
        import jax
        self.parameters = jax.device_put(self.parameters, device)
        return self

    def sample_debiasing(self, batch, *, num_sampling_steps=100, reverse_flow=False):
        """Invoke the original RK4 sampler and physical-unit reconstruction.

        batch follows sampling_from_batch: x_0, channel:mean/channel:std,
        output_mean/output_std, with time flattened before 8-frame chunking.
        Preserves the author's arange schedule (last time is 1-1/N).
        """
        if self.stage != 'debiasing':
            raise ValueError('sample_debiasing requires the debiasing checkpoint')
        from types import SimpleNamespace
        from swirl_dynamics.projects.genfocal.debiasing import inference_utils
        return inference_utils.sampling_from_batch(batch,
            model=SimpleNamespace(flow_model=self.network),
            trained_state=SimpleNamespace(model_variables={'params':self.parameters}),
            num_sampling_steps=num_sampling_steps, time_chunk_size=8,
            reverse_flow=reverse_flow, use_jit_dynamics_fn=True)

    def trajectory_sampler(self, **kwargs):
        """Construct the author's SR trajectory sampler with loaded EMA weights.

        Supply its explicit region, schedule, dataloader, baseline/statistics
        and store arguments; see upstream run_sampling.py. No custom sampler.
        """
        if self.stage != 'super_resolution':
            raise ValueError('trajectory_sampler requires the super_resolution checkpoint')
        from swirl_dynamics.projects.genfocal.super_resolution.sampling import TrajectorySamplerParallel
        return TrajectorySamplerParallel(denoise_fn=self, **kwargs)


class GenFocalOfficial:
    @staticmethod
    def debiasing(config_path, checkpoint_directory=None):
        """Return the exact upstream ReFlow model and optional restored state.

        config_path is the training run's config.json, not the inference
        config. checkpoint_directory is that run's checkpoints directory.
        Use upstream sampling_from_batch with these objects and its documented
        batch schema for normalization and integration.
        """
        from ml_collections import ConfigDict
        from swirl_dynamics.projects.genfocal.debiasing import inference_utils, trainers
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if isinstance(config, str):
            config = json.loads(config)
        model = inference_utils.build_model_from_config(ConfigDict(config))
        state = None if checkpoint_directory is None else trainers.TrainState.restore_from_orbax_ckpt(
            str(checkpoint_directory), step=None)
        return model, state

    @staticmethod
    def super_resolution(config_path, checkpoint_directory=None):
        """Return exact upstream EDM U-ViT and optional EMA denoising callable.

        Accepts official conus_train.yaml/conus_sample.yaml (model section).
        Checkpoint path is the parent containing the numeric step directory.
        Without a checkpoint, returns an uninitialized Flax module, not weights.
        """
        import yaml
        from swirl_dynamics.lib import diffusion
        from swirl_dynamics.projects.genfocal.super_resolution import training
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))["model"]
        mapping = {"out_channels": "num_output_channels", "resize_to_shape": "sample_resize",
                   "num_channels": "num_latent_channels", "downsample_ratio": "downsample_ratio",
                   "num_blocks": "num_blocks", "noise_embed_dim": "noise_embedding_dim",
                   "input_proj_channels": "num_input_projection_channels",
                   "output_proj_channels": "num_output_projection_channels", "padding": "padding",
                   "use_spatial_attention": "use_spatial_attention", "use_temporal_attention": "use_temporal_attention",
                   "use_position_encoding": "use_positional_encoding", "num_heads": "num_attention_heads",
                   "normalize_qk": "normalize_qk", "cond_resize_method": "cond_resize_method",
                   "cond_embed_dim": "cond_embedding_dim"}
        backbone = diffusion.PreconditionedDenoiserUNet3d(**{k: cfg[v] for k, v in mapping.items()})
        denoise = None
        if checkpoint_directory is not None:
            state = training.DenoisingModelTrainState.restore_from_orbax_ckpt(str(checkpoint_directory))
            denoise = training.DenoisingTrainer.inference_fn_from_state_dict(state, denoiser=backbone, use_ema=True)
        return backbone, denoise
