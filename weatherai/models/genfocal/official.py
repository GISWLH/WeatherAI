"""Optional upstream JAX route; preserves full architectures and Orbax weights.

Requires the official swirl-dynamics runtime (Python >=3.11; Linux/3.12 is
recommended upstream). No dependencies are imported until a factory is called.
This route is separate from the native, approximate PyTorch lite networks.
"""
import json
from pathlib import Path


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
