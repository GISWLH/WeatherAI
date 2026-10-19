"""Official pretrained GraphCast and signed input gradients for gray-swan studies.

The study is a diagnostic methodology, not a new model/checkpoint. This adapter
uses DeepMind's unchanged GraphCast and the wrapper order in the authors' ERF
notebook (Zenodo 18111128). It does not reproduce the complete Dubai experiment.
"""
import dataclasses
from pathlib import Path
import sys

CHECKPOINTS = {
    'operational': 'GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz',
    '37levels': 'GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz',
    'small': 'GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz',
}


class Translocation:
    implementation = 'official'

    @classmethod
    def from_pretrained(cls, variant='operational', *, cache_dir=None):
        from ...official.assets import checkout, download, cache_root, gcs_url, SOURCES
        if variant not in CHECKPOINTS:
            raise ValueError(f'variant must be one of {tuple(CHECKPOINTS)}')
        source = checkout('graphcast', cache_dir)
        if 'graphcast' in sys.modules:
            import graphcast as imported
            if Path(imported.__file__).resolve().parent.parent != source.resolve():
                raise RuntimeError('Another graphcast source is already imported; start a fresh process')
        sys.path.insert(0, str(source))
        import haiku as hk
        import xarray as xr
        from graphcast import checkpoint, graphcast, casting, normalization, autoregressive
        path = (Path(cache_dir) if cache_dir else cache_root()) / ('graphcast_small' if variant == 'small' else 'graphcast' if variant == 'operational' else 'graphcast_37levels')
        file, digest = download(gcs_url('dm_graphcast', 'params/'+CHECKPOINTS[variant]), path/'checkpoint.npz')
        with file.open('rb') as stream:
            ckpt = checkpoint.load(stream, graphcast.CheckPoint)
        stats = {}
        for name in ('diffs_stddev_by_level','mean_by_level','stddev_by_level'):
            file, _ = download(gcs_url('dm_graphcast','stats/'+name+'.nc'), path/(name+'.nc'))
            stats[name] = xr.load_dataset(file)

        @hk.transform_with_state
        def forward(inputs, targets, forcings):
            model = graphcast.GraphCast(ckpt.model_config, ckpt.task_config)
            model = casting.Bfloat16Cast(model)
            model = normalization.InputsAndResiduals(model, **stats)
            model = autoregressive.Predictor(model, gradient_checkpointing=True)
            return model(inputs, targets_template=targets, forcings=forcings)

        obj = cls()
        import jax
        obj.params, obj.forward = ckpt.params, forward
        obj._apply = jax.jit(forward.apply)
        obj.task_config, obj.model_config = ckpt.task_config, ckpt.model_config
        obj.provenance = {'implementation':'official', 'pretrained':True,
                          'source_revision':SOURCES['graphcast']['revision'],
                          'checkpoint':CHECKPOINTS[variant], 'sha256':digest,
                          'erf_reference':'https://doi.org/10.5281/zenodo.18111128'}
        return obj

    def prepare(self, dataset, *, lead_time='6h'):
        from graphcast import data_utils
        return data_utils.extract_inputs_targets_forcings(dataset,
            target_lead_times=slice('6h',lead_time), **dataclasses.asdict(self.task_config))

    def __call__(self, inputs, targets_template, forcings):
        import jax
        return self._apply(self.params, {}, jax.random.PRNGKey(0),
                                  inputs, targets_template, forcings)[0]

    def effective_receptive_field(self, inputs, *, targets_template, forcings,
                                  variable='total_precipitation_6hr', location):
        """Signed d(selected physical output)/d(inputs), using JAX reverse AD.

        location is a named positional index dict, e.g. batch=0,time=-1,lat=260,
        lon=220. All output dimensions must be selected to yield one scalar.
        Gradient units depend on each physical input field. No causal claim.
        """
        import jax
        from graphcast import xarray_jax
        def objective(x):
            prediction = self(x, targets_template, forcings)
            scalar = prediction[variable].isel(**location)
            if scalar.size != 1:
                raise ValueError('location must select exactly one output value')
            return xarray_jax.unwrap_data(scalar).reshape(())
        value, gradient = jax.value_and_grad(objective)(inputs)
        return {'gradient':gradient, 'selected_prediction':value, 'provenance':self.provenance}
