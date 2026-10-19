"""Actual official checkpoints and official example weather data. No lite fallback."""
import dataclasses
import json
from pathlib import Path
import sys
import time

GRAPHCAST_FILES = {
    "graphcast": "GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz",
    "graphcast_small": "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz",
}


def prepare_assets(stage, log=print):
    from weatherai.official.assets import checkout, download, genfocal_parameters, cache_root, gcs_url
    root = cache_root()
    if stage == 'graphcast_small_erf':
        stage = 'graphcast_small'
    if stage.startswith('genfocal'):
        source = checkout('genfocal')
        kind = 'super_resolution' if stage == 'genfocal_sr' else 'debiasing'
        manifest = genfocal_parameters(kind, load=False, log=log)
        result = {k: v for k, v in manifest.items() if k != 'tensors'}
        if stage == 'genfocal_sr':
            # Author's example ERA5 inputs and normalization, with the author's
            # dataloader (global daily conditions, regional residual output).
            import numpy as np
            sys.path.insert(0, str(source))
            from swirl_dynamics.projects.genfocal.super_resolution import data
            # Same public objects, without probing Google application credentials.
            base = 'https://storage.googleapis.com/genfocal/super_resolution/inference/example_inputs/'
            variables = tuple(data.DatasetVariable(name=n, indexers={}, rename=n) for n in ('T2m','W10m','Q1000','MSL'))
            loader = data.InputLoader(date_range=('2015-08-01','2015-08-07'),
                daily_dataset_path=base+'era5_1p5deg_2015_global_t2m_w10m_q1000_msl_z200_z500.zarr',
                daily_variables=variables,
                daily_stats_path=base+'era5_1p5deg_stats_global_t2m_w10m_q1000_msl_z200_z500.zarr',
                dims_order=('time','longitude','latitude'))
            days = np.repeat(np.arange(np.datetime64('2015-08-01'), np.datetime64('2015-08-08')), 12)
            condition = loader.get_days(days)['channel:daily_mean'][None].astype(np.float32)
            np.save(root/'genfocal/super_resolution/era5_condition.npy', condition)
            result['condition_shape'] = list(condition.shape)
            result['input_data'] = 'official ERA5 example, 2015-08-01..07'
        return result
    source = checkout('graphcast')
    path = root / stage
    path.mkdir(parents=True, exist_ok=True)
    outputs = {}
    objects = {'checkpoint.npz': 'params/'+GRAPHCAST_FILES[stage]}
    resolution = '0.25' if stage == 'graphcast' else '1.0'
    objects['example.nc'] = f'dataset/source-era5_date-2022-01-01_res-{resolution}_levels-13_steps-01.nc'
    for name in ['diffs_stddev_by_level.nc', 'mean_by_level.nc', 'stddev_by_level.nc']:
        objects[name] = 'stats/'+name
    for filename, key in objects.items():
        log(key)
        file, digest = download(gcs_url('dm_graphcast',key), path/filename)
        outputs[filename] = {'source':key, 'sha256':digest, 'bytes':file.stat().st_size}
    (path/'assets.json').write_text(json.dumps(outputs,indent=2))
    return outputs


def run(stage):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from weatherai.official.assets import cache_root
    gpu = jax.devices('gpu')[0]
    start = time.perf_counter()
    if stage == 'graphcast_small_erf':
        return _erf(gpu,start)
    if stage.startswith('genfocal'):
        from weatherai.models.genfocal import GenFocal
        kind = 'super_resolution' if stage == 'genfocal_sr' else 'debiasing'
        print('Loading official tensors directly onto GPU:', kind, flush=True)
        model = GenFocal.from_pretrained(kind, device=gpu)
        print('Official weights loaded:', model.provenance, flush=True)
        if kind == 'super_resolution':
            c = np.load(cache_root()/'genfocal/super_resolution/era5_condition.npy')
            cond = {'channel:daily_mean':jax.device_put(c,gpu)}
            # Official orientation is rotated: (time, latitude, longitude).
            x = jax.device_put(np.zeros((1,84,120,240,4), np.float32), gpu)
            sigma = jax.device_put(np.ones(1,np.float32),gpu)
            scope = 'official EMA denoiser, full CONUS grid; real ERA5 conditions, zero residual at sigma=1; not a sampled forecast'
        else:
            x = jax.device_put(np.zeros((1,8,240,121,10),np.float32),gpu)
            cond = {'channel:mean':jnp.zeros_like(x),'channel:std':jnp.ones_like(x)}
            sigma = jax.device_put(np.full(1,.5,np.float32),gpu)
            scope = 'official pretrained full ReFlow vector field; synthetic normalized inputs; not an end-to-end climate run'
        print('Compiling/evaluating original full network', flush=True)
        output = jax.jit(lambda p, a, s, c: model.network.apply({'params':p},x=a,sigma=s,cond=c,is_training=False))(
            model.parameters,x,sigma,cond)
        output.block_until_ready()
        return {'status':'PASS' if bool(jnp.isfinite(output).all()) else 'FAIL', 'model':stage,
                **model.provenance,'scope':scope,'output_shape':list(output.shape),
                'output_mean':float(output.mean()),'output_std':float(output.std()),
                'gpu':str(gpu),'seconds':time.perf_counter()-start}
    return _graphcast(stage,gpu,start)


def _erf(gpu,start):
    import numpy as np
    import xarray as xr
    from weatherai import Translocation
    from weatherai.official.assets import cache_root
    model=Translocation.from_pretrained('small')
    dataset=xr.load_dataset(cache_root()/'graphcast_small/example.nc',decode_timedelta=True)
    inputs,targets,forcings=model.prepare(dataset)
    # Geographical selection near Dubai on the official 1-degree sample grid.
    location={'batch':0,'time':0,'lat':int(np.abs(targets.lat.values-25).argmin()),
              'lon':int(np.abs(targets.lon.values-55).argmin())}
    result=model.effective_receptive_field(inputs,targets_template=targets*np.nan,
                                         forcings=forcings,location=location)
    summary={name:{'finite':bool(np.isfinite(a.values).all()),
                   'l1':float(np.abs(a.values).sum())} for name,a in result['gradient'].data_vars.items()}
    return {'status':'PASS' if all(v['finite'] for v in summary.values()) and any(v['l1']>0 for v in summary.values()) else 'FAIL',
            'model':'graphcast_small_erf',**model.provenance,
            'scope':'official pretrained Small model; actual Translocation API and signed input gradients; ERA5 2022 sample, not Dubai 2024 event',
            'location':location,'selected_precipitation':float(result['selected_prediction']),
            'gradients':summary,'gpu':str(gpu),'seconds':time.perf_counter()-start}


def _graphcast(stage,gpu,start):
    import jax
    import numpy as np
    import xarray as xr
    import haiku as hk
    from weatherai.official.assets import checkout,cache_root,SOURCES
    source=checkout('graphcast'); sys.path.insert(0,str(source))
    from graphcast import checkpoint,graphcast,casting,normalization,data_utils
    path=cache_root()/stage
    with (path/'checkpoint.npz').open('rb') as stream:
        ckpt=checkpoint.load(stream,graphcast.CheckPoint)
    stats={name:xr.load_dataset(path/(name+'.nc')) for name in ['diffs_stddev_by_level','mean_by_level','stddev_by_level']}
    dataset=xr.load_dataset(path/'example.nc', decode_timedelta=True)
    inputs,targets,forcings=data_utils.extract_inputs_targets_forcings(dataset,target_lead_times=slice('6h','6h'),**dataclasses.asdict(ckpt.task_config))
    @hk.transform_with_state
    def forward(inputs,targets,forcings):
        predictor=graphcast.GraphCast(ckpt.model_config,ckpt.task_config)
        predictor=casting.Bfloat16Cast(predictor)
        predictor=normalization.InputsAndResiduals(predictor,**stats)
        return predictor(inputs,targets_template=targets,forcings=forcings)
    params=jax.device_put(ckpt.params,gpu)
    prediction,_=jax.jit(forward.apply)(params,{},jax.random.PRNGKey(0),inputs,targets*np.nan,forcings)
    # xarray materialization synchronizes device execution.
    summary={name:{'shape':list(a.shape),'finite':bool(np.isfinite(a.values).all()),
                   'mean':float(a.values.mean())} for name,a in prediction.data_vars.items()}
    return {'status':'PASS' if all(v['finite'] for v in summary.values()) else 'FAIL',
            'model':stage,'implementation':'official','pretrained':True,'source_revision':SOURCES['graphcast']['revision'],
            'checkpoint':GRAPHCAST_FILES[stage],'model_config':dataclasses.asdict(ckpt.model_config),
            'input_data':'official ERA5 example 2022-01-01','scope':'one-step 6h physical forecast, no skill claim',
            'outputs':summary,'gpu':str(gpu),'seconds':time.perf_counter()-start}
