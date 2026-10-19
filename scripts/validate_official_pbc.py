"""Execute original projector/ensemble scripts on a transparent numerical fixture.

This verifies original implementation execution, not real-weather skill or
historical fitting. The author's climatology is also run on its native grid.
"""
import json
from pathlib import Path
import tempfile
import numpy as np
import xarray as xr
from weatherai import PBC


def validate():
    with tempfile.TemporaryDirectory(prefix='pbc-official-') as folder:
        pbc = PBC(folder)
        root = Path(folder)
        pbc.run('climatology','era5-f1_mslp',19,target_dates='20200106')
        models = [('tuned_ecmwfpp','tuned_ecmwfpp_on_years3_marginNone',[-.2,.7,.3,1.2]),
                  ('perpp_ecmwf','perpp_ecmwf-yearsall_marginNone_clim20',[.1,.2,.6,.9])]
        outputs = []
        for model, submodel, values in models:
            for q, value in enumerate(values,1):
                task = f'era5-f{q}_mslp_19'
                path = root/'models'/model/'submodel_forecasts'/submodel/task/(task+'-20200106.nc')
                path.parent.mkdir(parents=True,exist_ok=True)
                xr.Dataset({f'f{q}_mslp':(('time','latitude','longitude'),np.full((1,2,2),value))},
                    coords={'time':[np.datetime64('2020-01-06')],'latitude':[0.,1.],'longitude':[10.,11.]}).to_netcdf(path)
            pbc.run('projector','era5-mslp',19,target_dates='20200106',arguments=('-mn',model,'-sn',submodel))
        for q, expected in enumerate([.05,.35,.55,.95],1):
            task = f'era5-f{q}_mslp_19'
            pbc.run('pbc_ecmwf',f'era5-f{q}_mslp',19,target_dates='20200106',arguments=('-e','True'))
            file=root/'models/pbc_ecmwf/submodel_forecasts/pbc_ecmwf-yearsall_marginNone_equal'/task/(task+'-20200106.nc')
            with xr.open_dataset(file) as ds:
                actual=ds[f'f{q}_mslp'].values
                np.testing.assert_allclose(actual,expected,atol=1e-7)
                outputs.append(float(actual.mean()))
        return {'status':'PASS','implementation':'official','provenance':pbc.provenance,
                'stages':['climatology','projector','pbc_ecmwf equal ensemble'],
                'input_cdfs':[m[2] for m in models], 'output_cdf':outputs,
                'scope':'original author scripts on synthetic NetCDF fixture; no historical fit or skill claim'}


if __name__ == '__main__':
    result=validate()
    destination=Path(__file__).resolve().parents[1]/'docs/results/pbc_official_execution.json'
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
