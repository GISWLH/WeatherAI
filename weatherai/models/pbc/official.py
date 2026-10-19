"""Run the unchanged, pinned Nature PBC implementation in its native layout."""
import os
from pathlib import Path
import shutil
import subprocess
import sys


class PBC:
    """Author scripts, fitted from historical forecasts/observations.

    There is no universal pretrained neural checkpoint. `run` executes the
    selected original stage; inputs/outputs use the author's NetCDF layout.
    Work directory and source are separate, so model/data products are durable.
    """
    implementation = 'official'
    stages = ('climatology', 'ecmwfpp', 'tuner', 'perpp_ecmwf', 'projector',
              'pbc_ecmwf', 'pbc_aifs', 'pbc_msn')

    def __init__(self, work_dir, *, cache_dir=None, python=None):
        from ...official.assets import checkout, SOURCES
        self.source = checkout('pbc', cache_dir)
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.python = python or sys.executable
        # Several official attribute modules read src/.../selected_submodel.json
        # relative to cwd. Copy only their configuration, never replace user data.
        config_files = list((self.source/'src').rglob('*.json')) + list((self.source/'src').rglob('attributes.py'))
        for file in config_files:
            target = self.work_dir/file.relative_to(self.source)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, target)
        self.provenance = {**SOURCES['pbc'], 'implementation': 'official',
                           'pretrained': False, 'weights': 'fit per task/date/grid using historical data'}

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise ValueError('PBC has no universal pretrained weights; use PBC(work_dir=...) and the official fitting stages')

    def run(self, stage, gt_id, horizon, *, target_dates, arguments=(), timeout=None):
        """Execute original CLI; missing required histories raise, never fallback."""
        if stage not in self.stages:
            raise ValueError(f'Choose an official stage: {self.stages}')
        script = self.source/'src/models'/stage/'batch_predict.py'
        if not script.is_file():
            raise ValueError(f'{stage} has no batch_predict entry; consult its official CLI')
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.source/'src') + os.pathsep + env.get('PYTHONPATH', '')
        command = [self.python, str(script), str(gt_id), str(horizon), '-t', str(target_dates), *map(str, arguments)]
        result = subprocess.run(command, cwd=self.work_dir, env=env, text=True,
                                capture_output=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f'Official PBC stage failed ({result.returncode}):\n{result.stdout}\n{result.stderr}')
        return {'provenance': self.provenance, 'command': command,
                'work_dir': str(self.work_dir), 'stdout': result.stdout, 'stderr': result.stderr}
