"""Host-side preparation/collection for a1_speed's gate1/g2vox VM modes."""
import json
from pathlib import Path
import numpy as np
from a1_common import write
from chatcarlo.gpu import bundle
from chatcarlo.gpu.core import health


def export_job(bundle_path,job_path,*,n,batch,seed,source_seed,grid=None):
    manifest=bundle.verify(bundle_path)
    job=dict(bundle_manifest=manifest,n_histories=n,batch_size=batch,seed=seed,source_seed=source_seed)
    if grid is not None:job['grid']=grid
    write(job_path,job);return job


def collect(path,job,arrays):
    """Reject mixed seeds/code, corrupt payloads and incomplete batch sets."""
    path=Path(path)
    if json.loads((path/'done.json').read_text())!=dict(status='complete',job=job):
        raise ValueError('remote completion identity mismatch')
    expected={f'batch_{s:09d}.npz' for s in range(0,job['n_histories'],job['batch_size'])}
    if {p.name for p in path.glob('batch_*.npz')}!=expected:raise ValueError('remote batch set mismatch')
    for start in range(0,job['n_histories'],job['batch_size']):
        n=min(job['batch_size'],job['n_histories']-start);file=path/f'batch_{start:09d}.npz'
        receipt=json.loads(file.with_suffix('.json').read_text())
        if (receipt['job']!=job or receipt['start']!=start or receipt['n']!=n or
                receipt['sha256']!=bundle.sha(file) or receipt['health']['status']!='pass' or
                receipt['health']['histories']!=n):raise ValueError('remote batch identity/health/SHA mismatch')
        with np.load(file,allow_pickle=False) as z:r={k:z[k] for k in z.files}
        health(r,n,int(arrays['nm']),arrays['detector'])
        if 'grid' in job:
            from chatcarlo.gpu.g2_core import health as g2_health,grid_from
            g2_health(r,grid_from(job['grid']))
        yield n,r,receipt['health']
