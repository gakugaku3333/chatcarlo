"""Inside ncu: P8f kernel compiled with --generate-line-info (otherwise the same options); warmup, then 20,000 histories."""
import sys, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import cupy as cp
from prototypes.mv_electron.gpu import p8f_core as core
core.CUDA_OPTIONS = core.CUDA_OPTIONS + ('--generate-line-info',)
from prototypes.mv_electron.gpu import p8f_colab
p8f_colab.core.CUDA_OPTIONS = core.CUDA_OPTIONS
b = Path(sys.argv[1]); out = Path(sys.argv[2])
m = json.loads((b/'p8f_manifest.json').read_text())
with np.load(b/'p8f_inputs.npz') as f: arrays = {k: f[k].copy() for k in f.files}
mats = []
for name in ('water', 'lung', 'bone', 'lead'):
    with np.load(b/(name+'_bundle.npz')) as f:
        mats.append(SimpleNamespace(**{k: f[k].item() if f[k].ndim == 0 else f[k].copy() for k in core.FIELDS}))
g = p8f_colab.GPU(cp, mats, arrays, m['capacity'])
(out/'compiled.cu').write_text(core.cuda_source())
(out/'line_launch.json').write_text(json.dumps(g.launches, indent=1))
g.run(1000, core.SEED2); g.run(20000, core.SEED2); cp.cuda.Device().synchronize(); print('line target done', flush=True)
