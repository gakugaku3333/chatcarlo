"""Runs inside ncu on the Colab VM: one warmup launch, then the profiled launch (20,000 histories).
Usage: python prof_target.py BUNDLE material|water"""
import sys, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import cupy as cp
from prototypes.mv_electron.gpu import p8f_core as core
from prototypes.mv_electron.gpu.p8f_colab import GPU
b = Path(sys.argv[1]); mode = sys.argv[2]
if mode == 'material':
    m = json.loads((b/'p8f_manifest.json').read_text())
    with np.load(b/'p8f_inputs.npz') as f: arrays = {k: f[k].copy() for k in f.files}
    mats = []
    for name in ('water', 'lung', 'bone', 'lead'):
        with np.load(b/(name+'_bundle.npz')) as f:
            mats.append(SimpleNamespace(**{k: f[k].item() if f[k].ndim == 0 else f[k].copy() for k in core.FIELDS}))
    g = GPU(cp, mats, arrays, m['capacity'])
    g.run(1000, core.SEED2); g.run(20000, core.SEED2)
else:
    from prototypes.mv_electron.gpu.p5k_colab import GPU as WaterGPU
    with np.load(b/'p8f_water_baseline.npz') as f: d = {k: f[k].copy() for k in f.files}
    w = WaterGPU(cp, d, variant='F', compile_fault=False)
    w.measure(1000); w.measure(20000)
cp.cuda.Device().synchronize(); print('target done', mode, flush=True)
