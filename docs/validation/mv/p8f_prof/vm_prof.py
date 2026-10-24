"""P8f profiling on a Colab T4 (measurement only; the frozen P8f kernel is not edited).
1) kernel resources, 2) same-session alternating timing (3 x 20,000, as P8f's timing.json),
3) ncu --set full of one p8f_kernel launch and one P5k F p5k_kernel launch."""
import glob, hashlib, json, subprocess, sys, zipfile, shutil, time
from pathlib import Path
B = Path('/content/p8f'); OUT = Path('/content/p8f/prof'); OUT.mkdir(parents=True, exist_ok=True)
for f in ['p8f_water_baseline.npz']:
    parts = sorted(glob.glob(f'/content/parts/{f}.part.*'))
    if parts: open(B/f, 'wb').write(b''.join(open(p, 'rb').read() for p in parts))
m = json.loads((B/'p8f_manifest.json').read_text())
need = ['water_bundle.npz','lung_bundle.npz','bone_bundle.npz','lead_bundle.npz','water_fluorescence.npz','lung_fluorescence.npz',
        'bone_fluorescence.npz','lead_fluorescence.npz','bundles.json','p8f_inputs.npz','p8f_inputs.json','p8f_source.zip','p8f_water_baseline.npz']
bad = [n for n in need if hashlib.sha256((B/n).read_bytes()).hexdigest() != m['files'][n]]
log = dict(bad=bad)
def save(): (OUT/'prof.json').write_text(json.dumps(log, indent=1, default=str))
if bad: save(); sys.exit('hash mismatch')
src = Path('/content/p8f_source'); zipfile.ZipFile(B/'p8f_source.zip').extractall(src); sys.path.insert(0, str(src))
shutil.copy('/content/prof_target.py', src/'prof_target.py')
import numpy as np, cupy as cp
from types import SimpleNamespace
from prototypes.mv_electron.gpu import p8f_core as core
from prototypes.mv_electron.gpu.p8f_colab import GPU, run_health
from prototypes.mv_electron.gpu.p5k_colab import GPU as WaterGPU
from prototypes.mv_electron.gpu.p5b_core import health as water_health
from prototypes.mv_electron.gpu.p5b_colab import environment
log['fingerprint_same'] = core.fingerprint() == m['identity']['sources']
log['environment'] = environment(cp); save()
with np.load(B/'p8f_inputs.npz') as f: arrays = {k: f[k].copy() for k in f.files}
mats = []
for name in ('water', 'lung', 'bone', 'lead'):
    with np.load(B/(name+'_bundle.npz')) as f:
        mats.append(SimpleNamespace(**{k: f[k].item() if f[k].ndim == 0 else f[k].copy() for k in core.FIELDS}))
g = GPU(cp, mats, arrays, m['capacity']); log['material_launches'] = g.launches
with np.load(B/'p8f_water_baseline.npz') as f: wd = {k: f[k].copy() for k in f.files}
w = WaterGPU(cp, wd, variant='F', compile_fault=False)
log['water_attributes'] = {k: int(v) for k, v in w.kernels['p5k_kernel'].attributes.items()}
t = []
for _ in range(3):
    r, tw = w.measure(20000); assert water_health(r)['status'] == 'pass'
    a, _ = g.run(20000, core.SEED2); assert run_health(a)
    t.append(dict(water_us=tw['kernel_ms']*1000/20000, material_us=float(a['seconds'])*1e6/20000))
log['timing'] = t; save(); print('timing', t, flush=True)
ncu = next((x for x in [shutil.which('ncu'), *glob.glob('/usr/local/cuda*/bin/ncu'), *glob.glob('/opt/nvidia/nsight-compute/*/ncu')] if x and Path(x).is_file()), None)
log['ncu'] = ncu; save()
for mode, kernel in (('material', 'p8f_kernel'), ('water', 'p5k_kernel')):
    rep = OUT/f'{mode}_profile'
    cmd = [ncu, '--set', 'full', '--replay-mode', 'kernel', '--cache-control', 'all', '--kernel-name', kernel,
           '--launch-skip', '1', '--launch-count', '1', '--export', str(rep), '-f', sys.executable, str(src/'prof_target.py'), str(B), mode]
    t0 = time.time(); r = subprocess.run(cmd, capture_output=True, text=True, cwd=src)
    log[f'ncu_{mode}'] = dict(cmd=cmd, rc=r.returncode, seconds=time.time()-t0, out=r.stdout[-3000:], err=r.stderr[-3000:]); save()
    for page, fmt in (('details', 'txt'), ('raw', 'csv')):
        x = subprocess.run([ncu, '--import', str(rep)+'.ncu-rep', '--page', page] + (['--csv'] if fmt == 'csv' else []), capture_output=True, text=True)
        (OUT/f'{mode}_{page}.{fmt}').write_text(x.stdout + x.stderr)
log['complete'] = True; save(); print('PROF DONE', flush=True)
