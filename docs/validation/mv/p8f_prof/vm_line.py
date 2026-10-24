"""Colab T4: (1) launch-size test of the unchanged p8f_kernel, (2) ncu SourceCounters line map (line-info build)."""
import glob, hashlib, json, subprocess, sys, zipfile, shutil, time
from pathlib import Path
B = Path('/content/p8f'); OUT = Path('/content/p8f/line'); OUT.mkdir(parents=True, exist_ok=True)
log = {}
def save(): (OUT/'line.json').write_text(json.dumps(log, indent=1, default=str))
m = json.loads((B/'p8f_manifest.json').read_text())
need = ['water_bundle.npz','lung_bundle.npz','bone_bundle.npz','lead_bundle.npz','bundles.json','p8f_inputs.npz','p8f_source.zip']
log['bad'] = [n for n in need if hashlib.sha256((B/n).read_bytes()).hexdigest() != m['files'][n]]; save()
if log['bad']: sys.exit('hash mismatch')
src = Path('/content/p8f_source'); zipfile.ZipFile(B/'p8f_source.zip').extractall(src); sys.path.insert(0, str(src))
shutil.copy('/content/line_target.py', src/'line_target.py')
import numpy as np, cupy as cp
from types import SimpleNamespace
from prototypes.mv_electron.gpu import p8f_core as core
from prototypes.mv_electron.gpu.p8f_colab import GPU, run_health
log['fingerprint_same'] = core.fingerprint() == m['identity']['sources']
with np.load(B/'p8f_inputs.npz') as f: arrays = {k: f[k].copy() for k in f.files}
mats = []
for name in ('water', 'lung', 'bone', 'lead'):
    with np.load(B/(name+'_bundle.npz')) as f:
        mats.append(SimpleNamespace(**{k: f[k].item() if f[k].ndim == 0 else f[k].copy() for k in core.FIELDS}))
g = GPU(cp, mats, arrays, m['capacity'])
nl, nb, nc, nf, nspec = g.sizes; dw = nb+nc+nf+37; iw = 79+nb
def launch(n, blocks, threads):
    hd = cp.zeros((1, dw)); hi = cp.zeros((1, iw), np.int64)
    groups = cp.zeros((100, dw)); absolute = cp.zeros((100, dw)); ints = cp.zeros((100, iw), np.uint64)
    nxt = cp.array([blocks*threads], np.int32)
    args = g.args+(g.geometry, g.mapping, *map(np.int32, g.sizes), np.uint64(core.SEED2), np.uint64(0), np.int32(n), nxt, np.int32(0), hd, hi, groups, ints, absolute)
    a, z = cp.cuda.Event(), cp.cuda.Event(); a.record(); g.kernels['p8f_kernel']((blocks,), (threads,), args); z.record(); z.synchronize()
    return float(cp.cuda.get_elapsed_time(a, z))/1000*1e6/n, cp.asnumpy(groups), cp.asnumpy(ints)
configs = [(40, 128), (80, 128), (160, 128), (40, 256), (80, 64), (320, 64)]
ref_us, ref_g, ref_i = launch(20000, 40, 128)
rows = []
for rep in range(3):
    for blocks, threads in configs:
        us, gr, gi = launch(20000, blocks, threads)
        rel = float(np.max(np.abs(gr.sum(0)-ref_g.sum(0))/np.maximum(np.abs(ref_g.sum(0)), 1e-300)))
        rows.append(dict(rep=rep, blocks=blocks, threads=threads, us=us, ints_total_equal=bool(np.array_equal(gi.sum(0), ref_i.sum(0))), max_rel_total_diff=rel))
        log['launch'] = rows; save()
print('launch done', flush=True)
ncu = next((x for x in [shutil.which('ncu'), *glob.glob('/usr/local/cuda*/bin/ncu')] if x and Path(x).is_file()), None)
rep = OUT/'line_profile'
cmd = [ncu, '--section', 'SourceCounters', '--replay-mode', 'kernel', '--cache-control', 'all', '--kernel-name', 'p8f_kernel',
       '--launch-skip', '1', '--launch-count', '1', '--export', str(rep), '-f', sys.executable, str(src/'line_target.py'), str(B), str(OUT)]
t0 = time.time(); r = subprocess.run(cmd, capture_output=True, text=True, cwd=src)
log['ncu'] = dict(rc=r.returncode, seconds=time.time()-t0, out=r.stdout[-2000:], err=r.stderr[-2000:]); save()
x = subprocess.run([ncu, '--import', str(rep)+'.ncu-rep', '--page', 'source', '--print-source', 'cuda,sass', '--csv',
                    '--resolve-source-file', str(OUT/'compiled.cu')], capture_output=True, text=True)
(OUT/'source.csv').write_text(x.stdout); (OUT/'export.log').write_text(x.stderr)
log['complete'] = True; save(); print('LINE DONE', flush=True)
