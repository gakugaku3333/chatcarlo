"""関門 C2′（計画 11 節 追補 1、Claude、2026-10-23）。使い方: judge_c2.py P_DIR Q_DIR OUT.json"""
import json, sys, filecmp
from pathlib import Path
import numpy as np

EXACT = ('endings', 'n_fluorescence', 'scatter_sum', 'box_sum', 'box_sum2')
CLOSE = ('deposits', 'kerma', 'h10')
INT_NPZ = ('n_batches', 'n_batches_hit', 'shape', 'origin_cm', 'voxel_size_cm')

def close(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype: return False
    if not np.array_equal(np.isnan(a), np.isnan(b)): return False
    m = ~np.isnan(a)
    if np.any((a[m] == 0) != (b[m] == 0)): return False
    return bool(np.allclose(a[m], b[m], rtol=1e-12, atol=0))

def maxrel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); m = (a != 0) & ~np.isnan(a)
    return float(np.max(np.abs(a[m]-b[m])/np.abs(a[m]))) if m.any() else 0.

def main(p, q, out):
    p, q = Path(p), Path(q); rep = dict(checks={}, detail={})
    rep['checks']['job_identical'] = (p/'job.json').read_bytes() == (q/'job.json').read_bytes()
    log = (q/'run.log').read_text()
    t = json.loads(log.split('計算時間・バッチ: ', 1)[1].splitlines()[0])
    rep['checks']['reused_8_new_12'] = t['reused_batches'] == 8 and t['new_batches'] == 12
    rep['checks']['reused_bytes'] = all(filecmp.cmp(p/'results'/f'batch_{i:06d}.npz', q/'results'/f'batch_{i:06d}.npz', shallow=False) for i in range(8))
    ex, cl, rel = True, True, {}
    for i in range(8, 20):
        a = np.load(p/'results'/f'batch_{i:06d}.npz'); b = np.load(q/'results'/f'batch_{i:06d}.npz')
        if a.files != b.files: ex = False; continue
        for k in a.files:
            if a[k].dtype != b[k].dtype or a[k].shape != b[k].shape: ex = False
        ex &= all(a[k].tobytes() == b[k].tobytes() for k in EXACT if k in a.files)
        cl &= all(close(a[k], b[k]) for k in CLOSE)
        for k in CLOSE: rel[k] = max(rel.get(k, 0.), maxrel(a[k], b[k]))
    rep['checks']['new_exact'] = ex; rep['checks']['new_close'] = cl; rep['detail']['new_max_rel'] = rel
    a = np.load(p/'dose.npz'); b = np.load(q/'dose.npz')
    rep['checks']['npz_keys'] = a.files == b.files
    rep['checks']['npz_exact'] = all(a[k].tobytes() == b[k].tobytes() for k in INT_NPZ)
    rep['checks']['npz_close'] = all(close(a[k], b[k]) for k in a.files if k not in INT_NPZ)
    rep['detail']['npz_max_rel'] = {k: maxrel(a[k], b[k]) for k in a.files if k not in INT_NPZ}
    for name in ('P', 'Q'):
        j = json.loads((Path(out).parent/f'gate_c2r_{name}.json').read_text())
        rep['checks'][f'judge_{name}'] = j['status'] == 'pass'
    rep['status'] = 'pass' if all(rep['checks'].values()) else 'fail'
    Path(out).write_text(json.dumps(rep, indent=1, ensure_ascii=False)+'\n'); print(rep)

if __name__ == '__main__':
    main(*sys.argv[1:])
