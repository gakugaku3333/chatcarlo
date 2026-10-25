"""Recompute the archived comparison from batch_tables.npz and .json only.

Arithmetic and output are copied unchanged from this stage's pool.py.
Byte identity is verified for the recorded extraction environment.
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import chi2

DIRECTORY = Path(__file__).resolve().parent
metadata = json.loads((DIRECTORY / 'batch_tables.json').read_text())
with np.load(DIRECTORY / 'batch_tables.npz', allow_pickle=False) as tables:
    def read_runs(group):
        records = [r for r in metadata['runs'] if r['group'] == group]
        if 'orig' in sys.argv:
            records = records[:1]
        return [dict(r, q=tables[r['id'] + '_q']) for r in records]
    egs, gpu = read_runs('egs'), read_runs('gpu')

NR = metadata['columns']['NR']

def merge(runs, k):
    out = []
    for r in runs:
        q = r['q']; q2 = np.concatenate([q[:, :120].reshape(100, -1, k).mean(-1), q[:, 120:].reshape(100, 3, -1, k).mean(-1).reshape(100, -1)], 1)
        out.append((q2.mean(0), q2.std(0, ddof=1)/10, r['N']))
    T = sum(n for *_, n in out)
    return sum(n/T*m for m, _, n in out), np.sqrt(sum((n/T*s)**2 for _, s, n in out)), out

def chi(a, b):
    c = float(np.sum((a[0]-b[0])**2/(a[1]**2+b[1]**2))); return c, a[0].size, float(chi2.sf(c, a[0].size))

def compare_(gm, gs, em, es, k):
    nb, no = 120//k, NR//k; res = {}
    gp, ep = gm[:nb], em[:nb]; im = int(ep.argmax())
    d = (gp/gp[im])/(ep/ep[im])-1; s = np.sqrt((gs[:nb]/gp)**2+(es[:nb]/ep)**2+(gs[im]/gp[im])**2+(es[im]/ep[im])**2)
    res['PDD'] = (d, s, (im+.5)*.25*k)
    D, S, per = [], [], []
    for j in range(3):
        sl = slice(nb+no*j, nb+no*(j+1)); g, e = gm[sl], em[sl]
        dd = (g/g[0])/(e/e[0])-1; ss = np.sqrt((gs[sl]/g)**2+(es[sl]/e)**2+(gs[sl][0]/g[0])**2+(es[sl][0]/e[0])**2); m = e/e[0] >= .2
        D += list(dd[m]); S += list(ss[m]); per.append(100*dd[m].mean())
    res['OCR'] = (np.array(D), np.array(S), [round(x, 2) for x in per])
    return {key: dict(n=len(d), mean_abs=round(100*abs(d).mean(), 2), max_abs=round(100*abs(d).max(), 2), sem_mean=round(100*s.mean(), 2),
                      sem_max=round(100*s.max(), 2), over_2sigma=int((abs(d) > 2*s).sum()), extra=x) for key, (d, s, x) in res.items()}

print('EGS5', [(r['name'], r['N'], r['seed']) for r in egs], 'GPU', [(r['name'], r['N'], r['seed']) for r in gpu])
_, _, eo = merge(egs, 1); _, _, go = merge(gpu, 1)
for lab, o, runs in (('EGS5', eo, egs), ('GPU', go, gpu)):
    for i in range(len(o)):
        for j in range(i+1, len(o)):
            c, n, p = chi(o[i], o[j]); print(f'G3 {lab} {runs[i]["name"]} - {runs[j]["name"]}: chi2 {c:.1f}/{n} p {p:.3g}')
for k in (1, 2):
    em, es, _ = merge(egs, k); gm, gs, _ = merge(gpu, k)
    print(f'{0.25*k} cm', json.dumps(compare_(gm, gs, em, es, k), ensure_ascii=False))
