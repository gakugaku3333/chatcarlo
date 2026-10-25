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
    grid = {k: tables[k] for k in ('bin_layer', 'edges_cm', 'mass_g_cm2')}
for r in egs:
    r['grid'] = grid

def ms(q): return q.mean(0), q.std(0, ddof=1)/10

def pool(runs):
    T = sum(r['N'] for r in runs); m = sum(r['N']/T*ms(r['q'])[0] for r in runs)
    s = np.sqrt(sum((r['N']/T*ms(r['q'])[1])**2 for r in runs)); return m, s

def chi(a, b):
    (ma, sa), (mb, sb) = ms(a['q']), ms(b['q']); c = float(np.sum((ma-mb)**2/(sa**2+sb**2))); return c, ma.size, float(chi2.sf(c, ma.size))

def compare(gm, gs, em, es, layer):
    w = np.flatnonzero(layer == 0); im = int(w[np.argmax(em[w])])   # reference: EGS5 maximum in the first water layer
    d = (gm/gm[im])/(em/em[im])-1; s = np.sqrt((gs/gm)**2+(es/em)**2+(gs[im]/gm[im])**2+(es[im]/em[im])**2)
    return dict(n=len(d), ref_bin=im, mean_abs=100*abs(d).mean(), max_abs=100*abs(d).max(), sem_mean=100*s.mean(), sem_max=100*s.max(), over_2sigma=int((abs(d) > 2*s).sum()))

layer = egs[0]['grid']['bin_layer']; edges = egs[0]['grid']['edges_cm']
for r in egs[1:]: assert all(np.array_equal(r['grid'][k], egs[0]['grid'][k]) for k in r['grid'])
for lab, runs in (('EGS5', egs), ('GPU', gpu)):
    for i in range(len(runs)):
        for j in range(i+1, len(runs)): print(f'chi2 {lab} {runs[i]["name"]}-{runs[j]["name"]}: %.1f/%d p %.3g' % chi(runs[i], runs[j]))
em, es = pool(egs); gm, gs = pool(gpu)
c = compare(gm, gs, em, es, layer); c['ref_depth_cm'] = (edges[c['ref_bin']]+edges[c['ref_bin']+1])/2
print('220 bins:', json.dumps(c))
# 2-bin merge within each batch, never across a layer boundary (lead 0.05 -> 0.1 cm, others 0.1 -> 0.2 cm)
pairs = [(i, i+1) for i in range(0, len(layer)-1, 2) if layer[i] == layer[i+1]]
assert sorted(set(sum(pairs, ()))) == list(range(len(layer))), 'odd bins in a layer'
def merge(runs):
    return [dict(r, q=np.stack([(r['q'][:, a]+r['q'][:, b])/2 for a, b in pairs], 1)) for r in runs]
em2, es2 = pool(merge(egs)); gm2, gs2 = pool(merge(gpu)); layer2 = np.array([layer[a] for a, _ in pairs])
c2 = compare(gm2, gs2, em2, es2, layer2); c2['ref_depth_cm'] = float(edges[pairs[c2['ref_bin']][0]]+edges[pairs[c2['ref_bin']][1]+1])/2
print('merged 2 bins:', json.dumps(c2))
if 'detail' in sys.argv:
    w = np.flatnonzero(layer == 0); im = int(w[np.argmax(em[w])])
    d = (gm/gm[im])/(em/em[im])-1; s = np.sqrt((gs/gm)**2+(es/em)**2+(gs[im]/gm[im])**2+(es[im]/em[im])**2)
    names = ['water', 'lung', 'water', 'bone', 'water', 'lead']
    for L in range(6):
        k = layer == L; z = d[k]/s[k]
        print(f'layer {L} {names[L]:5s} {edges[:-1][k][0]:5.2f}-{edges[1:][k][-1]:5.2f} cm  mean diff {100*d[k].mean():+.3f}%  mean |diff| {100*abs(d[k]).mean():.3f}%  >2σ {int((abs(z)>2).sum())}/{k.sum()}  mean z {z.mean():+.2f}')
    j = int(np.argmax(abs(d))); print('max |diff| at %.3f cm (layer %d): %+.3f%% ± %.3f%%' % ((edges[j]+edges[j+1])/2, layer[j], 100*d[j], 100*s[j]))
