"""4 MV: pool EGS5 (P7 condition C + additions) and entry GPU runs with weights N_i/N, compare at 0.25 and 0.5 cm.
Usage: pool.py [orig]   (orig = only the original 2.6e7 runs, to reproduce the abstract's earlier numbers)"""
import sys, json, hashlib
from pathlib import Path
import numpy as np
from scipy.stats import chi2
from prototypes.mv_electron import analysis_p7 as st
from prototypes.mv_electron.entry import bundle, compare
HOME = Path.home()/'.cache/chatcarlo'
P7 = Path('docs/validation/mv/p7/egs5/main_egs_C')
EGS = [(P7, 26000000, 20291823)]
GPU = [Path('runs/mv_water_4mv_20261024/entry')]
if 'orig' not in sys.argv:
    EGS += [(HOME/f'stats4mv/out/egs5/add_C_{s}', 37000000, s) for s in (20292301, 20292302)]
    GPU += [HOME/'mv_entry/96782f068a81']
ref_cfg = {k: v for k, v in json.load(open(P7/'config.json')).items() if k not in ('N', 'seed')}
ref_tests = set(json.load(open(P7/'summary.json'))['tests'])
NR = 36  # radial bins per side compared (GPU entry layout: +-9 cm)

def quantities(d, s):
    g = st.dose(d, s)
    return np.concatenate([g['axis_coarse'], st.fold(g['profile'])[:, :, :NR].reshape(100, -1)], axis=1)

def load_egs(path, N, seed):
    summ = json.load(open(path/'summary.json')); cfg = json.load(open(path/'config.json'))
    assert summ['status'] == '合格' and set(summ['tests']) == ref_tests and all(summ['tests'].values()), path
    assert {k: v for k, v in cfg.items() if k not in ('N', 'seed')} == ref_cfg and cfg['N'] == N and cfg['seed'] == seed, path
    v = dict(np.load(path/'result.npz')); assert int(v['N']) == N and int(v['seed']) == seed
    d = {k: v[k] for k in ('axis_coarse', 'axis_fine', 'slab_coarse', 'profile', 'profile_fine')}
    d.update(N=np.array(N), nbatch=np.array(100))
    return dict(name=path.name, N=N, seed=seed, q=quantities(d, st.spec(5.)),
                sha=hashlib.sha256((path/'result.npz').read_bytes()).hexdigest())

def load_gpu(e):
    c = bundle.read_json(e/'bundle/condition.json'); m = bundle.read_json(e/'bundle/materials.json')
    d = compare.stacked(bundle.collect(e/'results', c, m), c)
    return dict(name=str(e), N=c['histories'], seed=c['seed'], q=quantities(d, st.spec(legacy=True)))

egs = [load_egs(*r) for r in EGS]; gpu = [load_gpu(e) for e in GPU]
for g in (egs, gpu): assert len({r['seed'] for r in g}) == len(g)

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
