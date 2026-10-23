"""Pool EGS5 runs (weights N_i/N) and compare with the entry GPU run, abstract definitions."""
import sys, json, hashlib
from pathlib import Path
import numpy as np
from scipy.stats import chi2
from prototypes.mv_electron import analysis_p3b3 as st, run_p3b3 as p3
from prototypes.mv_electron.entry import bundle, compare
HOME = Path.home()/'.cache/chatcarlo'
ENTRY = HOME/'mv_entry/9734038057e4'
RUNS = [('docs/validation/mv/p3b/egs5/main_egs', 46000000, 20291763)]
if 'pooled' in sys.argv:
    RUNS += [(str(HOME/f'stats10mv/out/egs5/add_{s}'), 27000000, s) for s in (20292101, 20292102)]
main_tests = set(json.load(open('docs/validation/mv/p3b/egs5/main_egs/summary.json'))['tests'])
main_cfg = json.load(open('docs/validation/mv/p3b/egs5/main_egs/config.json'))
assert len({s for _, _, s in RUNS}) == len(RUNS)

def quantities(d):
    """axis_coarse (100,120) and folded profile (100,3,36) in Gy/photon, per batch."""
    g = st.dose(d)
    return np.concatenate([g['axis_coarse'], st.fold(g['profile']).reshape(100, -1)], axis=1)

runs = []
for path, N, seed in RUNS:
    path = Path(path); summ = json.load(open(path/'summary.json')); cfg = json.load(open(path/'config.json'))
    assert set(summ['tests']) == main_tests, path
    assert {k: v for k, v in cfg.items() if k not in ('N', 'seed')} == {k: v for k, v in main_cfg.items() if k not in ('N', 'seed')}, path
    d = p3.validate(dict(np.load(path/'result.npz')), 'egs', N, seed, summ)
    q = quantities(d); m, s = st.mean_sem(q)
    runs.append(dict(name=path.name, N=N, seed=seed, m=m, s=s, sha=hashlib.sha256((path/'result.npz').read_bytes()).hexdigest(),
                     cpu=summ['CPU_seconds'], wall=summ['wall_seconds']))
Ntot = sum(r['N'] for r in runs)
em = sum(r['N']/Ntot*r['m'] for r in runs); es = np.sqrt(sum((r['N']/Ntot*r['s'])**2 for r in runs))

# G3: pairwise consistency on absolute dose, 228 points
g3 = []
for i in range(len(runs)):
    for j in range(i+1, len(runs)):
        a, b = runs[i], runs[j]; c = float(np.sum((a['m']-b['m'])**2/(a['s']**2+b['s']**2)))
        g3.append(dict(pair=[a['name'], b['name']], chi2=c, dof=a['m'].size, p=float(chi2.sf(c, a['m'].size))))

# GPU: entry runs, verified loading path of the entry
GPUS = [ENTRY] + ([HOME/'mv_entry/b521f4485fa5'] if 'gpu_pooled' in sys.argv else [])
gruns = []
for e in GPUS:
    cond = bundle.read_json(e/'bundle/condition.json'); mat = bundle.read_json(e/'bundle/materials.json')
    q = quantities(compare.stacked(bundle.collect(e/'results', cond, mat), cond)); m, s_ = st.mean_sem(q)
    gruns.append(dict(name=e.name, N=cond['histories'], seed=cond['seed'], m=m, s=s_))
assert len({r['seed'] for r in gruns}) == len(gruns)
NG = sum(r['N'] for r in gruns)
gm = sum(r['N']/NG*r['m'] for r in gruns); gs = np.sqrt(sum((r['N']/NG*r['s'])**2 for r in gruns))
for i in range(len(gruns)):
    for j in range(i+1, len(gruns)):
        a, b = gruns[i], gruns[j]; c = float(np.sum((a['m']-b['m'])**2/(a['s']**2+b['s']**2)))
        g3.append(dict(pair=[a['name'], b['name']], chi2=c, dof=a['m'].size, p=float(chi2.sf(c, a['m'].size))))

def table(gm, gs, em, es):
    out = {}
    gp, ep, sgp, sep = gm[:120], em[:120], gs[:120]/gm[:120], es[:120]/em[:120]
    im = int(ep.argmax()); d = (gp/gp[im])/(ep/ep[im])-1
    s = np.sqrt(sgp**2+sep**2+sgp[im]**2+sep[im]**2)
    out['PDD'] = d, s
    D, S = [], []
    for k in range(3):
        sl = slice(120+36*k, 156+36*k); g, e = gm[sl], em[sl]; sg, se = gs[sl]/g, es[sl]/e
        dd = (g/g[0])/(e/e[0])-1; ss = np.sqrt(sg**2+se**2+sg[0]**2+se[0]**2); m = e/e[0] >= .2
        D += list(dd[m]); S += list(ss[m])
    out['OCR'] = np.array(D), np.array(S)
    res = {}
    for k, (d, s) in out.items():
        n2 = int((abs(d) > 2*s).sum())
        res[k] = dict(n=len(d), mean_abs=100*abs(d).mean(), max_abs=100*abs(d).max(), sem_mean=100*s.mean(), sem_max=100*s.max(),
                      over_2sigma=n2, expected_by_chance=round(0.0455*len(d), 1))
    res['PDD']['ref_depth_cm'] = (int(em[:120].argmax())+.5)*.25
    return res

out = dict(runs=[{k: v for k, v in r.items() if k not in ('m', 's')} for r in runs], N_EGS5=Ntot, N_GPU=NG, gpu_runs=[{k: v for k, v in r.items() if k not in ('m', 's')} for r in gruns],
           G3=g3, comparison=table(gm, gs, em, es))
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
