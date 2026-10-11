"""Outside the pre-registered analysis of 追補1: descriptive values only, no verdict.

Build-up (0.5 mm bins 6-43) mean differences / fixed Dref, 6 MeV, using the new
12M-history EGS5 runs as well as the saved P2 runs.
"""
import json
from pathlib import Path
import numpy as np
from prototypes.mv_electron.workflow_p2 import load
ROOT=Path(__file__).resolve().parents[5]
P2=ROOT/'docs/validation/mv/p2';B=ROOT/'docs/validation/mv/p2b'
D=json.loads((P2/'pilot_lock.json').read_text())['windows']['6.0']['Dref'];bins=list(range(6,44))
def fine(v):
    v=dict(v);return np.asarray(v['fine'])
runs={'new S+':fine(np.load(B/'egs5/p2b_Splus/result.npz')),'new C-':fine(np.load(B/'egs5/p2b_Cminus/result.npz')),
      'P2 main':fine(load(P2/'egs5/main_ref_2_0/result.npz')),'P2 sensitivity':fine(load(P2/'egs5/main_ref_2_4/result.npz'))}
for b in ('dcslib','mf','elsepa'):runs[b]=fine(load(P2/f'runs/main_{b}_2/result.npz'))
def diff(l,r):
    a=runs[l][:,bins].mean(1);q=runs[r][:,bins].mean(1)
    return dict(pair=f'{l} - {r}',delta_pct_Dref=float((a.mean()-q.mean())/D*100),
                SEM_pct_Dref=float(np.sqrt(a.var(ddof=1)/len(a)+q.var(ddof=1)/len(q))/D*100))
pairs=[('new C-','new S+'),('P2 sensitivity','P2 main'),('new S+','P2 main'),('new C-','P2 sensitivity')]
pairs+=[(b,'P2 main') for b in ('dcslib','mf','elsepa')]+[(b,'new S+') for b in ('dcslib','mf','elsepa')]
rows=[diff(*p) for p in pairs]
(B/'supplementary/outside_plan.json').write_text(json.dumps(dict(note='追補1の事前登録の外。記述のみで判定ではない。',rows=rows),ensure_ascii=False,indent=2)+'\n')
for r in rows:print(f"{r['pair']:28s} {r['delta_pct_Dref']:+.3f} ± {r['SEM_pct_Dref']:.3f}")
