"""判定の外の実測値（事後の集計）。事前登録の判定(q5–q7.json)を置き換えない。

保存済みのresult.npzと固定済みの評価bin(pilot_lock.json)から、Q7未達で
「未評価」になったbackendも含めて、EGS5主計算との差を同じ式で集計する。
実行: .venv/bin/python docs/validation/mv/p2/supplementary_outside_verdict.py
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import norm
from prototypes.mv_electron.compare_photon_p2 import select
from prototypes.mv_electron.workflow_p2 import load

OUT=Path(__file__).resolve().parent
lock=json.loads((OUT/'pilot_lock.json').read_text())
rows=[]
for j,E in enumerate((1.25,2.0,6.0,10.0)):
    w=lock['windows'][str(E)];ref=w['Dref']
    get=lambda p:select(*[load(OUT/p/'result.npz')[k] for k in ('coarse','fine')],w)
    r,rs,tol=get(f'egs5/main_ref_{j}_0');s,ss,_=get(f'egs5/main_ref_{j}_4')
    def stats(d,sem):
        z=d/sem;bonf=norm.ppf(1-.05/(2*len(z)))
        return dict(max_abs_over_Dref=float(max(abs(d))/ref),mean_over_Dref=float(np.mean(d)/ref),rms_z=float(np.sqrt(np.mean(z**2))),
                    max_abs_z=float(max(abs(z))),bins_beyond_bonferroni=int(np.sum(abs(z)>bonf)),bins=len(z),
                    bins_beyond_dose_tolerance=int(np.sum(abs(d)/ref>tol)))
    rows.append(dict(E=E,pair='EGS5 sensitivity(IUNRST=4) - main',**stats(s-r,np.hypot(rs,ss))))
    for b in ('dcslib','mf','elsepa'):
        p,ps,_=get(f'runs/main_{b}_{j}')
        rows.append(dict(E=E,pair=f'{b} - EGS5 main',**stats(p-r,np.hypot(ps,rs)),max_prototype_SEM_over_Dref=float(max(ps)/ref),max_reference_SEM_over_Dref=float(max(rs)/ref)))
(OUT/'supplementary_outside_verdict.json').write_text(json.dumps(dict(
    scope='事後の集計。事前登録の判定ではない。線量許容(2%/3%)の超過bin数だけでは工学的判定にならない(頂点・SEM・基準幅の条件が別にある)。',rows=rows),ensure_ascii=False,indent=2)+'\n')
for r in rows:print(r['E'],r['pair'],f"max {r['max_abs_over_Dref']*100:.2f}% mean {r['mean_over_Dref']*100:+.3f}% rms z {r['rms_z']:.2f} bonf {r['bins_beyond_bonferroni']}/{r['bins']} tol超過 {r['bins_beyond_dose_tolerance']}")
