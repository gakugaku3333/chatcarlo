"""基準v2を決める前にClaudeが見た追加の試算（2026-10-11）の再現用記録。

事後解析の透明性のために、基準の選択に使った統計量の式・対象bin・出力を残す。
判定ではない。式は当時のまま（系統rmsの95%上限はbinを独立とみなした近似で、
bin間の相関を含まない。基準v2では上限を使わない）。
実行: .venv/bin/python -m docs.validation.mv.p2.posthoc_v2.trial_before_criteria
"""
import json
from pathlib import Path
import numpy as np
from prototypes.mv_electron.workflow_p2 import load

OUT=Path(__file__).resolve().parents[1]
lock=json.loads((OUT/'pilot_lock.json').read_text())

def regions(v,w):
    c,f=v['coarse'],v['fine']
    # 領域平均はbinの単純平均（build=0.5 mm bin、peak=窓の2 mm bin、deep=深部の2 mm bin）
    return dict(build=f[:,w['build']].mean(1),peak=c[:,w['near']].mean(1),deep=c[:,w['deep']].mean(1)),np.c_[c[:,w['deep']+w['near']],f[:,w['build']]]

def ms(x):return x.mean(0),x.std(0,ddof=1)/np.sqrt(len(x))

rows=[]
for j,E in enumerate((1.25,2.0,6.0,10.0)):
    w=lock['windows'][str(E)];ref=w['Dref']
    R,Rb=regions(load(OUT/f'egs5/main_ref_{j}_0/result.npz'),w)
    for name,path in [('EGS5 sensitivity',f'egs5/main_ref_{j}_4')]+[(b,f'runs/main_{b}_{j}') for b in ('dcslib','mf','elsepa')]:
        P,Pb=regions(load(OUT/path/'result.npz'),w);row=dict(E=E,pair=name+' - EGS5 main')
        for k in ('build','peak','deep'):
            m1,s1=ms(P[k]);m0,s0=ms(R[k])
            row[k+'_mean_pct']=float((m1-m0)/ref*100);row[k+'_SEM_pct']=float(np.hypot(s1,s0)/ref*100)
        mp,sp=ms(Pb);mr,sr=ms(Rb);d=(mp-mr)/ref;v=(sp**2+sr**2)/ref**2
        sys2=float(np.mean(d**2)-np.mean(v));se=float(np.sqrt(2*np.mean(v**2)/len(d)+4*np.mean(d**2*v)/len(d)))
        row.update(systematic_ms=sys2,systematic_rms_upper95_pct_independent_bins=float(np.sqrt(max(0,sys2+1.645*se))*100),
                   per_bin_stat_rms_pct=float(np.sqrt(np.mean(v))*100),max_abs_pct=float(max(abs(d))*100),bins_over_1pct=int(np.sum(abs(d)>.01)),bins=len(d))
        rows.append(row)
(Path(__file__).with_suffix('.json')).write_text(json.dumps(dict(note='基準v2の決定前に見た試算。判定ではない。',rows=rows),ensure_ascii=False,indent=2)+'\n')
for r in rows:print(r['E'],r['pair'],*(f"{k} {r[k+'_mean_pct']:+.2f}±{r[k+'_SEM_pct']:.2f}" for k in ('build','peak','deep')),f"sys上限 {r['systematic_rms_upper95_pct_independent_bins']:.2f}")
