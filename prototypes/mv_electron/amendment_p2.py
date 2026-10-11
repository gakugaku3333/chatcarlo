"""Approved amendment 1: new verdict only; old Q1 and photon data stay intact."""
import json
import numpy as np
from .p2_common import OUT,ROOT,write,sha,progress
from .photon_data_p2 import TABLE,coefficients,kn_total

def q1_amended():
    old=OUT/'q1.json'
    if (OUT/'q1_amendment1.json').exists():
        result=json.loads((OUT/'q1_amendment1.json').read_text())
        if result['old_q1_sha256']!=sha(old) or result['table_sha256']!=sha(TABLE):
            raise RuntimeError('Q1 inputs changed')
        return result
    historical=json.loads(old.read_text())
    p=dict(np.load(TABLE));mid=p['mid'];direct=p['mid_direct']
    mu=np.array([kn_total(float(e)) for e in mid])+direct[:,2:5].sum(axis=1)
    rows=[]
    for j,name in enumerate(('photoelectric','pair_nuclear','pair_electron'),1):
        value=np.array([coefficients(float(e),p['E'],p['water'],p['thresholds'])[j] for e in mid])
        ref=direct[:,j+1];error=abs(value-ref)
        limit=.002*ref if j==1 else np.maximum(.002*ref,1e-6*mu)
        finite=bool(np.all(np.isfinite(value)) and np.all(np.isfinite(ref)) and np.all(np.isfinite(mu)))
        zero=bool(np.all(value[ref==0]==0))
        if j>1:zero=zero and bool(np.all(value[mid<=p['thresholds'][j-2]]==0))
        eligible=ref>0;rel=np.divide(error,ref,out=np.zeros_like(ref),where=eligible)
        ratio=np.divide(error,limit,out=np.zeros_like(error),where=limit>0)
        worst=int(np.argmax(rel))
        rows.append(dict(process=name,finite=finite,strict_zero=zero,failures=int(np.sum(error>limit)),
            max_relative=float(rel.max()),worst_MeV=float(mid[worst]),worst_absolute=float(error[worst]),
            worst_error_over_mu=float(error[worst]/mu[worst]),max_error_over_limit=float(ratio.max()),
            points=int(len(mid)),positive_points=int(eligible.sum()),
            status='合格' if finite and zero and np.all(error<=limit) else '不合格'))
    tests=dict(historical['tests']);tests['c']=all(r['status']=='合格' for r in rows)
    result=dict(status='合格' if all(tests.values()) else '不合格',tests=tests,interpolation=rows,
        old_q1_sha256=sha(old),table_sha256=sha(TABLE),historical_status=historical['status'],
        rule='photoelectric: 0.002 sigma_ref; pairs: max(0.002 sigma_ref, 1e-6 mu_ref)',
        mu_ref='direct XCOM photoelectric + nuclear/electron pair coefficients + analytic free-electron KN',
        disclosure='Criterion approved after the two worst near-threshold values were known; no table/grid/interpolation/seed change; Q1(a,b,d,e) saved unchanged.')
    write('q1_amendment1.json',result)
    progress('C1','complete',result=result['status'],historical_q1='q1.json retained unchanged',new_verdict='q1_amendment1.json')
    return result

def cutoff_check():
    from .physics import stopping
    rows=[]
    # RM read from EGS5 block data, not from memory; explicitly distinct from P1 constant.
    import re
    source=ROOT/'docs/egs5_crosscheck/egs5/egs/egs5_block_data.f'
    if not source.exists():
        source=ROOT/'docs/egs5_crosscheck/egs5/egs/egs5_block_set.f'
    text=source.read_text()
    match=re.search(r'(?i)\brm\s*/\s*([\d.]+(?:d[+-]?\d+)?)',text)
    if not match:raise RuntimeError('EGS RM source pattern not found')
    rm=float(np.float32(match[1])) if 'd' not in match[1].lower() else float(match[1].lower().replace('d','e'))
    for path in sorted((OUT/'egs5').glob('selfcheck_*/summary.json')):
        s=json.loads(path.read_text());c=s['cutoffs'];kin=c['ECUT']-rm
        ok=abs(c['AE']-.521)<1e-12 and abs(c['ECUT']-.521)<1e-12 and c['ECUT']>=c['AE']
        rows.append(dict(case=path.parent.name,AE=c['AE'],ECUT=c['ECUT'],RM_MeV=rm,
            kinetic_cutoff_keV=kin*1000,difference_eV=(kin-.01)*1e6,
            estimated_extra_range_nm=(kin-.01)/float(stopping(.01))*1e7,status='合格' if ok else '不合格'))
    result=dict(status='合格' if len(rows)==8 and all(r['status']=='合格' for r in rows) else '不合格',rows=rows,
        source='egs/egs5_block_data.f RM; egs5_hatch.f ECUT=max(ECUT,AE)',
        boundary_caveat='Remaining ~10 keV can move to adjacent bin if the boundary is crossed; bin difference is not bounded by 1 eV.')
    write('t5_cutoffs_amendment1.json',result);progress('C2','complete',result=result['status']);return result

def stopping_amended():
    """D1: judge the SAME saved 401 nodes, preserving the historical verdict."""
    old=OUT/'t5_stopping_amendment1.json'
    sources=[old]+[OUT/f't5_stopping_audit_{mode}.npz' for mode in (0,4)]
    hashes={p.name:sha(p) for p in sources}
    target=OUT/'t5_stopping_amendment2.json'
    if target.exists():
        result=json.loads(target.read_text())
        if result['source_hashes']!=hashes:raise RuntimeError('D1 saved inputs changed')
        return result
    history=json.loads(old.read_text());rows=[]
    for mode in (0,4):
        with np.load(OUT/f't5_stopping_audit_{mode}.npz') as v:
            t=v['T'];cp=v['collision_prototype'];rp=v['radiation_prototype']
            if len(t)!=401 or not np.allclose(t,np.geomspace(.01,10.,401),rtol=1e-12):
                raise RuntimeError('D1 requires the original 401 nodes')
            dc=abs(v['collision_transport']-cp);dr=abs(v['radiation_full']-rp);total=cp+rp
            tests=np.column_stack((dc<=.005*cp,dr<=np.maximum(.005*rp,.0005*total),(dc+dr)<=.0005*total))
            if not np.all(np.isfinite([cp,rp,dc,dr,total])) or np.any(total<=0):
                raise RuntimeError('invalid D1 inputs')
            bad=~np.all(tests,axis=1)
            previous=next(r for r in history['rows'] if r['IUNRST']==mode)
            rows.append(dict(IUNRST=mode,status='判定保留' if np.any(bad) else '合格',
                tests=dict(collision=bool(tests[:,0].all()),radiation=bool(tests[:,1].all()),sum_absolute=bool(tests[:,2].all())),
                max_collision_relative=float(max(dc/cp)),max_radiation_relative=float(max(dr/rp)),
                max_radiation_over_total=float(max(dr/total)),max_sum_absolute_over_total=float(max((dc+dr)/total)),
                exceeding_energies_MeV=t[bad].tolist(),affected_source_energies=[e for e in (1.25,2.,6.,10.) if np.any(bad&(t<=e))],
                warning_functions=previous['warning_functions'],pegs_warnings=previous['pegs_warnings']))
    result=dict(status='合格' if all(r['status']=='合格' for r in rows) else '判定保留',rows=rows,
        source_hashes=hashes,historical_status=history['status'],
        affected_source_energies=sorted(set(e for r in rows for e in r['affected_source_energies'])),
        criterion='D1: dc <= .005 Scoll; dr <= max(.005 Srad,.0005 Stot); (dc+dr)/Stot <= .0005; Stot=prototype restricted collision + full radiation',
        disclosure='Approved after max absolute-component sum/total=0.0286% was known. Only stopping comparison passes; not a bound on bin dose differences.',
        reference_uncertainty='PEGS ESIG/ESTEPMAX and other fit warnings remain unquantified; no refit. P1–P1f reference tables carried the same warnings.')
    write(target,result)
    progress('D1','complete',result=result['status'],active_verdict=target.name,historical_verdict=old.name)
    return result
