"""P1b saved-data replay. Never writes p1 and never runs EGS5."""
import argparse,sys,json,hashlib,contextlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from prototypes.mv_electron.run_m1_p1b import verify_p1
P=ROOT/'docs/validation/mv/p1b'

def reanalyse_p1():
    from prototypes.mv_electron.check_transport import main as transport
    from prototypes.mv_electron.check_egs5 import main as egs
    d=P/'p1_reanalysis';d.mkdir(exist_ok=True)
    with (d/'reanalysis.log').open('w') as f,contextlib.redirect_stdout(f):transport(d);egs(d)
    checks={name:json.loads((d/(name+'.json')).read_text())==json.loads((ROOT/'docs/validation/mv/p1'/(name+'.json')).read_text()) for name in ['transport_checks','egs5_checks']}
    # Fixed-energy displacement statistics are recomputed from the saved samples.
    import numpy as np
    fixed=json.loads((ROOT/'docs/validation/mv/p1/fixed_checks.json').read_text())
    fixed_ok=True
    for row in fixed['K9i']['rows']:
        raw=np.load(ROOT/'docs/validation/mv/p1/data'/f"fixed_{row['path_cm']}_{row['K']}.npz")
        ref,pro=raw['reference'],raw['prototype'];n=len(ref)
        diff=(pro.mean(0)-ref.mean(0))/ref.mean(0)
        sem=np.sqrt(pro.var(0,ddof=1)/n+ref.var(0,ddof=1)/n)/ref.mean(0)
        for key,value in [('reference',ref.mean(0)),('prototype',pro.mean(0)),('relative',diff),('difference_sem_relative',sem)]:
            fixed_ok &= bool(np.array_equal(value,np.asarray(row[key])))
        fixed_ok &= row['pass']==bool(max(abs(diff[:2]))<=.02 and max(sem[:2])<=.02/3)
    checks['fixed_checks_values']=fixed_ok
    original=json.loads((ROOT/'docs/validation/mv/p1/gs_checks.json').read_text())['K3']
    raw=np.load(ROOT/'docs/validation/mv/p1/data/gs.npz')
    if (d/'gs_value_verification.json').exists():checks['gs_report_values_and_decisions']=json.loads((d/'gs_value_verification.json').read_text())['all_reported_values_and_decisions_identical']
    checks['gs_saved_table_values']=bool(float(max(abs(raw['normalization']-1)))==original['normalization_max_error'] and bool(np.all(np.diff(raw['y'],axis=2)>=0))==original['cdf_monotone'])
    (P/'m1_reanalysis.json').write_text(json.dumps(checks,indent=2)+'\n')
    if not all(checks.values()):raise RuntimeError('M1 saved reanalysis mismatch: stop')
    return checks

def replay_m4(name):
    """Recompute decisions and chi-square from saved statistical aggregates."""
    import numpy as np
    from scipy.stats import chisquare
    path=P/f'm4_{name}.json';r=json.loads(path.read_text())
    if 'K3_rows' not in r:return r
    for row in r['K3_rows']:
        actual=np.asarray(row['moments']);expected=np.asarray(row['expected']);sem=np.asarray(row['sem'])
        tol=np.where(expected>=.05,.01*expected,.001)
        assert row['relative']==((actual-expected)/expected).tolist()
        assert row['pass']==bool(np.all(abs(actual-expected)<=tol) and np.all(sem<=tol/3))
    for row in r['refinement']:
        change=float(max(abs(np.asarray(row['moments'])-row['refined_moments'])))
        inverse=float(max(abs(np.asarray(row['inverse_table_moments'])-row['refined_inverse_table_moments'])))
        assert change==row['moment_change'] and inverse==row['inverse_table_moment_change']
        assert row['pass']==bool(change<=1e-4 and inverse<=1e-4)
    for row in r['cdf_tests']:
        p=float(chisquare(row['observed'],row['expected']).pvalue)
        assert p==row['chi_p']
        assert row['pass']==bool(p>.001 and min(row['expected'])>=100)
    decision=all(a['pass'] for a in r['K3_rows']+r['refinement']+r['cdf_tests']) and r['normalization_max_error']<=1e-6 and r['cdf_monotone']
    assert r['status']==('pass' if decision else 'fail')
    return r

def replay_m5(name,out):
    """Recompute every numerical M5 decision from saved raw histories/samples."""
    import numpy as np
    from prototypes.mv_electron import check_transport
    original=json.loads((P/f'm5_{name}.json').read_text())
    old=check_transport.OUT
    try:
        check_transport.OUT=out
        with (out/'transport_replay.log').open('w') as f,contextlib.redirect_stdout(f):check_transport.main(out)
    finally:check_transport.OUT=old
    result=json.loads((out/'transport_checks.json').read_text())
    fixed=json.loads((out/'fixed_checks.json').read_text())
    for row in fixed['K9i']['rows']:
        raw=np.load(out/'data'/f"fixed_{row['path_cm']}_{row['K']}.npz")
        ref,pro=raw['reference'],raw['prototype'];n=len(ref)
        diff=(pro.mean(0)-ref.mean(0))/ref.mean(0)
        sem=np.sqrt(pro.var(0,ddof=1)/n+ref.var(0,ddof=1)/n)/ref.mean(0)
        row.update(reference=ref.mean(0).tolist(),prototype=pro.mean(0).tolist(),relative=diff.tolist(),difference_sem_relative=sem.tolist(),statistics_ok=bool(max(sem[:2])<=.02/3),pass_=bool(max(abs(diff[:2]))<=.02 and max(sem[:2])<=.02/3))
        row['pass']=row.pop('pass_')
    fixed['K9i']['status']='pass' if all(r['pass'] for r in fixed['K9i']['rows']) else 'fail'
    result.update(fixed)
    result['status']='pass' if all(result[k]['status']=='pass' for k in ['K5','K5b','K7','K8','K9i','K9ii']) else 'fail'
    if result!=original:raise RuntimeError(f'{name}: saved M5 values do not reproduce')
    (P/f'm5_{name}.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def saved():
    verify_p1();reanalyse_p1()
    identities=json.loads((P/'m1_identity.json').read_text())
    if not all(identities.values()):raise RuntimeError('M1 failed')
    from prototypes.mv_electron.check_dcs_p1b import inputs,scattering
    for name in ['dcslib','eedl']:
        result=inputs(name);original=json.loads((P/f'm2_{name}.json').read_text())
        if result!=original:raise RuntimeError(f'{name}: M2 saved inputs disagree')
    if scattering()!=json.loads((P/'m3.json').read_text()):raise RuntimeError('M3 saved inputs disagree')
    summary={'M1':{'status':'pass','identity':identities,'p1_reanalysis':json.loads((P/'m1_reanalysis.json').read_text()),'p1_untouched':verify_p1()},'M3':json.loads((P/'m3.json').read_text()),'backends':{}}
    from prototypes.mv_electron.compare_p1b import main as comparison
    for name in ['dcslib','eedl']:
        out=P/name;row={}
        replay_m4(name)
        if (P/f'm5_{name}.json').exists():replay_m5(name,out)
        for number in [2,4,5]:
            f=P/f'm{number}_{name}.json';row[f'M{number}']=json.loads(f.read_text()) if f.exists() else {'status':'not-evaluated'}
        eligible=all(row[f'M{n}']['status']=='pass' for n in [2,4,5])
        if eligible:
            result=comparison(name,out);row.update(result)
            (P/f'comparison_{name}.json').write_text(json.dumps(result,indent=2)+'\n')
        else:
            row['M6']={'status':'not-evaluated','reason':'M2/M4/M5 prerequisite not met'};row['M7']=row['M6'].copy()
            row['reduction']={'status':'not-evaluated','reason':'prerequisite not met; no residual conclusion'}
        if (out/'benchmark.json').exists():row['benchmark']=json.loads((out/'benchmark.json').read_text())
        summary['backends'][name]=row
    (P/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    from prototypes.mv_electron.report_p1b import report,plots,manifest
    report(summary);plots(summary);manifest()
    print('P1b replay complete; physical pass/fail statuses in summary.json and RESULTS.md')

def main():
    a=argparse.ArgumentParser();a.add_argument('--from-saved',action='store_true');a.add_argument('--verify-p1-untouched',action='store_true');v=a.parse_args()
    if v.verify_p1_untouched:print(json.dumps(verify_p1(),indent=2));return
    if not v.from_saved:a.error('choose --from-saved or --verify-p1-untouched')
    saved()
if __name__=='__main__':main()
