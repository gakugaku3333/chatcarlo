"""P1c saved-data replay; no EGS5 execution and no p1/p1b writes."""
import argparse,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.stats import chisquare
from prototypes.mv_electron.regression_p1c import P,write,verify,KEYS

def replay_n1():
    checks={}
    for name in ['rutherford','dcslib','eedl']:
        for suffix in ['']+([] if name=='rutherford' else ['_table']):
            a=np.load(P/'regression'/f'{name}{suffix}_before.npz');b=np.load(P/'regression'/f'{name}{suffix}_after.npz')
            checks[name+suffix]=set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
    old=np.load(ROOT/'docs/validation/mv/p1b/m1_baseline.npz');a=np.load(P/'regression/rutherford_before.npz')
    checks['p1b_rutherford_baseline']=all(np.array_equal(old[k],a[k]) for k in KEYS)
    result=json.loads((P/'n1.json').read_text())
    assert all(checks.values()) and result['status']=='pass' and all(all(v.values()) for v in result['checks'].values())
    return result

def replay_n4():
    r=json.loads((P/'n4.json').read_text());provider=__import__('prototypes.mv_electron.dcs',fromlist=['get_backend']).get_backend('mf')
    from prototypes.mv_electron.dcs import moments
    for row in r['K3_rows']:
        G,*_=moments(provider,row['T']);expected=np.exp(-row['K']*G[1:3]/G[1]);a=np.asarray(row['moments']);sem=np.asarray(row['sem']);tol=np.where(expected>=.05,.01*expected,.001)
        assert expected.tolist()==row['expected'] and ((a-expected)/expected).tolist()==row['relative']
        assert row['pass']==bool(np.all(abs(a-expected)<=tol) and np.all(sem<=tol/3))
    for row in r['refinement']:
        change=float(max(abs(np.asarray(row['moments'])-row['refined_moments'])))
        inverse=float(max(abs(np.asarray(row['inverse_table_moments'])-row['refined_inverse_table_moments'])))
        assert change==row['moment_change'] and inverse==row['inverse_table_moment_change']
        assert row['pass']==bool(change<=1e-4 and inverse<=1e-4)
    for row in r['cdf_tests']:
        p=float(chisquare(row['observed'],row['expected']).pvalue)
        assert p==row['chi_p'] and row['pass']==bool(p>.001 and min(row['expected'])>=100)
    g=np.load(P/'mf/data/gs.npz')
    assert r['normalization_max_error']==float(max(abs(g['normalization']-1)))
    assert r['cdf_monotone']==bool(np.all(np.diff(g['y'],axis=2)>=0))
    isolated=r['rutherford_call_isolation'];passed=all(x['pass'] for x in r['K3_rows']+r['refinement']+r['cdf_tests']) and r['normalization_max_error']<=1e-6 and r['cdf_monotone'] and isolated['exit_code']==0 and 'skipped' not in isolated['output']
    assert r['status']==('pass' if passed else 'fail')
    assert len(r['K3_rows'])==24 and len(r['refinement'])==24 and len(r['cdf_tests'])==7
    return r

def auxiliary(comparison):
    from prototypes.mv_electron.check_egs5 import parse
    from prototypes.mv_electron.analysis import dref
    from prototypes.mv_electron.dcs import get_backend,moments
    ref=parse('base_2');D=dref(ref['batches'].mean(0));mask=ref['q']>=.2*dref(ref['q']);z=ref['z']
    mf=np.load(P/'mf/runs/base_2.npz')['batches'].mean(0);dc=np.load(ROOT/'docs/validation/mv/p1b/dcslib/runs/base_2.npz')['batches'].mean(0)
    diff=mf-dc;eligible=np.flatnonzero(mask);i=int(eligible[np.argmax(abs(diff[mask]))]);r50=comparison['M6']['reference_ranges_cm'][0];tail=z>r50;deep=mask&tail
    p0=[]
    for name in ['mf','dcslib']:
        b=get_backend(name);g=np.load(P/'mf/data/gs.npz' if name=='mf' else ROOT/'docs/validation/mv/p1b/dcslib/data/gs.npz')
        for T in [.01,.02,2.]:
            G,total,_=moments(b,T)
            for K in [.05,float(g['K'].min())]:
                exponent=K*total/G[1];p0.append(dict(backend=name,T=T,K=K,total=total,G1=float(G[1]),negative_log_p0=float(exponent),p0=float(np.exp(-exponent))))
    result={'mf_minus_dcslib':{'Dref':D,'evaluation_mask':'same EGS5 >=0.2 Dref mask as N6','max_difference_over_Dref':float(abs(diff[i])/D),'signed_difference_over_Dref':float(diff[i]/D),'depth_cm':float(z[i]),'beyond_R50_all_bins_max_over_Dref':float(max(abs(diff[tail]))/D),'beyond_R50_eligible_bins_max_over_Dref':float(max(abs(diff[deep]))/D),'reference_R50_cm':r50,'depth_cm_array':z.tolist(),'difference_over_Dref':(diff/D).tolist()},'p0':p0}
    write('auxiliary.json',result);return result

def saved():
    verify();summary={'N1':replay_n1()}
    from prototypes.mv_electron.check_dcs_p1c import inputs,scattering,interpolation_midpoints
    old2=json.loads((P/'n2.json').read_text());old3=json.loads((P/'n3.json').read_text())
    g=np.load(P/'mf/data/gs.npz') if (P/'mf/data/gs.npz').exists() else None
    mid=[] if g is None else interpolation_midpoints(g)
    summary['N2']=inputs(mid);summary['N3']=scattering()
    assert summary['N2']==old2 and summary['N3']==old3
    summary['N4']=replay_n4() if (P/'n4.json').exists() else {'status':'not-evaluated'}
    if (P/'n5.json').exists():
        from prototypes.mv_electron.run_p1c import fixed_checks,transport_checks
        original=json.loads((P/'n5.json').read_text());fixed_checks();summary['N5']=transport_checks();assert summary['N5']==original
    else:summary['N5']={'status':'not-evaluated'}
    eligible=all(summary[k]['status']=='pass' for k in ['N1','N2','N4','N5'])
    if eligible and (P/'mf/runs/base_10.npz').exists():
        from prototypes.mv_electron.compare_p1b import main as comparison
        c=comparison('mf',P/'mf');summary.update(N6=c['M6'],N7=c['M7'],reduction=c['reduction'])
        write('comparison.json',c)
    else:
        failed=[k for k in ['N1','N2','N4','N5'] if summary[k]['status']!='pass']
        reason=('、'.join(failed)+'が未達：事前登録の停止規則に従い未判定') if failed else '比較用runが未完了'
        for key in ['N6','N7']:summary[key]={'status':'not-evaluated','reason':reason}
    if (P/'mf/runs/base_2.npz').exists():
        # Input reference R50 is already fixed in P1. These are record-only
        # prototype-to-prototype and DCS metrics, never an N6/N7 decision.
        baseline=json.loads((ROOT/'docs/validation/mv/p1/egs5_checks.json').read_text())
        summary['auxiliary']=auxiliary({'M6':baseline['K6']})
    if (P/'benchmark.json').exists():summary['benchmark']=json.loads((P/'benchmark.json').read_text())
    write('summary.json',summary)
    from prototypes.mv_electron.report_p1c import report,plots,manifest
    report(summary);plots(summary);manifest();print('P1c saved replay complete',flush=True)
    return summary

def main():
    a=argparse.ArgumentParser();a.add_argument('--from-saved',action='store_true');a.add_argument('--verify-untouched',action='store_true');v=a.parse_args()
    if v.verify_untouched:print(json.dumps(verify(),indent=2));return
    if not v.from_saved:a.error('choose --from-saved or --verify-untouched')
    saved()
if __name__=='__main__':main()
