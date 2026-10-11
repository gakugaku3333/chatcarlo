"""P1e exact frozen replay, stop gates, reports and main calculation entry."""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from prototypes.mv_electron.p1e_common import *

def prerequisites():
    for name in ('s1.json','s2.json'):
        if not (OUT/name).exists() or json.loads((OUT/name).read_text())['status']!='pass':
            raise RuntimeError(name+' not passed; S4 onwards forbidden')

def r1():
    rows=[]
    for length in (.05,.2):
        paths=[OUT/'elsepa/fixed'/f'{length}_{c}.npz' for c in (False,True)]
        if not all(p.exists() for p in paths):continue
        values=[]
        for p,c in zip(paths,(False,True)):
            receipt=json.loads(p.with_suffix('.json').read_text());assert receipt['sha256']==sha(p)
            with np.load(p) as a:
                assert int(a['N'])==1000000 and int(a['seed'])==(FIXED_SEED if c else SEED) and float(a['K'])==.025
                values.append(a['values'])
        ref,pro=values;n=len(ref);difference=(pro.mean(0)-ref.mean(0))/ref.mean(0)
        sem=np.sqrt(pro.var(0,ddof=1)/n+ref.var(0,ddof=1)/n)/ref.mean(0)
        check=dict(statistics_ok=bool(max(sem[:2])<=.005),**{'pass':bool(max(abs(difference[:2]))<=.02)})
        rows.append(dict(path_cm=length,relative=difference.tolist(),difference_SEM_relative=sem.tolist(),reference=ref.mean(0).tolist(),prototype=pro.mean(0).tolist(),N=n,reference_seed=SEED,prototype_seed=FIXED_SEED,status=status(check)))
    return dict(status=aggregate([r['status'] for r in rows]+([] if len(rows)==2 else ['not-evaluated'])),rows=rows)

def validate_run(name):
    p=OUT/'elsepa/runs'/f'{name}.npz';receipt=json.loads(p.with_suffix('.json').read_text())
    expected=dict(**CONFIGS[name],seed=SEED,backend='elsepa')
    assert receipt['sha256']==sha(p) and receipt['conditions']==expected
    with np.load(p) as a:
        for k,v in expected.items():assert a[k]==v
    return p

def r2():
    needed=list(CONFIGS)[:-1];available=[n for n in needed if (OUT/'elsepa/runs'/f'{n}.npz').exists()]
    for n in available:validate_run(n)
    if len(available)!=len(needed):return dict(status='not-evaluated',completed=available,pending=[n for n in needed if n not in available])
    from prototypes.mv_electron import check_transport
    old=check_transport.OUT
    try:
        check_transport.OUT=OUT/'elsepa'
        with contextlib.redirect_stdout(io.StringIO()):check_transport.main(OUT/'elsepa')
    finally:check_transport.OUT=old
    result=json.loads((OUT/'elsepa/transport_checks.json').read_text())
    result['K5b']=json.loads((OUT/'k5b.json').read_text()) if (OUT/'k5b.json').exists() else dict(status='not-evaluated')
    for k in ('K7','K8'):
        for row in result[k]['rows'].values():row['status']=status(row)
        result[k]['status']=aggregate(row['status'] for row in result[k]['rows'].values())
    result['status']=aggregate(result[k]['status'] for k in ('K5','K5b','K7','K8','K9ii'))
    return result

def comparison(T):
    from prototypes.mv_electron.check_egs5 import parse
    from prototypes.mv_electron.analysis import compare,dref
    ref=parse(f'base_{T}');path=validate_run(f'base_{T}');mask=ref['q']>=.2*dref(ref['q'])
    with np.load(path) as p:
        c=compare(ref['batches'],p['batches'],ref['z'],tol=.03,range_tol=.03,mask=mask,bootstrap=2000 if T==2 else 1000)
        c['back_energy_fraction']=float(p['ledger'][:,1].sum()/(T*int(p['N'])))
    c['status']=status(c) if T==2 else 'not-evaluated';c['mask_indices']=np.flatnonzero(mask).tolist()
    c['EGS5_back_energy_fraction']=ref['ledger'][2]/ref['ledger'][0]
    return c

def auxiliary():
    from prototypes.mv_electron.check_egs5 import parse
    from prototypes.mv_electron.analysis import dref,ranges
    from prototypes.mv_electron.run_checks_p1d import record_curve
    result={};ref=parse('base_2');D=dref(ref['batches'].mean(0));mask=ref['q']>=.2*dref(ref['q'])
    for b in ('elsepa','dcslib','mf'):
        path=OUT/'elsepa/runs/base_2.npz' if b=='elsepa' else ROOT/f'docs/validation/mv/p1d/{b}/runs/base_2.npz'
        if path.exists():result[b]=record_curve(path,ref)
    our=OUT/'elsepa/runs/base_2.npz';control=ROOT/'docs/validation/mv/p1d/dcslib/runs/base_2.npz'
    if our.exists() and control.exists():
        with np.load(our) as a,np.load(control) as b:
            aa,bb=a['batches'],b['batches'];diff=aa.mean(0)-bb.mean(0);sem=np.sqrt(aa.var(0,ddof=1)/len(aa)+bb.var(0,ddof=1)/len(bb))
            ii=np.flatnonzero(mask);i=int(ii[np.argmax(abs(diff[mask]))])
            result['elsepa_minus_dcslib']=dict(max_difference_over_Dref=float(abs(diff[i])/D),depth_cm=float(ref['z'][i]),difference_over_combined_SEM=float(diff[i]/sem[i]),difference_over_Dref=(diff/D).tolist(),combined_SEM_over_Dref=(sem/D).tolist(),comparison='elsepa seed20261011 vs P1d saved dcslib seed20261009')
    return result

def replay_s2():
    from prototypes.mv_electron.check_elsepa import s2,s3
    # loaded() must be read-only: every required direct case already completed.
    if not (OUT/'direct_cases.json').exists():raise RuntimeError('saved direct cases incomplete; no computation in replay')
    from prototypes.mv_electron.generate_elsepa import prepare,energies
    # Validate completeness before any loaded() call: no executable may be started.
    for row in json.loads((OUT/'direct_cases.json').read_text())['rows']:
        group='ihef0' if row.get('IHEF')==0 else 'midpoints';h=row.get('IHEF',1)
        base=WORK/'cases'/group/f"z{row['Z']}_q{row['charge']}_e{float(row['T']).hex()}_h{h}"
        if not (base/'complete.json').exists():raise RuntimeError('saved direct receipt missing')
    from prototypes.mv_electron import generate_elsepa
    previous=generate_elsepa.READ_ONLY;generate_elsepa.READ_ONLY=True
    try:return s2(),s3()
    finally:generate_elsepa.READ_ONLY=previous

def regression_saved():
    r=json.loads((OUT/'regression.json').read_text());assert r['status']=='pass'
    before=json.loads((OUT/'regression_before.json').read_text())
    for b in TABLES:
        p=OUT/'regression'/f'{b}_before.npz'
        assert sha(p)==before[b]['sha256']
        with np.load(p) as a,np.load(p.with_name(f'{b}_after.npz')) as z:
            assert set(a.files)==set(z.files) and all(np.array_equal(a[k],z[k]) for k in a.files)
    return r

def evaluate():
    frozen=verify();initial=verify_initial()
    reg=regression_saved()
    summary={'S0':dict(status='pass',freeze=frozen,initial=initial,regression=reg)}
    for key in ('S1','S2','S3','S4'):
        p=OUT/(key.lower()+'.json');summary[key]=json.loads(p.read_text()) if p.exists() else dict(status='not-evaluated')
    permitted=all(summary[k]['status']=='pass' for k in ('S0','S1','S2'))
    if not permitted:
        summary['S4']=dict(status='not-evaluated',reason='S2不合格（または未完了）。停止規則によりS4以降を判定しない。')
        summary['S5']=dict(status='not-evaluated',reason='S2停止規則')
    else:
        one=r1();two=r2();summary['S5']=dict(status=aggregate([one['status'],two['status']]),R1=one,R2=two)
    summary['S6']=dict(status='not-evaluated',reason='S0,S1,S2,S4,S5合格時のみ判定')
    if all(summary[k]['status']=='pass' for k in ('S0','S1','S2','S4','S5')):
        reference=json.loads((ROOT/'docs/validation/mv/p1/egs5_checks.json').read_text())['K4']['status']
        if reference=='pass' and (OUT/'elsepa/runs/base_2.npz').exists():summary['S6']=comparison(2)
    summary['S7']=comparison(10) if permitted and (OUT/'elsepa/runs/base_10.npz').exists() else dict(status='not-evaluated',reason='参考run未実行。合否規則なし。')
    summary['auxiliary']=auxiliary() if permitted else dict(status='not-evaluated',reason='S2停止規則。新規輸送と補助EGS5比較を実行せず。')
    write('summary.json',summary)
    from prototypes.mv_electron.report_p1e import report
    report(summary);return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--from-saved',action='store_true');p.add_argument('--verify-untouched',action='store_true');p.add_argument('--calculate',action='store_true');p.add_argument('--build-gs',action='store_true');a=p.parse_args()
    if a.build_gs:
        if (OUT/'freeze_hashes.json').exists():raise RuntimeError('frozen inputs cannot be extended with GS; build before freeze')
        verify_initial()
        from prototypes.mv_electron.elsepa import Elsepa
        from prototypes.mv_electron.gs_backend import build
        if not (WORK/'gs.npz').exists():
            pending=WORK/'gs.pending.npz';build(Elsepa(),pending);pending.replace(WORK/'gs.npz')
        print('ELSEPA GS prepared, no S4 judgment');return
    if a.freeze:
        verify_initial();regression('after')
        if not (OUT/'s1.json').exists() or json.loads((OUT/'s1.json').read_text())['status']!='pass':raise RuntimeError('S1 incomplete; no freeze')
        if (OUT/'freeze_hashes.json').exists():verify()
        else:write('freeze_hashes.json',targets())
        print('S0 implementation frozen',verify());return
    if a.verify_untouched:
        regression_saved();result=verify();result['initial']=verify_initial();write('untouched.json',result);print(json.dumps(result,ensure_ascii=False));return
    if a.from_saved:
        verify()
        from prototypes.mv_electron.generate_elsepa import verify_saved
        verify_saved()
        if (OUT/'direct_cases.json').exists():replay_s2()
        s=evaluate();print(json.dumps({k:LABELS[s[k]['status']] for k in ('S0','S1','S2','S3','S4','S5','S6','S7')},ensure_ascii=False));return
    if a.calculate:
        verify()
        from prototypes.mv_electron.check_elsepa import point_data,s2,s3
        point_data();result=s2();s3()
        if result['status']=='pass':
            prerequisites()
            if not (WORK/'gs.npz').exists():raise RuntimeError('GS must be built and frozen before main calculation')
            from prototypes.mv_electron.run_p1e import table
            from prototypes.mv_electron.check_gs_p1e import main as check_gs
            g,_,_=table();write('s4.json',check_gs(__import__('prototypes.mv_electron.elsepa',fromlist=['Elsepa']).Elsepa(),g))
        evaluate();return
    p.error('choose --freeze, --from-saved, --verify-untouched or --calculate')

if __name__=='__main__':main()
