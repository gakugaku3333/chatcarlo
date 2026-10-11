"""P1c dedicated gated, resumable calculations. Never executes EGS5.
Checkpoints: GS per energy, N4 per case, transport per completed run,
K9(i) per reference/prototype sample and timing per replicate. Interrupted
active transport run restarts from its fixed seed; completed runs are reused.
"""
import argparse,sys,json,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from prototypes.mv_electron.regression_p1c import P,write,verify,main as regression
from prototypes.mv_electron.component_checks import SEED
OUT=P/'mf'

def tests(node,log):
    command=['.venv/bin/python','-m','pytest',node,'-q']
    r=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    # Sanitize before writing, including pytest error traces.
    text=(r.stdout+r.stderr).replace(str(ROOT),'.').replace(str(Path.home()),'~')
    (P/log).write_text(text)
    return {'command':' '.join(command),'exit_code':r.returncode,'output':text.strip()}

def fixed_checks():
    from prototypes.mv_electron.reference import fixed
    g=np.load(OUT/'data/gs.npz');args=tuple(g[k] for k in ['T','K','u','y']);kw=dict(backend_rates=g['rates'],backend_single=g['single']);rows=[]
    for path in [.05,.2]:
        n=1000000
        refpath=OUT/'data'/f'fixed_reference_{path}.npz'
        if not refpath.exists():
            fixed(2,path,1,SEED,False,.05,*args,**kw);t=time.perf_counter()
            ref=fixed(2,path,n,SEED,False,.05,*args,**kw)
            np.savez(refpath,reference=ref,seconds=time.perf_counter()-t)
        a=np.load(refpath);ref=a['reference'];rt=float(a['seconds'])
        for k in [.05,.025]:
            f=OUT/'data'/f'fixed_{path}_{k}.npz'
            if not f.exists():
                fixed(2,path,1,SEED+1,True,k,*args,**kw);t=time.perf_counter()
                pro=fixed(2,path,n,SEED+1,True,k,*args,**kw)
                np.savez(f,reference=ref,prototype=pro,prototype_seconds=time.perf_counter()-t)
            a=np.load(f);pro=a['prototype'];pt=float(a['prototype_seconds'])
            diff=(pro.mean(0)-ref.mean(0))/ref.mean(0);sem=np.sqrt(pro.var(0,ddof=1)/n+ref.var(0,ddof=1)/n)/ref.mean(0)
            rows.append(dict(path_cm=path,K=k,N=n,reference=ref.mean(0).tolist(),prototype=pro.mean(0).tolist(),relative=diff.tolist(),difference_sem_relative=sem.tolist(),statistics_ok=bool(max(sem[:2])<=.02/3),pass_=bool(max(abs(diff[:2]))<=.02 and max(sem[:2])<=.02/3),reference_seconds=rt,prototype_seconds=pt))
            rows[-1]['pass']=rows[-1].pop('pass_');write('n5_fixed_checkpoint.json',{'rows':rows});print('N5 K9i',path,k,rows[-1]['pass'],flush=True)
    result={'K9i':{'status':'pass' if all(r['pass'] for r in rows) else 'fail','rows':rows},'seed':SEED}
    (OUT/'fixed_checks.json').write_text(json.dumps(result,indent=2)+'\n');return result

def transport_checks():
    from prototypes.mv_electron import check_transport
    import contextlib,io
    old=check_transport.OUT
    try:
        check_transport.OUT=OUT
        with contextlib.redirect_stdout(io.StringIO()):check_transport.main(OUT)
    finally:check_transport.OUT=old
    r=json.loads((OUT/'transport_checks.json').read_text());r.update(json.loads((OUT/'fixed_checks.json').read_text()))
    # Never accept the legacy unconditional K5b status.
    evidence=json.loads((P/'n5_k5b.json').read_text())
    r['K5b']={'status':'pass' if evidence['exit_code']==0 and 'skipped' not in evidence['output'] else 'fail','evidence':evidence}
    r['status']='pass' if all(r[k]['status']=='pass' for k in ['K5','K5b','K7','K8','K9i','K9ii']) else 'fail'
    (OUT/'transport_checks.json').write_text(json.dumps(r,indent=2)+'\n')
    write('n5.json',r);return r

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--benchmark-only',action='store_true');a=parser.parse_args()
    verify()
    if not (P/'n1_before.json').exists():raise RuntimeError('freeze N1 before code changes')
    if not (P/'n1.json').exists():regression()
    if json.loads((P/'n1.json').read_text())['status']!='pass':raise RuntimeError('N1 failed: stop')
    (OUT/'data').mkdir(parents=True,exist_ok=True);(OUT/'runs').mkdir(exist_ok=True)
    if a.benchmark_only:
        from prototypes.mv_electron.benchmark_p1c import main as bm
        bm(OUT);verify();return
    from prototypes.mv_electron.check_dcs_p1c import inputs,scattering,interpolation_midpoints
    from prototypes.mv_electron.dcs import get_backend
    from prototypes.mv_electron.build_gs_p1c import build
    from prototypes.mv_electron.check_gs_p1c import main as gs_checks
    from prototypes.mv_electron.run_transport import execute
    scattering()
    if inputs()['status']!='pass':return
    table=OUT/'data/gs.npz'
    if not table.exists():build(get_backend('mf'),table)
    g=np.load(table);mid=interpolation_midpoints(g)
    if inputs(mid)['status']!='pass':return
    if not (P/'n4.json').exists():
        r=gs_checks(get_backend('mf'),g);isolation=tests('prototypes/mv_electron/tests/test_mf.py::test_no_rutherford_with_actual_mf_table','n4_isolation.log');r['rutherford_call_isolation']=isolation
        if isolation['exit_code'] or 'skipped' in isolation['output']:r['status']='fail'
        write('n4.json',r)
    if json.loads((P/'n4.json').read_text())['status']!='pass':return
    if not (P/'n5_k5b.json').exists():write('n5_k5b.json',tests('prototypes/mv_electron/tests/test_mf.py::test_k5b_with_actual_mf_table','n5_k5b.log'))
    configs=[('base_2',dict(n=1000000)),('split_aligned',dict(n=1000000,split=.1)),('split_shifted',dict(n=1000000,split=.1,shift=.037)),('ms_half',dict(n=1000000,k=.025)),('energy_half',dict(n=1000000,fe=.025)),('reference_2',dict(n=300000,mode=1))]
    for name,kw in configs:
        if not (OUT/'runs'/f'{name}.npz').exists():
            # Remove only unfinished source snapshot for this run; no data lost.
            source=OUT/'runs'/f'{name}_source'
            if source.exists():
                import shutil
                shutil.rmtree(source)
            execute(name,output_dir=OUT,backend='mf',**kw)
        write('progress.json',{'last_completed_run':name,'resume_command':'.venv/bin/python prototypes/mv_electron/run_p1c.py'})
    fixed_checks()
    if transport_checks()['status']!='pass':
        from prototypes.mv_electron.run_checks_p1c import saved
        saved();verify();return
    if not (OUT/'runs/base_10.npz').exists():
        source=OUT/'runs/base_10_source'
        if source.exists():
            import shutil
            shutil.rmtree(source)
        execute('base_10',T=10.,n=1000000,output_dir=OUT,backend='mf')
    from prototypes.mv_electron.run_checks_p1c import saved
    saved();verify()
if __name__=='__main__':main()
