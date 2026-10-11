"""Fixed P1f run checkpoints. No EGS5 execution, no added histories/seeds."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from prototypes.mv_electron.p1f_common import *

def table():
    g=np.load(WORK/'gs.npz')
    validate_gs(g)
    return g,tuple(g[x] for x in ('T','K','u','y')),dict(backend_rates=g['rates'],backend_single=g['single'])

def checkpoint(path,conditions,compute):
    verify()
    receipt=path.with_suffix('.json')
    pending=path.with_suffix('.pending.npz')
    if not path.exists() and pending.exists() and receipt.exists():
        r=json.loads(receipt.read_text())
        if r['conditions']!=conditions or r['sha256']!=sha(pending) or r['table_sha256']!=sha(WORK/'gs.npz') or r['elsepa_table_hashes']!=table_hashes():raise RuntimeError('pending completion mismatch')
        pending.replace(path)
    if path.exists():
        r=json.loads(receipt.read_text())
        if r['conditions']!=conditions or r['sha256']!=sha(path) or r['table_sha256']!=sha(WORK/'gs.npz') or r['elsepa_table_hashes']!=table_hashes():raise RuntimeError('completed run mismatch')
        return
    write(OUT/'progress'/f'{path.parent.name}_{path.stem}.json',dict(active=display(path),conditions=conditions,resume_command='.venv/bin/python prototypes/mv_electron/run_p1f.py'))
    start=time.perf_counter();values=compute();seconds=time.perf_counter()-start
    pending=path.with_suffix('.pending.npz');save(pending,**values,seconds=seconds)
    write(receipt,dict(conditions=conditions,sha256=sha(pending),seconds=seconds,table_sha256=sha(WORK/'gs.npz'),elsepa_table_hashes=table_hashes(),IHEF=0))
    pending.replace(path); verify()
    print('run saved',path.name,round(seconds,3),flush=True)

def fixed_runs():
    from prototypes.mv_electron.reference import fixed
    _,args,kw=table()
    for length in (.05,.2):
        for condensed in (False,True):
            seed=FIXED_SEED if condensed else SEED
            path=OUT/'elsepa/fixed'/f'{length}_{condensed}.npz'
            conditions=dict(T=2.,N=1000000,seed=seed,K=.025,path_cm=length,condensed=condensed)
            def compute():
                fixed(2.,length,1,seed,condensed,.025,*args,**kw)
                return dict(values=fixed(2.,length,1000000,seed,condensed,.025,*args,**kw),**conditions)
            checkpoint(path,conditions,compute)

def transport_run(name):
    from prototypes.mv_electron.transport import run
    g,args,kw=table(); cfg=CONFIGS[name]
    validate_gs(g,cfg['T']);T,n,k,fe=[cfg[x] for x in ('T','N','K','fe')]
    thick,dz=(1.2,.01) if T==2 else (6.,.05)
    faces=np.array([0.,thick]) if not cfg['split'] else np.unique(np.r_[0.,np.arange(cfg['shift'] or cfg['split'],thick,cfg['split']),thick])
    conditions=dict(**cfg,seed=SEED,backend='elsepa')
    def compute():
        start=time.perf_counter();run(T,1,SEED,fe,k,faces,dz,*args,mode=cfg['mode'],**kw);jit=time.perf_counter()-start
        r=run(T,n,SEED,fe,k,faces,dz,*args,mode=cfg['mode'],**kw)
        v=dict(zip(KEYS,r));v['batches']=r[0]/(n/100*dz)
        return dict(**v,**conditions,faces=faces,dz=dz,jit_seconds=jit)
    checkpoint(OUT/'elsepa/runs'/f'{name}.npz',conditions,compute)

def benchmark():
    from prototypes.mv_electron.transport import run
    g,args,kw=table();target=OUT/'elsepa/benchmark.json'
    saved=json.loads(target.read_text()) if target.exists() else dict(rows=[])
    for T in (2.,10.):
        for scoring in (True,False):
            row=next((r for r in saved['rows'] if r['T']==T and r['scoring']==scoring),None)
            if row is None:
                row=dict(T=T,scoring=scoring,N=20000,K=.025,fe=.05,seed=SEED,wall_seconds=[]);saved['rows'].append(row)
            faces=np.array([0.,1.2 if T==2 else 6.]);dz=.01 if T==2 else .05
            if len(row['wall_seconds'])==3:continue
            run(T,1,SEED,.05,.025,faces,dz,*args,scoring=scoring,**kw)
            while len(row['wall_seconds'])<3:
                verify();start=time.perf_counter();run(T,20000,SEED,.05,.025,faces,dz,*args,scoring=scoring,**kw)
                row['wall_seconds'].append(time.perf_counter()-start);row['us_per_primary']=float(np.median(row['wall_seconds'])/20000*1e6)
                saved['method']='single process, median 3 x 20000, fixed seed, JIT excluded, statistics not pooled'
                saved['table_bytes']=sum(g[k].nbytes for k in g.files);write(target,saved)

def main():
    from prototypes.mv_electron.run_checks_p1f import evaluate,prerequisites
    p=argparse.ArgumentParser();p.add_argument('--run',choices=('fixed',*CONFIGS));a=p.parse_args()
    verify();prerequisites()
    if json.loads((OUT/'t4.json').read_text())['status']!='pass':raise RuntimeError('T4 not passed; no transport')
    if a.run=='fixed':fixed_runs();return
    if a.run:
        from prototypes.mv_electron.run_checks_p1f import r1
        if r1()['status']!='pass':raise RuntimeError('K9(i) not passed; no transport')
        transport_run(a.run);return
    fixed_runs();s=evaluate()
    if s['T5']['R1']['status']!='pass':return
    from prototypes.mv_electron.run_checks_p1f import k5b
    k5b(execute=True)
    # Each run has an independent process and reseeds its own Numba stream.
    # Parent reports only after all jobs complete; workers never rewrite shared reports.
    import subprocess
    from concurrent.futures import ThreadPoolExecutor,as_completed
    def worker(name):
        command=['.venv/bin/python','prototypes/mv_electron/run_p1f.py','--run',name]
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        (OUT/'elsepa'/f'{name}.log').write_text(sanitize(result.stdout+result.stderr))
        write(OUT/'elsepa'/f'{name}_command.json',dict(command=' '.join(command),exit_code=result.returncode))
        if result.returncode:raise RuntimeError('run worker stopped: '+name)
        return name
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(worker,name) for name in CONFIGS]
        for future in as_completed(futures):
            name=future.result();print('completed',name,flush=True)
            write('progress.json',dict(last_completed=name,resume_command='.venv/bin/python prototypes/mv_electron/run_p1f.py'))
    evaluate()
    benchmark();evaluate();write('progress.json',dict(status='complete'))

if __name__=='__main__':main()
