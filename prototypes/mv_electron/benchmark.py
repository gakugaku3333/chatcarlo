"""T6 isolated, JIT-excluded scalar per-primary measurements; fixed seeds."""
import json,time,hashlib,datetime
from pathlib import Path
import numpy as np
from .physics import ROOT
from .component_checks import OUT,SEED
from .transport import run,FIELDS
def main():
    t=time.perf_counter();g=np.load(OUT/'data/gs.npz');args=tuple(g[x] for x in ['T','K','u','y']);init=time.perf_counter()-t
    rows=[];n=20000
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    for T in [2.,10.]:
        dz=.01 if T==2 else .05;faces=np.array([0.,1.2 if T==2 else 6.])
        for fe,k in [(.05,.05),(.05,.025),(.025,.05),(.025,.025)]:
            for scoring in [True,False]:
                t=time.perf_counter();run(T,1,SEED,fe,k,faces,dz,*args,scoring);jit=time.perf_counter()-t
                times=[];cs=[];ledgers=[]
                for rep in range(3):
                    t=time.perf_counter();b,l,c,*_=run(T,n,SEED+rep,fe,k,faces,dz,*args,scoring);times.append(time.perf_counter()-t);cs.append(c/n);ledgers.append(float(max(abs(l[:,3]))/T))
                rows.append({'T':T,'fe':fe,'K':k,'scoring':scoring,'N':n,'replicates':3,'wall_seconds':times,'us_per_primary':float(np.median(times)/n*1e6),'counts_per_primary':np.mean(cs,axis=0).tolist(),'warmup_seconds_excluded':jit,'max_ledger_relative':max(ledgers)})
                print(T,fe,k,scoring,rows[-1]['us_per_primary'],flush=True)
    result={'rows':rows,'table_load_initialization_seconds':init,'table_build_seconds':float(g['build_seconds']),'table_bytes':sum(g[x].nbytes for x in ['T','K','u','y','p0','G1','total']),'stack_allocated_bytes':4096*FIELDS*8,'seed':SEED,'started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},'method':'Single process, scalar Numba; median of 3 timed runs per condition. Compilation/warmup excluded; other validation subprocesses must have completed before starting.'}
    (OUT/'benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
