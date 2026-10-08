"""Identical P1 T6 measurement matrix, selected-provider rate/sampling overhead included."""
import json,time,datetime
from pathlib import Path
import numpy as np
from .component_checks import SEED
from .transport import run,FIELDS

def main(backend,out,resume_log=None):
    from .dcs import get_backend
    from .gs_backend import build
    # Measure table generation isolated, preserve the table used for validation runs.
    if resume_log is None:build(get_backend(backend),out/'data/benchmark_gs.npz')
    built=np.load(out/'data/benchmark_gs.npz')
    t=time.perf_counter();g=np.load(out/'data/gs.npz');args=tuple(g[x] for x in ['T','K','u','y']);kw={'backend_rates':g['rates'],'backend_single':g['single']};init=time.perf_counter()-t
    for key in ['T','K','u','y','p0','G1','total','rates','single']:np.testing.assert_array_equal(g[key],built[key])
    rows=[];n=20000
    if resume_log is not None:
        import re
        for T,fe,k,scoring,value in re.findall(r'^'+backend+r' (2\.0|10\.0) (\S+) (\S+) (True|False) (\S+)$',Path(resume_log).read_text(),re.M):
            rows.append({'T':float(T),'fe':float(fe),'K':float(k),'scoring':scoring=='True','N':n,'replicates':3,'wall_seconds':None,'us_per_primary':float(value),'counts_per_primary':None,'warmup_seconds_excluded':None,'max_ledger_relative':None,'provenance':'completed median recovered from interrupted stdout; per-replicate times/counts were only in process memory and were lost'})
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    for T in [2.,10.]:
        dz=.01 if T==2 else .05;faces=np.array([0.,1.2 if T==2 else 6.])
        for fe,k in [(.05,.05),(.05,.025),(.025,.05),(.025,.025)]:
            for scoring in [True,False]:
                if any(r['T']==T and r['fe']==fe and r['K']==k and r['scoring']==scoring for r in rows):continue
                t=time.perf_counter();run(T,1,SEED,fe,k,faces,dz,*args,scoring,**kw);jit=time.perf_counter()-t
                times=[];cs=[];ledgers=[]
                for rep in range(3):
                    t=time.perf_counter();b,l,c,*_=run(T,n,SEED+rep,fe,k,faces,dz,*args,scoring,**kw);times.append(time.perf_counter()-t);cs.append(c/n);ledgers.append(float(max(abs(l[:,3]))/T))
                rows.append({'T':T,'fe':fe,'K':k,'scoring':scoring,'N':n,'replicates':3,'wall_seconds':times,'us_per_primary':float(np.median(times)/n*1e6),'counts_per_primary':np.mean(cs,axis=0).tolist(),'warmup_seconds_excluded':jit,'max_ledger_relative':max(ledgers)})
                print(backend,T,fe,k,scoring,rows[-1]['us_per_primary'],flush=True)
                (out/'benchmark_checkpoint.json').write_text(json.dumps({'rows':rows,'started_utc':started},indent=2)+'\n')
    result={'backend':backend,'rows':rows,'table_load_initialization_seconds':init,'table_build_seconds':float(built['build_seconds']),'table_bytes':sum(g[x].nbytes for x in ['T','K','u','y','p0','G1','total','rates','single']),'stack_allocated_bytes':4096*FIELDS*8,'seed':SEED,'started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':'Single process, median of 3 x 20000 histories; JIT excluded; no other validation processes during timing.'}
    if resume_log is not None:
        result['interruption_note']='13 completed medians recovered from stdout; original individual times/counts unavailable. Remaining 3 conditions freshly measured with original 3x20000 method; table build time recovered from saved benchmark_gs.npz, initialization remeasured on resume.'
        result['recovered_log']=str(resume_log)
    (out/'benchmark.json').write_text(json.dumps(result,indent=2)+'\n');return result
