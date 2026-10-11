"""P1b identical 16-condition timing, checkpoint every completed replicate."""
import json,time
import numpy as np
from .component_checks import SEED
from .transport import run,FIELDS
from .dcs import get_backend
from .regression_p1c import P,write

def main(out):
    from .build_gs_p1c import build
    # Separate directory keeps table construction checkpoints independent.
    dest=out/'benchmark_data';dest.mkdir(exist_ok=True);path=dest/'gs.npz'
    if not path.exists():build(get_backend('mf'),path)
    built=np.load(path);t=time.perf_counter();g=np.load(out/'data/gs.npz');args=tuple(g[x] for x in ['T','K','u','y']);kw=dict(backend_rates=g['rates'],backend_single=g['single']);init=time.perf_counter()-t
    for k in ['T','K','u','y','p0','G1','total','rates','single']:np.testing.assert_array_equal(g[k],built[k])
    checkpoint=P/'benchmark_checkpoint.json';saved=json.loads(checkpoint.read_text()) if checkpoint.exists() else {'rows':[],'active':None};rows=saved['rows'];n=20000
    for T in [2.,10.]:
        dz=.01 if T==2 else .05;faces=np.array([0.,1.2 if T==2 else 6.])
        for fe,k in [(.05,.05),(.05,.025),(.025,.05),(.025,.025)]:
            for scoring in [True,False]:
                key=[T,fe,k,scoring]
                if any(r['key']==key for r in rows):continue
                t=time.perf_counter();run(T,1,SEED,fe,k,faces,dz,*args,scoring,**kw);jit=time.perf_counter()-t
                active=saved.get('active');active=active if active and active['key']==key else dict(key=key,times=[],counts=[],ledgers=[])
                for rep in range(len(active['times']),3):
                    t=time.perf_counter();b,l,c,*_=run(T,n,SEED+rep,fe,k,faces,dz,*args,scoring,**kw)
                    active['times'].append(time.perf_counter()-t);active['counts'].append((c/n).tolist());active['ledgers'].append(float(max(abs(l[:,3]))/T))
                    saved=dict(rows=rows,active=active);write('benchmark_checkpoint.json',saved)
                rows.append(dict(key=key,T=T,fe=fe,K=k,scoring=scoring,N=n,replicates=3,wall_seconds=active['times'],us_per_primary=float(np.median(active['times'])/n*1e6),counts_per_primary=np.mean(active['counts'],axis=0).tolist(),warmup_seconds_excluded=jit,max_ledger_relative=max(active['ledgers'])))
                saved=dict(rows=rows,active=None);write('benchmark_checkpoint.json',saved);print('mf benchmark',key,rows[-1]['us_per_primary'],flush=True)
    result=dict(backend='mf',rows=rows,table_load_initialization_seconds=init,table_build_seconds=float(built['build_seconds']),table_bytes=sum(g[x].nbytes for x in ['T','K','u','y','p0','G1','total','rates','single']),stack_allocated_bytes=4096*FIELDS*8,seed=SEED,method='Single process; median 3 x 20000 histories; JIT excluded; timing run after validation processes finish. Active incomplete replicate restarts; completed replicates saved.')
    write('benchmark.json',result);return result
