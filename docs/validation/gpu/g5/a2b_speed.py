"""B5 validation-only adapter: equal 0.5 mm grids, chunks and recording on one T4."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from a3_entry_speed import ratio,timed_main,cache_used,SharedClient,SITE_CUSTOMIZE
from chatcarlo.gpu import seed_geometry as core,seed_bundle,contracts
from chatcarlo.gpu.seed_cuda import GPU


def kernel(cp,materials,inputs,*,n=71429,seed=9502001,repeats=3):
    from prototypes.mv_electron.gpu.p9c_colab import GPU as Legacy
    from prototypes.mv_electron.gpu.p9c_core import Grid as OldGrid
    if n!=71429 or repeats<2:raise ValueError('B5 fixes one registered door chunk and at least two measured repeats')
    grids=tuple(OldGrid(g.lo,g.step,g.shape) for g in (core.FINE,core.COARSE))
    kw=dict(densities=inputs['densities'],edges=inputs['mohan10_edges'],cum=inputs['mohan10_cum'],beam='prostate',grids=grids)
    old=Legacy(cp,materials,**kw)
    new=GPU(cp,materials,**kw,constants=inputs['constants'],strength=inputs['strength'])
    # Warm both in the same VM; alternate measurement order, same door/state.
    for e in (old,new):e.run(20,seed)
    rows=[]
    for repeat in range(repeats):
        r={}
        for label,e in ((('old',old),('new',new)) if repeat%2==0 else (('new',new),('old',old))):
            result=e.run(n,seed,start=repeat*n);r[label]=float(result['seconds'])
        rows.append(r)
    a=float(np.median([r['old'] for r in rows]));b=float(np.median([r['new'] for r in rows]))
    return dict(ratio(a,b),rows=rows,old_s=a,new_s=b,chunk=n,grids=[g.record() for g in (core.FINE,core.COARSE)],same_session=True,recording='per-history doubles/ints, six-site ledger, two grids; no P7 region tensor')


def entry(old_main,new_main,old_argv,new_argv):
    """Time concrete old/new CLI adapters, cold then fresh warm worker.

    Caller supplies A3 SharedClient for the SAME owned T4 and distinct fixed
    CUPY_CACHE_DIR per version. Return compiler/cache records alongside timings;
    cache_used() must pass before labelling the second run cached.
    """
    timings={}
    for label,main,argv in (('old',old_main,old_argv),('new',new_main,new_argv)):
        timings[label]=[timed_main(main,argv) for _ in range(2)]
    return dict(first=ratio(timings['old'][0],timings['new'][0]),cached=ratio(timings['old'][1],timings['new'][1]),seconds=timings)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bundle',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    import cupy as cp
    m,i,_=seed_bundle.load(a.bundle);contracts.write(a.out,kernel(cp,m,i))
