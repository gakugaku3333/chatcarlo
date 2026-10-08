"""P1 K5/K5b/K7/K8/K9 conditions using one selected DCS provider."""
from pathlib import Path
import json,time,contextlib
import numpy as np
from .component_checks import SEED
from .reference import fixed
from .analysis import compare
from .run_transport import execute
from . import check_transport

def fixed_checks(out,g):
    args=tuple(g[k] for k in ['T','K','u','y']);kw={'backend_rates':g['rates'],'backend_single':g['single']};rows=[]
    for path in [.05,.2]:
        n=1000000
        fixed(2,path,1,SEED,False,.05,*args,**kw);fixed(2,path,1,SEED,True,.05,*args,**kw)
        t=time.perf_counter();ref=fixed(2,path,n,SEED,False,.05,*args,**kw);rt=time.perf_counter()-t
        for k in [.05,.025]:
            t=time.perf_counter();pro=fixed(2,path,n,SEED+1,True,k,*args,**kw);pt=time.perf_counter()-t
            diff=(pro.mean(0)-ref.mean(0))/ref.mean(0);sem=np.sqrt(pro.var(0,ddof=1)/n+ref.var(0,ddof=1)/n)/ref.mean(0)
            row={'path_cm':path,'K':k,'N':n,'reference':ref.mean(0).tolist(),'prototype':pro.mean(0).tolist(),'relative':diff.tolist(),'difference_sem_relative':sem.tolist(),'statistics_ok':bool(max(sem[:2])<=.02/3),'pass':bool(max(abs(diff[:2]))<=.02 and max(sem[:2])<=.02/3),'reference_seconds':rt,'prototype_seconds':pt};rows.append(row)
            np.savez(out/f'data/fixed_{path}_{k}.npz',reference=ref,prototype=pro)
            print('K9i',row,flush=True)
    result={'K9i':{'status':'pass' if all(r['pass'] for r in rows) else 'fail','rows':rows},'seed':SEED}
    (out/'fixed_checks.json').write_text(json.dumps(result,indent=2)+'\n');return result

def main(backend,out):
    out=Path(out);g=np.load(out/'data/gs.npz')
    configs=[('base_2',dict(n=1000000)),('split_aligned',dict(n=1000000,split=.1)),('split_shifted',dict(n=1000000,split=.1,shift=.037)),('ms_half',dict(n=1000000,k=.025)),('energy_half',dict(n=1000000,fe=.025)),('reference_2',dict(n=300000,mode=1))]
    for name,kw in configs:
        if not (out/'runs'/(name+'.npz')).exists():execute(name,output_dir=out,backend=backend,**kw)
    old=check_transport.OUT
    try:
        check_transport.OUT=out;check_transport.main(out)
    finally:check_transport.OUT=old
    if not (out/'fixed_checks.json').exists():fixed_checks(out,g)
    r=json.loads((out/'transport_checks.json').read_text());r.update(json.loads((out/'fixed_checks.json').read_text()))
    keys=['K5','K5b','K7','K8','K9i','K9ii'];r['status']='pass' if all(r[k]['status']=='pass' for k in keys) else 'fail'
    (out/'m5.json').write_text(json.dumps(r,indent=2)+'\n');return r
