import numpy as np,json,time
from .reference import fixed
from .component_checks import OUT,SEED

def main():
    g=np.load(OUT/'data/gs.npz');args=tuple(g[k] for k in ['T','K','u','y']);rows=[]
    for path in [.05,.2]:
        n=1000000
        fixed(2,path,1,SEED,False,.05,*args);fixed(2,path,1,SEED,True,.05,*args)
        t=time.perf_counter();ref=fixed(2,path,n,SEED,False,.05,*args);rt=time.perf_counter()-t
        for k in [.05,.025]:
            t=time.perf_counter();pro=fixed(2,path,n,SEED+1,True,k,*args);pt=time.perf_counter()-t
            diff=(pro.mean(0)-ref.mean(0))/ref.mean(0)
            sem=np.sqrt(pro.var(0,ddof=1)/n+ref.var(0,ddof=1)/n)/ref.mean(0)
            row={'path_cm':path,'K':k,'N':n,'reference':ref.mean(0).tolist(),'prototype':pro.mean(0).tolist(),'relative':diff.tolist(),'difference_sem_relative':sem.tolist(),'statistics_ok':bool(max(sem[:2])<=.02/3),'pass':bool(max(abs(diff[:2]))<=.02 and max(sem[:2])<=.02/3),'reference_seconds':rt,'prototype_seconds':pt};rows.append(row)
            np.savez(OUT/f'data/fixed_{path}_{k}.npz',reference=ref,prototype=pro)
    ans={'K9i':{'status':'pass' if all(r['pass'] for r in rows) else 'fail','rows':rows},'seed':SEED}
    (OUT/'fixed_checks.json').write_text(json.dumps(ans,indent=2)+'\n');print(json.dumps(ans,indent=2))
if __name__=='__main__':main()
