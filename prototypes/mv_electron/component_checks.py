"""Independent component checks. Saved results are the report inputs."""
import json,time
import numpy as np
from scipy.integrate import quad
from scipy.stats import chisquare
from numba import njit
from .physics import *
SEED=20261007
OUT=ROOT/'docs/validation/mv/p1'
@njit
def energies(T,n,seed):
    np.random.seed(seed);a=np.empty(n)
    for i in range(n):a[i]=sample_W(T)
    return a

def k1k2():
    start=time.perf_counter()
    dif=np.array([collision(t) for t in ET])/estar['collision']-1
    K1={'max_relative':float(max(abs(dif))),'points':len(ET),'status':'pass' if max(abs(dif))<=.01 else 'fail'}
    results=[]
    rng=np.random.default_rng(SEED)
    for t in [.05,.1,1,2,10,np.nextafter(.02,0),.02,np.nextafter(.02,np.inf)]:
        rate,sh=hard(t)
        num=quad(lambda w:w*dcs_moller(t,w),DELTA,t/2,epsabs=1e-16,epsrel=1e-11)[0] if t>2*DELTA else 0.
        rel=abs(sh-num)/abs(num) if num else abs(sh-num)
        row={'T':float(t),'S_hard_analytic':float(sh),'S_hard_numeric':float(num),'integral_relative':float(rel),'integral_pass':bool(rel<=1e-6)}
        if t>.0200001:
            n=1000000
            flight=rng.exponential(1/rate,n);r=abs(1/flight.mean()/rate-1)
            ws=energies(t,n,SEED)
            edges=np.geomspace(DELTA,t/2,21);observed=np.histogram(ws,edges)[0]
            expected=np.array([quad(lambda w:dcs_moller(t,w),a,b,epsabs=1e-12)[0]/rate*n for a,b in zip(edges[:-1],edges[1:])]);expected*=n/expected.sum()
            chi=chisquare(observed,expected)
            maxp=0.;maxe=0.
            for w in np.r_[DELTA,np.nextafter(t/2,0),ws[:1000]]:
                dp,ds=moller_directions(t,w,0.,0.,1.,.71)
                v=np.sqrt((t-w)*(t-w+2*M))*np.array(dp)+np.sqrt(w*(w+2*M))*np.array(ds)
                maxp=max(maxp,float(np.max(abs(v-[0,0,np.sqrt(t*(t+2*M))]))))
                maxe=max(maxe,abs(t-(t-w)-w))
            row.update({'free_path_relative':float(r),'free_path_sem_relative':float(flight.std(ddof=1)/np.sqrt(n)/flight.mean()),'chi_p':float(chi.pvalue),'min_expected':float(min(expected)),'momentum_residual_MeV_c':maxp,'energy_residual_MeV':maxe,'N':n})
        results.append(row)
    ok=all(r['integral_pass'] and r.get('free_path_relative',0)<=.005 and r.get('chi_p',1)>.001 and r.get('min_expected',100)>=100 and r.get('momentum_residual_MeV_c',0)<1e-12 for r in results)
    answer={'K1':K1,'K2':{'status':'pass' if ok else 'fail','rows':results},'seconds':time.perf_counter()-start,'seed':SEED}
    (OUT/'components.json').write_text(json.dumps(answer,indent=2)+'\n')
    np.savez(OUT/'data/k1.npz',T=ET,relative=dif)
    print(json.dumps(answer,indent=2),flush=True)
if __name__=='__main__':k1k2()
