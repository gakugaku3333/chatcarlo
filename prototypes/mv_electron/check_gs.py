import json,time
import numpy as np
from scipy.integrate import quad
from scipy.stats import chisquare
from .physics import *
from .gs import coefficients,distribution
from .sampling import sample_moments
from .component_checks import SEED,OUT
from numba import njit
@njit
def short_gs(T,s,n,seed):
    # Exact Poisson-convolution representation of GS for rare-collision validation.
    np.random.seed(seed);eta,B,total,G1=elastic(T);v=np.empty(n);counts=np.empty(n,np.int64)
    for i in range(n):
        count=np.random.poisson(s*np.sum(total));counts[i]=count
        ux=0.;uy=0.;uz=1.
        for j in range(count):ux,uy,uz=rotate(ux,uy,uz,single_mu(T),2*np.pi*np.random.random())
        v[i]=uz
    return v,counts

def main(output_dir=None):
    start=time.perf_counter();g=np.load(OUT/'data/gs.npz');args=tuple(g[k] for k in ['T','K','u','y'])
    rows=[]
    for t in [.02,.1,1,2,10]:
        G,total=coefficients(t,2)
        # Independent quadrature verifies Q derivative moment implementation.
        eta,B,_,_=elastic(t)
        numeric=[]
        for l in [1,2]:
            f=lambda y: (y if l==1 else 3*y-1.5*y*y)*sum(B/(y+2*eta)**2)
            numeric.append(quad(lambda v:f(np.exp(v))*np.exp(v),np.log(1e-16),np.log(2),epsabs=1e-9,epsrel=1e-10)[0])
        for k in [.001,.01,.05,.2]:
            mu=sample_moments(t,k,1000000,SEED,*args);p2=(3*mu*mu-1)/2
            actual=np.array([mu.mean(),p2.mean()]);expected=np.exp(-k*G[1:3]/G[1]);sem=np.array([mu.std(ddof=1),p2.std(ddof=1)])/np.sqrt(len(mu))
            tol=np.where(expected>=.05,.01*expected,.001)
            rows.append({'T':t,'K':k,'moments':actual.tolist(),'expected':expected.tolist(),'relative':((actual-expected)/expected).tolist(),'sem':sem.tolist(),'statistics_ok':bool(np.all(sem<=tol/3)),'pass':bool(np.all(abs(actual-expected)<=tol) and np.all(sem<=tol/3)),'moment_quadrature_relative':(np.array(numeric)/G[1:3]-1).tolist(),'p0_analytic':float(np.exp(-k*total/G[1]))})
    short=[]
    # Validate the converged GS series (not just its first two moments) in rare-collision limit.
    # Conditional draws avoid spending 1/p(scattered) CPU on the analytically known zero atom.
    t=.02;eta,B,total,G1=elastic(t);mean=1e-5;s=mean/np.sum(total)
    y,cdf,p0,G,tot,diag=distribution(t,s*G1,L=2048,n=32769)
    n=10000000;rng=np.random.default_rng(SEED)
    vals=np.interp(rng.random(n),cdf,y)
    # Equally spaced logarithms of screened angle coordinate cover both physical endpoints.
    offset=2*min(eta);edges=np.geomspace(offset,2+offset,21)-offset;edges[0]=0;edges[-1]=2
    obs=np.histogram(vals,edges)[0]
    expected=np.array([np.sum(B*(1/(a+2*eta)-1/(b+2*eta)))/np.sum(total)*n for a,b in zip(edges[:-1],edges[1:])]);expected*=sum(obs)/sum(expected)
    chi=chisquare(obs,expected)
    short.append({'T':t,'mean_collisions':mean,'conditional_scattered_N':n,'p0_analytic':float(np.exp(-mean)),'p0_table':p0,'min_expected':float(min(expected)),'chi_p':float(chi.pvalue),'pass':bool(chi.pvalue>.001 and min(expected)>=100),'coordinate':'log(1-mu+2*eta_min),20 bins','normalization':diag['norm']})
    result={'K3':{'status':'pass' if all(r['pass'] for r in rows+short) else 'fail','rows':rows,'short':short,'normalization_max_error':float(max(abs(g['normalization']-1))),'cdf_monotone':bool(np.all(np.diff(g['y'],axis=2)>=0))},'seconds':time.perf_counter()-start,'seed':SEED}
    ((OUT if output_dir is None else output_dir)/'gs_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
