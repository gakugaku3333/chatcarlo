"""M4: P1 K3, deterministic refinement and forward/backward single-CDF tests."""
import numpy as np
from numba import njit
from scipy.stats import chisquare
from scipy.integrate import quad
from .component_checks import SEED
from .sampling import gs_mu_cached
from .backend_sampling import table_rates,table_single
from .gs_backend import distribution,single_cdf
from .dcs import moments
@njit
def samples(T,K,n,ts,ks,us,ys,rates):
    np.random.seed(SEED);total,G1=table_rates(T,ts,rates);m=np.empty(n)
    for i in range(n):m[i]=gs_mu_cached(T,K,ts,ks,us,ys,total,G1)
    return m

@njit
def single_samples(T,n,ts,us,ys):
    np.random.seed(SEED);v=np.empty(n)
    for i in range(n):v[i]=1-table_single(T,ts,us,ys)
    return v

def cdf_moments(y,cdf,p0):
    # Exact integration of P1 and P2 against piecewise linear CDF.
    a=1-y[:-1];b=1-y[1:];mass=np.diff(cdf)
    return p0+(1-p0)*np.array([np.sum(mass*(a+b)/2),np.sum(mass*((a*a+a*b+b*b)/2-.5))])

def main(provider,g):
    rows=[]
    for t in [.02,.1,1.,2.,10.]:
        G,total,_=moments(provider,t)
        for k in [.001,.01,.05,.2]:
            mu=samples(t,k,1000000,*(g[x] for x in ['T','K','u','y','rates']));p2=(3*mu*mu-1)/2
            actual=np.array([mu.mean(),p2.mean()]);expected=np.exp(-k*G[1:3]/G[1]);sem=np.array([mu.std(ddof=1),p2.std(ddof=1)])/np.sqrt(len(mu));tol=np.where(expected>=.05,.01*expected,.001)
            rows.append({'T':t,'K':k,'moments':actual.tolist(),'expected':expected.tolist(),'relative':((actual-expected)/expected).tolist(),'sem':sem.tolist(),'pass':bool(np.all(abs(actual-expected)<=tol) and np.all(sem<=tol/3))})
    refine=[]
    for t in [.02,.1,1.,2.,10.]:
        for k in [.001,.01,.05,.2]:
            L=1024
            while True:
                try:a=distribution(provider,t,k,L,16385,16);break
                except ValueError:
                    L*=2
                    if L>32768:raise
            b=distribution(provider,t,k,2*L,32769,32)
            ma=cdf_moments(*a[:3]);mb=cdf_moments(*b[:3]);change=float(max(abs(ma-mb)))
            u=g['u'];uf=np.unique(np.r_[u,(u[:-1]+u[1:])/2])
            qa=np.interp(u,a[1],a[0]);qb=np.interp(uf,b[1],b[0])
            def inverse_moments(u,y,p0):
                aa=1-y[:-1];bb=1-y[1:];mass=np.diff(u)
                return p0+(1-p0)*np.array([sum(mass*(aa+bb)/2),sum(mass*((aa*aa+aa*bb+bb*bb)/2-.5))])
            im= inverse_moments(u,qa,a[2]);jm=inverse_moments(uf,qb,b[2]);tablechange=float(max(abs(im-jm)))
            r={'T':t,'K':k,'L':L,'moment_change':change,'moments':ma.tolist(),'refined_moments':mb.tolist(),'inverse_table_moment_change':tablechange,'inverse_table_moments':im.tolist(),'refined_inverse_table_moments':jm.tolist(),'pass':change<=1e-4 and tablechange<=1e-4}
            if t==2 and k==.05:
                r['cdf_max_change']=float(max(abs(a[1]-np.interp(a[0],b[0],b[1]))));r['back_change']=float(abs(np.interp(1,a[0],a[1])-np.interp(1,b[0],b[1])))
            refine.append(r)
    rng=np.random.default_rng(SEED);tests=[];n=10000000
    from numpy.polynomial.legendre import leggauss
    def check_cdf(T,vals,kind):
        total=moments(provider,T)[1]
        offset=min(provider.eta(T,int(z)) for z in provider.atomic) if provider.name=='eedl' else 1e-6
        edges=np.geomspace(offset,2+offset,21)-offset;edges[0]=0;edges[-1]=2
        expected=[];x,w=leggauss(32);native=provider.edges(T)
        for a,b in zip(edges[:-1],edges[1:]):
            cuts=np.unique(np.r_[a,native[(native>a)&(native<b)],b])
            y=(cuts[:-1,None]+cuts[1:,None])/2+(cuts[1:,None]-cuts[:-1,None])*x/2
            expected.append(np.sum(provider.density(T,y)*w*(cuts[1:,None]-cuts[:-1,None])/2)/total*n)
        # Predetermined expected-count merging, never based on observations.
        # Retains both endpoints even when backward events are rare at high T.
        boundaries=[len(expected)];acc=0.
        for i in range(len(expected)-1,-1,-1):
            acc+=expected[i]
            if acc>=100:boundaries.append(i);acc=0.
        if boundaries[-1]!=0:boundaries[-1]=0
        indices=np.array(sorted(boundaries));ee=edges[indices]
        exp=np.array([sum(expected[a:b]) for a,b in zip(indices[:-1],indices[1:])])
        obs=np.histogram(vals,ee)[0];exp*=obs.sum()/exp.sum();chi=chisquare(obs,exp)
        return {'T':T,'kind':kind,'chi_p':float(chi.pvalue),'minimum_expected':float(exp.min()),'N':n,'coordinate':'log(1-mu+offset),20 initial bins; expected-only merging to >=100','offset':float(offset),'observed':obs.tolist(),'expected':exp.tolist(),'edges':ee.tolist(),'pass':bool(chi.pvalue>.001 and exp.min()>=100)}
    for t in [.02,.1,1.,2.,10.]:
        vals=single_samples(t,n,g['T'],g['u'],g['single'])
        tests.append(check_cdf(t,vals,'runtime-single'))
        print(provider.name,'M4 runtime single',t,tests[-1]['chi_p'],flush=True)
    t=.02;G,total,_=moments(provider,t);mean=1e-5
    yy,cc,p0,*_=distribution(provider,t,mean/total*G[1],L=2048,n=32769)
    vals=np.interp(rng.random(n),cc,yy);tests.append(check_cdf(t,vals,'short-GS'))
    norm=float(max(abs(g['normalization']-1)));monotone=bool(np.all(np.diff(g['y'],axis=2)>=0))
    passed=all(r['pass'] for r in rows+refine+tests) and norm<=1e-6 and monotone
    return {'status':'pass' if passed else 'fail','K3_rows':rows,'refinement':refine,'cdf_tests':tests,'normalization_max_error':norm,'cdf_monotone':monotone}
