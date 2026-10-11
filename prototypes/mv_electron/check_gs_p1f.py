"""N4: P1 K3, deterministic refinement and forward/backward single-CDF tests."""
import numpy as np
from numba import njit
from scipy.stats import chisquare
from scipy.integrate import quad
from .p1f_common import SEED, status, aggregate, WORK, sha, table_hashes
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
    from .p1f_common import OUT as P,write
    import json
    file=P/'t4_checkpoint.json'
    saved=json.loads(file.read_text()) if file.exists() else {}
    identity=dict(seed=SEED,IHEF=0,table_hashes=table_hashes(),GS_sha256=sha(WORK/'gs.npz'),N_K3=1000000,N_CDF=10000000)
    if saved and saved.get('identity')!=identity:raise RuntimeError('T4 saved conditions/source hash mismatch')
    rows=saved.get('K3_rows',[])
    def checkpoint():write('t4_checkpoint.json',{'identity':identity,'K3_rows':rows,'refinement':refine,'cdf_tests':tests})
    refine=saved.get('refinement',[]);tests=saved.get('cdf_tests',[])
    for t in [.01,.02,.1,1.,2.,10.]:
        G,total,_=moments(provider,t)
        for k in [.001,.01,.05,.2]:
            if any(r['T']==t and r['K']==k for r in rows):continue
            mu=samples(t,k,1000000,*(g[x] for x in ['T','K','u','y','rates']));p2=(3*mu*mu-1)/2
            actual=np.array([mu.mean(),p2.mean()]);expected=np.exp(-k*G[1:3]/G[1]);sem=np.array([mu.std(ddof=1),p2.std(ddof=1)])/np.sqrt(len(mu));tol=np.where(expected>=.05,.01*expected,.001)
            rows.append({'T':t,'K':k,'moments':actual.tolist(),'expected':expected.tolist(),'relative':((actual-expected)/expected).tolist(),'sem':sem.tolist(),'pass':bool(np.all(abs(actual-expected)<=tol) and np.all(sem<=tol/3))})
            checkpoint()
    # Resume deterministic refinement.
    for t in [.01,.02,.1,1.,2.,10.]:
        for k in [.001,.01,.05,.2]:
            if any(r['T']==t and r['K']==k for r in refine):continue
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
            refine.append(r);checkpoint()
    rng=np.random.default_rng(SEED);n=10000000
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
    for t in [.01,.02,.1,1.,2.,10.]:
        if any(r['T']==t and r['kind']=='runtime-single' for r in tests):continue
        vals=single_samples(t,n,g['T'],g['u'],g['single'])
        tests.append(check_cdf(t,vals,'runtime-single'))
        checkpoint()
        print(provider.name,'N4 runtime single',t,tests[-1]['chi_p'],flush=True)
    if not any(r['kind']=='short-GS' for r in tests):
        t=.02;G,total,_=moments(provider,t);mean=1e-5
        yy,cc,p0,*_=distribution(provider,t,mean/total*G[1],L=2048,n=32769)
        vals=np.interp(rng.random(n),cc,yy);tests.append(check_cdf(t,vals,'short-GS'));checkpoint()
    norm=float(max(abs(g['normalization']-1)));monotone=bool(np.all(np.diff(g['y'],axis=2)>=0))
    for r in rows:
        expected=np.asarray(r['expected']);tol=np.where(expected>=.05,.01*expected,.001)
        r['statistics_ok']=bool(np.all(np.asarray(r['sem'])<=tol/3));r['status']=status(r)
    for r in refine:r['status']='pass' if r['pass'] else 'fail'
    for r in tests:
        r['statistics_ok']=r['minimum_expected']>=100;r['status']=status(r)
    state=aggregate([r['status'] for r in rows+refine+tests]+['pass' if norm<=1e-6 and monotone else 'fail'])
    return {'status':state,'identity':identity,'K3_rows':rows,'refinement':refine,'cdf_tests':tests,'normalization_max_error':norm,'cdf_monotone':monotone}

def from_saved(provider,g,result):
    """Rejudge saved sufficient statistics only; no draws or extra histories."""
    if result['identity']!=dict(seed=SEED,IHEF=0,table_hashes=table_hashes(),GS_sha256=sha(WORK/'gs.npz'),N_K3=1000000,N_CDF=10000000):raise RuntimeError('T4 replay identity mismatch')
    if len(result['K3_rows'])!=24 or len(result['refinement'])!=24 or len(result['cdf_tests'])!=7:
        raise RuntimeError('T4 saved checkpoints incomplete; no execution in replay')
    states=[]
    for r in result['K3_rows']:
        G,total,_=moments(provider,r['T']);expected=np.exp(-r['K']*G[1:3]/G[1])
        if not np.array_equal(expected,np.array(r['expected'])):raise RuntimeError('T4 expected moments changed')
        tol=np.where(expected>=.05,.01*expected,.001)
        stats=bool(np.all(np.array(r['sem'])<=tol/3));passed=bool(np.all(abs(np.array(r['moments'])-expected)<=tol) and stats)
        states.append(status(dict(statistics_ok=stats,**{'pass':passed})))
    for r in result['refinement']:
        change=max(abs(np.array(r['moments'])-r['refined_moments']));inverse=max(abs(np.array(r['inverse_table_moments'])-r['refined_inverse_table_moments']))
        states.append('pass' if change<=1e-4 and inverse<=1e-4 else 'fail')
    for r in result['cdf_tests']:
        p=chisquare(r['observed'],r['expected']).pvalue
        if float(p)!=r['chi_p']:raise RuntimeError('T4 saved CDF p-value mismatch')
        stats=min(r['expected'])>=100
        states.append(status(dict(statistics_ok=stats,**{'pass':bool(p>.001 and stats)})))
    states.append('pass' if max(abs(g['normalization']-1))<=1e-6 and np.all(np.diff(g['y'],axis=2)>=0) else 'fail')
    if aggregate(states)!=result['status']:raise RuntimeError('T4 saved verdict mismatch')
    return result
