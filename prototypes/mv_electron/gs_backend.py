"""Numerical full-order GS for tabulated providers; original Rutherford untouched."""
import time
import numpy as np
from numba import njit
from scipy.integrate import cumulative_trapezoid
from numpy.polynomial.legendre import leggauss
from .dcs import moments

@njit
def integrate_moments(y,w,L):
    G=np.zeros(L+1)
    for j in range(len(y)):
        mu=1-y[j];d0=0.;d1=y[j]
        if L>0:G[1]+=w[j]*y[j]
        for l in range(2,L+1):
            d=((2*l-1)*(y[j]+mu*d1)-(l-1)*d0)/l
            G[l]+=w[j]*d;d0=d1;d1=d
    return G

def coefficients(provider,T,L,order=16):
    # Split native intervals only for oscillatory high-order integration accuracy.
    edges=provider.edges(T)
    theta=np.arccos(1-edges)
    target=order/max(L,1)
    fine=[]
    for a,b,ta,tb in zip(edges[:-1],edges[1:],theta[:-1],theta[1:]):
        count=max(1,int(np.ceil((tb-ta)/target)))
        fine.extend(np.linspace(a,b,count+1)[:-1])
    fine=np.r_[fine,edges[-1]]
    x,w=leggauss(order);y=((fine[:-1,None]+fine[1:,None])/2+(fine[1:,None]-fine[:-1,None])*x/2).ravel()
    w=((fine[1:,None]-fine[:-1,None])*w/2).ravel()*provider.density(T,y)
    return integrate_moments(y,w,L),float(w.sum())

def distribution(provider,T,k,L=4096,n=16385,order=16):
    G,total=coefficients(provider,T,L,order);return from_coefficients(G,total,k,n,provider,T)

def from_coefficients(G,total,k,n=16385,provider=None,T=None):
    L=len(G)-1;s=k/G[1];p0=np.exp(-s*total)
    a=np.exp(-s*G)-p0
    if provider is not None:a-=p0*s*(total-G)
    coeff=(2*np.arange(L+1)+1)/2*a
    y=np.r_[0,np.geomspace(1e-12,2,n-1)]
    if provider is not None and provider.name=="eedl":
        # Respect piecewise-linear knots and the unforced branch discontinuities.
        native=provider.edges(T)
        seams=np.array([provider.parameters(T,int(z))[2] for z in provider.atomic])
        y=np.unique(np.r_[y,native,np.nextafter(seams,0),np.nextafter(seams,2)])
    mu=1-y
    density=np.polynomial.legendre.legval(mu,coeff)
    # Exact Poisson single-collision term, separated to avoid truncating sharp
    # features of the tabulated DCS. Remaining >=2-collision GS series unchanged.
    if provider is not None:density+=p0*s*provider.density(T,y)
    norm=np.trapezoid(density,y)+p0;negative=float(np.min(density))
    if not np.isfinite(norm) or negative < -1e-7 or abs(norm-1)>1e-6:
        raise ValueError(f'GS numerical convergence L={L},norm={norm},min={negative}')
    cdf=cumulative_trapezoid(density,y,initial=0)/(1-p0)
    if np.min(np.diff(cdf)) < -1e-12:raise ValueError('Nonmonotone GS CDF')
    cdf[-1]=1
    return y,cdf,p0,G,total,{'norm':float(norm),'min_density':negative,'L':L}

def single_cdf(provider,T,n=32769):
    # Integrate in each native interval; dense cumulative grid for inverse sampling.
    edges=provider.edges(T);extra=np.r_[0,np.geomspace(1e-14,2,n-1)]
    grid=np.unique(np.r_[edges,extra]);x,w=leggauss(8)
    y=(grid[:-1,None]+grid[1:,None])/2+(grid[1:,None]-grid[:-1,None])*x/2
    areas=np.sum(provider.density(T,y)*w*(grid[1:,None]-grid[:-1,None])/2,axis=1)
    cdf=np.r_[0,np.cumsum(areas)];cdf/=cdf[-1]
    return grid,cdf

def build(provider,dest):
    start=time.perf_counter();ts=np.geomspace(.01,20,49)
    ks=np.array([.0005,.001,.003125,.00625,.0125,.025,.05,.1,.2])
    from .single_table import build as single_build
    u,rates,single,lookup_diagnostics=single_build(provider)
    q=np.empty((len(ts),len(ks),len(u)))
    p0s=np.empty((len(ts),len(ks)));g1=np.empty(len(ts));tot=np.empty(len(ts));norms=[];orders=[]
    for i,t in enumerate(ts):
        L=1024
        while True:
            G,total=coefficients(provider,t,L)
            try:rows=[from_coefficients(G,total,k,provider=provider,T=t) for k in ks];break
            except ValueError:
                L*=2
                if L>32768:raise
        for j,(y,cdf,p0,G,total,d) in enumerate(rows):
            q[i,j]=np.interp(u,cdf,y);p0s[i,j]=p0;norms.append(d['norm']);orders.append(L)
        g1[i]=G[1];tot[i]=total
        # Single CDF has its own converged energy grid, from this same provider.
        print(provider.name,'GS',i+1,len(ts),float(t),L,flush=True)
    np.savez(dest,T=ts,K=ks,u=u,y=q,p0=p0s,G1=g1,total=tot,rates=rates,single=single,lookup_diagnostics=__import__('json').dumps(lookup_diagnostics),build_seconds=time.perf_counter()-start,normalization=norms,L=orders,backend=provider.name)
    return dest
