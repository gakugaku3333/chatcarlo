"""Goudsmit--Saunderson Legendre tables, with exact unscattered atom.
SLAC-R-730 eqs 2.352,2.362. No Gaussian scattering replacement.
Screened Rutherford moments use Q_l'(z): integral P_l(mu)/(z-mu)^2=-2Q_l'(z).
"""
import time
from pathlib import Path
import numpy as np
from numba import njit
from scipy.integrate import cumulative_trapezoid
from .physics import elastic,ROOT
OUT=ROOT/'docs/validation/mv/p1/data/gs.npz'

@njit
def qderivative(a,L):
    z=1+a
    # Backward continued fraction is stable near z=1 and at large order.
    extra=int(25/np.sqrt(2*a))+100
    r=np.zeros(L+1);rn=0.
    for l in range(L+extra,0,-1):
        rn=l/((2*l+1)*z-(l+1)*rn)
        if l<=L:r[l]=rn
    q=np.empty(L+1);q[0]=.5*np.log((2+a)/a)
    for l in range(1,L+1):q[l]=q[l-1]*r[l]
    dq=np.empty(L+1);dq[0]=-1/(a*(2+a))
    for l in range(1,L+1):dq[l]=l*(z*q[l]-q[l-1])/(a*(2+a))
    return dq

def coefficients(T,L):
    eta,B,total,G1=elastic(T)
    G=np.zeros(L+1)
    for a,b in zip(2*eta,B):
        dq=qderivative(a,L)
        G+=b*(1/a-1/(2+a)+2*dq)
    G[0]=0;G[1]=G1
    return G,float(np.sum(total))

def distribution(T,k,L=4096,n=16385):
    G,total=coefficients(T,L);s=k/G[1];p0=np.exp(-s*total)
    coeff=(2*np.arange(L+1)+1)/2*(np.exp(-s*G)-p0)
    # Include endpoint and forward peak on a logarithmic 1-mu mesh.
    y=np.r_[0,np.geomspace(1e-12,2,n-1)];mu=1-y
    density=np.polynomial.legendre.legval(mu,coeff)
    norm=np.trapezoid(density,y)+p0
    negative=float(np.min(density))
    # Reject unconverged truncation; never hide negative series with clipping.
    if not np.isfinite(norm) or negative < -1e-7 or abs(norm-1)>1e-6:
        raise ValueError(f'GS numerical convergence T={T},K={k},L={L},norm={norm},min={negative}')
    cdf=cumulative_trapezoid(density,y,initial=0)/(1-p0)
    if np.min(np.diff(cdf)) < -1e-12: raise ValueError('Nonmonotone GS CDF')
    cdf[-1]=1
    return y,cdf,p0,G,total,{'norm':float(norm),'min_density':negative,'L':L}

def build():
    start=time.perf_counter()
    ts=np.geomspace(.01,20,49)
    ks=np.array([.0005,.001,.003125,.00625,.0125,.025,.05,.1,.2])
    # inverse CDF nonuniform probability mesh resolves rare large-angle tails.
    u=np.unique(np.r_[np.linspace(0,1,2049),1-np.geomspace(1e-8,.1,257),np.geomspace(1e-8,.1,257)])
    q=np.empty((len(ts),len(ks),len(u)));p0s=np.empty((len(ts),len(ks)));g1=np.empty(len(ts));tot=np.empty(len(ts)); diagnostics=[]
    for i,t in enumerate(ts):
        for j,k in enumerate(ks):
            L=1024
            while True:
                try:y,cdf,p0,G,total,d=distribution(t,k,L=L);break
                except ValueError:
                    L*=2
                    if L>32768:raise
            q[i,j]=np.interp(u,cdf,y);p0s[i,j]=p0;g1[i]=G[1];tot[i]=total;diagnostics.append(d)
        print('GS energy',i+1,len(ts),float(t),flush=True)
    np.savez(OUT,T=ts,K=ks,u=u,y=q,p0=p0s,G1=g1,total=tot,build_seconds=time.perf_counter()-start,normalization=np.array([d['norm'] for d in diagnostics]),L=np.array([d['L'] for d in diagnostics]))
    return OUT
if __name__=='__main__': build()
