"""Converged lookup of one provider's rates/single CDF, independent of GS energy mesh.
Only numerical lookup resolution is refined. Native DCS energy interpolation unchanged.
"""
import numpy as np
from .dcs import moments
from .gs_backend import single_cdf
TOL=1e-4

def inverse_moments(u,q):
    a=q[:-1];b=q[1:];du=np.diff(u)
    return np.array([np.sum(du*(a+b)/2),np.sum(du*(3*(a+b)/2-.5*(a*a+a*b+b*b)))])

def build(provider):
    native=[]
    if provider.name=='dcslib':native=provider.ts
    else:
        for atom in provider.atomic.values():
            native.extend(atom.ts)
            for arr in atom.xs.values():native.extend(arr[:,0]*1e-6)
    ts=np.unique(np.r_[np.geomspace(.01,20,49),native,[.02,.1,.256,1.,2.,10.]])
    ts=ts[(ts>=.01)&(ts<=20)]
    cache={}
    def node(t):
        t=float(t)
        if t not in cache:
            G,total,_=moments(provider,t);y,c=single_cdf(provider,t)
            cache[t]=(np.array([total,G[1],G[2]]),y,c)
        return cache[t]
    factor=8
    while True:
        u=np.unique(np.r_[np.linspace(0,1,2048*factor+1),1-np.geomspace(32*np.finfo(float).eps,.1,256*factor+1),np.geomspace(32*np.finfo(float).eps,.1,256*factor+1)])
        worst=0.
        for t in ts:
            r,y,c=node(t);q=np.interp(u,c,y)
            worst=max(worst,float(max(abs(r[0]*inverse_moments(u,q)/r[1:]-1))))
        print(provider.name,'single probability refinement',factor,worst,flush=True)
        if worst<=TOL/4:break
        factor*=2
        if factor>64:raise ValueError('single probability lookup did not converge')
    rounds=[]
    for iteration in range(12):
        additions=[];maxerror=0.
        for a,b in zip(ts[:-1],ts[1:]):
            t=np.sqrt(a*b);ra,ya,ca=node(a);rb,yb,cb=node(b);r,y,c=node(t)
            interp=np.sqrt(ra*rb);q=(np.interp(u,ca,ya)+np.interp(u,cb,yb))/2
            err=max(float(max(abs(interp/r-1))),float(max(abs(interp[0]*inverse_moments(u,q)/r[1:]-1))))
            maxerror=max(maxerror,err)
            if err>TOL:additions.append(t)
        rounds.append({'points':len(ts),'max_relative_error':maxerror,'added':len(additions)})
        print(provider.name,'single lookup refinement',rounds[-1],flush=True)
        if not additions:break
        ts=np.unique(np.r_[ts,additions])
    else:raise ValueError('single energy lookup did not converge')
    rates=np.empty((len(ts),3));single=np.empty((len(ts),len(u)+1))
    for i,t in enumerate(ts):
        r,y,c=node(t);rates[i]=[t,r[0],r[1]];single[i,0]=t;single[i,1:]=np.interp(u,c,y)
    return u,rates,single,{'tolerance':TOL,'probability_factor':factor,'native_probability_max_relative_error':worst,'energy_refinement':rounds,'rate_points':len(ts)}
