"""P1b elastic providers. Atomic cm²/dmu; water effective cm^-1/dmu.
Research-only dcslib reads original files, never copies or rescales them.
"""
from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy.interpolate import CubicSpline
from numpy.polynomial.legendre import leggauss
from .physics import ROOT, Z, NJ, elastic

DCSLIB=ROOT/'docs/egs5_crosscheck/egs5/data/dcslib'

def inigrd():
    # Literal recurrence in pegs/inigrd.f, including original cosine subtraction.
    th=[0.,1e-4]
    while len(th)<606:
        t=th[-1]
        inc=next((d for lim,d in [(0.9999e-3,2.5e-5),(0.9999e-2,2.5e-4),(0.9999e-1,2.5e-3),(.9999,2.5e-2),(9.999, .1),(24.999,.25)] if t<lim),.5)
        th.append(t+inc)
    return (1-np.cos(np.array(th)*np.pi/180))/2

def read_dcslib(z):
    lines=(DCSLIB/f'eeldx{z:03}.tab').read_text().splitlines()
    es=[];head=[];dcs=[];i=0
    while i<len(lines):
        h=lines[i].split();i+=1
        if int(h[0])!=-1 or int(h[1])!=z:raise ValueError('dcslib identity')
        es.append(float(h[2])*1e-6);head.append([float(v) for v in h[3:6]])
        vals=[]
        while len(vals)<606:
            vals.extend(map(float,lines[i].split()));i+=1
        if len(vals)!=606:raise ValueError('dcslib angle count')
        dcs.append(vals)
    a=np.array(dcs)
    if np.any(a<=0):raise ValueError('nonpositive dcslib')
    return np.array(es),a,np.array(head)

class Dcslib:
    name='dcslib'
    def __init__(self):
        self.rmu=inigrd();self.atomic={int(z):read_dcslib(int(z)) for z in Z}
        self.ts=self.atomic[1][0]
        # Convert number densities to stoichiometric indices before summing.
        # The final number-density scale is applied after log-energy interpolation.
        scale=NJ[1];stoich=NJ/scale
        combined=sum(stoich[j]*(z+1)/z*self.atomic[int(z)][1] for j,z in enumerate(Z))
        self.combined=CubicSpline(np.log(self.ts),np.log(combined),axis=0,bc_type='natural')
        self.scale=scale
        self.atomic_splines={z:CubicSpline(np.log(e),np.log(a),axis=0,bc_type='natural') for z,(e,a,h) in self.atomic.items()}
    @lru_cache(maxsize=256)
    def spline(self,T,z=None):
        v=(self.combined if z is None else self.atomic_splines[z])(np.log(T))
        return CubicSpline(self.rmu,v,bc_type='natural')
    def density(self,T,y,z=None):
        return 2*np.pi*np.exp(self.spline(float(T),z)(np.asarray(y)/2))*(self.scale if z is None else 1.)
    def edges(self,T,z=None):return 2*self.rmu
    def quadrature(self,T,order=16,z=None):
        edges=self.edges(T,z);x,w=leggauss(order)
        y=((edges[:-1,None]+edges[1:,None])/2+(edges[1:,None]-edges[:-1,None])*x/2).ravel()
        weights=((edges[1:,None]-edges[:-1,None])*w/2).ravel()*self.density(T,y,z)
        return y,weights

class Rutherford:
    name='rutherford'
    def density(self,T,y,z=None):
        eta,B,total,g=elastic(T)
        a=np.asarray(y)
        if z is None:return np.sum(B[:,None]/(a.ravel()[None,:]+2*eta[:,None])**2,axis=0).reshape(a.shape)
        j=list(Z).index(z)
        return B[j]/NJ[j]/((z+1)/z)/(a+2*eta[j])**2
    def quadrature(self,T,order=16,z=None):
        eta,*_=elastic(T);edges=np.r_[0,np.geomspace(min(eta)*1e-5,2,606)]
        x,w=leggauss(order)
        y=((edges[:-1,None]+edges[1:,None])/2+(edges[1:,None]-edges[:-1,None])*x/2).ravel()
        return y,((edges[1:,None]-edges[:-1,None])*w/2).ravel()*self.density(T,y,z)

def moments(provider,T,L=2,order=16,z=None):
    y,w=provider.quadrature(T,order,z);mu=1-y
    G=np.zeros(L+1);p0=np.ones_like(y);p1=mu.copy()
    if L>=1:G[1]=np.dot(w,y)
    for l in range(2,L+1):
        p=((2*l-1)*mu*p1-(l-1)*p0)/l
        G[l]=np.dot(w,1-p);p0,p1=p1,p
    return G,float(w.sum()),float(w[y>1].sum())

def get_backend(name):
    if name=='rutherford':return Rutherford()
    if name=='dcslib':return Dcslib()
    if name=='eedl':
        from .eedl import Eedl
        return Eedl()
    raise ValueError(name)
