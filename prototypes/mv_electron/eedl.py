"""ENDF MF23 / MF26 LAW=2 reader: obey file interpolation, no normalization fitting."""
from pathlib import Path
import numpy as np
from functools import lru_cache
from numpy.polynomial.legendre import leggauss
from .physics import ROOT,Z,NJ,M,ALPHA
DATA=ROOT/'docs/validation/mv/p1b/data'

def fields(line):
    return [float(line[i:i+11].strip() or '0') for i in range(0,66,11)]

class Records:
    def __init__(self,lines):self.lines=lines;self.i=0
    def row(self):
        a=fields(self.lines[self.i]);self.i+=1;return a
    def values(self,n):
        a=[]
        while len(a)<n:a.extend(self.row())
        return np.array(a[:n])
    def tab1(self):
        h=self.row();nr,np_=int(h[4]),int(h[5]);regions=self.values(2*nr).reshape(-1,2).astype(int);v=self.values(2*np_).reshape(-1,2)
        return h,regions,v

class EndfAtom:
    def __init__(self,z):
        lines=(DATA/f'ZA{z:03}000').read_text().splitlines();self.z=z
        def section(mf,mt):return Records([l for l in lines if l[70:72].strip()==str(mf) and l[72:75].strip()==str(mt)])
        self.xs={};self.regions23={}
        for mt in [525,526]:
            r=section(23,mt);r.row();h,regions,v=r.tab1()
            if np.any(regions[:,1]!=2):raise ValueError('unsupported MF23 interpolation')
            self.xs[mt]=v;self.regions23[mt]=regions.tolist()
        r=section(26,525);head=r.row();h,reg,yield_=r.tab1()
        if int(head[4])!=1 or int(h[3])!=2:raise ValueError('MF26 requires one product, LAW=2')
        tab2=r.row();regions=r.values(2*int(tab2[4])).reshape(-1,2).astype(int)
        if np.any(regions[:,1]!=2):raise ValueError('unsupported MF26 energy interpolation')
        self.angular=[]
        for i in range(int(tab2[5])):
            h=r.row()
            if int(h[2])!=12 or int(h[4])!=2*int(h[5]):raise ValueError('requires LANG12')
            v=r.values(int(h[4])).reshape(-1,2);self.angular.append((h[1]*1e-6,v))
        self.ts=np.array([a[0] for a in self.angular])
        self.regions26=regions.tolist()
    def cross(self,T,mt):
        v=self.xs[mt];return float(np.interp(T*1e6,v[:,0],v[:,1])*1e-24)
    @lru_cache(maxsize=256)
    def angular_at(self,T):
        j=np.searchsorted(self.ts,T)-1;j=max(0,min(j,len(self.ts)-2));a=(T-self.ts[j])/(self.ts[j+1]-self.ts[j])
        v0=self.angular[j][1];v1=self.angular[j+1][1]
        if v0[-1,0]!=v1[-1,0]:raise ValueError('varying angular endpoint requires explicit ENDF treatment')
        mu=np.unique(np.r_[v0[:,0],v1[:,0]])
        p=(1-a)*np.interp(mu,v0[:,0],v0[:,1])+a*np.interp(mu,v1[:,0],v1[:,1])
        return mu,p

class Eedl:
    name='eedl'
    def __init__(self):self.atomic={int(z):EndfAtom(int(z)) for z in Z}
    def eta(self,T,z):
        # EEDL1991 cites Seltzer ETRAN screening. Original cited chapter Eq. 7.8 visually verified.
        # Denominator here uses eta_EEDL = 2 eta_Moliere.
        # Seltzer empirical correction sqrt(tau/(tau+1)); not P1 screening.
        tau=T/M;p2=T*(T+2*M);b2=p2/(T+M)**2
        return .5*(ALPHA*M*z**(1/3)/(.885*np.sqrt(p2)))**2*(1.13+3.76*(ALPHA*z)**2/b2*np.sqrt(tau/(tau+1)))
    def parameters(self,T,z):
        atom=self.atomic[z];mu,p=atom.angular_at(float(T));yc=1-mu[-1];eta=self.eta(T,z)
        total=atom.cross(T,526);large=atom.cross(T,525)
        A=(total-large)/(1/eta-1/(eta+yc))
        return mu,p,yc,eta,A,large,total
    def density(self,T,y,z=None):
        a=np.asarray(y)
        if z is None:return sum(NJ[j]*(zz+1)/zz*self.density(T,a,int(zz)) for j,zz in enumerate(Z))
        mu,p,yc,eta,A,large,total=self.parameters(T,z)
        return np.where(a>=yc,large*np.interp(1-a,mu,p),A/(eta+a)**2)
    def edges(self,T,z=None):
        if z is None:return np.unique(np.concatenate([self.edges(T,int(zz)) for zz in Z]))
        mu,p,yc,eta,A,large,total=self.parameters(T,z)
        return np.unique(np.r_[0,np.geomspace(min(eta,yc)*1e-5,yc,129),1-mu])
    def quadrature(self,T,order=16,z=None):
        edges=self.edges(T,z);x,w=leggauss(order)
        y=((edges[:-1,None]+edges[1:,None])/2+(edges[1:,None]-edges[:-1,None])*x/2).ravel()
        return y,((edges[1:,None]-edges[:-1,None])*w/2).ravel()*self.density(T,y,z)
