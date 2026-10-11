"""Analytic electron McKinley–Feshbach effective DCS, water H/O only.
Formula/provenance and approximations: approved P1c plan, facts 2/3 and design.
P1 Moliere eta/B and Z(Z+1) unchanged. No positron or high-Z implementation.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from .physics import Z,NJ,M,ALPHA,elastic

def correction(T,y,z,born_only=False):
    beta2=T*(T+2*M)/(T+M)**2;s=np.sqrt(np.asarray(y)/2)
    return 1-beta2*s*s+(0 if born_only else np.pi*ALPHA*z*np.sqrt(beta2)*s*(1-s))

class McKinleyFeshbach:
    name='mf'
    native_energies=()
    def __init__(self,variant='full'):
        if variant not in ('full','nuclear_only','born_only'):raise ValueError(variant)
        self.variant=variant
        if not np.array_equal(Z,np.array([1.,8.])):raise ValueError('P1c is restricted to water H/O')
    def density(self,T,y,z=None):
        eta,B,*_=elastic(T);a=np.asarray(y);flat=a.ravel()
        R=np.array([correction(T,flat,int(zz),self.variant=='born_only') for zz in Z])
        if self.variant=='nuclear_only':R=(Z[:,None]*R+1)/(Z[:,None]+1)
        values=B[:,None]*R/(flat[None,:]+2*eta[:,None])**2
        if z is None:return values.sum(axis=0).reshape(a.shape)
        j=list(Z).index(z)
        return (values[j]/NJ[j]/((z+1)/z)).reshape(a.shape)
    def edges(self,T,z=None):
        eta,*_=elastic(T)
        return np.unique(np.r_[0,np.geomspace(min(eta)*1e-5,2,606),1.])
    def quadrature(self,T,order=16,z=None):
        edges=self.edges(T,z);x,w=leggauss(order)
        y=((edges[:-1,None]+edges[1:,None])/2+(edges[1:,None]-edges[:-1,None])*x/2).ravel()
        weights=((edges[1:,None]-edges[:-1,None])*w/2).ravel()
        return y,weights*self.density(T,y,z)

def born_g1(T):
    """Independent closed-form integrals I1 and I2, no angular quadrature."""
    eta,B,*_=elastic(T);a=2*eta;log=np.log1p(2/a)
    I1=log+a/(a+2)-1
    I2=2-2*a*log+a-a*a/(a+2)
    beta2=T*(T+2*M)/(T+M)**2
    return float(np.sum(B*(I1-beta2/2*I2)))
