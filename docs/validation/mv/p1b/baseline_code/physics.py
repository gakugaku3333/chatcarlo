"""MeV/cm water electron model. Equations: SLAC-R-730 §§2.10,2.13,2.14.
Constants are extracted by fetch_data.py; density effect and I from NIST.
"""
from pathlib import Path
import json
import numpy as np
from numba import njit
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'docs/validation/mv/p1/data'
META=json.loads((DATA/'metadata.json').read_text())
M=META['electron_mass_MeV']; RE=META['classical_radius_cm']; ALPHA=META['alpha']
I=META['I_MeV']; NA=META['avogadro']; DELTA=.01
# Z/A taken directly from ESTAR's response; elemental number densities from fetched composition.
import re
ZA=float(re.search(r'Z/A.*?<br>\s*([\d.]+)',(DATA/'estar_water.html').read_text()).group(1))
NE=NA*ZA
Z=np.array([int(z) for z in META['weights']],dtype=float)
NJ=np.array([NA*META['weights'][str(int(z))]/META['atomic_weights'][str(int(z))] for z in Z])
estar=np.load(DATA/'estar.npz'); ET=estar['T']; ED=estar['delta']; ER=estar['radiative']
LOGET=np.log(ET)
C=2*np.pi*RE**2*M*NE
@njit
def collision(T):
    tau=T/M; g=1+tau; b2=1-1/(g*g)
    # unrestricted Berger/Seltzer electron F^-(tau,tau/2), eqs 2.257/2.268.
    d=tau/2
    F=-1-b2+np.log((tau-d)*d)+tau/(tau-d)+(d*d/2+(2*tau+1)*np.log1p(-d/tau))/(g*g)
    density=np.interp(np.log(T),LOGET,ED)
    return C/b2*(np.log(2*(tau+2)/(I/M)**2)+F-density)
@njit
def moller_params(T):
    g=1+T/M; b2=1-1/g**2
    return (g-1)**2/g**2,(2*g-1)/g**2,C/(b2*T)
@njit
def primitive0(x,a,b):
    return a*x-1/x+1/(1-x)-b*np.log(x/(1-x))
@njit
def primitive1(x,a,b):
    return a*x*x/2+np.log(x)+1/(1-x)+(1+b)*np.log1p(-x)
@njit
def hard(T):
    if T<=2*DELTA:return 0.,0.
    a,b,c=moller_params(T); x=DELTA/T
    # Near threshold avoid cancellation with explicit local quadrature.
    if .5-x<1e-7:
        h=(T/2-DELTA)/T; mid=.5-h/2
        v=a+1/mid**2+1/(1-mid)**2-b/(mid*(1-mid))
        return c*h*v,c*T*h*mid*v
    return c*(primitive0(.5,a,b)-primitive0(x,a,b)),c*T*(primitive1(.5,a,b)-primitive1(x,a,b))
@njit
def dcs_moller(T,W):
    a,b,c=moller_params(T);x=W/T
    return c/T*(a+1/x**2+1/(1-x)**2-b/(x*(1-x)))
@njit
def stopping(T,enable_hard=True):
    sh=hard(T)[1] if enable_hard else 0.
    return collision(T)-sh+np.interp(np.log(T),LOGET,ER)
@njit
def sample_W(T):
    a,b,c=moller_params(T); lo=DELTA/T
    while True:
        x=1/(1/lo-np.random.random()*(1/lo-2))
        # Proposal 1/x²; bound 2+a/4 follows x/(1-x)<=1, negative interference.
        v=1+a*x*x+(x/(1-x))**2-b*x/(1-x)
        if np.random.random()*(2+a/4)<=v:return T*x
@njit
def rotate(ux,uy,uz,mu,phi):
    sn=np.sqrt(max(0.,1-mu*mu)); r=np.sqrt(max(0.,1-uz*uz))
    if r>1e-12:
        ax=ux*uz/r;ay=uy*uz/r;az=-r; bx=-uy/r;by=ux/r
    else:
        ax=1.;ay=0.;az=0.;bx=0.;by=1. if uz>=0 else -1.
    cp=np.cos(phi);sp=np.sin(phi)
    return mu*ux+sn*(cp*ax+sp*bx),mu*uy+sn*(cp*ay+sp*by),mu*uz+sn*cp*az
@njit
def moller_directions(T,W,ux,uy,uz,phi):
    # Two-body kinematics: incident + target at rest, opposite transverse momenta.
    k=(T+2*M)/T
    mp=np.sqrt((T-W)/(T-W+2*M)*k);ms=np.sqrt(W/(W+2*M)*k)
    return rotate(ux,uy,uz,mp,phi),rotate(ux,uy,uz,ms,phi+np.pi)
@njit
def elastic(T):
    p2=T*(T+2*M);b2=p2/(T+M)**2
    # Moliere screening, eqs 2.309--2.314: eta = chi_a²/4.
    eta=(ALPHA*M*Z**(1/3)/(.885*np.sqrt(p2)))**2*(1.13+3.76*(ALPHA*Z/np.sqrt(b2))**2)/4
    # dSigma/dmu = B/(1-mu+2 eta)^2, Z(Z+1) includes soft e-e approximation.
    B=2*np.pi*RE**2*M*M*Z*(Z+1)*NJ/(p2*b2)
    total=B*(1/(2*eta)-1/(2+2*eta))
    G1=np.sum(B*(np.log1p(1/eta)+eta/(1+eta)-1))
    return eta,B,total,G1
@njit
def single_cached(eta,total):
    r=np.random.random()*np.sum(total)
    j=0
    while j<len(total)-1 and r>total[j]:r-=total[j];j+=1
    a=2*eta[j];u=np.random.random()
    y=1/(1/a-u*(1/a-1/(a+2)))-a
    return 1-y
@njit
def single_mu(T):
    eta,B,total,g=elastic(T)
    return single_cached(eta,total)

@njit
def elastic_rates(T):
    p2=T*(T+2*M);b2=p2/(T+M)**2
    total=0.;G1=0.
    for j in range(len(Z)):
        eta=(ALPHA*M*Z[j]**(1/3)/(.885*np.sqrt(p2)))**2*(1.13+3.76*(ALPHA*Z[j]/np.sqrt(b2))**2)/4
        B=2*np.pi*RE**2*M*M*Z[j]*(Z[j]+1)*NJ[j]/(p2*b2)
        total+=B*(1/(2*eta)-1/(2+2*eta))
        G1+=B*(np.log1p(1/eta)+eta/(1+eta)-1)
    return total,G1
