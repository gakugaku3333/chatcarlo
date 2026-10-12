"""Water bremsstrahlung, B normalization (no EGS5 data or correction tables).

SLAC-R-730 §2.7, Born/Butcher–Messel screening, eqs. 2.50–2.51;
§2.7 electron–electron factor xi and radiation logarithms. The implemented
PEGS brmsfz convention uses 20.209 for phi2 (the report prints 20.029).
The six H/low-Z logarithm constants are analytic formula constants, not data
copied from an EGS5 table. Existing fetched metadata supplies m, alpha, r0,
and water atom densities. Only the below-50-MeV branch is implemented.
"""
from functools import lru_cache
import numpy as np
from numba import njit
from scipy.integrate import quad
from .physics import M, RE, ALPHA, Z, NJ, ET, LOGET, ER, collision, hard, rotate
CUT = .01
MAX_T = 20.
# SLAC-R-730 §2.7: radiation logarithms and screening scale.
ALRAD_H, ALRADP_H = 5.31, 6.144
LOG_NUCLEAR, LOG_ELECTRON = 184.15, 1194.
SCREEN_SCALE, SCREEN_OFFSET = 136., .952

@njit
def shape(T, k):
    """Unnormalized k*dSigma/dk, summed over H and O, cm^-1."""
    if T > MAX_T or T <= 0.:raise ValueError('brems electron energy outside (0,20] MeV')
    if k < 0. or k > T: return 0.
    E = T+M; rem = E-k; y = rem/E; result = 0.
    for j in range(len(Z)):
        z = Z[j]; az2 = (ALPHA*z)**2
        fc = az2*(1/(1+az2)+.20206+az2*(-.0369+az2*(.0083-.002*az2)))
        xi = ALRADP_H/(ALRAD_H-fc) if z==1 else np.log(LOG_ELECTRON*z**(-2/3))/(np.log(LOG_NUCLEAR*z**(-1/3))-fc)
        xl = 4/3*np.log(z); d = SCREEN_SCALE*z**(-1/3)*M/E*k/rem
        if d >= np.exp((21.12-xl)/4.184)-SCREEN_OFFSET:continue
        if d <= 1:
            s1 = 20.867+d*(-3.242+.625*d)-xl
            s2 = 20.209+d*(-1.930-.086*d)-xl
        else:
            s1 = 21.12-4.184*np.log(d+SCREEN_OFFSET)-xl; s2 = s1
        result += NJ[j]*RE**2*ALPHA*z*(z+xi)*((1+y*y)*s1-2/3*y*s2)
    return result


def integral(T, lo, hi, logarithmic=False):
    # Split at analytic branch switches and the zero endpoint for each element.
    E = T+M; points = []
    for z in Z:
        c = SCREEN_SCALE*z**(-1/3)*M/E
        for d in (1.,np.exp((21.12-4/3*np.log(z))/4.184)-SCREEN_OFFSET):
            k = d*E/(c+d)
            if lo < k < hi:points.append(np.log(k) if logarithmic else k)
    if logarithmic:
        return quad(lambda u:shape(T,np.exp(u)),np.log(lo),np.log(hi),
                    points=sorted(points),epsabs=1e-12,epsrel=2e-10,limit=100)[0]
    return quad(lambda k:shape(T,k),lo,hi,points=sorted(points),
                epsabs=1e-12,epsrel=2e-10,limit=100)[0]

@lru_cache(maxsize=1)
def tables():
    """Memory only; ESTAR nodes plus fixed refinements (before validation)."""
    base = ET[(ET>=CUT)&(ET<=MAX_T)]
    # Max log spacing .003; cutoff offsets resolve log(rate)'s singularity.
    dense = np.geomspace(CUT,MAX_T,int(np.ceil(np.log(MAX_T/CUT)/.003))+1)
    near = CUT+np.geomspace(1e-8,1e-3,401)
    energies = np.unique(np.r_[base,dense,near,CUT,MAX_T])
    v = np.zeros((len(energies),5))  # lambda, hard stopping, soft, raw envelope, A
    for i,T in enumerate(energies):
        rad = np.interp(np.log(T),LOGET,ER)
        norm = rad/integral(T,0.,T); soft = norm*integral(T,0.,min(T,CUT))
        rate = norm*integral(T,CUT,T,True) if T>CUT else 0.
        discrete = rad-soft if T>CUT else 0.
        envelope = max(shape(T,x) for x in np.geomspace(CUT,T,1025))*1.01
        v[i] = rate,discrete,soft,envelope,norm
    return energies,v

@njit
def quantities(T, energies, values, disabled=False):
    if T > MAX_T:raise ValueError('brems electron energy above 20 MeV')
    rad = np.interp(np.log(max(T,CUT)),LOGET,ER)
    if disabled or T<=CUT:return 0.,0.,rad,0.
    i = min(len(energies)-2,max(0,np.searchsorted(energies,T)-1))
    u = (np.log(T)-np.log(energies[i]))/(np.log(energies[i+1])-np.log(energies[i]))
    if i==0:
        rate = u*values[1,0]; discrete = u*values[1,1]
        return rate,discrete,rad-discrete,u*values[1,3]+(1-u)*values[0,3]
    rate = np.exp((1-u)*np.log(values[i,0])+u*np.log(values[i+1,0]))
    soft = np.exp((1-u)*np.log(values[i,2])+u*np.log(values[i+1,2]))
    discrete = np.exp((1-u)*np.log(values[i,1])+u*np.log(values[i+1,1]))
    envelope = (1-u)*values[i,3]+u*values[i+1,3]
    return rate,discrete,soft,envelope

@njit
def restricted_stopping(T, energies, values, disabled=False, enable_hard=True):
    # G6b preserves the exact original operation order.
    soft = quantities(T,energies,values,disabled)[2]
    return collision(T)-(hard(T)[1] if enable_hard else 0.)+soft

@njit
def sample_k(T, energies, values):
    if T<=CUT or T>MAX_T:raise ValueError('invalid brems emission energy')
    envelope = quantities(T,energies,values)[3]; excess = 0
    lo = np.log(CUT); width = np.log(T/CUT)
    while True:
        k = min(T,max(CUT,np.exp(lo+width*np.random.random())))
        f = shape(T,k)
        if f>envelope:excess+=1
        if np.random.random()*envelope<=f:return k,excess

@njit
def direction(T, ux, uy, uz, phi):
    theta = M/(T+M)
    return rotate(ux,uy,uz,np.cos(theta),phi)

@njit
def new_remaining(enabled):
    # No RNG draw when rate is disabled (G6b).
    return -np.log(np.random.random()) if enabled else np.inf

@njit
def advance_remaining(remaining, step, rate, active=True):
    return max(0.,remaining-step*rate) if active else remaining

@njit
def event_order(bb, eb, mb, hb, rb, ab):
    """Stable priority for exact ties, also used by deterministic tests."""
    d = np.array([bb,eb,mb,hb,rb,ab])
    return np.argmin(d)
