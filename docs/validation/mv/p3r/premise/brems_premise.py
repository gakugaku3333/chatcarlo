'''P3r premise check (plan section 0.3 and 1). Formula of PEGS5 brmsdz/brmsfz (SLAC-R-730), aprim=1, water.
Quadrature: scipy.integrate.quad. Run from the repository root: .venv/bin/python docs/validation/mv/p3r/premise/brems_premise.py
'''
import numpy as np, json
from scipy.integrate import quad
RM=0.51099895069; FSC=1/137.035999; R0=2.8179403e-13; AN=6.02214076e23
Zs=[1.,8.]; WA=[1.00794,15.9994]; w=[0.111894,0.888106]
nj=[AN*w[i]/WA[i] for i in range(2)]            # atoms/cm3 (rho=1)
ALRAD={1:5.31}; ALRADP={1:6.144}
def fc(z):
    a=(FSC*z)**2; return a*(1/(1+a)+0.20206+a*(-0.0369+a*(0.0083+a*(-0.002))))
def xsi(z):
    if z<=4: return ALRADP[int(z)]/(ALRAD[int(z)]-fc(z))
    return np.log(1194.*z**(-2/3))/(np.log(184.15*z**(-1/3))-fc(z))
def kds(z,n,E,k):   # k*dSigma/dk [1/cm], E total energy, below 50 MeV
    delc=136*z**(-1/3)*RM/E; xl=4/3*np.log(z); dm=np.exp((21.12-xl)/4.184)-0.952
    em=E-k; d=delc*k/em
    if d>=dm: return 0.
    if d<=1: s1=20.867+d*(-3.242+d*0.625)-xl; s2=20.209+d*(-1.930+d*(-0.086))-xl
    else: s1=21.12-4.184*np.log(d+0.952)-xl; s2=s1
    ee=em/E
    return n*R0**2*FSC*z*(z+xsi(z))*((1+ee*ee)*s1-2/3*ee*s2)
def srad(T,k0=0.,k1=None):
    E=T+RM; k1=T if k1 is None else min(k1,T)
    return sum(quad(lambda k:kds(z,n,E,k),k0,k1,limit=200)[0] for z,n in zip(Zs,nj))
def sig(T,kc=0.01):
    E=T+RM
    return sum(quad(lambda u:kds(z,n,E,np.exp(u)),np.log(kc),np.log(T),limit=200)[0] for z,n in zip(Zs,nj))
e=np.load('docs/validation/mv/p1/data/estar.npz')
print(' T     BH_srad   ESTAR_rad  ratio(A_own)  Srad(k<10keV)/tot  sigma(k>10keV)[1/cm]  mean k[MeV]')
for T in [0.05,0.1,0.2,0.5,1,2,5,10]:
    s=srad(T); er=np.interp(np.log(T),np.log(e['T']),e['radiative']); lo=srad(T,0,0.01); sg=sig(T)
    print(f'{T:5.2f} {s:.5e} {er:.5e} {er/s:.4f}   {lo/s:.4f}   {sg:.4e}   {(s-lo)/sg:.4f}')
# EGS5 side of section 0.2 is read from the external PEGS listing (pgs5job.pegs5lst), not stored here.

# --- mixture-level sampling shape (egs5_brems.f / bremfr) vs per-element sum
al2=np.log(2.); al183=np.log(184.15)
# Mixture variables as in PEGS5 subroutine MIX (PZ = atoms per unit mass, normalisation cancels).
_pz=[w[i]/WA[i] for i in range(2)]; _zzx=[_pz[i]*Zs[i]*(Zs[i]+xsi(Zs[i])) for i in range(2)]
ZT=sum(_zzx); ZB=sum(_zzx[i]*np.log(Zs[i]**(-1/3)) for i in range(2)); ZG=ZB/ZT; ZP=ZB/(al183*ZT)
print('ZG,ZP',ZG,ZP,'(PEGS listing: -6.550096E-01 -1.255830E-01)')
alphi=al2*(4/3+1/(9*al183*(1+ZP))); delcm=136*np.exp(ZG)*RM
delpos=(np.exp((21.12+4*ZG)/4.184)-0.952)/delcm
f10=4*al183; f20=f10-2/3; a1=3*f10-f20+8*ZG; b1=f10+4*ZG
def kmix(E,k):
    eps=k/E; d0=eps/(E*(1-eps))
    if d0>=delpos: return 0.
    d=delcm*d0
    if d<=1:
        A=((3*20.867-20.209+8*ZG)+d*((3*-3.242+1.930)+d*(3*0.625-0.086)))/a1
        B=((20.867+4*ZG)+d*(-3.242+d*0.625))/b1
    else:
        A=((2*21.12+8*ZG)+2*-4.184*np.log(d+0.952))/a1; B=((21.12+4*ZG)-4.184*np.log(d+0.952))/b1
    return k*(alphi*(1-eps)/eps/al2*A+eps*B)/E
print('\n T   mean k (element sum)  mean k (mixture)  rel diff   max|shape ratio-1| (unit-normalised, k>10keV)')
for T in [0.05,0.1,0.5,1,2,5,10]:
    E=T+RM; u=np.linspace(np.log(0.01),np.log(T),4001)[:-1]; k=np.exp(u)
    fe=np.array([sum(kds(z,n,E,x) for z,n in zip(Zs,nj)) for x in k]); fm=np.array([kmix(E,x) for x in k])
    pe=fe/np.trapezoid(fe,u); pm=fm/np.trapezoid(fm,u)
    me=np.trapezoid(pe*k,u); mm=np.trapezoid(pm*k,u)
    ok=(pe>1e-3*pe.max())
    print(f'{T:5.2f} {me:.5f} {mm:.5f} {mm/me-1:+.4%}  {np.abs(pm[ok]/pe[ok]-1).max():.4%}')
