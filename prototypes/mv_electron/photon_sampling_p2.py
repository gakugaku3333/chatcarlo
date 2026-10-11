"""P2 photon/secondary samplers; pair defaults follow distributed egs5_pair.f."""
import numpy as np
from numba import njit
from .physics import M,rotate

@njit
def compton(E,ux,uy,uz):
    # Composition/rejection using q + 1/q as envelope for dσ/dq.
    qmin=1/(1+2*E/M);a=np.log(1/qmin);b=(1-qmin*qmin)/2
    while True:
        if np.random.random()*(a+b)<a:q=np.exp(-np.random.random()*a)
        else:q=np.sqrt(qmin*qmin+np.random.random()*(1-qmin*qmin))
        mu=1-M/E*(1/q-1);sin2=max(0.,1-mu*mu)
        if np.random.random()*(q+1/q)<=q+1/q-sin2:break
    phi=2*np.pi*np.random.random();gx,gy,gz=rotate(ux,uy,uz,mu,phi);ep=E*q;T=E-ep
    px=E*ux-ep*gx;py=E*uy-ep*gy;pz=E*uz-ep*gz;p=np.sqrt(px*px+py*py+pz*pz)
    if p==0:return ep,gx,gy,gz,0.,ux,uy,uz
    return ep,gx,gy,gz,T,px/p,py/p,pz/p

@njit
def photoelectric(E,ux,uy,uz):
    return E,ux,uy,uz

@njit
def pair_small_total(E,dl,bpar,delcm,delpos):
    if E<2*M:raise ValueError('pair below physical threshold')
    if E<=2.1:return M+np.random.random()*(E/2-M)
    # lvx=1 and lvl0=0 over all P2 energies (<=20 MeV).
    while True:
        r1=np.random.random();r=np.random.random()
        if r>=bpar:
            br=.5*(1-max(r1,np.random.random(),np.random.random()));lvl=0
        else:br=r1*.5;lvl=2
        if E*br<M:continue
        d=1/(E*br*(1-br))
        if d>=delpos:continue
        delta=delcm*d
        if delta<1:rej=dl[lvl,0]+delta*(dl[lvl,1]+delta*dl[lvl,2])
        else:rej=dl[lvl,3]+dl[lvl,4]*np.log(delta+dl[lvl,5])
        if np.random.random()<=rej:return E*br

@njit
def pair(E,ux,uy,uz,dl,bpar,delcm,delpos):
    e2=pair_small_total(E,dl,bpar,delcm,delpos);e1=E-e2
    mu=np.cos(M/E);phi=2*np.pi*np.random.random()
    d1=rotate(ux,uy,uz,mu,phi);d2=rotate(ux,uy,uz,mu,phi+np.pi)
    charge1=1. if np.random.random()<=.5 else -1.
    return e1-M,d1[0],d1[1],d1[2],charge1,e2-M,d2[0],d2[1],d2[2],-charge1

@njit
def annihilation():
    mu=2*np.random.random()-1;phi=2*np.pi*np.random.random();sn=np.sqrt(1-mu*mu)
    return sn*np.cos(phi),sn*np.sin(phi),mu

@njit
def samples(E,n,seed,dl,bpar,delcm,delpos):
    np.random.seed(seed);c=np.empty((n,8));p=np.empty((n,10))
    for i in range(n):
        c[i]=compton(E,0.,0.,1.)
        p[i]=pair(E,0.,0.,1.,dl,bpar,delcm,delpos)
    return c,p

@njit
def annihilation_samples(n,seed):
    np.random.seed(seed);a=np.empty((n,3))
    for i in range(n):a[i]=annihilation()
    return a

def pair_parameters():
    from .p2_common import ROOT,CACHE,OUT,sha,portable,write,save
    source=ROOT/'docs/validation/mv/p1/egs5/base_2/pgs5job.pegs5dat'
    # READ format of HATCH: 36 DL entries followed by DELCM, ALPHI/BPAR/DELPOS.
    lines=source.read_text().splitlines()
    values=[]
    for line in lines[6:]:
        if not line.strip():continue
        try:values.extend(map(float,line.split()))
        except ValueError:break
        if len(values)>=43:break
    values=np.array(values)
    if len(values)<43:raise RuntimeError('PEGS pair constants could not be read')
    dl=values[:36].reshape(6,6);tail=values[36:43]
    target=CACHE/'pair_parameters.npz'
    if target.exists():
        with np.load(target) as saved:
            if not np.array_equal(saved['dl'],dl) or float(saved['bpar'])!=tail[2] or float(saved['delcm'])!=tail[0] or float(saved['delpos'])!=tail[3]:
                raise RuntimeError('cached pair constants differ from protected PEGS source')
    else:save(target,dl=dl,bpar=tail[2],delcm=tail[0],delpos=tail[3])
    if not (OUT/'pair_parameters.json').exists():write('pair_parameters.json',dict(source=portable(source),sha256=sha(source),line_start=7,line_end=15,
          dl=dl.tolist(),bpar=float(tail[2]),delcm=float(tail[0]),delpos=float(tail[3]),
          note='material-dependent EGS5/PEGS screening constants; no dose fitting. Same H2O/rho=1. AP does not enter DIFFER.'))
    return dl,float(tail[2]),float(tail[0]),float(tail[3])

@njit
def photoelectric_check(E,n,seed):
    np.random.seed(seed);max_energy=0.;max_direction=0.
    for i in range(n):
        ux,uy,uz=annihilation()
        T,ex,ey,ez=photoelectric(E,ux,uy,uz)
        max_energy=max(max_energy,abs(T-E))
        max_direction=max(max_direction,abs(ex-ux),abs(ey-uy),abs(ez-uz))
    return max_energy,max_direction
