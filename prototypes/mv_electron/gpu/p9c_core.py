"""P9c geometry, grid ABI, fixed patient coordinates and source contracts."""
import ctypes as ct
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from . import p9b_core as previous
from .p9b_core import D,I,DP,IP
core=previous.core
HERE=Path(__file__).resolve().parent
ANGLES=(0,45,90,135,225,270,315)
AXES=np.array([2.25,1.75,1.5])

@dataclass(frozen=True)
class Grid:
    lo: tuple
    step: tuple
    shape: tuple
    def __post_init__(self):
        if len(self.lo)!=3 or len(self.step)!=3 or len(self.shape)!=3 or not np.isfinite([*self.lo,*self.step]).all() or min(self.step)<=0 or any(type(n) is not int or n<=0 for n in self.shape):raise ValueError('grid')
    @property
    def hi(self):return np.array(self.lo)+np.array(self.step)*self.shape
    @property
    def size(self):return math.prod(self.shape)
    @property
    def volume(self):return math.prod(self.step)
    def index(self,p):
        # Round only representational noise at a mathematically exact face.
        u=(np.asarray(p)-self.lo)/self.step
        near=np.rint(u);u=np.where(abs(u-near)<=4*np.spacing(np.maximum(1.,abs(u))),near,u)
        i=np.floor(u).astype(int)
        if np.any(i<0) or np.any(i>=self.shape):raise ValueError('outside grid')
        return tuple(i)
    def centers(self):return tuple(self.lo[k]+(np.arange(self.shape[k])+.5)*self.step[k] for k in range(3))
    def record(self):return dict(lo=list(self.lo),step=list(self.step),shape=list(self.shape))
FINE=Grid((-3.,-2.5,-2.5),(.025,)*3,(240,200,300))
COARSE=Grid((-18.2,-15.2,-12.2),(.2,)*3,(182,152,122))
class GridABI(ct.Structure):
    _fields_=[('lo',ct.c_double*3),('step',ct.c_double*3),('n',ct.c_int*3),('values',DP),('adds',ct.POINTER(ct.c_ulonglong))]
class Geometry(previous.Geometry):
    _fields_=[('outer',ct.c_int),('beam',ct.c_int),('radii',ct.c_double*3),('angle',ct.c_double),('grids',GridABI*2)]

def seeds(material=4):
    result=[]
    for i in range(-4,5):
        for j in range(-3,4):
            if (i+j)%2:continue
            for y in (-.9,-.2,.5,1.2):
                if all((i*.5/2.25)**2+(j*.5/1.5)**2+((y+end)/1.75)**2<=1. for end in (-.25,.25)):
                    result.append((i*.5,y,j*.5,.05,.25,material))
    if len(result)!=58 or len(set((s[0],s[2]) for s in result))!=19:raise AssertionError('58 seeds / 19 needles')
    return tuple(result)

def geometry(densities,edges=None,cum=None,energy=1.,angle=0,beam='rectangle',with_seeds=True,grids=(FINE,COARSE),cylinders=None,radii=(18.,15.,12.),layers=None,**kw):
    if beam not in ('rectangle','prostate') or not np.isfinite(angle) or len(grids)!=2:raise ValueError('source/grids')
    radii=np.asarray(radii,float)
    if radii.shape!=(3,) or not np.isfinite(radii).all() or min(radii)<=0:raise ValueError('radii')
    kw.setdefault('field_half',3.5)
    ss=seeds(4 if with_seeds else 0) if cylinders is None else tuple(cylinders)
    # Reuse the strict non-contact seed constructor in a containing box. Shift its
    # validation coordinate only; transport coordinates stay fixed in the patient.
    if layers is None:
        shift=radii[2];ls=((1,2*shift,.2),)
        shifted=[(x,y,z+shift,r,h,m) for x,y,z,r,h,m in ss]
        old,arrays,layout=previous.geometry(ls,densities,edges,cum,energy=energy,seeds=shifted,half=radii,**kw)
        for x,y,z,r,h,m in ss:
            if ((abs(x)+r)/radii[0])**2+((abs(z)+r)/radii[2])**2>=1 or abs(y)+h>=radii[1]:raise ValueError('seed outside body')
        arrays['seeds'][:,2]-=shift
        arrays['faces']-=shift
    else:
        old,arrays,layout=previous.geometry(layers,densities,edges,cum,energy=energy,seeds=ss,**kw)
    g=Geometry();ct.memmove(ct.addressof(g),ct.addressof(old),ct.sizeof(old))
    g.outer=int(layers is None);g.beam=int(beam=='prostate');g.radii[:]=radii;g.angle=np.deg2rad(angle)
    g.nb=g.nc=g.nf=0;g.scoring_enabled=2 if kw.get('score_field') else 1
    for j,grid in enumerate(grids):g.grids[j].lo[:]=grid.lo;g.grids[j].step[:]=grid.step;g.grids[j].n[:]=grid.shape
    return g,arrays,grids

def frame(angle):
    t=np.deg2rad(angle);s=np.array([100*np.sin(t),0.,-100*np.cos(t)])
    return s,-s/100,np.array([np.cos(t),0.,np.sin(t)]),np.array([0.,1.,0.])
def ray(angle,a,b):
    s,_,u,v=frame(angle);d=a*u+b*v-s;return s,d/np.linalg.norm(d)
def intersects(s,d):
    x=np.asarray(s)/AXES;u=np.asarray(d)/AXES;B=x@u;return bool(B<0 and B*B-(u@u)*(x@x-1)>=0)
def entry(s,d,radii=(18.,15.,12.)):
    x=np.asarray(s)[[0,2]]/np.asarray(radii)[[0,2]];u=np.asarray(d)[[0,2]]/np.asarray(radii)[[0,2]]
    A=u@u;B=x@u;C=x@x-1;D=B*B-A*C
    if D<=0:return None
    lo=max(0.,(-B-np.sqrt(D))/A);hi=(-B+np.sqrt(D))/A
    if d[1]:
        a,b=sorted(((-radii[1]-s[1])/d[1],(radii[1]-s[1])/d[1]));lo=max(lo,a);hi=min(hi,b)
    elif abs(s[1])>=radii[1]:return None
    return np.asarray(s)+lo*np.asarray(d) if hi>lo else None

def seed_mask(grid,cylinders=None):
    mask=np.zeros(grid.shape,bool);ss=seeds() if cylinders is None else cylinders
    lo=np.array(grid.lo);step=np.array(grid.step)
    for x,y,z,r,h,*_ in ss:
        bounds=np.array([[x-r,y-h,z-r],[x+r,y+h,z+r]])
        # Include touching cells on both sides of a grid face.
        u=(bounds-lo)/step;near=np.rint(u);u=np.where(abs(u-near)<32*np.finfo(float).eps*np.maximum(1.,abs(u)),near,u)
        start=np.maximum(0,np.ceil(u[0]).astype(int)-1)
        stop=np.minimum(grid.shape,np.floor(u[1]).astype(int)+1)
        if np.any(stop<=start):continue
        xs=lo[0]+np.arange(start[0],stop[0])*step[0];zs=lo[2]+np.arange(start[2],stop[2])*step[2]
        dx=np.maximum(np.maximum(xs-x,x-xs-step[0]),0);dz=np.maximum(np.maximum(zs-z,z-zs-step[2]),0)
        radial=dx[:,None]**2+dz[None,:]**2<=r*r+8*np.spacing(r*r)
        mask[tuple(slice(a,b) for a,b in zip(start,stop))]|=radial[:,None,:]
    return mask

def water_mask(grid,coarse=False,cylinders=None,radii=(18.,15.,12.)):
    mask=~seed_mask(grid,cylinders)
    if coarse:
        coords=grid.centers();x=abs(coords[0])+grid.step[0]/2;z=abs(coords[2])+grid.step[2]/2;y=abs(coords[1])+grid.step[1]/2
        mask&=((x[:,None,None]/radii[0])**2+(z[None,None,:]/radii[2])**2<=1)&(y[None,:,None]<=radii[1])
    return mask

def fingerprint():
    names=set(previous.fingerprint())|{p.name for p in HERE.glob('p9c*') if p.suffix in ('.py','.h','.cpp','.cu') and p.name not in ('p9c_cli.py','p9c_colab.py')}
    return {n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in sorted(names)}
def expand(name,seen=None):
    seen=set() if seen is None else seen;lines=[]
    for line in (HERE/name).read_text().splitlines():
        if line.startswith('#include "'):
            child=line.split('"')[1]
            if child not in seen:seen.add(child);lines.append(expand(child,seen))
        else:lines.append(line)
    return '\n'.join(lines)
def cuda_source():return expand('p9c_gpu.cu')
