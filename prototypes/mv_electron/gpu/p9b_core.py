"""P9b geometry ABI, strict construction, and CPU/CUDA shared contracts."""
import ctypes as ct
from pathlib import Path
import hashlib
import numpy as np
from . import p7_core, p8g2_core as core
from .p8g import D, I, DP, IP, Geometry as SlabGeometry

HERE=Path(__file__).resolve().parent
CAPS=(220,160,640)
class Geometry(SlabGeometry):
    _fields_=[('seeds',DP),('parents',IP),('axis',ct.c_int),('ns',ct.c_int),('field',ct.c_int),('scoring_enabled',ct.c_int),
              ('half',ct.c_double*3),('field_half',ct.c_double),('ssd',ct.c_double),('scoring',p7_core.p6.Geometry)]

def geometry(layers,densities,edges=None,cum=None,binding=True,fluorescence=True,energy=1.,axis=2,half=(1e6,1e6,1e6),seeds=(),field=False,field_half=2.5,ssd=100.,score_field=None):
    from ..p9a import layout
    from ..photon_transport_p8e import BOUNDARY_STRENGTH_EDGES
    if type(axis) is not int or axis not in (0,1,2):raise ValueError('axis')
    half=np.asarray(half,float)
    if half.shape!=(3,) or not np.isfinite(half).all() or np.any(half<=0):raise ValueError('box half widths')
    if field and axis!=2:raise ValueError('field requires +z')
    if not np.isfinite([energy,field_half,ssd]).all() or not 0<energy<=10 or min(field_half,ssd)<=0:raise ValueError('source')
    grid=layout(layers,densities);faces=np.r_[0.,np.cumsum([t for _,t,_ in layers])]
    ids=[m-1 for m,_,_ in layers];nl=len(ids);rho=[densities[j] for j in ids]
    nb=len(grid['mass_g_cm2']);nc=round(faces[-1]/.2);nf=round(faces[-1]/.05)
    if min(nb,nc,nf)<=0 or any(a>b for a,b in zip((nb,nc,nf),CAPS)) or nb+nc+nf+37>1200 or 79+nb>1200:raise ValueError('scoring/record capacity')
    if (edges is None)!=(cum is None):raise ValueError('spectrum')
    e=np.empty(0) if edges is None else np.asarray(edges,float);c=np.empty(0) if cum is None else np.asarray(cum,float)
    if cum is not None and (c.ndim!=1 or not 1<=len(c)<=64 or e.shape!=(len(c)+1,) or c[-1]!=1 or c[0]<=0 or np.any(np.diff(c)<=0) or np.any(np.diff(e)<=0) or e[0]<0 or e[-1]>10 or not np.isfinite(e).all() or not np.isfinite(c).all()):raise ValueError('spectrum')
    cylinders=[];parents=[]
    for seed in seeds:
        # centre (x,y,z), radius, half-length, zero-based material ID
        a=np.asarray(seed,float)
        if a.shape!=(6,) or not np.isfinite(a).all() or min(a[3:5])<=0 or a[5]!=int(a[5]) or not 0<=a[5]<len(densities):raise ValueError('seed')
        extent=np.array([a[3],a[4],a[3]]);lo=a[:3]-extent;hi=a[:3]+extent
        candidates=np.flatnonzero((lo[axis]-faces[:-1]>=1e-4)&(faces[1:]-hi[axis]>=1e-4))
        if len(candidates)!=1:raise ValueError('seed crosses/touches layer face (gap < 1e-4 cm)')
        for k in range(3):
            if k!=axis and min(lo[k]+half[k],half[k]-hi[k])<1e-4:raise ValueError('seed touches box face')
        for b in cylinders:
            radial=np.hypot(a[0]-b[0],a[2]-b[2])-a[3]-b[3]
            axial=abs(a[1]-b[1])-a[4]-b[4]
            if np.hypot(max(0.,radial),max(0.,axial))<1e-4:raise ValueError('seed overlap/contact (gap < 1e-4 cm)')
        cylinders.append(a[:5]);parents.append(int(candidates[0]));ids.append(int(a[5]));rho.append(densities[int(a[5])])
    arrays=dict(faces=faces,rho=np.array(rho),bins=grid['edges_cm'],strength=BOUNDARY_STRENGTH_EDGES,edges=e,cum=c,ids=np.array(ids,np.int64),owner=grid['bin_layer'].astype(np.int64),seeds=np.asarray(cylinders,float).reshape(-1,5),parents=np.asarray(parents,np.int64))
    arrays={k:np.ascontiguousarray(v) for k,v in arrays.items()}
    g=Geometry()
    for k,t in SlabGeometry._fields_[:8]:setattr(g,k,arrays[k].ctypes.data_as(t))
    for k,v in dict(nl=nl,nb=nb,nc=nc,nf=nf,nspec=len(c),binding=int(binding),fluorescence=int(fluorescence),energy=energy,axis=axis,ns=len(cylinders),field=int(field),scoring_enabled=int(field if score_field is None else score_field),field_half=field_half,ssd=ssd).items():setattr(g,k,v)
    g.seeds=arrays['seeds'].ctypes.data_as(DP);g.parents=arrays['parents'].ctypes.data_as(IP);g.half[:]=half;g.scoring=p7_core.geometry()[0]
    return g,arrays,grid

def fingerprint():
    names=set(core.fingerprint())|{'p7_field.h','p6_field.h','p9b_geometry.h','p9b_kernel.h','p9b_cpu.cpp','p9b_gpu.cu','p9b_core.py','p9b_cases.py'}
    pending=['p9b_kernel.h'];seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name)
        for line in (HERE/name).read_text().splitlines():
            if line.startswith('#include "'):
                child=line.split('"')[1]
                names.add(child);pending.append(child)
    return {n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in sorted(names)}

def cuda_source():
    def expand(name,seen):
        lines=[]
        for line in (HERE/name).read_text().splitlines():
            if line.startswith('#include "'):
                child=line.split('"')[1]
                if child not in seen:seen.add(child);lines.append(expand(child,seen))
            else:lines.append(line)
        return '\n'.join(lines)
    probe=(HERE/'p9b_cpu.cpp').read_text().split('P5_FN void p9b_probe_one',1)[1].split('extern "C" void p9b_probe(',1)[0]
    score='P5_FN void p9b_score_one'+(HERE/'p9b_cpu.cpp').read_text().split('P5_FN void p9b_score_one',1)[1].split('extern "C" void p9b_score_probe',1)[0]
    injected='P5_FN void p9b_injected_one'+(HERE/'p9b_cpu.cpp').read_text().split('P5_FN void p9b_injected_one',1)[1].split('extern "C" void p9b_injected',1)[0]
    return (HERE/'p9b_gpu.cu').read_text().replace('//@@COMMON@@',expand('p9b_kernel.h',set())).replace('//@@PROBE@@','P5_FN void p9b_probe_one'+probe).replace('//@@SCORE@@',score).replace('//@@INJECTED@@',injected)
