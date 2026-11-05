"""Observe actual native-launch values, independently of the scene condition."""
import ctypes as ct
import numpy as np
from .catalog import digest, P8_DEFINITIONS
from . import physics_sets
from .therapy import NAME, Geometry, FieldGeometry


def descriptor(arrays,fg,half,ssd,seed,start,n,observed):
    result=dict(physics_set=observed,inputs={k:digest(arrays[k]) for k in ('faces','rho','bins','strength','edges','cum','ids','owner','constants','sizes')},
                geometry_sha256=digest(np.frombuffer(bytes(fg),np.uint8)),half=list(map(float,half)),ssd=float(ssd),seed=int(seed),start=int(start),count=int(n))
    if '_table_hashes' in arrays:result['tables']=arrays['_table_hashes']
    return result


def cpu_observation(driver,args):
    # These are the tuple positions passed to p8g2_histories, not driver.half/ssd.
    g=ct.cast(args[3],ct.POINTER(Geometry)).contents
    arrays={}
    sizes=dict(faces=g.nl+1,rho=g.nl,bins=g.nb+1,strength=17,edges=g.nspec+1 if g.nspec else 0,cum=g.nspec,ids=g.nl,owner=g.nb)
    for k,n in sizes.items():
        arrays[k]=np.ctypeslib.as_array(getattr(g,k),shape=(n,))
    arrays['constants']=args[2];arrays['sizes']=np.array([g.nl,g.nb,g.nc,g.nf,g.nspec],np.int64)
    arrays['_table_hashes']=table_hashes(args[0],args[1])
    fg=ct.cast(args[11],ct.POINTER(FieldGeometry)).contents
    observed=physics_sets.observation(NAME,driver.lib.t2_set_id(),driver.lib.t2_set_tag())
    return descriptor(arrays,fg,args[12:14],args[14],args[4],args[5],args[6],observed)


def gpu_observation(driver,args):
    cp=driver.cp
    # Read the actual device buffers and scalar positions before launching.
    from .therapy import FieldGeometry
    nl,nb,nc,nf,nspec=map(int,args[5:10]);flat=cp.asnumpy(args[3]);mapping=cp.asnumpy(args[4]);p=0;arrays={}
    for k,n in zip(('faces','rho','bins','strength','edges','cum'),(nl+1,nl,nb+1,17,nspec+1 if nspec else 0,nspec)):
        arrays[k]=flat[p:p+n];p+=n
    arrays.update(ids=mapping[:nl],owner=mapping[nl:],constants=cp.asnumpy(args[2]),sizes=np.array([nl,nb,nc,nf,nspec],np.int64))
    if isinstance(args[18],np.uint64):
        # Depth-only A2 use has no field geometry. Still expose identity/range.
        return dict(physics_set=driver.observed,seed=int(args[10]),start=int(args[11]),count=int(args[12]))
    arrays['_table_hashes']=table_hashes(cp.asnumpy(args[0]),cp.asnumpy(args[1]))
    fg=FieldGeometry.from_buffer_copy(cp.asnumpy(args[18]).tobytes())
    out=cp.zeros(2,np.uint64);driver.kernels['t2_identity']((1,),(1,),(out,));cp.cuda.Device().synchronize();ids=cp.asnumpy(out)
    return descriptor(arrays,fg,args[19:21],args[21],args[10],args[11],args[12],physics_sets.observation(NAME,int(ids[0]),int(ids[1])))


def table_hashes(data,meta):
    result={}
    for mid,row in enumerate(meta):
        for j,name in enumerate(P8_DEFINITIONS):
            if name=='photon':
                offset=int(row[133])+37;p=data[offset:];tail=int(p[1]);size=tail+1+3*int(p[0])+4*int(p[tail])
            else:offset,size=map(int,row[4*j:4*j+2])
            if offset<0 or size<0 or offset+size>len(data):raise ValueError('native table descriptor range')
            result[str(mid)+'.'+name]=digest(data[offset:offset+size])
    return result
