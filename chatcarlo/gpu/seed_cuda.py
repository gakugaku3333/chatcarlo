"""Seed CUDA launcher; table pointers and geometry read back before launch."""
import ctypes as ct
import numpy as np
from . import seed_geometry as core,therapy
from .seed_core import DP,IP,bind_grids,field_seed,check_result,cuda_source,unpack,observation

class GPU:
    def __init__(self,cp,materials,*,constants,strength,fault=0,**kw):
        from . import therapy,physics_sets
        self.cp=cp;self.g,self.arrays,self.grids=core.geometry(**kw)
        self.arrays['strength']=np.ascontiguousarray(strength,float);self.g.strength=self.arrays['strength'].ctypes.data_as(DP)
        self.values=[cp.zeros(g.shape) for g in self.grids];self.adds=None
        self.device_arrays={k:cp.asarray(v) for k,v in self.arrays.items()}
        data,meta=therapy.pack(materials)
        self.args=(cp.asarray(data),cp.asarray(meta),cp.asarray(constants))
        self.module=cp.RawModule(code=cuda_source(),options=('-DP5_GPU','-DP5_RNG_PHILOX','--fmad=false','-DT2_SEED=1','-DP8G_STACK_CAP=18','-DT2S_CHECK=0',f'-DT2_SEED_FAULT={fault}',f'-DG5_SET_ID={physics_sets.declaration(therapy.NAME)["id"]}',f'-DG5_SET_TAG={physics_sets.declaration(therapy.NAME)["tag"]}ULL'),backend='nvrtc')
        self.module.get_function('t2_seed_init')((1,),(1,),self.args+(np.int32(len(materials)),));self.upload()
        out=cp.zeros(2,np.uint64);self.module.get_function('t2_seed_identity')((1,),(1,),(out,));v=cp.asnumpy(out)
        self.observed=dict(name=therapy.NAME,id=int(v[0]),tag=int(v[1]));physics_sets.check(self.observed,physics_sets.declaration(therapy.NAME))
    def upload(self):
        cp=self.cp;g=core.Geometry.from_buffer_copy(bytes(self.g))
        for k,t in therapy.Geometry._fields_[:8]:setattr(g,k,ct.cast(int(self.device_arrays[k].data.ptr),t))
        g.seeds=ct.cast(int(self.device_arrays['seeds'].data.ptr),DP);g.parents=ct.cast(int(self.device_arrays['parents'].data.ptr),IP)
        bind_grids(g,self.grids,self.values,self.adds,device=True)
        self.geometry=cp.asarray(np.frombuffer(bytes(g),np.uint8).copy())
    def clear(self):
        for x in self.values:x.fill(0)
        if self.adds is not None:
            for x in self.adds:x.fill(0)
    def boundary(self,states):
        cp=self.cp;a=cp.asarray(np.ascontiguousarray(states,float).reshape(-1,23));out=cp.zeros((len(a),6))
        self.module.get_function('t2_seed_boundary_probe')(((len(a)+31)//32,),(32,),(self.geometry,a,np.int32(len(a)),out));return cp.asnumpy(out)
    def score(self,a,b,loss=1.,site=3,repeats=1):
        cp=self.cp;self.module.get_function('t2_seed_score_probe')((40,),(256,),(self.geometry,cp.asarray(a,dtype=float),cp.asarray(b,dtype=float),np.float64(loss),np.int32(site),np.int32(repeats)))
        return self.values
    def run(self,n,seed,start=0,field=0,maxsteps=10000000,injected=None,regions=False):
        cp=self.cp
        if type(n) is not int or not 0<n<=200000 or not 0<=start<start+n<2**64:raise ValueError('history range')
        self.clear();d=cp.zeros((n,37));i=cp.zeros((n,79),np.int64);ledger=cp.zeros((n,6));sites=cp.zeros((n,6),np.int64);counts=cp.zeros((n,4),np.int64)
        r=cp.zeros((n,3435)) if regions else None;x=None if injected is None else cp.asarray(injected,dtype=float)
        begin,end=cp.cuda.Event(),cp.cuda.Event()
        args=self.args+(self.geometry,np.uint64(field_seed(seed,field)),np.uint64(start),np.int32(n),np.int32(maxsteps),np.uint64(0) if x is None else x,d,i,ledger,sites,counts,np.uint64(0) if r is None else r)
        self._invoke(args,timing=(begin,end))
        end.synchronize();hd,hi=cp.asnumpy(d),cp.asnumpy(i);ld=cp.asnumpy(ledger);sc=cp.asnumpy(counts)
        from .seed_checks import raw
        h=raw(hd,hi,ld,sc,start);check_result(h,ld,sc)
        if not self.g.beam and np.any(sc[:,1]):raise AssertionError('rectangle body miss: investigate source geometry')
        if abs(float(self.values[1].sum())-h['dep'].sum())>1e-9*h['sources'].sum():raise AssertionError('coarse grid sum / ledger')
        return dict(d=hd,i=hi,observed=self.last_observation,histories=h,ledger=ld,sites=cp.asnumpy(sites),source_counts=sc,regions=None if r is None else cp.asnumpy(r[:,:3420]),grids=[cp.asnumpy(v) for v in self.values],seconds=cp.cuda.get_elapsed_time(begin,end)/1000)

    def _invoke(self,args,timing=None):
        self.last_observation=self.observe(int(args[4])//8,int(args[5]),int(args[6]),int(args[4])%8,args=args)
        if timing:timing[0].record()
        self.module.get_function('t2_seed_kernel')((40,),(256,),args)
        if timing:timing[1].record()

    def source(self,s,d):
        cp=self.cp;out=cp.zeros(5)
        self.module.get_function('t2_seed_source_probe')((1,),(1,),(self.geometry,cp.asarray(s,dtype=float),cp.asarray(d,dtype=float),out));return cp.asnumpy(out)

    def probe(self,states,directions=None,mode=0,seed=9502101,start=0):
        cp=self.cp;a=np.ascontiguousarray(states,float).reshape(-1,23);directions=np.ascontiguousarray(a[:,3:6] if directions is None else directions,float)
        out=cp.zeros((len(a),74));self.module.get_function('t2_seed_probe')(((len(a)+31)//32,),(32,),self.args+(self.geometry,cp.asarray(a),cp.asarray(directions),np.int32(mode),np.uint64(seed),np.uint64(start),np.int32(len(a)),out));return cp.asnumpy(out)

    def observe(self,seed,start,n,field,args=None):
        cp=self.cp
        args=self.args+(self.geometry,) if args is None else args
        g=core.Geometry.from_buffer_copy(cp.asnumpy(args[3]).tobytes())
        def read(ptr,size,dtype):
            if not size:return np.empty(0,dtype)
            owner=cp.cuda.UnownedMemory(ct.cast(ptr,ct.c_void_p).value,size*np.dtype(dtype).itemsize,self)
            return cp.asnumpy(cp.ndarray((size,),dtype=dtype,memptr=cp.cuda.MemoryPointer(owner,0)))
        arrays={}
        for name,ptr_type in therapy.Geometry._fields_[:8]:
            shape=self.arrays[name].shape;arrays[name]=read(getattr(g,name),self.arrays[name].size,self.arrays[name].dtype).reshape(shape)
        arrays['seeds']=read(g.seeds,g.ns*5,float).reshape(-1,5);arrays['parents']=read(g.parents,g.ns,np.int64)
        out=cp.zeros(2,np.uint64);self.module.get_function('t2_seed_identity')((1,),(1,),(out,));v=cp.asnumpy(out)
        declared=dict(name=therapy.NAME,id=int(v[0]),tag=int(v[1]))
        return observation(g,arrays,tuple(cp.asnumpy(a) for a in args[:3]),seed,start,n,field,declared)
