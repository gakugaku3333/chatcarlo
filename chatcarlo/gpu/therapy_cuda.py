"""Package-owned G5 therapy CUDA launcher; imports CuPy only in its caller."""
import numpy as np

class T4:
    """One compiled build of the unified kernel; same launch shape as production."""
    def __init__(self,cp,materials,arrays,fg=None,half=(0.,0.),ssd=100.,*,cap=18,fault=0,check=0,blocks=40,threads=256,source_check=False,_test_legacy_physics=False):
        from chatcarlo.gpu import therapy,physics_sets
        self.cp,self.blocks,self.threads=cp,blocks,threads
        self.source_check=source_check
        self.sizes=tuple(map(int,arrays['sizes']));self.half,self.ssd=half,ssd
        data,meta=therapy.pack(materials);self.declared=physics_sets.declaration(therapy.NAME, specialization=physics_sets.ELECTRON);s=self.declared
        self.args=(cp.asarray(data),cp.asarray(meta),cp.asarray(np.ascontiguousarray(arrays['constants'],float)))
        self.geometry=cp.asarray(np.concatenate([arrays[k] for k in ('faces','rho','bins','strength','edges','cum')]))
        self.mapping=cp.asarray(np.r_[arrays['ids'],arrays['owner']])
        self.fg=np.uint64(0) if fg is None else cp.asarray(np.frombuffer(bytes(fg),np.uint8))
        o=(f'-DG5_TEST_LEGACY_ELECTRON={int(_test_legacy_physics)}','-DP5_GPU','-DP5_RNG_PHILOX','--fmad=false',f'-DP8G_STACK_CAP={cap}',f'-DT2_FAULT={fault}',f'-DP8G2_CHECK={check}',
           f'-DT2_FIELD={int(fg is not None)}',f'-DT2_SOURCE_CHECK={int(source_check)}',f'-DG5_SET_ID={s["id"]}',f'-DG5_SET_TAG={s["tag"]}ULL')
        src=therapy.cuda_source();self.kernels={}
        for n in ('p8g2_kernel','p8g2_init','t2_identity','t2_score_kernel'):
            self.kernels[n]=cp.RawKernel(src,n,options=o,backend='nvrtc');self.kernels[n].compile()
        self.kernel=self.kernels['p8g2_kernel']
        self.kernels['p8g2_init']((1,),(1,),self.args+(np.int32(len(materials)),));cp.cuda.Device().synchronize()
        out=cp.zeros(2,np.uint64);self.kernels['t2_identity']((1,),(1,),(out,));cp.cuda.Device().synchronize();x=cp.asnumpy(out)
        self.observed=physics_sets.observation(therapy.NAME,int(x[0]),int(x[1]))
        physics_sets.check(self.observed,s)

    def run(self,n,seed,start=0,maxsteps=10000000,injected=None,energy=1.,history=0):
        cp=self.cp;nl,nb,nc,nf,_=self.sizes;dw=nb+nc+nf+37;iw=79+nb
        field=cp.zeros(3420);ledger=cp.zeros(15);adds=cp.zeros(3420,np.uint64);m=n if history else 1
        hd=cp.zeros((m,dw));hi=cp.zeros((m,iw),np.int64);groups=cp.zeros((100,dw));ints=cp.zeros((100,iw),np.uint64)
        absolute=cp.zeros((100,dw));nxt=cp.zeros(1,np.int32);visits=cp.zeros(n,np.int64);hits=cp.zeros(80,np.uint64)
        inj=np.uint64(0) if injected is None else cp.asarray(np.ascontiguousarray(injected,float))
        a=self.args+(self.geometry,self.mapping,*map(np.int32,self.sizes),np.uint64(seed),np.uint64(start),np.int32(n),nxt,
                     np.int32(history),np.int32(maxsteps),np.int32(0),inj,self.fg,np.float64(self.half[0]),np.float64(self.half[1]),
                     np.float64(self.ssd),np.float64(energy),field,ledger,adds,hd,hi,groups,ints,absolute,visits,hits)
        sources=cp.zeros((n,6)) if self.source_check else None
        if self.source_check:a+= (sources,)
        from .therapy_observation import gpu_observation
        self.last_observation=gpu_observation(self,a)
        t0,t1=cp.cuda.Event(),cp.cuda.Event();t0.record();self.kernel((self.blocks,),(self.threads,),a);t1.record();t1.synchronize()
        r={k:cp.asnumpy(v) for k,v in dict(d=hd,i=hi,visits=visits,hits=hits,groups=groups,group_ints=ints,field=field,ledger=ledger,field_additions=adds).items()}
        r['sources']=cp.asnumpy(sources) if self.source_check else np.empty((0,6));r['kernel_s']=float(cp.cuda.get_elapsed_time(t0,t1))/1000
        return r

    def score(self,rows):
        cp=self.cp;n=len(rows);out=cp.zeros((n,3420));sums=cp.zeros((n,15))
        self.kernels['t2_score_kernel'](((n+255)//256,),(256,),(self.fg,cp.asarray(np.ascontiguousarray(rows,float)),np.int32(n),out,sums))
        cp.cuda.Device().synchronize();return cp.asnumpy(out),cp.asnumpy(sums)

