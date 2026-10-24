"""Host replay of the exported CUDA driver (not a GPU gate)."""
import ctypes as ct
import shutil
import subprocess
import sys
import numpy as np
import pytest
from prototypes.mv_electron.gpu import p8g as p,p8f,p8g_core as core

STUB='''#include <cmath>
#include <cstring>
#define __device__
#define __global__
struct Dim {int x;};Dim blockIdx,blockDim,threadIdx;
static double __longlong_as_double(unsigned long long x){double y;memcpy(&y,&x,8);return y;}
static long long __double_as_longlong(double x){long long y;memcpy(&y,&x,8);return y;}
template<class T,class U> T atomicAdd(T* p,U v){T x=*p;*p+=v;return x;}
template<class T,class U> T atomicMax(T* p,U v){T x=*p;if(*p<v)*p=v;return x;}
using std::isfinite;
#define P5_GPU
#define P5_RNG_PHILOX
#define P8G_STACK_CAP 16
'''
WRAP='''
extern "C" void emulate(double *data,const long long *meta,const double *constants,
 const double *geometry,const long long *mapping,const int *sizes,int n,int threads,unsigned long long start,
 double *hd,long long *hi,double *groups,unsigned long long *ints){
 double absolute[100*1200]={};int next=40*threads;blockDim.x=threads;blockIdx.x=threadIdx.x=0;
 p8g_init(data,meta,constants,4);
 for(int slot=0;slot<40*threads;++slot){blockIdx.x=slot/threads;threadIdx.x=slot%threads;
 p8g_kernel(data,meta,constants,geometry,mapping,sizes[0],sizes[1],sizes[2],sizes[3],sizes[4],2026102102ULL,start,n,&next,1,hd,hi,groups,ints,absolute);}
}
'''

def test_p8g_cuda_driver_host_replay(tmp_path):
    materials,i,_,_=p8f.inputs();cpu=p.CPU(tmp_path/'cpu',materials,densities=i['densities'],edges=np.array([.007,.008]),cum=np.ones(1),rng='philox',cap=16)
    a=cpu.arrays;geom=np.concatenate([a[k] for k in ('faces','rho','bins','strength','edges','cum')]);mapping=np.r_[a['ids'],a['owner']]
    sizes=np.array([cpu.g.nl,cpu.g.nb,cpu.g.nc,cpu.g.nf,cpu.g.nspec],np.int32);dw=cpu.g.nb+cpu.g.nc+cpu.g.nf+37;iw=79+cpu.g.nb
    for check in (0,1):
        source=tmp_path/f'gpu{check}.cpp';source.write_text(STUB+f'\n#define P8G_CHECK {check}\n'+core.cuda_source()+WRAP)
        library=source.with_suffix('.dylib' if sys.platform=='darwin' else '.so')
        result=subprocess.run([shutil.which('clang++'),*core.CPU_FLAGS,str(source),'-o',str(library)],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        lib=ct.CDLL(str(library));lib.emulate.argtypes=[p.D,p.I,p.D,p.D,p.I,np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS'),ct.c_int,ct.c_int,ct.c_uint64,p.D,p.I,p.D,np.ctypeslib.ndpointer(np.uint64,flags='C_CONTIGUOUS')];lib.emulate.restype=None
        for threads in (128,256):
            for n,start in ((40*threads-1,0),(40*threads,0),(40*threads+1,0),(40*threads+1,7000),(40*threads+1,7000)):
                hd=np.zeros((n,dw));hi=np.zeros((n,iw),np.int64);groups=np.zeros((100,dw));ints=np.zeros((100,iw),np.uint64)
                lib.emulate(cpu.data,cpu.meta,cpu.constants,geom,mapping,sizes,n,threads,start,hd,hi,groups,ints)
                actual=core.unpack(hd,hi,cpu.g.nb,cpu.g.nc,cpu.g.nf)
                core.exact_histories(actual,cpu.histories(n,core.SEED2,start),'host driver')
                aggregate=core.reduce_records(groups,ints,cpu.g.nb,cpu.g.nc,cpu.g.nf,n,cpu.grid['mass_g_cm2'])
                aggregate.update(_groups_d=groups,_groups_i=ints)
                core.check_group_records(aggregate,actual,n)
                broken={**aggregate,'_groups_i':ints.copy()};broken['_groups_i'][0,3]+=1
                with pytest.raises(AssertionError):core.check_group_records(broken,actual,n)
                broken={**aggregate,'_groups_d':groups.copy()};broken['_groups_d'][0,cpu.g.nb+cpu.g.nc+cpu.g.nf+9]+=1.
                with pytest.raises(AssertionError):core.check_group_records(broken,actual,n)
