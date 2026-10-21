"""P8f hand-off contracts and host emulation; no CUDA device is run."""
import ctypes as ct
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest
from prototypes.mv_electron.gpu import p8f as p,p8f_core as core


def test_p8f_seal_rejects_metadata_and_content_changes(tmp_path):
    file=tmp_path/'batch.npz';p.save(file,dict(a=np.arange(3)),dict(seed=core.SEED3,n=100000,start=0,batch=0))
    assert p.load_sealed(file,dict(seed=core.SEED3))[0]['a'].tolist()==[0,1,2]
    with pytest.raises(ValueError,match='receipt'):p.load_sealed(file,dict(start=100000))
    with file.open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='hash'):p.load_sealed(file)


def test_p8f_fixed_dref_and_paired_sem():
    from prototypes.mv_electron.run_p8e import layout,LAYERS
    grid=layout(LAYERS,[1.,.26,1.85,11.35]);a=np.ones((100,220))*.06
    b=a+np.linspace(-.0001,.0001,100)[:,None]
    result=core.paired_blocks(a,b,grid)
    expected=np.std(np.linspace(-.0001,.0001,100),ddof=1)/10/core.DREF
    np.testing.assert_allclose(result['SEM'],expected,rtol=1e-12)
    assert core.judge(a,a,grid,dref=core.DREF)['status']==['pass']*19
    with pytest.raises(ValueError,match='fixed'):core.judge(a,a,grid,dref=.06047850726023682)
    b=a+core.DREF*.006
    assert core.paired_blocks(a,b,grid)['status']=='fail'


def test_p8f_partial_receipt_does_not_satisfy_gate(tmp_path):
    with pytest.raises(FileNotFoundError):p.require_gate(tmp_path,'gate1',{})
    p.write_json(tmp_path/'gate1.json',dict(status='pass',identity={'sources':1}))
    with pytest.raises(ValueError,match='current'):p.require_gate(tmp_path,'gate1',{})


def test_p8f_notebook_and_identity_exclude_drivers():
    notebook=json.loads((core.HERE/'p8f_colab.ipynb').read_text())
    code=''.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
    assert 'p8f_manifest.json' in code and 'run(bundle' in code and 'drive.mount' in code
    assert not any('colab' in k or k=='p8f.py' for k in core.fingerprint())
    assert 'atomicAdd(next,1)' in core.cuda_source()


STUB='''#include <cmath>
#include <cstring>
#define __device__
#define __global__
struct Dim { int x; };Dim blockIdx,blockDim,threadIdx;
static double __longlong_as_double(unsigned long long x){double y;memcpy(&y,&x,8);return y;}
static long long __double_as_longlong(double x){long long y;memcpy(&y,&x,8);return y;}
template<class T,class U> T atomicAdd(T* p,U v){T x=*p;*p+=v;return x;}
template<class T,class U> T atomicMax(T* p,U v){T x=*p;if(*p<v)*p=v;return x;}
using std::isfinite;
#define P5_GPU
#define P5_RNG_PHILOX
#define P8F_STACK_CAP 16
'''
WRAP='''
extern "C" void emulate(const double *data,const long long *meta,const double *constants,
    const double *geometry,const long long *mapping,const int *sizes,int n,
    double *hd,long long *hi,double *groups,unsigned long long *ints){
    double absolute[100*1200]={};int next=5120;blockDim.x=128;
    for(int slot=0;slot<5120;++slot){blockIdx.x=slot/128;threadIdx.x=slot%128;
        p8f_kernel(data,meta,constants,geometry,mapping,sizes[0],sizes[1],sizes[2],sizes[3],sizes[4],2026102102ULL,0,n,&next,1,hd,hi,groups,ints,absolute);}
}
'''


def test_p8f_gpu_driver_host_emulation_tail_and_reset(tmp_path):
    # This checks the *actual* exported driver, 40 x 128 slots and 15,377
    # histories, with fast sub-CUT photons. It is not a T4 verification gate.
    m,i,_,_=p.inputs();edges=np.array([.007,.008]);cum=np.ones(1)
    cpu=p.CPU(tmp_path/'cpu',m,densities=i['densities'],edges=edges,cum=cum,rng='philox',cap=16)
    n=15377;ref=cpu.histories(n,core.SEED2)
    a=cpu.arrays;geom=np.concatenate([a[k] for k in ('faces','rho','bins','strength','edges','cum')]);mapping=np.r_[a['ids'],a['owner']]
    sizes=np.array([cpu.g.nl,cpu.g.nb,cpu.g.nc,cpu.g.nf,cpu.g.nspec],np.int32)
    dw=cpu.g.nb+cpu.g.nc+cpu.g.nf+37;iw=79+cpu.g.nb
    for fault in (0,7):
        source=tmp_path/f'emulate{fault}.cpp';source.write_text(STUB+f'\n#define P8F_FAULT {fault}\n'+core.cuda_source()+WRAP)
        library=source.with_suffix('.dylib' if __import__('sys').platform=='darwin' else '.so')
        subprocess.run([shutil.which('clang++'),'-O2','-std=c++11','-ffp-contract=off','-shared','-fPIC',str(source),'-o',str(library)],check=True,capture_output=True,text=True)
        lib=ct.CDLL(str(library));lib.emulate.argtypes=[p.D,p.I,p.D,p.D,p.I,np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS'),ct.c_int,p.D,p.I,p.D,np.ctypeslib.ndpointer(np.uint64,flags='C_CONTIGUOUS')];lib.emulate.restype=None
        hd=np.zeros((n,dw));hi=np.zeros((n,iw),np.int64);groups=np.zeros((100,dw));ints=np.zeros((100,iw),np.uint64)
        lib.emulate(cpu.data,cpu.meta,cpu.constants,geom,mapping,sizes,n,hd,hi,groups,ints)
        actual=core.unpack(hd,hi,cpu.g.nb,cpu.g.nc,cpu.g.nf)
        if fault:
            with pytest.raises(AssertionError,match='local bins'):core.check_local_bins(actual['histories'],actual['dep'])
        else:
            for key in ref:core.bitwise(actual[key],ref[key],'host GPU driver '+key)
            r=core.reduce_records(groups,ints,cpu.g.nb,cpu.g.nc,cpu.g.nf,n,cpu.grid['mass_g_cm2'])
            core.check_local_bins(actual['histories'],actual['dep']);core.check_aggregation(r['p8_batches'],actual['histories'],actual['additions'])
