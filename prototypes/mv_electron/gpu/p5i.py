"""Local P5i gates. gate1 is explicit; local runs never imply the 100-batch gate."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from . import p5b_core as core, p5d_core as d, p5g, p5i_core as h
DP = ct.POINTER(ct.c_double)
UP = ct.POINTER(ct.c_ulonglong)


SITES=('brems_e','brems_v','rates_ratio','rates_values','ts_ratio','pes','water2','pair0','pair1')


def layout(data):
    nr=len(data['rates']) if data['rates'].shape[1]==3 else len(data['ts'])
    lengths=np.array([len(data['brems_e']),3*len(data['brems_e']),nr-1,2*nr,
                      len(data['ts'])-1,*([len(data['pes'])]*4)],np.int32)
    return dict(lengths=lengths,offsets=np.r_[0,np.cumsum(lengths)[:-1]].astype(np.int32))


class CPU(d.CPU):
    def __init__(self,work,data,check=False):
        import time
        core.fingerprints();core.check_inputs(data)
        self.data=data;self.tables=self.make_tables(data);self.maps=layout(data)
        self.logs=np.empty(int(self.maps['lengths'].sum()))
        self.counts=np.zeros((9,2),np.uint64)
        self.flags=d.FLAGS+[f'-DP5I_CHECK={int(check)}']
        exe=shutil.which('clang++')
        if exe is None:raise core.Stop('clang++ unavailable')
        self.compiler=subprocess.check_output([exe,'--version'],text=True).splitlines()[0]
        self.identity=core.sha(json.dumps(dict(identity=h.identity(data),flags=self.flags,
            compiler=self.compiler),sort_keys=True).encode())
        work=Path(work);work.mkdir(parents=True,exist_ok=True)
        path=work/('p5i_'+self.identity[:16]+('.dylib' if sys.platform=='darwin' else '.so'))
        build=subprocess.run([exe,*self.flags,str(core.HERE/'p5i_cpu.cpp'),'-o',str(path)],capture_output=True,text=True)
        (work/'build.log').write_text(build.stdout+build.stderr);build.check_returncode()
        self.lib=ct.CDLL(str(path))
        dp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        ip=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
        self.lib.p5a_run.argtypes=[ct.POINTER(d.Tables),ct.c_uint64,ct.c_uint64,
            ct.c_int,ct.c_int,ct.c_int,dp,dp,dp,dp,ip,ip,ip,ip,dp,ip,dp,ct.c_int,ip]
        self.lib.p5a_run.restype=None
        self.lib.p5i_configure.argtypes=[ct.POINTER(d.Tables),dp,UP,ct.c_int]
        self.lib.p5i_gate0.argtypes=[ct.c_int,dp,ct.c_int,dp]
        self.lib.p5i_diagnostic.argtypes=[dp]
        self.lib.p5i_initialization_diagnostic.argtypes=[dp]
        for name in ('p5i_configure','p5i_gate0','p5i_diagnostic','p5i_initialization_diagnostic'):getattr(self.lib,name).restype=None
        begin=time.perf_counter();self.configure(True);self.initialization_s=time.perf_counter()-begin

    def configure(self,initialize=False):
        self.lib.p5i_configure(ct.byref(self.tables),self.logs,self.counts.ctypes.data_as(UP),int(initialize))

    def run(self,*args,**kwargs):
        self.counts.fill(0);self.configure()
        return super().run(*args,**kwargs)

    def compare(self,which,x):
        self.configure();x=np.ascontiguousarray(x);out=np.zeros((len(x),40))
        self.lib.p5i_gate0(which,x,len(x),out)
        return out


def vectors(grid,extras=(),randoms=1000):
    grid=np.asarray(grid);extras=np.asarray(extras);rng=np.random.default_rng(20261016)
    values=np.concatenate((grid,np.nextafter(grid,-np.inf),np.nextafter(grid,np.inf),
        [grid[0]*.5,grid[-1]*1.1],extras,np.nextafter(extras,-np.inf),np.nextafter(extras,np.inf),
        (grid[:-1,None]+(grid[1:]-grid[:-1])[:,None]*rng.random((len(grid)-1,randoms))).ravel()))
    return np.ascontiguousarray(np.column_stack((values,np.resize([.025,.05,.1,1e-12,1.],len(values)))))


def synthetic(data):
    """Formal gate0 fixtures: non-three-column rates and rounded-log grid ties."""
    result={k:v.copy() for k,v in data.items()}
    result['rates']=np.ascontiguousarray(np.column_stack((np.linspace(1,2,len(data['ts'])),
                                                      np.linspace(.1,.2,len(data['ts'])))))
    result['single']=np.ones((len(data['ts']),len(data['us'])+1))
    result['pes']=np.array([.01,.1,10.,np.nextafter(10.,np.inf),20.])
    result['water']=np.ones((5,7));result['thresholds']=np.array([1.,2.])
    return result


def gate0(cpu,work=None,randoms=1000,variants=True,functions=range(6)):
    cpu.counts.fill(0);checks={}
    grids=(cpu.data['brems_e'],cpu.data['rates'][:,0] if cpu.data['rates'].shape[1]==3 else cpu.data['ts'],
           cpu.data['ts'],cpu.data['pes'],cpu.data['pes'],cpu.data['pes'])
    for which in functions:
        grid=grids[which]
        x=vectors(grid,cpu.data['thresholds'],randoms)
        out=cpu.compare(which,x)
        check=core.verdict(core.bits(out[:,:20].copy(),out[:,20:].copy()),inputs=len(x),outputs=20)
        checks[str(which)]=check;core.require(check,'CPU gate0 function '+str(which))
    count=h.counters(cpu.counts,require_coverage=len(functions)==6);core.require(count,'gate0 unit coverage')
    diag=np.empty((len(cpu.logs),5));cpu.lib.p5i_diagnostic(diag)
    init_diag=np.empty_like(diag);cpu.lib.p5i_initialization_diagnostic(init_diag)
    diagnostic=dict(intermediates_equal=core.bits(diag,init_diag),equal=core.bits(diag[:,0].copy(),cpu.logs),note='rewritten expressions; auxiliary only',nonpositive_arguments=int(np.sum(diag[:,4]<=0)),
                    recorded_columns=['log','quotient','q','q_cube','argument'])
    report=dict(status='pass',checks=checks,counters=count,auxiliary=diagnostic,
                identity=h.identity(cpu.data),randoms_per_interval=randoms,
                arrays=core.describe({'logs':cpu.logs}),initialization_s=cpu.initialization_s,
                extra_bytes=cpu.logs.nbytes)
    if variants:
        import tempfile
        with tempfile.TemporaryDirectory(prefix='p5i_gate0_') as folder:
            variant=CPU(Path(folder)/'build',synthetic(cpu.data),check=True)
            report['synthetic']=gate0(variant,None,randoms,False,(1,3,4,5))
        report['synthetic']['purpose']='ratecols=2 and rounded-log adjacent grid ties'
    if work is not None:
        core.save_npz(Path(work)/'p5i_tables.npz',dict(logs=cpu.logs,auxiliary=diag,initialization_records=init_diag,**cpu.maps))
        core.write_json(Path(work)/'gate0_cpu.json',report)
    return report


def local(work, data):
    work=Path(work);new=CPU(work/'build',data);checked=CPU(work/'checked_build',data,check=True)
    gate0(checked,work)
    old=p5g.CPU(work/'p5g_build',data)
    expected=old.run(2000,core.SEED,start=7000,diagnostic=True)
    checks={}
    for label,cpu in (('ordinary',new),('checked',checked)):
        out=cpu.run(2000,core.SEED,start=7000,diagnostic=True)
        np.savez_compressed(work/(label+'.npz'),**out)
        checks[label]=h.exact(out,expected,(*d.OUTPUTS,'diagnostic','diagnostic_meta'))
        core.require(checks[label],label)
    np.savez_compressed(work/'p5g.npz',**expected)
    checks['counters']=h.counters(checked.counts);core.require(checks['counters'],'log coverage')
    report=dict(status='pass',n=2000,start=7000,checks=checks,identity=h.identity(data),
                compiler=new.compiler,flags=new.flags,checked_flags=checked.flags,
                note='local substitute only; gate1 100 batches not executed')
    core.write_json(work/'local_2000.json',report)
    return report


def gate1(work,saved):
    from . import p5b
    work=Path(work);data,config=p5b.load_local_inputs(Path(saved))
    new=CPU(work/'build',data);checked=CPU(work/'checked_build',data,check=True)
    gate0(checked,work)
    if new.compiler!=config['compiler']:raise core.Stop('saved compiler mismatch')
    report=dict(status='incomplete',batches=100,identity=h.identity(data),checks={},
                compiler=new.compiler,flags=new.flags,checked_flags=checked.flags)
    try:
        for b in range(100):
            ref=p5b.saved_batch(Path(saved),b,config);v={}
            for label,cpu in (('ordinary',new),('checked',checked)):
                out=cpu.run(100000,core.SEED,start=100000*b)
                core.save_npz(work/f'{label}_{b:03}.npz',out)
                v[label]=h.exact(out,ref,d.OUTPUTS);core.require(v[label],label)
            v['counters']=h.counters(checked.counts);core.require(v['counters'],'coverage')
            report['checks'][str(b)]=v;core.write_json(work/'gate1.json',report)
        report['status']='pass'
    except Exception as exc:
        report.update(status='fail',stop=str(exc));raise
    finally:core.write_json(work/'gate1.json',report)
    return report


def export_sources(work):
    """Self-contained NumPy/CuPy sources for the run-all Colab notebook."""
    import zipfile
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(work/'p5i_sources.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(core.HERE.iterdir()):
            if path.suffix in ('.py','.h','.cpp','.cu'):
                z.write(path,Path('prototypes/mv_electron/gpu')/path.name)
    return work/'p5i_sources.zip'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('local','gate0','gate1','export'))
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--saved',type=Path)
    a=p.parse_args()
    if a.command=='export':
        export_sources(a.work)
    elif a.command=='gate1':
        if a.saved is None:p.error('--saved required for gate1')
        gate1(a.work,a.saved)
    else:
        from .p5a import load_inputs
        data=load_inputs(full=True)
        if a.command=='local':local(a.work,data)
        else:gate0(CPU(a.work/'build',data,check=True),a.work)


if __name__=='__main__':main()
