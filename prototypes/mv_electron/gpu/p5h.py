"""Local P5h gates. gate1 is explicit; local runs never imply the 100-batch gate."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from . import p5b_core as core, p5d_core as d, p5g, p5h_core as h, p5h_tables as maps
DP = ct.POINTER(ct.c_double)
IP = ct.POINTER(ct.c_int)
UP = ct.POINTER(ct.c_ulonglong)


class CPU(d.CPU):
    def __init__(self, work, data, check=False):
        core.fingerprints();core.check_inputs(data)
        self.data=data;self.tables=self.make_tables(data)
        self.maps=maps.generate(data);self.counts=np.zeros((len(maps.SITES),3),np.uint64)
        self.flags=d.FLAGS+[f'-DP5H_CHECK={int(check)}']
        exe=shutil.which('clang++')
        if exe is None: raise core.Stop('clang++ unavailable')
        self.compiler=subprocess.check_output([exe,'--version'],text=True).splitlines()[0]
        self.identity=core.sha(json.dumps(dict(identity=h.identity(data),flags=self.flags,
            compiler=self.compiler),sort_keys=True).encode())
        work=Path(work);work.mkdir(parents=True,exist_ok=True)
        path=work/('p5h_'+self.identity[:16]+('.dylib' if sys.platform=='darwin' else '.so'))
        build=subprocess.run([exe,*self.flags,str(core.HERE/'p5h_cpu.cpp'),'-o',str(path)],capture_output=True,text=True)
        (work/'build.log').write_text(build.stdout+build.stderr);build.check_returncode()
        self.lib=ct.CDLL(str(path))
        dp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        ip=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
        self.lib.p5a_run.argtypes=[ct.POINTER(d.Tables),ct.c_uint64,ct.c_uint64,
            ct.c_int,ct.c_int,ct.c_int,dp,dp,dp,dp,ip,ip,ip,ip,dp,ip,dp,ct.c_int,ip]
        self.lib.p5a_run.restype=None
        self.lib.p5h_configure.argtypes=[ct.POINTER(d.Tables),IP,DP,IP,UP]
        self.lib.p5h_configure.restype=None
        self.lib.p5h_gate0.argtypes=[dp,ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.c_int,dp,ct.c_int,ct.c_int,ip]
        self.lib.p5h_gate0.restype=None
        self.configure()

    def configure(self):
        self.lib.p5h_configure(ct.byref(self.tables),self.maps['ranges'].ctypes.data_as(IP),
            self.maps['meta'].ctypes.data_as(DP),self.maps['pair_start'].ctypes.data_as(IP),
            self.counts.ctypes.data_as(UP))

    def run(self,*args,**kwargs):
        self.counts.fill(0);self.configure()
        return super().run(*args,**kwargs)

    def lookup(self, raw, x, map_id, site, start=0, narrow=False):
        self.configure();raw=np.ascontiguousarray(raw);x=np.ascontiguousarray(x)
        out=np.zeros((len(x)+3*len(raw),2),np.int64)
        self.lib.p5h_gate0(raw,len(raw),1,int(map_id==4),start,map_id,site,x,len(x),int(narrow),out)
        return out


def gate0(cpu, work=None, randoms=1000000):
    cpu.counts.fill(0);checks={}
    for site,(name,map_id) in enumerate(zip(
        ('loget','loget','brems_e','pes','pes','pes','rates','ts','us'),(3,3,1,4,4,4,2,5,0))):
        raw=maps.grids(cpu.data)[map_id]
        x=maps.vectors(raw,cpu.maps['edges_'+name],map_id==4,randoms)
        start=int(cpu.maps['pair_start'][site-4]) if site in (4,5) else 0
        for narrow in (False,True):
            out=cpu.lookup(raw,x,map_id,site,start,narrow)
            check=core.verdict(bool(np.array_equal(out[:,0],out[:,1])),inputs=len(out))
            checks[f'{maps.SITES[site]}/{"narrow" if narrow else "ordinary"}']=check
            core.require(check,'gate0 '+maps.SITES[site])
    count=h.counters(cpu.counts);core.require(count,'gate0 coverage')
    if not np.all(cpu.counts[:,1]>0):raise core.Stop('fallback path not covered')
    report=dict(status='pass',checks=checks,counters=count,identity=h.identity(cpu.data),randoms=randoms)
    if work is not None:
        core.save_npz(Path(work)/'p5h_tables.npz',cpu.maps)
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
        core.save_npz(work/(label+'.npz'),out)
        checks[label]=h.exact(out,expected,(*d.OUTPUTS,'diagnostic','diagnostic_meta'))
        core.require(checks[label],label)
    core.save_npz(work/'p5g.npz',expected)
    checks['counters']=h.counters(checked.counts);core.require(checks['counters'],'lookup coverage')
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
    with zipfile.ZipFile(work/'p5h_sources.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(core.HERE.iterdir()):
            if path.suffix in ('.py','.h','.cpp','.cu'):
                z.write(path,Path('prototypes/mv_electron/gpu')/path.name)
    return work/'p5h_sources.zip'


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
