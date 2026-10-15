"""Local P5e CPU stack check and independent artifact rejudgement."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np
from . import p5b_core as core, p5d_core as d, p5e_core as e


class CPU16(d.CPU):
    def __init__(self, work, data):
        core.fingerprints(); core.check_inputs(data)
        work=Path(work); work.mkdir(parents=True,exist_ok=True)
        self.flags=d.FLAGS[:-1]+['-DP5_STACK_CAP=16']
        self.compiler=subprocess.check_output(['clang++','--version'],text=True).splitlines()[0]
        self.identity=core.sha(json.dumps(dict(compiler=self.compiler,flags=self.flags,sources=core.FROZEN),sort_keys=True).encode())
        library=(work/('p5e_'+self.identity[:16]+('.dylib' if sys.platform=='darwin' else '.so'))).resolve()
        build=subprocess.run(['clang++',*self.flags,str(core.HERE/'p5_cpu.cpp'),'-o',str(library)],capture_output=True,text=True)
        (work/'build.log').write_text(build.stdout+build.stderr); build.check_returncode()
        self.data,self.tables=data,self.make_tables(data)
        self.lib=ct.CDLL(str(library))
        f=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        i=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
        self.lib.p5a_run.argtypes=[ct.POINTER(d.Tables),ct.c_uint64,ct.c_uint64,ct.c_int,ct.c_int,ct.c_int,f,f,f,f,i,i,i,i,f,i,f,ct.c_int,i]
        self.lib.p5a_run.restype=None


def cpu_part(args):
    from . import p5a,p5b
    work,saved,part=args
    data,config=p5b.load_local_inputs(saved)
    cpu=CPU16(Path(work)/f'build_{part}',data)
    checks=[]
    for b in range(part,100,2):
        out=cpu.run(100000,core.SEED,start=100000*b)
        expected=p5b.saved_batch(saved,b,config)
        check=p5a.compare_outputs(expected,out)
        core.require(check,'CPU16 all ten outputs')
        core.require(d.cpu_health(out,100000),'CPU16 health')
        core.require(p5a.health(out),'CPU16 P5a health')
        core.save_npz(Path(work)/f'batch_{b:03}.npz',{k:out[k] for k in p5a.OUTPUTS},dict(batch=b,check=check,identity=cpu.identity))
        checks.append(b)
    return checks


def cpu_check(work,saved):
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        done=list(pool.map(cpu_part,[(work,saved,p) for p in range(2)]))
    core.write_json(Path(work)/'cpu16.json',dict(status='pass',batches=sorted(sum(done,[])),scope='tested slab and history ranges only'))


def rejudge(bundle,results,mode):
    _,ref,manifest=d.load_bundle(bundle)
    results=Path(results)
    if mode=='explore':
        report=json.loads((results/'explore.json').read_text())
        if not report['complete']: raise core.Stop('incomplete session: repeat all three timing rounds')
        times={}
        baseline=core.read_npz(results/'V0_e1.npz')
        for v,r in report['variants'].items():
            if r['status']!='pass': continue
            identity=r['identity']
            if identity!=e.identity(v,manifest,report['environment'],identity['launches']): raise core.Stop('fingerprint mismatch')
            core.require(e.e1(core.read_npz(results/f'{v}_e1.npz'),baseline),'E1')
            core.require(e.e2(core.read_npz(results/f'{v}_e2.npz'),ref),'E2')
            if e.VARIANTS[v]['allocation']=='D':
                T=identity['launches']['p5e_visits_kernel']['T']
                conditions=((2,0),(T-1,0),(T,0),(T+1,0),(3*T+17,0),(T+1,7000),(T+1,7000))
                for i,(n,start) in enumerate(conditions):
                    core.require(e.visits_check(**core.read_npz(results/f'{v}_e3_{i}.npz'),n=n,start=start),'E3 visits')
                core.require(e.e1(core.read_npz(results/f'{v}_e3_transport.npz'),core.read_npz(results/f'{v}_e3_static.npz')),'E3 transport')
            raw=[x for x in report['measurements'] if x['label']=='round' and x['variant']==v]
            if [x['repeat'] for x in raw]!=[0,1,2] or any(x['session']!=report['session'] for x in raw): raise core.Stop('timing session mismatch')
            times[v]=[x['kernel_ms'] for x in raw]
        selected=e.select(times)
        if times!=report['selection_input'] or selected!=report['selected']: raise core.Stop('selection mismatch')
        return dict(selected=selected,t4=report['t4'],status='pass')
    identity=json.loads((results/'identity.json').read_text())
    if identity!=e.identity(identity['variant'],manifest,identity['environment'],identity['launches']): raise core.Stop('fingerprint mismatch')
    def load(name):
        out=core.read_npz(results/(name+'.npz'))
        meta=json.loads(str(out.pop('metadata')))
        if meta!=dict(identity=identity,outputs=core.describe(out)): raise core.Stop('artifact identity mismatch')
        return out
    checks=dict(C0=core.verdict(True),C1=core.c1(ref,load('c1')))
    histories=load('c2')
    checks['C2']=core.c2({k:ref['history_'+k] for k in core.HISTORY_ORDER},histories,float(ref['Dref']),float(ref['conversion']))
    normal=core.c3(histories,load('c3'),20); fault=core.c3(histories,load('c3_fault'),20)
    checks['C3']=core.verdict(normal['status']=='pass' and fault['status']=='fail' and core.split_check(histories,load('c3_split'))['status']=='pass')
    found=core.resume(results/'batches',identity)
    checks['C4']=core.c4(ref,[core.read_checkpoint(found[b],identity,b) for b in range(100)],20)
    return dict(status='pass' if all(x['status']=='pass' for x in checks.values()) else 'not adopted',checks=checks)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('cpu16','explore','adopt'))
    p.add_argument('--work',type=Path)
    p.add_argument('--saved',type=Path,default=Path('~/.cache/chatcarlo/p5/p5a/philox').expanduser())
    p.add_argument('--bundle',type=Path)
    p.add_argument('--results',type=Path)
    a=p.parse_args()
    if a.command=='cpu16':
        if a.work is None: p.error('--work required')
        cpu_check(a.work,a.saved)
    else:
        if a.bundle is None or a.results is None: p.error('--bundle and --results required')
        report=rejudge(a.bundle,a.results,a.command)
        core.write_json(a.results/'local_judgement.json',report)
        print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
