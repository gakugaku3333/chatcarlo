"""P8g-3 local CPU gate; Claude-only export and final verification."""
import argparse
from pathlib import Path
import shutil
import tempfile
import zipfile
import numpy as np
from . import p8g3_core as core

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[2]/'docs/validation/mv/p8g3'
CACHE=Path.home()/'.cache/chatcarlo/p8g3'


def local():
    import pytest
    code=pytest.main([str(HERE.parent/'tests'),'-q','-k','p8g3'])
    if code:raise SystemExit(code)
    from . import p8g2
    m,i,e,c=p8g2.inputs();identity=p8g2.run_identity(i,e,c)
    schedule=core.spec(8,2000,'cpu')
    with tempfile.TemporaryDirectory(prefix='p8g3-local-') as temp:
        work=Path(temp)
        cpu=p8g2.CPU(work/'build',m,densities=i['densities'],edges=e,cum=c)
        def runner(n,seed,start):
            import time
            begin=time.perf_counter()
            _,visits,_,result=cpu.histories(n,seed,start,details=True)
            result['seconds']=np.array(time.perf_counter()-begin)
            return result,visits
        roots=[work/'continuous',work/'resumed'];sessions=[]
        for root in roots:
            sid=core.new_session(root,identity,schedule,'CPU',dict(status='pass'),dict(status='new'))
            sessions.append(sid)
        core.drive(roots[0],runner,identity,schedule,sessions[0])
        core.drive(roots[1],runner,identity,schedule,sessions[1],limit=3)
        resumed=core.resume_check(roots[1],runner,identity,schedule)
        sid=core.new_session(roots[1],identity,schedule,'CPU',dict(status='pass'),resumed)
        core.drive(roots[1],runner,identity,schedule,sid)
        for root in roots:core.verify(root,identity,schedule)
        for j in range(8):
            a,_=core.load_call(roots[0],j,identity,schedule)
            b,_=core.load_call(roots[1],j,identity,schedule)
            for key in a:
                if key!='seconds':core.kernel.bitwise(a[key],b[key],'CPU resume '+key)
        core.write_json(OUT/'gate0.json',dict(status='pass',identity=identity,
            schedule=schedule,continuous_calls=8,interrupted_after=3,
            resumed_calls=5,numeric_arrays='byte-identical',resume=resumed,
            faults='see pytest.log and fault_observations.json',
            deferred=['T4 initial gate','T4 VM stop/collect/resend/resume','production','verify']))


def export(work, bundle):
    """Copy the verified P8g-2 export unchanged; add driver code separately."""
    from .p8g2_colab import load
    manifest,_,_=load(bundle)
    gate=core.read_json(OUT/'gate0.json')
    if gate['status']!='pass' or gate['identity']!=manifest['identity']:
        raise ValueError('matching local gate required')
    dest=Path(work)/'export';dest.mkdir(parents=True,exist_ok=True)
    for name in (*manifest['files'],'p8g2_manifest.json'):
        target=dest/name
        if target.exists() and core.kernel.sha(target)!=core.kernel.sha(Path(bundle)/name):
            raise ValueError('export conflict '+name)
        shutil.copyfile(Path(bundle)/name,target)
    with zipfile.ZipFile(dest/'p8g3_driver.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(HERE.glob('p8g3*.py')):
            archive.write(path,'prototypes/mv_electron/gpu/'+path.name)
    names=(*manifest['files'],'p8g2_manifest.json','p8g3_driver.zip')
    core.write_json(dest/'p8g3_manifest.json',dict(status='pass',identity=manifest['identity'],
        files={n:core.kernel.sha(dest/n) for n in names}))
    print(dest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('local','export','verify'))
    parser.add_argument('--work',type=Path,default=CACHE)
    parser.add_argument('--bundle',type=Path,default=Path.home()/'.cache/chatcarlo/p8g2/export')
    parser.add_argument('--results',type=Path)
    args=parser.parse_args()
    if args.command=='local':local()
    elif args.command=='export':export(args.work,args.bundle)
    else:
        from .p8g2_colab import load
        manifest,_,_=load(args.bundle)
        result=core.verify(args.results or args.work/'colab/results',manifest['identity'])
        print('P8G3 verified',len(result['calls']),'calls',result['n'],'histories')

if __name__=='__main__':main()
