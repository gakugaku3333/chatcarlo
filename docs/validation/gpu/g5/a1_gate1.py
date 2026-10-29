"""G1 independent 51-quantity gate; CPU twin or CUDA, with healthy batch replay."""
import argparse
import tempfile
from pathlib import Path
import numpy as np
import a1_common
from chatcarlo.gpu import bundle,validation
from chatcarlo.gpu.aggregate import aggregate
from chatcarlo.gpu.core import Core
from chatcarlo.gpu.source_host import sample_source_photons
from a1_common import HERE,ROOT,CACHE,SUMMARY,write,module,code_identity,checked_cached
from a1_health import checked_run

# Read-only helpers; never call the old gate1() which overwrites G1 records.
legacy=module('a1_g1_readonly',HERE.parent/'g1/run_gates.py')


def twin(scene,plane,n,batch,seed,source_seed,device,remote=None):
    with tempfile.TemporaryDirectory(prefix='g5-a1-g1-') as tmp:
        bundle.create(scene,plane,tmp);a,src,spec,settings,_=bundle.load(tmp)
        if remote is not None:
            from a1_remote import export_job,collect
            job=export_job(tmp,Path(tmp)/'job.json',n=n,batch=batch,seed=seed,source_seed=source_seed)
            rows=list(collect(remote,job,a))
            return aggregate(((n,r) for n,r,h in rows),settings['material_names'],plane),[h for n,r,h in rows]
        driver=Core(a,device,physics_declaration=settings['physics_set'])
        rng=np.random.default_rng(source_seed);reports=[]
        def batches():
            for start in range(0,n,batch):
                size=min(batch,n-start)
                inputs=sample_source_photons(src,size,rng,spec)
                r,h=checked_run(driver,inputs,seed=seed,start=start,fluorescence=settings['fluorescence'])
                reports.append(h);yield size,r
        result=aggregate(batches(),settings['material_names'],plane)
        return result,reports


def run(out=CACHE/'gate1',summary=SUMMARY/'gate1_cpu.json',device='cpu',n=1000000,batch=200000,runs=10,remote=None):
    from chatcarlo.transport import run_transport
    attempts=[];code=code_identity()
    for retry in (False,True):
        reference={};candidate={};offset=100 if retry else 0
        for sid in ('s1','s2','s3'):
            scene,plane,digest=legacy.inputs(sid)
            for k in range(runs):
                for engine,values,seed in [('numpy',reference,5000+offset+k),('gpu',candidate,6000+offset+k)]:
                    identity=dict(code=code,scene_sha256=digest,engine=engine,device=device if engine=='gpu' else None,
                                  seed=seed,source_seed=7000+offset+k if engine=='gpu' else None,n_histories=n,batch_size=batch)
                    def compute():
                        if engine=='numpy':
                            result=run_transport(scene,n_histories=n,batch_size=batch,seed=seed,detector=plane,
                                                 engine='numpy',n_workers=4,track_uncertainty=True)
                            health=None
                        else:
                            result,health=twin(scene,plane,n,batch,seed,identity['source_seed'],device,None if remote is None else Path(remote)/f'{retry}_{sid}_{k:02d}')
                        return dict(quantities=legacy.collect(result,sid,n),health=health)
                    row=checked_cached(Path(out)/f'{device}_{retry}_{sid}_{engine}_{k:02d}.json',identity,compute)
                    if engine=='gpu' and (not row['health'] or any(h['status']!='pass' for h in row['health'])):
                        raise ValueError('gate1 failed batch health')
                    for name,value in row['quantities'].items():values.setdefault(sid+':'+name,[]).append(value)
        report=validation.independent_judge(reference,candidate)
        report.update(retry=retry,n_histories=n,batch_size=batch,runs=runs,code=code)
        attempts.append(report)
        if report['status']!='retry' or retry:break
    final=dict(attempts[-1],attempts=attempts)
    if final['status']=='retry':final['status']='fail'
    final['production_conditions']=n==1000000 and batch==200000 and runs==10
    write(summary,final);return final


def export(out):
    from a1_remote import export_job
    out=Path(out)
    for sid in ('s1','s2','s3'):
        scene,plane,_=legacy.inputs(sid);bundle.create(scene,plane,out/'bundles'/sid)
        for retry in (False,True):
            offset=100 if retry else 0
            for k in range(10):
                export_job(out/'bundles'/sid,out/'jobs'/f'{retry}_{sid}_{k:02d}.json',n=1000000,
                           batch=200000,seed=6000+offset+k,source_seed=7000+offset+k)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--out',type=Path,default=CACHE/'gate1');p.add_argument('--summary',type=Path,default=None)
    p.add_argument('--export',action='store_true');p.add_argument('--remote',type=Path)
    a=p.parse_args()
    if a.export:export(a.out)
    else:
        r=run(a.out,a.summary or SUMMARY/f'gate1_{a.device}.json',a.device,remote=a.remote);raise SystemExit(r['status']!='pass')
