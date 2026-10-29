"""Fixed-reference G2 voxel gate; the original judge is imported without edits."""
import argparse
from pathlib import Path
import tempfile
import numpy as np
import a1_common
from chatcarlo.gpu import bundle
from chatcarlo.gpu.g2_core import Core
from chatcarlo.gpu.source_host import sample_source_photons
from a1_common import HERE,ROOT,CACHE,SUMMARY,module,write,space,code_identity
from a1_health import checked_run

judge=module('a1_g2_readonly',HERE.parent/'g2/run_gates.py').judge
REFERENCE_HASHES={
 'ref_water_batches.npz':'a515113841e1c6e55dbae32423e435b92f53d9d33465dbd693adbcc8957d43d0',
 'ref_lead_batches.npz':'cb12f5c8fad79fc9fb36c48763ed09fce3ae6c082870c4f2ba4aac84df5673aa',
 'reference.json':'740a73d9a9cd54fae14c957b5de54040bede61fe934ce51ceb6b2f0cb2731d13'}
SCENES={'water':ROOT/'examples/water_phantom_pdd_ocr.yaml','lead':HERE.parent/'g2/scene_lead_slab.yaml'}


def verify_reference(path):
    for name,digest in REFERENCE_HASHES.items():
        if bundle.sha(Path(path)/name)!=digest:raise ValueError('fixed reference SHA256 mismatch: '+name)


def retry_allowed(results):
    failures=[]
    for scene,quantities in results.items():
        for q,report in quantities.items():
            failed={k for k,v in report['checks'].items() if not v}
            if failed:failures.append(failed)
            if failed-{'outliers','spread'}:return False
    return len(failures)==1


def batches(scene,grid,n,count,base,device,remote=None):
    with tempfile.TemporaryDirectory(prefix='g5-a1-g2-') as tmp:
        bundle.create(scene,None,tmp,g2_tally=True);a,src,spec,settings,_=bundle.load(tmp)
        driver=Core(a,device,grid=grid,physics_declaration=settings['physics_set']) if remote is None else None
        values={'kerma':[],'h10':[]};reports=[]
        for k in range(count):
            seed=base+k;ss=int(np.random.SeedSequence(seed).generate_state(1,np.uint64)[0])
            inputs=sample_source_photons(src,n,np.random.default_rng(ss),spec)
            if remote is None:
                r,h=checked_run(driver,inputs,seed=seed,fluorescence=settings['fluorescence'])
            else:
                from a1_remote import export_job,collect
                spec_grid=dict(origin_cm=[float(v) for v in grid.origin_cm],shape=[int(v) for v in grid.shape],voxel_size_cm=float(grid.voxel_size_cm),layout_version='g2-grid-v1')
                job=export_job(tmp,Path(tmp)/'job.json',n=n,batch=n,seed=seed,source_seed=ss,grid=spec_grid)
                rows=list(collect(Path(remote)/str(k),job,a));_,r,h=rows[0]
            for q in values:values[q].append(r[q].reshape(grid.shape))
            reports.append(h)
        return {q:np.asarray(v) for q,v in values.items()},reports


def regions_for(scene,grid,name):
    from chatcarlo.geometry import Geometry
    material=Geometry(scene.raw['geometry']).material_at(grid.voxel_centers()).reshape(grid.shape)
    regions={m:material==m for m in ('water','air')+(('bone',) if name=='lead' else ())}
    if name=='lead':
        zs=grid.origin_cm[2]+np.arange(grid.shape[2])*2
        layer=np.broadcast_to(((zs<0)&(zs+2>-.05))[None,None,:],grid.shape)
        regions={m:mask&~layer for m,mask in regions.items()};regions['lead_layer']=layer
    return regions


def run(ref=Path.home()/'.cache/chatcarlo/g5/a0/g2_ref',out=CACHE/'g2vox',summary=SUMMARY/'g2vox_cpu.json',device='cpu',remote=None):
    from chatcarlo.scene import load_scene
    from chatcarlo.tally import VoxelGrid
    verify_reference(ref);attempts=[];code=code_identity()
    for retry in (False,True):
        reports={}
        for name,base in [('water',221100),('lead',221200)]:
            space(out);scene=load_scene(str(SCENES[name]))
            with np.load(Path(ref)/f'ref_{name}_batches.npz') as z:
                if str(z['scene_sha256'])!=bundle.sha(SCENES[name]):raise ValueError('reference scene mismatch')
                if int(z['histories_per_batch'])!=200000 or float(z['resolution'])!=2:raise ValueError('reference configuration mismatch')
                reference={q:z[q] for q in ('kerma','h10')};grid=VoxelGrid(z['origin'],tuple(z['shape']),2)
            values,health=batches(scene,grid,200000,20,base+(10000 if retry else 0),device,None if remote is None else Path(remote)/f'{retry}_{name}')
            if any(not np.isfinite(v).all() for v in values.values()):raise ValueError('nonfinite voxel batches')
            Path(out).mkdir(parents=True,exist_ok=True)
            file=Path(out)/f'{device}_{name}_{retry}.npz';np.savez(file,**values)
            reports[name]={q:judge(reference[q],values[q],regions_for(scene,grid,name)) for q in values}
            write(file.with_suffix('.json'),dict(code=code,health=health,seed_base=base+(10000 if retry else 0),sha256=bundle.sha(file)))
        attempts.append(reports)
        if all(r['status']=='pass' for s in reports.values() for r in s.values()):break
        if retry or not retry_allowed(reports):break
    status='pass' if all(r['status']=='pass' for s in reports.values() for r in s.values()) else 'fail'
    final=dict(status=status,attempts=attempts,code=code);write(summary,final);return final


def export(ref,out):
    from chatcarlo.scene import load_scene
    from a1_remote import export_job
    verify_reference(ref);out=Path(out)
    for name,base in [('water',221100),('lead',221200)]:
        bundle.create(load_scene(str(SCENES[name])),None,out/'bundles'/name,g2_tally=True)
        with np.load(Path(ref)/f'ref_{name}_batches.npz') as z:
            grid=dict(origin_cm=z['origin'].tolist(),shape=z['shape'].tolist(),voxel_size_cm=2.,layout_version='g2-grid-v1')
        for retry in (False,True):
            for k in range(20):
                seed=base+(10000 if retry else 0)+k
                ss=int(np.random.SeedSequence(seed).generate_state(1,np.uint64)[0])
                export_job(out/'bundles'/name,out/'jobs'/f'{retry}_{name}_{k}.json',n=200000,batch=200000,
                           seed=seed,source_seed=ss,grid=grid)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--reference',type=Path,default=Path.home()/'.cache/chatcarlo/g5/a0/g2_ref')
    p.add_argument('--out',type=Path,default=CACHE/'g2vox');p.add_argument('--summary',type=Path,default=None)
    p.add_argument('--export',action='store_true');p.add_argument('--remote',type=Path)
    a=p.parse_args()
    if a.export:export(a.reference,a.out)
    else:
        r=run(a.reference,a.out,a.summary or SUMMARY/f'g2vox_{a.device}.json',a.device,a.remote);raise SystemExit(r['status']!='pass')
