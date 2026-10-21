"""Request-D compatible detector image runner (CLI integration waits for G4)."""
import argparse
import json
from pathlib import Path
import time
import numpy as np

SCENE=Path(__file__).resolve().parents[2]/'docs/jsmp133/request_d_history_noise/scene_portrait.yaml'
DET=dict(center_cm=[0.,0.,0.],normal=[0.,0.,1.],u_axis=[1.,0.,0.],size_cm=(15.,18.),shape=(100,120))

def save(result,out,n,tag,seed,batch,wall):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);stem=out/f'N{n:09d}_{tag}';t=result.detector
    np.savez_compressed(f'{stem}.npz',category_fluence=t.category_fluence,photon_count=t.photon_count,energy_sum_keV=t.energy_sum_keV,energy_sum2_keV2=t.energy_sum2_keV2,total_rel_err=t.total_relative_error(),n_batches_hit=t.n_batches_hit,n_histories=t.n_histories,n_batches=t.n_batches,pixel_area_cm2=t.plane.pixel_area_cm2)
    meta=dict(n_histories=n,tag=tag,seed=seed,workers=1,batch_size=batch,n_batches=t.n_batches,wall_s=wall,detected=float(t.photon_count.sum()))
    meta.update(result.gpu_metadata)
    (Path(f'{stem}.json')).write_text(json.dumps(meta,indent=1)+'\n');return meta

def main(argv=None):
    from ..scene import load_scene
    from ..detector import DetectorPlane
    from ..transport import run_transport
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=float,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--workers',type=int,default=1);ap.add_argument('--tag',default='A');ap.add_argument('--out',type=Path,required=True);ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');ap.add_argument('--source-seed',type=int)
    a=ap.parse_args(argv);n=int(round(a.n));batch=int(min(200000,max(1,-(-n//10))))
    if a.device=='cuda':
        gate_path=SCENE.parents[2]/'validation/gpu/g1/gate2.json'
        if not gate_path.exists() or json.loads(gate_path.read_text()).get('status')!='pass':raise ValueError('gate 2 must pass before request-D CUDA execution')
    t0=time.perf_counter();res=run_transport(load_scene(str(SCENE)),n_histories=n,seed=a.seed,source_seed=a.source_seed,n_workers=a.workers,batch_size=batch,track_uncertainty=True,detector=DetectorPlane(**DET),engine='gpu',gpu_device=a.device)
    print(json.dumps(save(res,a.out,n,a.tag,a.seed,batch,time.perf_counter()-t0)))

if __name__=='__main__':main()

def collect_image(bundle_path,job,results,out,tag='A'):
    """On the host, assemble verified Colab batches into request-D files."""
    from types import SimpleNamespace
    from ..detector import DetectorPlane,DetectorTally
    from .bundle import load
    from .colab_driver import read_result,expected_identity
    a,src,spec,settings,manifest=load(bundle_path)
    p=a['detector'];plane=DetectorPlane(p[:3],p[3:6],p[6:9],tuple(p[12:14]),tuple(p[14:16].astype(int)))
    tally=DetectorTally(plane,True)
    if job['mode']!='batches' or job['device']!='cuda':raise ValueError('expected a CUDA image job')
    for b,start in enumerate(range(0,job['n_histories'],job['batch_size'])):
        n=min(job['batch_size'],job['n_histories']-start);name=f'batch_{b:06d}.npz';r=read_result(results,name,expected_identity(manifest,job,name))
        if r is None:raise ValueError('missing verified image batch '+name)
        from .core import health
        health(r,n,len(settings['material_names']),p)
        tally.category_fluence+=r['images'][:4];tally.photon_count+=r['images'][4];tally.energy_sum_keV+=r['images'][5];tally.energy_sum2_keV2+=r['images'][6];tally.end_batch(n)
    metadata=dict(engine='gpu',gpu_device='cuda',seed=job['seed'],source_seed=job['source_seed'],n_workers=1,bundle=manifest)
    return save(SimpleNamespace(detector=tally,gpu_metadata=metadata),out,job['n_histories'],tag,job['seed'],job['batch_size'],None)
