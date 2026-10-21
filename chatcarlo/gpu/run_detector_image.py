"""Request-D compatible detector image runner using the shared G4 output layer."""
import argparse
import json
from pathlib import Path
import time
import numpy as np

SCENE=Path(__file__).resolve().parents[2]/'docs/jsmp133/request_d_history_noise/scene_portrait.yaml'
DET=dict(center_cm=[0.,0.,0.],normal=[0.,0.,1.],u_axis=[1.,0.,0.],size_cm=(15.,18.),shape=(100,120))

def save(result,out,n,tag,seed,batch,wall):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);stem=out/f'N{n:09d}_{tag}'
    from .outputs import save as save_outputs
    meta=dict(n_histories=n,tag=tag,seed=seed,workers=1,batch_size=batch,wall_s=wall)
    meta.update(result.gpu_metadata)
    return save_outputs(result,stem,meta)

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
    from ..detector import DetectorPlane
    from .bundle import load
    from .approval import verify_batches
    from .aggregate import aggregate
    a,src,spec,settings,manifest=load(bundle_path)
    p=a['detector']
    plane=None if not p.size else DetectorPlane(p[:3],p[3:6],p[6:9],tuple(p[12:14]),tuple(p[14:16].astype(int)))
    if job['mode']!='batches' or job['device']!='cuda':raise ValueError('expected a CUDA image job')
    batches,timing=verify_batches(bundle_path,job,results)
    result=aggregate(batches,settings['material_names'],plane)
    result.gpu_metadata=dict(engine='gpu',gpu_device='cuda',seed=job['seed'],source_seed=job['source_seed'],n_workers=1,bundle=manifest,timing=timing)
    return save(result,out,job['n_histories'],tag,job['seed'],job['batch_size'],timing['elapsed_s'])
