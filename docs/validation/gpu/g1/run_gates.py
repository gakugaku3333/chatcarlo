"""Claude entry point: gate 1, export, gate 2 collection. Never auto-adjust criteria."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import numpy as np

from chatcarlo.gpu import bundle,validation
from chatcarlo.gpu.run_detector_image import DET
from chatcarlo.detector import DetectorPlane
from chatcarlo.scene import load_scene
from chatcarlo.transport import run_transport

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
PORTRAIT=ROOT/'docs/jsmp133/request_d_history_noise/scene_portrait.yaml'
S3_SHA='12c46b401c37508349a71d624b57bf202d0f885b1834df3f872e77d08e7bb892'

def inputs(scene_id):
    path=PORTRAIT if scene_id=='s3' else HERE/f'scenes/{scene_id}.yaml'
    if scene_id=='s3' and bundle.sha(path)!=S3_SHA:raise ValueError('committed S3 scene SHA256 changed')
    scene=load_scene(str(path));plane=DetectorPlane(**DET) if scene_id=='s3' else DetectorPlane([0,0,0],[0,0,1],[1,0,0],(30,30),(60,60))
    if not scene.ok:raise ValueError(scene.errors)
    return scene,plane,bundle.sha(path)

def collect(result,scene_id,n):
    names=list(dict.fromkeys(['air']+[g['material'] for g in inputs(scene_id)[0].raw['geometry']]))
    t=result.detector
    raw=dict(deposits=np.array([result.energy_deposited_MeV.get(name,0)*1000 for name in names]),endings=np.array([result.fraction_absorbed*n,result.fraction_escaped*n,t.photon_count.sum()]),images=np.concatenate((t.category_fluence,t.photon_count[None],t.energy_sum_keV[None],t.energy_sum2_keV2[None])))
    return validation.quantities(raw,names,scene_id,n)

def gate1(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);attempts=[]
    for retry in (False,True):
        reference={};twin={};scenes={};offset=100 if retry else 0
        for sid in ('s1','s2','s3'):
            scene,plane,sha=inputs(sid);scenes[sid]=sha
            with tempfile.TemporaryDirectory(prefix='chatcarlo-g1-gate1-') as temp:
                manifest=bundle.create(scene,plane,temp)
            reference_sources={name:bundle.sha(ROOT/'chatcarlo'/name) for name in ('transport.py','kernel.py','physics.py','geometry.py','source.py','spectrum.py','materials.py','detector.py','tally.py')}
            for i in range(10):
                for engine,values,seed in (('numpy',reference,5000+offset+i),('gpu',twin,6000+offset+i)):
                    filename=out/f'gate1_{"retry" if retry else "initial"}_{sid}_{engine}_{i:02d}.json'
                    identity=dict(scene_sha256=sha,bundle=manifest,reference_sources=reference_sources,engine=engine,seed=seed,source_seed=7000+offset+i if engine=='gpu' else None,n_histories=1000000,batch_size=200000,n_workers=4 if engine=='numpy' else 1)
                    if filename.exists():
                        data=json.loads(filename.read_text())
                        if data['identity']!=identity:raise ValueError('gate1 resume identity mismatch')
                        q=data['quantities']
                    else:
                        kw=dict(n_histories=1000000,batch_size=200000,seed=seed,detector=plane,track_uncertainty=True,engine=engine,n_workers=identity['n_workers'])
                        if engine=='gpu':kw.update(gpu_device='cpu',source_seed=identity['source_seed'])
                        r=run_transport(scene,**kw);q=collect(r,sid,1000000)
                        filename.write_text(json.dumps(dict(identity=identity,quantities=q),indent=2)+'\n')
                    for k,v in q.items():values.setdefault(sid+':'+k,[]).append(v)
        report=validation.independent_judge(reference,twin);report.update(scene_sha256=scenes,seeds=dict(reference=list(range(5000+offset,5010+offset)),transport=list(range(6000+offset,6010+offset)),source=list(range(7000+offset,7010+offset))),n_histories_per_run=1000000,n_runs=10,batch_size=200000,retry=retry)
        attempts.append(report)
        if report['status']!='retry' or retry:break
    final=dict(report);final['attempts']=attempts
    if final['status']=='retry':final['status']='fail'
    if final['status']=='pass':final['interpretation']='この精度では差を検出しなかった。同等性の証明ではない。'
    (HERE/'gate1.json').write_text(json.dumps(final,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=final['status'],K=final['K'],sum_z2_over_K=final['sum_z2_over_K'])))

def export(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for sid in ('s1','s2','s3'):
        scene,plane,sha=inputs(sid);bundle.create(scene,plane,out/sid)
        if sid!='s1':
            job=dict(mode='gate2',scene_id=sid,seed=8000,source_seed=8001,n_histories=10000000,batch_size=1000000,device='cuda',scene_yaml_sha256=sha)
            (out/f'{sid}_gate2_job.json').write_text(json.dumps(job,indent=2)+'\n')

def gate2_merge(out):
    out=Path(out);reports={sid:json.loads((out/sid/'gate2.json').read_text()) for sid in ('s2','s3')}
    for sid,r in reports.items():
        if r['scene_id']!=sid or r['seeds']!=dict(transport=8000,source=8001) or r['n_batches']!=10 or r['batch_size']!=1000000:raise ValueError('gate2 identity mismatch')
        scene,plane,sha=inputs(sid)
        receipt=json.loads((out/sid/'gate2.json.receipt.json').read_text())
        if receipt['identity']['job']['scene_yaml_sha256']!=sha or receipt['sha256']!=bundle.sha(out/sid/'gate2.json'):raise ValueError('gate2 receipt mismatch')
    final=dict(status='pass' if all(r['status']=='pass' for r in reports.values()) else 'fail',scenes=reports)
    (HERE/'gate2.json').write_text(json.dumps(final,indent=2,allow_nan=False)+'\n');print(final['status'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['gate1','export','gate2-merge']);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    {'gate1':gate1,'export':export,'gate2-merge':gate2_merge}[a.action](a.out)

if __name__=='__main__':main()
