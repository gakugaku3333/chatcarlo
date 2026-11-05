"""S1 rerun of the unchanged G1 51-quantity gate; never rewrites G1 evidence."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
ROOT=Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from chatcarlo.gpu import bundle, validation
from chatcarlo.transport import run_transport
HERE=ROOT/'docs/validation/gpu/g1'
spec=importlib.util.spec_from_file_location('a4u_g1_reference',HERE/'run_gates.py')
reference_gate=importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference_gate)
inputs=reference_gate.inputs
collect=reference_gate.collect

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
    (Path(__file__).with_name('a4u_s1_gate1.json')).write_text(json.dumps(final,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=final['status'],K=final['K'],sum_z2_over_K=final['sum_z2_over_K'])))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    gate1(parser.parse_args().out)
