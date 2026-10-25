"""Approved P9c launch and strict metadata verification (no native argument observation)."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import os
import time
import numpy as np
from .condition import digest, check_equal
from . import seed_condition as condition, seed_preview as preview, seed_analysis as analysis
from ..gpu import p9c_cli as client, p9c_colab as colab, p9c_core as core
from ..gpu.p9c_stats import Checkpoint
from ..gpu.p5b_core import write_json


def shapes():
    return dict(fine=core.FINE.shape,coarse=core.COARSE.shape,aggregate=(2,),
                lines=(3,core.FINE.shape[2]),representative=(3,core.FINE.shape[2]),shadow=(3,))


def verify(results, cond):
    results = Path(results); job = condition.read(results/'job.json')
    for key,value in dict(status='complete',mode='production',beams=['prostate'],sources=cond['sources']).items():
        check_equal(job.get(key),value,'job '+key)
    if not isinstance(job.get('gates'), dict) or not set(condition.GPU_GATES) <= job['gates'].keys():
        raise ValueError('missing required GPU gates')
    try:
        colab.validate_gates(job.get('gates'),'production',required=True)
    except AssertionError as exc:
        raise ValueError(str(exc)) from exc
    cp = Checkpoint(results/'prostate.npz',condition.checkpoint_identity(cond),shapes())
    check_equal(cp.completed,list(range(100)),'checkpoint completed')
    summary = condition.read(results/'prostate_summary.json')
    for label,s in (('summary',summary),('job results',job.get('results',{}).get('prostate',{}))):
        check_equal(s.get('batches'),100,label+' batches')
        check_equal(s.get('checkpoint_sha256'),condition.io.sha(results/'prostate.npz'),label+' checkpoint_sha256')
        check_equal(s.get('V3_CUDA_acceptance',{}).get('status'),'pass',label+' V3_CUDA_acceptance')
    # Reuse the full bundle checks, including V2a, V6 and V6b. Recheck T0 as well.
    check_equal(condition.identities(cond['scene']),cond,'post-run conditions')
    return dict(local=condition.read(Path(cond['export'])/'input.json')['local'],gpu=job['gates'],
                V3_CUDA_acceptance=summary['V3_CUDA_acceptance'])


@contextmanager
def lock(path):
    try:
        fd = os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError as exc:
        raise ValueError('実行ロック: 二重実行は拒否します') from exc
    try:
        os.write(fd,str(os.getpid()).encode()); os.close(fd)
        yield
    finally:
        Path(path).unlink()


def run(scene, directory=None):
    before = time.perf_counter(); cond = condition.identities(scene)
    directory = Path(directory) if directory is not None else condition.default_out(cond)
    approval = preview.require_approval(directory,cond)  # before creating any output
    timings = dict(bundle_verification=time.perf_counter()-before)
    results = directory/'results'  # HTML/approval never uploaded to the GPU
    with lock(directory/'execution.lock'):
        marker = results/'condition_hash.json'
        if results.exists() and any(results.iterdir()):
            if not marker.is_file():
                raise ValueError('existing output: condition hash missing')
            check_equal(condition.read(marker),digest(cond),'existing output condition_hash')
        results.mkdir(parents=True,exist_ok=True)
        write_json(marker,digest(cond))
        # Last read-only check immediately before calling the existing supervisor.
        check_equal(condition.identities(scene),cond,'pre-launch conditions')
        before = time.perf_counter()
        status = client.run(SimpleNamespace(session='p9c'+digest(cond)[:8],export=Path(cond['export']),
                            output=results,t0=Path(cond['t0_path']),job=cond['job'],
                            first_wait=300,interval=600,stall_minutes=15,max_restarts=6))
        timings['colab'] = time.perf_counter()-before
        if status != 0:
            raise ValueError('P9c supervisor did not complete; same-condition resume is permitted')
        before = time.perf_counter(); gates = verify(results,cond)
        timings['verification'] = time.perf_counter()-before
        before = time.perf_counter(); values = analysis.analyse(results/'prostate.npz',directory)
        timings['analysis'] = time.perf_counter()-before
        record = dict(condition_hash=digest(cond),approval_time=approval['time'],
                      histories_per_field=scene['run']['histories_per_field'],histories_per_side=7*scene['run']['histories_per_field'],
                      seed=scene['run']['seed'],timings=timings,gates=gates,results=values,
                      carried_not_independently_observed=list(scene),
                      provenance='Metadata comparison only; actual GPU arguments were not independently observed.',
                      notes=analysis.NOTES+[preview.EXCLUDED])
        write_json(directory/'run_record.json',record)
    return record


def main(args, scene):
    if args.command == 'validate':
        print(f'OK: {args.scene} は有効な治療領域シーンです（Mohan 10 MV、seed_ratio、58本、前立腺形7門、SAD 100 cm）')
    elif args.command == 'approve':
        preview.approve(scene,args.preview)
        print('承認記録: '+str(args.preview/'approval.json'))
    elif args.command == 'preview':
        directory = args.out or condition.default_out(condition.identities(scene))
        preview.write(scene,directory)
        print('3D表示を確認して承認してください: '+str(directory/'preview.html'))
    else:
        cond = condition.identities(scene); directory = args.out or condition.default_out(cond)
        run(scene,directory)
        print('結果: '+str(directory/'run_record.json'))
    return 0
