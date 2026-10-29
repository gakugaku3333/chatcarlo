"""A1 same-session old/new measurement and worker modes; no transport copies."""
import argparse
import importlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import uuid
import numpy as np
from a0_speed import EventDelegate,resources,sha,write_json
from a1_health import checked_run

HERE=Path(__file__).resolve().parent


def load_configs(path):
    cfg=json.loads(Path(path).read_text())
    tasks=['old-g2b:500000','new-g2b:500000','old-g2b:100000','new-g2b:100000','old-g4:100000','new-g4:100000']
    orders=[tasks[i:]+tasks[:i] for i in (0,2,4)]
    orders[1]=[v for j in range(0,6,2) for v in orders[1][j:j+2][::-1]]
    if (cfg['format']!='g5-a1-speed-v1' or cfg['warmup_order']!=tasks or
            cfg['round_orders']!=orders or cfg['tolerance_ratio']!=1.0571):
        raise ValueError('predeclared A1 orders/config required')
    # Bind the full A0 configuration, including seeds, geometry and flags.
    baseline=HERE/'a0_configs.json'
    if sha(baseline)!=cfg['a0_configs_sha256']:raise ValueError('A0 config SHA256 mismatch')
    if cfg['configurations']!=json.loads(baseline.read_text())['configurations'][:2]:
        raise ValueError('A0 diagnostic configuration mismatch')
    return cfg,sha(path)


def group_call(command,timeout):
    """A shell wrapper and all profiler descendants share a new process group."""
    p=subprocess.Popen(command,start_new_session=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        stdout,stderr=p.communicate(timeout=timeout)
        return subprocess.CompletedProcess(command,p.returncode,stdout,stderr)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGKILL);p.communicate();raise


def install_runtimes(base):
    """Verify before copying; the old top-level package has relative imports."""
    base=Path(base)
    for version,package in [('old','chatcarlo_g5_old'),('new','chatcarlo/gpu')]:
        first=None
        for cid in ('g2b','g4'):
            src=base/'bundles'/version/cid
            manifest=json.loads((src/'manifest.json').read_text())
            for name,digest in manifest['files'].items():
                if Path(name).name!=name or sha(src/name)!=digest:raise ValueError('runtime transfer SHA/path mismatch: '+name)
            if first is None:first=src
            else:
                for name in manifest['files']:
                    if Path(name).suffix in ('.py','.h','.cu','.cpp') and (first/name).exists() and sha(first/name)!=sha(src/name):
                        raise ValueError('runtime bundles disagree: '+name)
        target=base/package;target.mkdir(parents=True,exist_ok=True)
        for name in json.loads((first/'manifest.json').read_text())['files']:
            if Path(name).suffix in ('.py','.h','.cu','.cpp'):(target/name).write_bytes((first/name).read_bytes())
    (base/'chatcarlo/__init__.py').write_text('')


def construct(c,version,bundles,device='cuda'):
    package='chatcarlo_g5_old' if version=='old' else 'chatcarlo.gpu'
    loader=importlib.import_module(package+'.bundle')
    a,src,spec,settings,manifest=loader.load(Path(bundles)/version/c['id'])
    mod=importlib.import_module(package+('.g2_core' if c['grid'] else '.core'))
    kwargs={}
    if version=='new':kwargs['physics_declaration']=settings['physics_set']
    if c['grid']:kwargs['grid']=mod.grid_from(c['grid'])
    obj=mod.Core(a,device,**kwargs)
    kernel=obj.native if c['grid'] else obj.fn
    if settings['scene_sha256']!=c['scene_model_sha256'] or settings['fluorescence']!=c['flags']['fluorescence']:
        raise ValueError('speed bundle scene/flags mismatch')
    if c['detector'] is None:
        if a['detector'].size:raise ValueError('unexpected detector')
    else:
        d=c['detector'];expected=np.r_[d['center_cm'],d['normal'],d['u_axis'],np.cross(d['normal'],d['u_axis']),d['size_cm'],d['shape']]
        if not np.array_equal(a['detector'],expected):raise ValueError('detector mismatch')
    timer=None
    if device=='cuda':
        timer=EventDelegate(kernel,obj.cp)
        if c['grid']:obj.native=timer
        else:obj.fn=timer
    return dict(obj=obj,kernel=kernel,timer=timer,source=src,spectrum=spec,
                sample=importlib.import_module(package+'.source_host').sample_source_photons)


def measure(c,state,n):
    inputs=state['sample'](state['source'],n,np.random.default_rng(c['source_seed']),state['spectrum'])
    state['timer'].seconds.clear();begin=time.perf_counter()
    measured=state['obj'].run(*inputs,seed=c['seed'],start=c['start'],**c['flags'])
    if isinstance(measured,tuple):measured=measured[0]
    call_s=time.perf_counter()-begin
    if len(state['timer'].seconds)!=1:raise ValueError('one timed kernel required')
    kernel_s=state['timer'].seconds[0]
    from a1_health import diagnostic_health
    replay=state['obj'].run(*inputs,seed=c['seed'],start=c['start'],**dict(c['flags'],record=True,observe=False))
    h=diagnostic_health(measured,replay,inputs[2])
    return dict(kernel_s=kernel_s,call_s=call_s,health=h)


def summarize(rows,cfg,digest):
    expected=[(task,r) for r,order in [(-1,cfg['warmup_order']),*enumerate(cfg['round_orders'])] for task in order]
    if [(r['task'],r['repeat']) for r in rows]!=expected:raise ValueError('measurement order/count mismatch')
    if any(r['config_sha256']!=digest or r['health']['status']!='pass' or not np.isfinite(r['kernel_s']) or r['kernel_s']<=0 for r in rows):
        raise ValueError('invalid timing/health/config identity')
    ratios={}
    for task in ('g2b:500000','g2b:100000','g4:100000'):
        med={v:statistics.median(r['kernel_s'] for r in rows if r['task']==v+'-'+task and r['repeat']>=0) for v in ('old','new')}
        ratios[task]=dict(medians=med,ratio=med['new']/med['old'])
    return dict(status='pass' if all(v['ratio']<=cfg['tolerance_ratio'] for v in ratios.values()) else 'fail',ratios=ratios)


def speed(args):
    import cupy as cp
    cfg,digest=load_configs(args.configs);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'measurements.json').exists():raise ValueError('fresh session required')
    props=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name=props['name'].decode() if isinstance(props['name'],bytes) else props['name']
    if 'T4' not in name or (props['major'],props['minor'])!=(7,5):raise ValueError('T4 sm_75 required')
    session=str(uuid.uuid4());states={};res={};rows=[]
    record=dict(status='running',session_id=session,config_sha256=digest,resources=res,measurements=rows,ncu=[])
    def save():write_json(out/'measurements.json',record)
    try:
        for c in cfg['configurations']:
            for version in ('old','new'):
                key=version+'-'+c['id'];states[key]=construct(c,version,args.bundles)
                res[key]=resources(cp,states[key]['kernel'],c);save()
            old=res['old-'+c['id']];new=res['new-'+c['id']]
            if new['attributes']['num_regs']>128 or new['manual']['blocks_per_sm']<old['manual']['blocks_per_sm'] or new['occupancy_api_blocks_per_sm']<old['occupancy_api_blocks_per_sm']:
                raise ValueError('A1b-1 resource cliff; stop before warmup')
        by_id={c['id']:c for c in cfg['configurations']}
        for repeat,order in [(-1,cfg['warmup_order']),*enumerate(cfg['round_orders'])]:
            for task in order:
                key,n=task.split(':');c=by_id[key.split('-',1)[1]]
                rows.append(dict(task=task,repeat=repeat,config_sha256=digest,session_id=session,
                                 seed=c['seed'],source_seed=c['source_seed'],start=c['start'],**measure(c,states[key],int(n))))
                save()
        record['summary']=summarize(rows,cfg,digest);save()
        # Report-only ncu, one production-size launch per old/new configuration.
        for c in cfg['configurations']:
            for version in ('old','new'):
                key=version+'-'+c['id'];stem=out/(key+'_full')
                command=['ncu','--set','full','--target-processes','all','--kernel-name',
                         'g2_histories_kernel' if c['grid'] else 'g1_histories_kernel','--launch-count','1',
                         '--force-overwrite','--export',str(stem),'--csv','--log-file',str(stem.with_suffix('.csv')),
                         sys.executable,str(Path(__file__).resolve()),'--mode','profile','--profile',key,
                         '--configs',str(args.configs),'--bundles',str(args.bundles),'--output',str(out)]
                receipt=dict(task=key,config_sha256=digest,status='unavailable')
                try:
                    r=group_call(command,1800);receipt.update(returncode=r.returncode,status='captured' if r.returncode==0 else 'unavailable')
                except (OSError,subprocess.TimeoutExpired) as exc:receipt['error']=str(exc)
                record['ncu'].append(receipt);save()
        record['status']='complete';save();(out/'G5A1_DONE').write_text(session+'\n')
    except Exception as exc:
        record.update(status='failed',error=str(exc));save();raise


def worker_batches(args):
    """T4 modes consume a host-prepared hash-verified bundle/job. Host judges."""
    from chatcarlo.gpu import bundle
    from chatcarlo.gpu.source_host import sample_source_photons
    job=json.loads(Path(args.job).read_text());a,src,spec,settings,manifest=bundle.load(args.bundle)
    if job['bundle_manifest']!=manifest:raise ValueError('worker bundle identity mismatch')
    from chatcarlo.gpu.core import Core
    kwargs={}
    if 'grid' in job:
        from chatcarlo.gpu.g2_core import Core,grid_from
        kwargs['grid']=grid_from(job['grid'])
    driver=Core(a,args.device,physics_declaration=settings['physics_set'],**kwargs)
    rng=np.random.default_rng(job['source_seed']);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    for start in range(0,job['n_histories'],job['batch_size']):
        n=min(job['batch_size'],job['n_histories']-start)
        inputs=sample_source_photons(src,n,rng,spec)
        result,health=checked_run(driver,inputs,seed=job['seed'],start=start,fluorescence=settings['fluorescence'])
        file=out/f'batch_{start:09d}.npz';np.savez(file,**result)
        write_json(file.with_suffix('.json'),dict(job=job,start=start,n=n,health=health,sha256=sha(file)))
    write_json(out/'done.json',dict(status='complete',job=job))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['speed','profile','install','gate1','g2vox','faults','g2gate3'],default='speed')
    p.add_argument('--configs',type=Path,default=HERE/'a1_speed_configs.json');p.add_argument('--bundles',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--base',type=Path);p.add_argument('--profile')
    p.add_argument('--bundle',type=Path);p.add_argument('--job',type=Path);p.add_argument('--reference',type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    if a.mode=='install':install_runtimes(a.base)
    elif a.mode=='speed':speed(a)
    elif a.mode=='profile':
        cfg,_=load_configs(a.configs);v,cid=a.profile.split('-',1);c=next(c for c in cfg['configurations'] if c['id']==cid)
        measure(c,construct(c,v,a.bundles),c['production_n'])
    elif a.mode in ('gate1','g2vox'):worker_batches(a)
    elif a.mode=='faults':
        from a1_faults import run
        run(a.reference,a.output/'faults.json',a.device)
    else:
        from a1_g2gate3 import legacy
        legacy.execute(a.bundle,a.output)


if __name__=='__main__':main()
