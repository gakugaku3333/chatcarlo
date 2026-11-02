"""A3 VM tasks. Called only by a3_t4_gates; no automatic GPU work on import."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT), str(HERE)]
from a0_speed import write_json, sha


def reject(action, message):
    try:
        action()
    except (ValueError, AssertionError) as exc:
        if message not in str(exc):
            raise AssertionError('unexpected rejection: '+str(exc)) from exc
        return dict(status='pass', detected=str(exc))
    raise AssertionError('fault accepted: '+message)


def h2(base, out):
    import cupy as cp
    from chatcarlo.gpu import therapy_inputs, therapy, therapy_checks, therapy_health, contracts
    from chatcarlo.gpu.therapy_cuda import T4
    from chatcarlo.gpu.therapy_observation import descriptor
    m, a, fg, _, _ = therapy_inputs.load(base/'therapy_bundle', 'mohan10')
    cpu = therapy.CPU(out/'build', m, a, field_geometry=fg, half=(5.,5.))
    gpu = T4(cp, m, a, fg, half=(5.,5.), check=1, source_check=True)
    result = dict(checks=therapy_checks.run(cpu,a,fg,20261101,gpu))
    # Change the actual launch tuple, not the approved condition or driver.half.
    native = gpu.kernel
    expected=descriptor(a,fg,(5.,5.),100.,20261101,0,8,gpu.declared)
    # observation is taken before kernel __call__, so recompute from the *tampered*
    # tuple in the delegate, exactly at the native boundary.
    from chatcarlo.gpu.therapy_observation import gpu_observation
    def changed_observed(grid,block,args,**kw):
        args=list(args);args[19]=np.float64(6.)
        gpu.last_observation=gpu_observation(gpu,tuple(args))
        return native(grid,block,tuple(args),**kw)
    gpu.kernel=changed_observed;gpu.run(8,20261101)
    result['field_half']=reject(lambda:contracts.check_observation(gpu.last_observation,expected,'native'),'half')
    gpu.kernel=native
    for key,value in (('ID','99'),('TAG','0ULL')):
        real=cp.RawKernel
        def bad(*args,options=(),**kw):
            opts=tuple(f'-DG5_SET_{key}={value}' if x.startswith(f'-DG5_SET_{key}=') else x for x in options)
            return real(*args,options=opts,**kw)
        with patch.object(cp,'RawKernel',bad):
            result[key.lower()]=reject(lambda:T4(cp,m,a,fg,half=(5.,5.)),'mismatch')
    write_json(out/'entry_faults.json',result)
    # Reuse all A2 path, fixed-scoring and health fault cases, but bind them to the
    # FINAL package launcher, never the historical validation-side T4 class.
    import a2_gates as gates
    import a2_common as common
    from prototypes.mv_electron.gpu import p8g2_colab
    _,materials,arrays=p8g2_colab.load(base/'bundles/p8g2')
    gates.WATER=base
    with patch.object(gates,'T4',T4):
        gates.checks(cp,materials,arrays,out,lambda *x:print(*x,flush=True))
    return dict(status='pass',entry=result)


def h5(base,out):
    """VM produces only A1-format GPU batches; references/judges stay on host.

    Both pre-exported seed sets are produced. The unchanged host judges decide
    whether the restricted second set may be consumed (as in A1 remote mode).
    """
    from a1_speed import worker_batches
    for gate in ('gate1','g2vox'):
        jobs=sorted((base/'h5'/gate/'jobs').glob('*.json'))
        if not jobs:raise ValueError('missing host-exported H5 jobs: '+gate)
        for job in jobs:
            retry,name,k=job.stem.split('_')
            target=out/gate/(job.stem if gate=='gate1' else f'{retry}_{name}')
            if gate=='g2vox':target=target/k
            worker_batches(SimpleNamespace(mode=gate,bundle=base/'h5'/gate/'bundles'/name,
                job=job,output=target,device='cuda'))
    return dict(status='pass',meaning='GPU batches produced; host statistics pending')


def speed(base,out):
    import cupy as cp
    import a1_speed,a2_gates,a2_speed
    from chatcarlo.gpu.therapy_cuda import T4
    from prototypes.mv_electron.gpu import p8g2_colab
    # Install ONLY the old package. a1_speed.install_runtimes would overwrite the
    # final package with a subset of an exported diagnostic bundle: never call it.
    target=ROOT/'chatcarlo_g5_old';target.mkdir(exist_ok=False)
    first=base/'diagnostic/old/g2b'
    manifest=json.loads((first/'manifest.json').read_text())
    for cid in ('g2b','g4'):
        source=base/'diagnostic/old'/cid
        m=json.loads((source/'manifest.json').read_text())
        baseline=json.loads((HERE/'a1_run/t4_speed/job.json').read_text())['bundles']
        if sha(source/'manifest.json')!=baseline['old/'+cid]['manifest.json']['sha256']:
            raise ValueError('old diagnostic baseline mismatch')
        if m['format']!='chatcarlo-g1-v1':raise ValueError('old diagnostic format')
        for name,digest in m['files'].items():
            if Path(name).name!=name or sha(source/name)!=digest:raise ValueError('old runtime SHA/path')
            if Path(name).suffix in ('.py','.h','.cpp','.cu'):
                if (first/name).exists() and sha(first/name)!=digest:raise ValueError('old runtime disagreement')
    for name in manifest['files']:
        if Path(name).suffix in ('.py','.h','.cpp','.cu'):(target/name).write_bytes((first/name).read_bytes())
    cfg=json.loads((HERE/'a1_speed_configs.json').read_text());cfg['round_orders']=cfg['round_orders'][:1]
    config=out/'diagnostic_config.json';write_json(config,cfg)
    a1_speed.speed(SimpleNamespace(configs=config,output=out/'diagnostic',bundles=base/'diagnostic'))
    diagnostic=json.loads((out/'diagnostic/measurements.json').read_text())['summary']
    if diagnostic['status']!='pass':raise RuntimeError('H6 diagnostic speed failed')
    _,materials,arrays=p8g2_colab.load(base/'bundles/p8g2');a2_gates.WATER=base
    therapy=therapy_speed(cp,base,materials,arrays,out)
    ratios={cid:dict(ratio=value,limit=1.0571,status='pass' if value<=1.0571 else 'fail') for cid,value in therapy['ratio'].items()}
    # Same-session direct old/new dose comparison. Reuse a2_oldnew's P7 unpack /
    # dose / compare and stack_dose / judge operations; that file has top-level
    # archive reads and writes to a2_run, so importing/running it would be unsafe.
    comparison=oldnew(cp,base,materials,arrays,out)
    return dict(status='pass' if all(v['status']=='pass' for v in ratios.values()) else 'fail',
                diagnostic=diagnostic,therapy=ratios,oldnew=comparison)


def remaining(base,out):
    import cupy as cp
    from prototypes.mv_electron.gpu import p8g2_colab
    import a2_gates
    _,materials,arrays=p8g2_colab.load(base/'bundles/p8g2');a2_gates.WATER=base
    return dict(status='pass',oldnew=oldnew(cp,base,materials,arrays,out))


def therapy_speed(cp,base,materials,arrays,out):
    """A2 speed order and measurements; a2_speed builds FINAL package launches."""
    import a0_speed,a2_speed,a2_gates,a2_common
    from chatcarlo.gpu.therapy_cuda import T4
    cfg={c['id']:c for c in json.loads((HERE/'a0_configs.json').read_text())['configurations']}
    cfg['stack']=cfg['p8g2'];states={};rows=[];resources={}
    # a2_speed.build_new consumes exactly the existing export/water inputs. Its
    # historical NewCore is replaced in memory, never edited on disk.
    for cid in ('p7A','p7B','p7C','stack'):
        c=cfg[cid]
        states['old-'+cid]=a0_speed.construct(cp,c,base/'bundles')
        extra=dict(p8g2=base/'bundles/p8g2',water=base/f'water_{cid[-1]}.npz')
        with patch.object(a2_speed,'NewCore',T4):
            states['new-'+cid]=a2_speed.build_new(cp,extra,cid)
        resources['old-'+cid]=a0_speed.resources(cp,states['old-'+cid]['kernel'],c)
        resources['new-'+cid]=a0_speed.resources(cp,states['new-'+cid].kernel,c)
    def one(version,cid,phase):
        c=cfg[cid];key=version+'-'+cid;n=c['production_n']
        if version=='old':row=a0_speed.measure(c,states[key],n)
        else:
            r=states[key].run(n,c['seed'])
            a=arrays if cid=='stack' else a2_gates.load_water(cid[-1])['arrays']
            health=a2_gates.batch_health(r,a,None if cid=='stack' else a2_common.field_geometry())
            row=dict(kernel_s=r['kernel_s'],health=health)
        rows.append(dict(row,task=key,phase=phase))
        write_json(out/'speed.json',dict(status='running',rows=rows,resources=resources))
    for phase in ('warmup','valid'):
        for cid in ('p7A','p7B','p7C','stack'):
            for version in ('old','new'):one(version,cid,phase)
    get=lambda key:next(r['kernel_s'] for r in rows if r['task']==key and r['phase']=='valid')
    from a3_entry_speed import ratio
    ratios={cid:ratio(get('old-'+cid),get('new-'+cid)) for cid in ('p7A','p7B','p7C','stack')}
    record=dict(status='complete',rows=rows,resources=resources,ratio={k:r['ratio'] for k,r in ratios.items()})
    write_json(out/'speed.json',record)
    return record


def oldnew(cp,base,materials,arrays,out):
    """Report only, independent of EGS5: same seeds/history ranges on one T4."""
    import a0_speed,a2_gates,a2_judge as J
    from chatcarlo.gpu.therapy_cuda import T4
    from prototypes.mv_electron.gpu import p7_core,p8f_core
    from prototypes.mv_electron import analysis_p7
    configs=json.loads((HERE/'a0_configs.json').read_text())['configurations']
    from prototypes.mv_electron import run_p8e
    grid=run_p8e.layout(run_p8e.LAYERS,[1.,.26,1.85,11.35])
    result={}
    for c in configs[2:]:
        cid=c['id'];n=c['production_n'];seed=c['seed'];batch=n//100
        old=a0_speed.construct(cp,c,base/'bundles')['obj']
        if cid=='p8g2':
            new=T4(cp,materials,arrays);a=arrays;fg=None
        else:
            w=a2_gates.load_water(cid[-1]);a=w['arrays'];fg=p7_core.geometry('p7')[0]
            new=T4(cp,materials,a,w['fg'],w['half'],w['ssd'])
        left=[];right=[]
        for b in range(100):
            if cid=='p8g2':
                r,_,visits,_=old.run(batch,seed,start=b*batch,**c['flags'])
                from prototypes.mv_electron.gpu.p8f_colab import run_health
                if not run_health(r) or not np.all(visits==1):raise ValueError('old stack health')
                left.append(r['p8_batches'].sum(0))
            else:
                r,_=old.measure(batch,seed,start=b*batch,**c['flags'])
                from prototypes.mv_electron.gpu import p7_colab
                p7_colab.core.require(p7_colab.h.health(r),'old P7 health')
                left.append(r['regions'].sum(0))
            fresh=new.run(batch,seed,start=b*batch)
            from a2_common import field_geometry
            a2_gates.batch_health(fresh,a,None if cid=='p8g2' else field_geometry())
            right.append(fresh['groups'][:,:int(a['sizes'][1])].sum(0) if cid=='p8g2' else fresh['field'])
        left,right=np.asarray(left),np.asarray(right)
        np.savez_compressed(out/f'oldnew_{cid}.npz',old=left,new=right)
        if cid=='p8g2':
            before=J.stack_dose(left,n,grid['mass_g_cm2'],grid)
            after=J.stack_dose(right,n,grid['mass_g_cm2'],grid)
            report=p8f_core.judge(after,before,grid,dref=p8f_core.DREF)
        else:
            lock=json.loads((ROOT/f'docs/validation/mv/p7/pilot_lock_{cid[-1]}.json').read_text())
            spec=lock['spec']
            before=analysis_p7.dose(J.p7_expand(left,n),spec)
            after=analysis_p7.dose(J.p7_expand(right,n),spec)
            report=analysis_p7.compare(after,before,lock['Dref'],spec)
        result[cid]=dict(status='report-only',histories=n,seed=seed,artifact=f'oldnew_{cid}.npz',comparison=report)
    write_json(out/'oldnew.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('task',choices=['h2','h5','speed','remaining']);p.add_argument('base',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    import cupy as cp
    props=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name=props['name'].decode() if isinstance(props['name'],bytes) else props['name']
    if 'T4' not in name or (props['major'],props['minor'])!=(7,5):raise ValueError('T4 sm_75 required')
    try:
        r=globals()[a.task](a.base,a.out);write_json(a.out/'verdict.json',r)
        if r['status']!='pass':raise RuntimeError(a.task+' failed')
        (a.out/'DONE').write_text('done\n')
    except Exception as exc:
        write_json(a.out/'failure.json',dict(status='fail',error=str(exc)));raise


if __name__=='__main__':main()
