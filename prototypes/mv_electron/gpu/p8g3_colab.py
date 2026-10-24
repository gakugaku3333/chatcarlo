"""Claude-only T4 driver. No kernel changes or local GPU execution."""
import argparse
from pathlib import Path
import shutil
import tempfile
import numpy as np
from . import p8g3_core as core, p8g2_colab as transport


def gpu_runner(gpu):
    def run(n, seed, start):
        result,_,visits,_ = gpu.run(n,seed,start)
        return result,visits
    return run


def initial_gate(cp, materials, arrays, gpu):
    original = transport.old.GPU(cp,materials,arrays,18,threads=256)
    _,expected = original.run(20000,core.kernel.SEED2,history=True)
    result,actual,visits,_ = gpu.run(20000,core.kernel.SEED2,history=True)
    core.kernel.check_visits(visits)
    if not core.run_health(result):
        raise AssertionError('initial gate health')
    core.kernel.exact_histories(actual,expected,'VM initial P8g1/P8g2')
    return dict(status='pass',n=20000,seed=core.kernel.SEED2,
                comparison='P8g1/P8g2 byte-identical')


def reject(action, label):
    try:
        action()
    except (ValueError,AssertionError,FileNotFoundError) as error:
        return dict(fault=label,rejected_by=str(error))
    raise AssertionError('fault not rejected '+label)


def numeric_faults(result, n):
    rows=[]
    for key,label in (('_groups_i','resume integer'),('_groups_d','resume float beyond bound')):
        bad={k:v.copy() for k,v in result.items()}
        bad[key][0,0] += 1
        rows.append(reject(lambda:core.check_resume(result,bad,n),label))
    for v in (0,2):
        visits=np.ones(n,np.int64);visits[0]=v
        rows.append(reject(lambda:core.kernel.check_visits(visits),'visits='+str(v)))
    return rows


def storage_faults(root, identity, schedule):
    """Exercise the new storage adapter on real T4 output, never its kernel."""
    root=Path(root)
    result,receipt=core.load_call(root,0,identity,schedule)
    sid=receipt['session_id'];rows=[]
    with tempfile.TemporaryDirectory(prefix='.faults-',dir=root) as tmp:
        work=Path(tmp)
        shutil.copyfile(core.session_path(root,sid),core.session_path(work,sid))
        def save(j=0,values=None,visits=None):
            return core.save_call(work,j,result if values is None else values,
                np.ones(schedule['n'],np.int64) if visits is None else visits,
                core.invocation(j,schedule),identity,schedule,sid)
        save()
        rows.append(reject(lambda:save(),'duplicate save'))
        rows.append(reject(lambda:core.completion(work,identity,schedule),'missing call'))
        path=core.call_path(work,0)
        data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
        rows.append(reject(lambda:core.resume_check(work,lambda **kw:(result,np.ones(schedule['n'])),identity,schedule),'one-byte corruption'))
        # Orphan data and orphan receipt both recover by recomputation.
        path.with_suffix('.json').unlink();save()
        path.unlink();save()
        core.load_call(work,0,identity,schedule)
        session=core.session_path(work,sid);good=core.read_json(session)
        session.unlink()
        rows.append(reject(lambda:core.available(work,identity,schedule),'missing session gate'))
        core.write_json(session,dict(good,status='fail'))
        rows.append(reject(lambda:core.available(work,identity,schedule),'failed session gate'))
        core.write_json(session,good)
        bad={k:v.copy() for k,v in result.items()};bad['violations'][0]=1
        rows.append(reject(lambda:save(1,bad),'health violation'))
        for v in (0,2):
            visits=np.ones(schedule['n'],np.int64);visits[0]=v
            rows.append(reject(lambda:save(1,visits=visits),'unsaved visits='+str(v)))
        if core.call_path(work,1).exists():raise AssertionError('unhealthy call saved')
    return rows


def qualify_trial(root, identity, schedule, sid, resume):
    """Run only after collection, VM stop and re-send to a distinct VM session."""
    complete=core.verify(root,identity,schedule)
    if resume['status']!='pass':
        raise ValueError('trial must resume collected calls')
    _,receipt=core.load_call(root,resume['call_id'],identity,schedule)
    if receipt['session_id']==sid:
        raise ValueError('trial needs a different session')
    result,_=core.load_call(root,0,identity,schedule)
    record=dict(status='pass',identity=identity,schedule=schedule,
                previous_session=receipt['session_id'],new_session=sid,
                resume=resume,complete=complete,
                faults=numeric_faults(result,schedule['n'])+storage_faults(root,identity,schedule),
                interrupted_storage='npz-only and receipt-only recovered')
    core.write_json(Path(root)/'trial_qualified.json',record,immutable=True)
    return record


def require_trial(root, identity):
    schedule=core.spec(4,2000)
    record=core.read_json(Path(root)/'trial_qualified.json')
    if (record['status']!='pass' or record['identity']!=identity or
        record['schedule']!=schedule or record['resume']['status']!='pass' or
        record['previous_session']==record['new_session'] or
        record['complete']!=core.verify(root,identity,schedule) or
        len(record['faults'])!=12 or
        record.get('interrupted_storage')!='npz-only and receipt-only recovered'):
        raise ValueError('mandatory T4 restart trial')
    core.require_session(root,record['new_session'],identity,schedule)
    return record


def run(bundle, output, mode='production', trial=None):
    import cupy as cp
    from .p5b_colab import environment
    manifest,materials,arrays=transport.load(bundle)
    identity=manifest['identity'];env=environment(cp)
    if 'T4' not in env['gpu']:
        raise ValueError('T4 required')
    schedule=core.spec() if mode=='production' else core.spec(4,2000)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    gpu=transport.GPU(cp,materials,arrays,blocks=40,threads=256)
    runner=gpu_runner(gpu)
    gate=initial_gate(cp,materials,arrays,gpu)
    if mode=='production':
        if trial is None:raise ValueError('T4 restart trial required')
        require_trial(trial,identity)
        gate['restart_trial']=core.read_json(Path(trial)/'trial_qualified.json')
    resume=core.resume_check(output,runner,identity,schedule)
    if mode=='trial-save' and resume['status']!='new':
        raise ValueError('trial-save requires fresh output')
    if mode=='trial-resume' and resume['status']!='pass':
        raise ValueError('trial-resume requires collected and resent calls')
    sid=core.new_session(output,identity,schedule,env['gpu'],gate,resume)
    core.drive(output,runner,identity,schedule,sid,limit=3 if mode=='trial-save' else None)
    if mode=='trial-resume':qualify_trial(output,identity,schedule,sid,resume)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--mode',choices=('production','trial-save','trial-resume'),default='production')
    parser.add_argument('--trial',type=Path)
    args=parser.parse_args();run(args.bundle,args.output,args.mode,args.trial)

if __name__=='__main__':main()
