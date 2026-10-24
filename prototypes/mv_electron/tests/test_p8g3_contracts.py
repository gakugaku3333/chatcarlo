"""Driver contracts and planted faults, without executing Colab or export."""
from pathlib import Path
import shutil
import numpy as np
import pytest
from prototypes.mv_electron.gpu import p8g3_core as c, p8g3_colab as vm
from prototypes.mv_electron.gpu.p8g3_cli import collect as transfer

IDENTITY={'sources':c.kernel.fingerprint(),'physics':'fixture'}
OBS=[]


@pytest.fixture(scope='module',autouse=True)
def observations():
    yield
    target=Path(__file__).resolve().parents[3]/'docs/validation/mv/p8g3/fault_observations.json'
    c.write_json(target,dict(status='observed',scope='CPU/NumPy fixtures; actual T4 gates deferred',faults=OBS))


def values(n=2000,gpu=True):
    d=np.zeros((100,795));i=np.zeros((100,299),np.uint64)
    d[:,0]=1
    result=c.kernel.reduce_records(d,i,220,108,430,n,np.ones(220))
    result['seconds']=np.array(.1)
    if gpu:result.update(_groups_d=d,_groups_i=i,_absolute=np.abs(d))
    return result


def setup(root,count=4,n=2000,mode='gpu'):
    schedule=c.spec(count,n,mode)
    gate=dict(status='pass',n=20000,seed=c.kernel.SEED2,comparison='P8g1/P8g2 byte-identical')
    sid=c.new_session(root,IDENTITY,schedule,'T4' if mode=='gpu' else 'CPU',gate,dict(status='new'))
    return schedule,sid


def runner(n,seed,start):return values(n),np.ones(n,np.int64)


def save(root,j,schedule,sid,result=None,visits=None):
    return c.save_call(root,j,values(schedule['n']) if result is None else result,
        np.ones(schedule['n'],np.int64) if visits is None else visits,
        c.invocation(j,schedule),IDENTITY,schedule,sid)


def rejected(label,action,pattern=None):
    with pytest.raises((ValueError,AssertionError,FileNotFoundError),match=pattern) as error:action()
    OBS.append(dict(fault=label,rejected_by=str(error.value)))


def snapshot(root,schedule):
    calls=c.available(root,IDENTITY,schedule)
    return dict(state='running',groups=[dict(name='results',schedule=schedule,
        sessions={r['session_id']:r['session_sha256'] for r in calls.values()},
        calls={str(j):dict(sha256=r['sha256'],receipt_sha256=c.kernel.sha(c.call_path(root,j).with_suffix('.json'))) for j,r in calls.items()},
        records={n:c.kernel.sha(root/n) for n in ('complete.json',) if (root/n).exists()})])


def test_p8g3_400_actual_arguments_and_p4(monkeypatch,tmp_path):
    schedule=c.spec();sid='unused';calls=[];receipts=[]
    class GPU:
        def run(self,n,seed,start):
            calls.append(dict(n=n,seed=seed,start=start))
            return {},None,None,None
    def sealed(root,j,result,visits,actual,identity,schedule,sid):
        receipts.append(dict(call_id=j,**actual))
    monkeypatch.setattr(c,'save_call',sealed)
    monkeypatch.setattr(c,'completion',lambda *args:{'status':'pass'})
    c.drive(tmp_path,vm.gpu_runner(GPU()),IDENTITY,schedule,sid)
    expected=[dict(n=200000,seed=820301,start=j*200000) for j in range(400)]
    assert calls==expected
    assert receipts==[dict(call_id=j,**v) for j,v in enumerate(calls)]
    assert calls[-1]['start']+calls[-1]['n']==80000000
    # The independent witness reads actual calls, not scheduled receipt labels.
    calls.clear();receipts.clear()
    def faulty(n,seed,start):return vm.gpu_runner(GPU())(n,seed,0)
    c.drive(tmp_path/'p4',faulty,IDENTITY,schedule,sid)
    rejected('p4 all actual start=0, receipts scheduled',lambda:c.kernel.bitwise(
        np.array([r['start'] for r in calls]),np.array([r['start'] for r in receipts]),'actual schedule'))


def test_p8g3_interruption_resume_and_minimum_id(tmp_path):
    schedule,sid=setup(tmp_path)
    c.drive(tmp_path,runner,IDENTITY,schedule,sid,limit=3)
    assert set(c.available(tmp_path,IDENTITY,schedule))=={0,1,2}
    calls=[]
    def spy(**kw):calls.append(kw);return runner(**kw)
    resume=c.resume_check(tmp_path,spy,IDENTITY,schedule)
    assert resume==dict(status='pass',call_id=0)
    gate=c.read_json(c.session_path(tmp_path,sid))['gate']
    sid2=c.new_session(tmp_path,IDENTITY,schedule,'T4',gate,resume)
    c.drive(tmp_path,spy,IDENTITY,schedule,sid2)
    assert [v['start'] for v in calls]==[0,6000]
    complete=c.verify(tmp_path,IDENTITY,schedule)
    assert complete['n']==8000 and len(complete['calls'])==4
    # Recovery may have only a later ID collected.
    for j in (0,1):
        c.call_path(tmp_path,j).unlink();c.call_path(tmp_path,j).with_suffix('.json').unlink()
    calls.clear();assert c.resume_check(tmp_path,spy,IDENTITY,schedule)['call_id']==2
    assert calls[0]['start']==4000


def test_p8g3_p1_hash_p2_duplicate_missing_and_p3_health(tmp_path):
    schedule,sid=setup(tmp_path)
    save(tmp_path,0,schedule,sid)
    rejected('p2 duplicate call save',lambda:save(tmp_path,0,schedule,sid),'duplicate')
    rejected('p2 missing call at completion',lambda:c.completion(tmp_path,IDENTITY,schedule),'incomplete')
    path=c.call_path(tmp_path,0);data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
    rejected('p1 one byte corruption at resume',lambda:c.resume_check(tmp_path,runner,IDENTITY,schedule),'hash')
    result=values();result['violations'][0]=1
    rejected('p3 health violation',lambda:save(tmp_path,1,schedule,sid,result),'run_health')
    assert not c.call_path(tmp_path,1).exists()


@pytest.mark.parametrize('v',[0,2])
def test_p8g3_visits_rejected_before_save(tmp_path,v):
    schedule,sid=setup(tmp_path);visits=np.ones(2000,np.int64);visits[71]=v
    rejected('visits='+str(v),lambda:save(tmp_path,0,schedule,sid,visits=visits),'output count')
    assert not c.call_path(tmp_path,0).exists()


@pytest.mark.parametrize('fault',['missing','failed','changed','conditions','actual','visits','id'])
def test_p8g3_receipt_session_faults(tmp_path,fault):
    schedule,sid=setup(tmp_path);save(tmp_path,0,schedule,sid)
    path=c.session_path(tmp_path,sid)
    if fault=='missing':path.unlink()
    elif fault in ('failed','changed'):
        rec=c.read_json(path);rec['status']='fail' if fault=='failed' else 'pass';rec['gpu']='T4 changed'
        c.write_json(path,rec)
    else:
        path=c.call_path(tmp_path,0).with_suffix('.json');rec=c.read_json(path)
        if fault=='conditions':rec['seed']+=1
        if fault=='actual':rec['actual']['start']=1
        if fault=='visits':rec['visits']['all_one']=False
        if fault=='id':rec['call_id']=1
        c.write_json(path,rec)
    rejected('receipt/session '+fault,lambda:c.available(tmp_path,IDENTITY,schedule))


def test_p8g3_immutable_session_and_complete(tmp_path):
    schedule,sid=setup(tmp_path,count=1);save(tmp_path,0,schedule,sid)
    path=c.session_path(tmp_path,sid);rec=c.read_json(path);rec['gpu']='changed'
    rejected('immutable session overwrite',lambda:c.write_json(path,rec,immutable=True),'immutable')
    c.write_json(tmp_path/'complete.json',c.completion(tmp_path,IDENTITY,schedule))
    rec=c.read_json(tmp_path/'complete.json');rec['calls']['0']['sha256']='wrong';c.write_json(tmp_path/'complete.json',rec)
    rejected('completion seal changed',lambda:c.verify(tmp_path,IDENTITY,schedule),'complete')
    shutil.copyfile(c.call_path(tmp_path,0),tmp_path/'call_400.npz')
    rejected('unexpected call ID',lambda:c.available(tmp_path,IDENTITY,schedule),'unexpected')


@pytest.mark.parametrize('kind',['npz','json','pending'])
def test_p8g3_orphan_save_recovery(tmp_path,kind):
    schedule,sid=setup(tmp_path,count=1);save(tmp_path,0,schedule,sid)
    path=c.call_path(tmp_path,0)
    if kind=='npz':path.with_suffix('.json').unlink()
    elif kind=='json':path.unlink()
    else:
        path.rename(path.with_name('call_000.pending.npz'));path.with_suffix('.json').unlink()
    assert c.available(tmp_path,IDENTITY,schedule)=={}
    c.drive(tmp_path,runner,IDENTITY,schedule,sid)
    assert c.verify(tmp_path,IDENTITY,schedule)['status']=='pass'


def test_p8g3_resume_faults_and_tolerance(tmp_path):
    schedule,sid=setup(tmp_path,count=1);save(tmp_path,0,schedule,sid)
    for key in ('_groups_i','_groups_d'):
        def bad(n,seed,start):
            r,v=runner(n,seed,start);r[key][0,0]+=1;return r,v
        rejected('resume '+key,lambda:c.resume_check(tmp_path,bad,IDENTITY,schedule),'resume')
    good=values();near={k:v.copy() for k,v in good.items()}
    near['_groups_d'][0,0]=np.nextafter(1.,2.)
    c.check_resume(good,near,2000)
    rows=vm.numeric_faults(good,2000);assert len(rows)==4;OBS.extend(rows)


def test_p8g3_schedule_receipt_fault(tmp_path):
    schedule,sid=setup(tmp_path)
    rejected('p4 save actual start mismatch',lambda:c.save_call(tmp_path,1,values(),np.ones(2000),
        dict(n=2000,seed=c.SEED,start=0),IDENTITY,schedule,sid),'actual')


def test_p8g3_transfer_incremental_sessions_first_and_conflict(tmp_path):
    remote=tmp_path/'remote';schedule,sid=setup(remote,count=2)
    c.drive(remote,runner,IDENTITY,schedule,sid)
    status=snapshot(remote,schedule);downloaded=[]
    def download(name,path):downloaded.append(name);shutil.copyfile(remote/Path(name).name,path)
    out=tmp_path/'local';transfer.collect(status,out,IDENTITY,download)
    assert downloaded[0].startswith('results/session_')
    c.verify(out/'results',IDENTITY,schedule)
    downloaded.clear();transfer.collect(status,out,IDENTITY,download)
    assert downloaded==['results/complete.json']
    status['groups'][0]['calls']['0']['sha256']='bad'
    rejected('transfer conflicting validated pair',lambda:transfer.collect(status,out,IDENTITY,download),'conflict')


def test_p8g3_transfer_interrupted_then_recovered(tmp_path):
    remote=tmp_path/'remote';schedule,sid=setup(remote,count=1);save(remote,0,schedule,sid)
    status=snapshot(remote,schedule);out=tmp_path/'local'
    def stopped(name,path):
        if name.endswith('.npz'):
            path.write_bytes(b'partial');raise ValueError('injected transfer stop')
        shutil.copyfile(remote/Path(name).name,path)
    rejected('transfer stop mid npz',lambda:transfer.collect(status,out,IDENTITY,stopped),'transfer stop')
    assert not c.call_path(out/'results',0).exists()
    assert c.session_path(out/'results',sid).exists()
    transfer.collect(status,out,IDENTITY,lambda name,path:shutil.copyfile(remote/Path(name).name,path))
    assert len(c.available(out/'results',IDENTITY,schedule))==1


def test_p8g3_watch_salvages_and_stops(monkeypatch,tmp_path):
    from prototypes.mv_electron.gpu.p8g3_cli import watch
    events=[]
    def failed(*args):events.append('collect');raise ValueError('injected acquisition failure')
    monkeypatch.setattr(watch,'poll_collect',failed)
    monkeypatch.setattr(watch,'call',lambda args:events.append(args[0]))
    with pytest.raises(ValueError):watch.watch('session',tmp_path,IDENTITY)
    assert events==['collect','collect','stop']


def test_p8g3_poll_final_failure_still_stops(monkeypatch,tmp_path):
    from prototypes.mv_electron.gpu.p8g3_cli import watch
    import subprocess
    events=[]
    def failed(args,**kwargs):
        events.append(args[0])
        if args[0]=='exec':raise subprocess.TimeoutExpired('colab',180)
        return ''
    monkeypatch.setattr(watch,'call',failed)
    with pytest.raises(RuntimeError,match='final status'):watch.watch('session',tmp_path,IDENTITY)
    assert events==['exec']*6+['stop']


def test_p8g3_mandatory_trial_and_resend_session(tmp_path):
    from prototypes.mv_electron.gpu.p8g3_cli.start import resend_files
    root=tmp_path/'trial';schedule,sid=setup(root)
    c.drive(root,runner,IDENTITY,schedule,sid,limit=3)
    rejected('production without T4 restart trial',lambda:vm.require_trial(root,IDENTITY))
    gate=c.read_json(c.session_path(root,sid))['gate']
    resume=c.resume_check(root,runner,IDENTITY,schedule)
    sid2=c.new_session(root,IDENTITY,schedule,'T4',gate,resume)
    c.drive(root,runner,IDENTITY,schedule,sid2)
    vm.qualify_trial(root,IDENTITY,schedule,sid2,resume)
    assert vm.require_trial(root,IDENTITY)['previous_session']==sid
    files=resend_files(tmp_path,IDENTITY)
    first_pair=next(i for i,p in enumerate(files) if p.suffix=='.npz')
    assert all(p.name.startswith('session_') for p in files[:first_pair])
    c.session_path(root,sid).unlink()
    rejected('resend missing gate',lambda:resend_files(tmp_path,IDENTITY))


def test_p8g3_vm_initial_gate_stops_on_mismatch(monkeypatch):
    expected={'field':np.arange(5)}
    class Original:
        def __init__(self,*args,**kwargs):pass
        def run(self,*args,**kwargs):return {},expected
    class Candidate:
        def __init__(self,bad=False):self.bad=bad
        def run(self,*args,**kwargs):
            return values(20000),{'field':np.arange(5)+int(self.bad)},np.ones(20000),None
    monkeypatch.setattr(vm.transport.old,'GPU',Original)
    assert vm.initial_gate(None,[],{},Candidate())['status']=='pass'
    rejected('VM initial history mismatch',lambda:vm.initial_gate(None,[],{},Candidate(True)),'byte mismatch')


def test_p8g3_production_session_requires_restart_trial(tmp_path):
    gate=dict(status='pass',n=20000,seed=c.kernel.SEED2,comparison='P8g1/P8g2 byte-identical')
    rejected('production session missing mandatory trial',lambda:c.new_session(tmp_path,IDENTITY,c.spec(),'T4',gate,dict(status='new')),'mandatory')
