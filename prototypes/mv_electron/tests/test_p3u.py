"""P3u U1–U9: deterministic batches, persistence, locks and stop semantics."""
import ast
import json
from pathlib import Path
import numpy as np
import pytest
from prototypes.mv_electron import batch_runner_p3u as b, batch_kernels_p3u as k
from prototypes.mv_electron import run_p3a as a


@pytest.fixture(autouse=True)
def threads(monkeypatch):
    for key in b.THREAD_KEYS:monkeypatch.setenv(key,'1')
    monkeypatch.setenv('CHATCARLO_P3U_FAKE_FAULTS','1')


def run(work,**kw):
    args=dict(kernel='fake',params={},N=80,seed=20291630,workers=1,nbatch=8)
    args.update(kw)
    return b.run_batches(work,**args)


def equal_runs(left,right,nbatch=8):
    assert b.same(a.load(left/'result.npz'),a.load(right/'result.npz'))
    for number in range(nbatch):
        assert b.same(a.load(left/f'batch_{number:03}.npz'),a.load(right/f'batch_{number:03}.npz'))


@pytest.mark.parametrize('seed',[0,42,20291630])
def test_u1_seeds(seed):
    assert b.batch_seeds(seed,100)==[int(x.generate_state(1)[0]) for x in np.random.SeedSequence(seed).spawn(100)]


def test_u2_fake(tmp_path):
    left=tmp_path/'w1';right=tmp_path/'w2'
    slow=b.batch_seeds(20291630,8)[0]
    params=dict(sleep=.1,sleep_seed=slow)
    run(left,params=params);run(right,workers=2,params=params)
    equal_runs(left,right)
    assert bool(a.load(left/'result.npz')['fresh'])
    before=(left/'result.npz').read_bytes()
    assert run(left,params=params)==left/'result.npz'
    assert (left/'result.npz').read_bytes()==before
    assert 'workers' not in json.loads((left/'config.json').read_text())


@pytest.fixture(scope='module')
def mf_inputs():
    from prototypes.mv_electron.p2_common import TABLES
    from prototypes.mv_electron.photon_data_p2 import TABLE
    missing=[str(path) for path in (TABLE,TABLES['mf']) if not path.is_file()]
    if missing:pytest.skip('required P2/GS data absent: '+', '.join(missing))
    from prototypes.mv_electron import run_p3s
    return run_p3s.inputs('mf')


def test_u2_slab(tmp_path,mf_inputs):
    params=dict(E=10.,backend='mf',brems_mode=2,spectrum=None)
    for workers in (1,2):
        run(tmp_path/f'w{workers}',kernel='p3t_slab',params=params,N=1600,workers=workers)
    equal_runs(tmp_path/'w1',tmp_path/'w2')


@pytest.mark.parametrize('mode',[1,2])
def test_u3_p3r(tmp_path,mf_inputs,mode):
    from prototypes.mv_electron import photon_transport_p3r as r, run_p3r, run_p3s
    params=dict(E=10.,backend='mf',brems_mode=mode,spectrum=None)
    work=tmp_path/'slab'
    run(work,kernel='p3t_slab',params=params,N=1600,workers=2)
    for number,seed in enumerate(b.batch_seeds(20291630,8)):
        values=r.run_coupled(10.,200,seed,.05,.025,30.,*mf_inputs,
            annihilate_in_flight=True,brems_mode=mode,score_kerma=False,score_pair=False,
            **run_p3s.rule_kwargs('own'))
        direct={key:np.asarray(value.sum(axis=0) if key in run_p3r.KEYS[:5] else value)
                for key,value in zip(run_p3r.KEYS,values)}
        saved=a.load(work/f'batch_{number:03}.npz')
        assert b.same(direct,{key:saved[key] for key in run_p3r.KEYS},skip=())


def test_u4_resume(tmp_path):
    full=tmp_path/'full';resume=tmp_path/'resume'
    run(full,workers=2)
    assert run(resume,workers=2,stop_after=3) is None
    assert sorted(p.name for p in resume.glob('batch_*.npz'))==[f'batch_{x:03}.npz' for x in range(3)]
    before=[(resume/f'batch_{x:03}.npz').read_bytes() for x in range(3)]
    run(resume)
    equal_runs(full,resume)
    assert not bool(a.load(resume/'result.npz')['fresh'])
    assert before==[(resume/f'batch_{x:03}.npz').read_bytes() for x in range(3)]


@pytest.mark.parametrize('change',[dict(N=160),dict(seed=7),dict(params={'sleep':0.}),dict(nbatch=4),
                                    dict(params={'fingerprint':'changed'})])
def test_u5_config(tmp_path,change):
    run(tmp_path,stop_after=3)
    before=(tmp_path/'config.json').read_bytes()
    with pytest.raises(RuntimeError,match='config mismatch'):run(tmp_path,**change)
    assert (tmp_path/'config.json').read_bytes()==before


def test_u5_environment_fingerprint(tmp_path,monkeypatch):
    run(tmp_path,stop_after=3)
    original=k.KERNELS['fake']
    def changed(params):return dict(original.fingerprint(params),python='different')
    monkeypatch.setitem(k.KERNELS,'fake',original._replace(fingerprint=changed))
    with pytest.raises(RuntimeError,match='config mismatch'):run(tmp_path)


@pytest.mark.parametrize('fault',['seed','missing','shape','dtype','nan'])
def test_u5_saved_batch(tmp_path,fault):
    run(tmp_path,stop_after=3)
    path=tmp_path/'batch_001.npz';row=a.load(path)
    if fault=='seed':row['seed']+=1
    elif fault=='missing':row.pop('counts')
    elif fault=='shape':row['value']=row['value'].ravel()
    elif fault=='dtype':row['value']=row['value'].astype(np.float32)
    else:row['value'][0,0]=np.nan
    a.save(path,**row);before=path.read_bytes()
    with pytest.raises(RuntimeError):run(tmp_path)
    assert path.read_bytes()==before


@pytest.mark.parametrize('fault',['N','seed','nbatch','batch_seeds','value','missing'])
def test_u5_saved_result(tmp_path,fault):
    path=run(tmp_path);row=a.load(path)
    if fault=='missing':row.pop('counts')
    elif fault=='value':row['value'][0,0,0]+=1
    else:row[fault]+=1
    a.save(path,**row);before=path.read_bytes()
    with pytest.raises(RuntimeError):run(tmp_path)
    assert path.read_bytes()==before


def test_u6_lock(tmp_path):
    with a.job_lock(tmp_path):
        with pytest.raises(RuntimeError,match='already executing'):run(tmp_path)


@pytest.mark.parametrize('change',[dict(workers=0),dict(workers=3),dict(workers=1.5),dict(workers=True),
    dict(N=0),dict(N=-8),dict(N=81),dict(N=80.),dict(nbatch=0),dict(nbatch=8.),
    dict(stop_after=0),dict(stop_after=9),dict(kernel='unknown')])
def test_u7_arguments(tmp_path,change):
    with pytest.raises(ValueError):run(tmp_path,**change)


@pytest.mark.parametrize('fault',['mode','unknown','size'])
def test_u7_slab(tmp_path,fault):
    params=dict(E=10.,backend='mf',brems_mode=2,spectrum=None)
    N=1600
    if fault=='mode':params['brems_mode']=0
    elif fault=='unknown':params['unknown']=1
    else:N=800+8
    with pytest.raises(ValueError):run(tmp_path,kernel='p3t_slab',params=params,N=N)


@pytest.mark.parametrize('key',b.THREAD_KEYS)
def test_u7_threads(tmp_path,monkeypatch,key):
    monkeypatch.setenv(key,'2')
    with pytest.raises(ValueError,match=key):run(tmp_path)


@pytest.mark.parametrize('fault',['nan_seed','unhealthy_seed'])
def test_u8_terminal(tmp_path,fault):
    params={fault:b.batch_seeds(20291630,8)[3]}
    with pytest.raises(RuntimeError):run(tmp_path,params=params)
    terminal=json.loads((tmp_path/'terminal.json').read_text())
    assert terminal['batch']==3
    assert terminal['reason']==('non-finite output' if fault=='nan_seed' else 'unhealthy output')
    before=[(tmp_path/f'batch_{i:03}.npz').read_bytes() for i in range(3)]
    assert not (tmp_path/'batch_003.npz').exists()
    with pytest.raises(RuntimeError,match='previously failed'):run(tmp_path,params=params)
    assert before==[(tmp_path/f'batch_{i:03}.npz').read_bytes() for i in range(3)]


@pytest.mark.parametrize('fault',['exception_seed','exit_seed','worker_fingerprint'])
def test_u8_interrupt(tmp_path,monkeypatch,fault):
    params={fault:'different' if fault=='worker_fingerprint' else b.batch_seeds(20291630,8)[3]}
    work=tmp_path/'resume';full=tmp_path/'full'
    with pytest.raises(RuntimeError):run(work,params=params)
    assert not (work/'terminal.json').exists()
    monkeypatch.setenv('CHATCARLO_P3U_FAKE_FAULTS','0')
    run(work,params=params);run(full,params=params)
    equal_runs(work,full)
    assert not bool(a.load(work/'result.npz')['fresh'])


def test_u8_save_interruption(tmp_path,monkeypatch):
    original=a.save
    def fail(path,**values):
        if Path(path).name=='batch_003.npz':raise OSError('injected save failure')
        original(path,**values)
    with monkeypatch.context() as patch:
        patch.setattr(a,'save',fail)
        with pytest.raises(RuntimeError,match='injected save failure'):run(tmp_path/'resume')
    assert not (tmp_path/'resume/terminal.json').exists()
    run(tmp_path/'resume');run(tmp_path/'full')
    equal_runs(tmp_path/'resume',tmp_path/'full')


def test_u8_waits_before_unlock(tmp_path,monkeypatch):
    # Exercise the real pool's shutdown ordering: the lock must still be held
    # while shutdown waits, including after a terminal result.
    original=b.ProcessPoolExecutor
    saved_before_wait={}
    class CheckedPool(original):
        def shutdown(self,*args,**kw):
            assert kw['wait'] is True
            with pytest.raises(RuntimeError,match='already executing'):
                with a.job_lock(tmp_path):pass
            saved_before_wait.update({p.name:p.read_bytes() for p in tmp_path.glob('batch_*.npz')})
            return super().shutdown(*args,**kw)
    monkeypatch.setattr(b,'ProcessPoolExecutor',CheckedPool)
    params=dict(nan_seed=b.batch_seeds(20291630,8)[0],sleep=.2)
    with pytest.raises(RuntimeError):run(tmp_path,params=params,workers=2)
    assert saved_before_wait=={p.name:p.read_bytes() for p in tmp_path.glob('batch_*.npz')}
    assert not (tmp_path/'batch_000.npz').exists()
    with a.job_lock(tmp_path):pass


def test_u9_no_deletion():
    banned={'unlink','rmtree','remove','rmdir','removedirs'}
    for name in ('batch_runner_p3u.py','batch_kernels_p3u.py','run_p3u.py'):
        tree=ast.parse((a.CODE/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                func=node.func
                assert not isinstance(func,ast.Name) or func.id not in banned
                assert not isinstance(func,ast.Attribute) or func.attr not in banned
            if isinstance(node,ast.Name):assert node.id!='subprocess'
            if isinstance(node,ast.Import):assert all(x.name!='subprocess' for x in node.names)
            if isinstance(node,ast.ImportFrom):assert node.module!='subprocess'
            if isinstance(node,ast.Attribute):
                assert not (isinstance(node.value,ast.Name) and node.value.id=='os' and node.attr=='system')


def test_same_bytes():
    assert not b.same({'x':np.array([0.])},{'x':np.array([-0.])})
    assert not b.same({'x':np.array([1],dtype=np.int64)},{'x':np.array([1],dtype=np.int32)})
    assert not b.same({'x':np.ones((1,2))},{'x':np.ones(2)})
    assert not b.same({'seconds':1.},{})
    assert b.same({'seconds':1.,'x':np.ones(2)},{'seconds':2.,'x':np.ones(2)})
