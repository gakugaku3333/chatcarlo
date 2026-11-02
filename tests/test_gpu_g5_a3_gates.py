"""A3 gate orchestration without Colab, CUDA, private tables or cache writes."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

HERE=Path(__file__).resolve().parents[1]/'docs/validation/gpu/g5'


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))
    import a3_entry_speed as entry
    import a3_t4_gates as gates
    import a3_t4_worker as worker
    return entry,gates,worker


@pytest.mark.parametrize('old,new,status',[(1.,1.0571,'pass'),(1.,1.05710001,'fail'),(2.,1.,'pass')])
def test_fixed_ratio(modules,old,new,status):
    assert modules[0].ratio(old,new)['status']==status


@pytest.mark.parametrize('value',[0,-1,float('nan'),float('inf')])
def test_invalid_timing(modules,value):
    with pytest.raises(ValueError):modules[0].ratio(1,value)


def proof():
    cold=dict(cache_dir='/vm/old',cache_after={'x.cubin':'hash'},compiler=dict(instrumented=True,
        rows=[dict(kind='nvrtc',source_nonempty=True,seconds=2.)]))
    warm=dict(cache_dir='/vm/old',cache_before={'x.cubin':'hash'},cache_after={'x.cubin':'hash'},
        compiler=dict(instrumented=True,rows=[dict(kind='RawKernel.compile',seconds=.01)]))
    return cold,warm


def test_cache_proof_requires_compilation_and_binary_evidence(modules):
    a,b=proof();assert modules[0].cache_used(a,b)['status']=='pass'
    assert modules[0].cache_used(a,b)['second_compile_s']==.01


@pytest.mark.parametrize('defect',['no_trace','no_files','different_cache','changed_binary','nvrtc_recompiled','no_second_compile','no_cold_compile'])
def test_cache_missing_or_recompiled_is_undetermined(modules,defect):
    a,b=proof()
    if defect=='no_trace':b['compiler']={}
    if defect=='no_files':a['cache_after']={}
    if defect=='different_cache':b['cache_dir']='/vm/new'
    if defect=='changed_binary':b['cache_after']={'x.cubin':'changed'}
    if defect=='nvrtc_recompiled':b['compiler']['rows'].append(dict(kind='nvrtc',source_nonempty=True,seconds=1.))
    if defect=='no_second_compile':b['compiler']['rows']=[]
    if defect=='no_cold_compile':a['compiler']['rows']=[]
    assert modules[0].cache_used(a,b)['status']=='undetermined'


def test_whole_main_timing_includes_work_after_transport(modules):
    ticks=iter([10.,25.]);calls=[]
    def main(argv):calls.extend(['parse','bundle','transport','collect','verify','analyse']);return 0
    assert modules[0].timed_main(main,[],clock=lambda:next(ticks))==15
    assert calls[-1]=='analyse'
    with pytest.raises(RuntimeError):modules[0].timed_main(lambda argv:1,[])


def test_shared_client_never_creates_or_stops_vm(modules,tmp_path):
    e=modules[0];owner=SimpleNamespace(raw=Mock(return_value='MV_ENTRY_JSON {"status":"ready"}'),call=Mock())
    client=e.SharedClient(owner,'/vm/run','/vm/cache')
    client.call('new','-s','mventry','--gpu','T4');client.call('stop','-s','mventry');client.new();client.stop()
    owner.call.assert_not_called()
    assert client.execute('pass')==dict(status='ready')
    client.upload('local','/vm/target');owner.call.assert_called_once_with('upload','local','/vm/target')
    client.call('download','-s','mventry','/vm/source',str(tmp_path/'dst'))
    assert owner.call.call_args.args==('download','/vm/source',str(tmp_path/'dst'))


def test_instrumentation_reaches_fresh_child_environment(modules,monkeypatch):
    e=modules[0];client=e.SharedClient(None,'/vm/new-cached','/vm/new-cache')
    calls=[]
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**kw:calls.append((a,kw)))
    code=client.instrument("import subprocess\nsubprocess.Popen(['worker'],cwd='/source')")
    exec(code,{})
    env=calls[0][1]['env']
    assert env['CUPY_CACHE_DIR']=='/vm/new-cache'
    assert env['A3_COMPILE_LOG']=='/vm/new-cached/compile.json'
    assert env['PYTHONPATH'].split(':')[0]=='/vm/new-cached/instrument'


@pytest.mark.parametrize('version',['old','new'])
def test_inject_client_into_real_main_call_only(modules,tmp_path,version):
    e=modules[0];calls=[]
    colab=SimpleNamespace(REMOTE='/frozen',POLL_SECONDS=600)
    def run(*args,**kw):calls.append(kw)
    colab.run=run
    def main(argv):
        assert argv==['run','scene.yaml','--out',str(tmp_path)]
        if version=='old':assert colab.POLL_SECONDS==60 and colab.REMOTE=='/fresh'
        colab.run('bundle','results');calls.append('analysis');return 0
    cli=SimpleNamespace(main=main);client=SimpleNamespace(remote='/fresh')
    e.cli_measure(version,'scene.yaml',tmp_path,client,old_cli=cli,old_colab=colab,new_cli=cli,new_colab=colab)
    assert calls[0]['client'] is client
    if version=='new':assert calls[0]['poll_seconds']==60
    assert calls[-1]=='analysis' and colab.run is run
    assert colab.REMOTE=='/frozen' and colab.POLL_SECONDS==600


def test_output_path_rejects_repository_and_cache(modules,tmp_path):
    _,g,_=modules
    for path in (g.ROOT/'private',Path.home()/'.cache'/'a3'):
        with pytest.raises(ValueError):g.writable(path)
    assert g.writable(tmp_path)==tmp_path.resolve()


def test_cache_write_guard_allows_reads_and_rejects_mutation(modules):
    import os
    guard=modules[1].no_cache_writes;p=str(Path.home()/'.cache'/'a3-never-created')
    guard('open',(p,'r',os.O_RDONLY))
    for event,args in [('open',(p,'w',os.O_WRONLY|os.O_CREAT)),('os.mkdir',(p,0o777,-1)),
                       ('os.rename',('/tmp/safe',p,-1,-1)),('os.remove',(p,-1))]:
        with pytest.raises(PermissionError):guard(event,args)


def test_prepare_identity_change_rejects_before_approval_or_vm(modules,tmp_path,monkeypatch):
    _,g,_=modules
    (tmp_path/'prepared.json').write_text(json.dumps(dict(sources={},histories=4_600_000,poll_seconds=60)))
    monkeypatch.setattr(g,'identity',lambda:{'final':'hash'})
    with pytest.raises(ValueError,match='changed'):g.require_prepared(tmp_path)


@pytest.mark.parametrize('fail_at',['h2','h5','speed',None])
def test_single_session_lifecycle_and_stop_on_failure(modules,tmp_path,monkeypatch,fail_at):
    _,g,_=modules;calls=[]
    record=dict(inputs={'therapy_bundle':str(tmp_path/'sealed')},sources={'code':'hash'})
    monkeypatch.setattr(g,'require_prepared',lambda p:record)
    monkeypatch.setattr(g,'identity',lambda:record['sources'])
    monkeypatch.setattr(g,'payload',lambda *a:tmp_path/'payload.tar')
    class Owner:
        def __init__(self,session,out):self.session=session
        def call(self,action,*a):calls.append(action)
    monkeypatch.setattr(g,'Owner',Owner);monkeypatch.setattr(g,'upload_payload',lambda *a:None)
    def collect(owner,remote,task,out):
        calls.append(task)
        if task==fail_at:raise RuntimeError('planted')
        return dict(status='pass')
    monkeypatch.setattr(g,'collect_task',collect)
    monkeypatch.setattr(g,'judge_h5',lambda *a:dict(status='pass'))
    monkeypatch.setattr(g,'g4',lambda *a:dict(status='pass'))
    monkeypatch.setattr(g,'entry_speed',lambda *a:dict(status='undetermined'))
    if fail_at:
        with pytest.raises(RuntimeError,match='planted'):g.run(SimpleNamespace(output=tmp_path))
    else:
        assert g.run(SimpleNamespace(output=tmp_path))['status']=='undetermined'
    assert calls.count('new')==calls.count('stop')==1 and calls[-1]=='stop'
    assert (tmp_path/'summary.json').exists()
    with pytest.raises(ValueError,match='single valid run'):g.run(SimpleNamespace(output=tmp_path))


def test_wrong_fault_rejection_does_not_pass(modules):
    reject=modules[2].reject
    with pytest.raises(AssertionError,match='fault accepted'):reject(lambda:None,'half')
    with pytest.raises(AssertionError,match='unexpected rejection'):reject(lambda:(_ for _ in ()).throw(ValueError('other')),'half')
    assert reject(lambda:(_ for _ in ()).throw(ValueError('native half mismatch')),'half')['status']=='pass'


def test_four_entry_measurements_use_distinct_results_and_version_caches(modules,tmp_path,monkeypatch):
    e,g,_=modules;calls=[];phase={'n':0}
    # Synthetic CLI timings must not include the machine's real NPZ I/O time.
    monkeypatch.setattr(g,'time',SimpleNamespace(perf_counter=lambda:0.))
    owner=SimpleNamespace(rows=[],raw=lambda code:None)
    def measure(version,scene,out,client):
        calls.append((version,Path(out),client.remote,client.cache));owner.rows.append(dict(operation='exec',seconds=.1,retry=version=='new'))
        return 2. if version=='old' else 2.1
    monkeypatch.setattr(g,'cli_measure',measure)
    def evidence(*a):
        phase['n']+=1
        # Cold workers compile, cached workers use files without compilation.
        warm=phase['n']>4
        rows=[dict(kind='RawKernel.compile',seconds=.01)]
        if not warm:rows.append(dict(kind='nvrtc',source_nonempty=True,seconds=1.))
        return dict(files={'a.cubin':'sha'},compiler=dict(instrumented=True,rows=rows))
    monkeypatch.setattr(g,'evidence',evidence)
    import a3_judge
    monkeypatch.setattr(a3_judge,'judge10',lambda field,n:dict(status='report-only'))
    import numpy as np
    for phase_name in ('cold','cached'):
        p=tmp_path/'entry/new'/phase_name/'results';p.mkdir(parents=True)
        np.savez(p/'dose.npz',field_MeV=np.zeros((100,3420)))
    result=g.entry_speed(owner,'/vm/session',tmp_path)
    assert [v for v,*_ in calls]==['old','new','old','new']
    assert all(row['transport_retried']==key.startswith('new') for key,row in result['rows'].items())
    assert all(row['transport_retry_attempts']==int(key.startswith('new')) for key,row in result['rows'].items())
    assert len({p for _,p,_,_ in calls})==len({r for _,_,r,_ in calls})==4
    assert calls[0][3]==calls[2][3] and calls[1][3]==calls[3][3] and calls[0][3]!=calls[1][3]
    assert result['status']=='pass' and all(r['histories']==4_600_000 for r in result['rows'].values())


def test_disk_floor_preserves_three_gib(modules,tmp_path,monkeypatch):
    _,g,_=modules
    monkeypatch.setattr(g.shutil,'disk_usage',lambda p:SimpleNamespace(free=4*1024**3))
    g.reserve(tmp_path,1024**3)
    with pytest.raises(RuntimeError,match='3 GiB'):g.reserve(tmp_path,1024**3+1)


def test_therapy_speed_reuses_a2_builder_with_final_launcher_and_one_valid_round(modules,tmp_path,monkeypatch):
    e,g,w=modules
    import a0_speed,a2_speed,a2_gates,a2_common
    import chatcarlo.gpu.therapy_cuda as native
    calls=[]
    class Final:
        kernel=object()
        def run(self,n,seed):calls.append(('new',n,seed));return dict(kernel_s=1.)
    monkeypatch.setattr(native,'T4',Final)
    def build(cp,extra,cid):
        assert a2_speed.NewCore is Final
        assert extra['p8g2']==tmp_path/'bundles/p8g2'
        return Final()
    monkeypatch.setattr(a2_speed,'build_new',build)
    monkeypatch.setattr(a0_speed,'construct',lambda cp,c,b:dict(kernel=object()))
    monkeypatch.setattr(a0_speed,'resources',lambda *a:{})
    def measured(c,state,n):calls.append(('old',n,c['seed']));return dict(kernel_s=1.,health=dict(status='pass'))
    monkeypatch.setattr(a0_speed,'measure',measured)
    monkeypatch.setattr(a2_gates,'batch_health',lambda *a:dict(status='pass'))
    monkeypatch.setattr(a2_gates,'load_water',lambda c:dict(arrays={}))
    monkeypatch.setattr(a2_common,'field_geometry',lambda:None)
    r=w.therapy_speed(None,tmp_path,[],{},tmp_path)
    assert len(calls)==16 and [x[0] for x in calls]==['old','new']*8
    valid=[row for row in r['rows'] if row['phase']=='valid']
    assert len(valid)==8 and len({row['task'] for row in valid})==8
    assert set(r['ratio'])=={'p7A','p7B','p7C','stack'}
    assert all(x==1. for x in r['ratio'].values())


@pytest.mark.parametrize('action,idempotent,code,tries',[
    ('upload',False,None,6),('download',False,None,6),('stop',False,None,6),
    ('exec',True,'print("status")',6),('exec',False,'print("status")',1),
    ('exec',True,'subprocess.Popen(["worker"])',1),('new',False,None,1)])
@pytest.mark.parametrize('error',[subprocess.TimeoutExpired,subprocess.CalledProcessError])
def test_owner_retry_scope_and_attempt_evidence(modules,tmp_path,monkeypatch,action,idempotent,code,tries,error):
    g=modules[1];calls=[];sleeps=[]
    def failed(*a,**kw):
        calls.append((a,kw))
        if error is subprocess.TimeoutExpired:raise error('colab',300)
        raise error(1,'colab',stderr='transport failure')
    monkeypatch.setattr(g.subprocess,'run',failed)
    monkeypatch.setattr(g.time,'sleep',sleeps.append)
    owner=g.Owner('fake',tmp_path)
    with pytest.raises(error):owner.call(action,code=code,idempotent=idempotent)
    assert len(calls)==len(owner.rows)==tries
    assert sleeps==[60]*(tries-1)
    assert [r['attempt'] for r in owner.rows]==list(range(1,tries+1))
    assert [r['retry'] for r in owner.rows]==[False]+[True]*(tries-1)
    assert all(r['error']==error.__name__ for r in owner.rows)
    assert json.loads((tmp_path/'transport_timings.json').read_text())==owner.rows


def test_owner_retry_recovers_and_keeps_success(modules,tmp_path,monkeypatch):
    g=modules[1];responses=iter([subprocess.TimeoutExpired('colab',300),subprocess.CalledProcessError(1,'colab'),'ok'])
    def call(*a,**kw):
        value=next(responses)
        if isinstance(value,Exception):raise value
        return SimpleNamespace(stdout=value)
    monkeypatch.setattr(g.subprocess,'run',call);monkeypatch.setattr(g.time,'sleep',lambda n:None)
    owner=g.Owner('fake',tmp_path)
    assert owner.call('download','remote','local')=='ok'
    assert [r['error'] for r in owner.rows]==['TimeoutExpired','CalledProcessError',None]


def test_shared_client_retries_only_exact_status_templates(modules):
    from prototypes.mv_electron.entry import colab_cli
    from chatcarlo.gpu import therapy_colab
    e=modules[0];owner=SimpleNamespace(raw=Mock(return_value=''))
    client=e.SharedClient(owner,'/fake')
    for code in (colab_cli.poll_code(),therapy_colab.listing('/fake')):
        client.call('exec',code=code)
        assert owner.raw.call_args.kwargs=={'idempotent':True}
    for code in ('subprocess.Popen(["worker"])','Path("new").mkdir()',colab_cli.bootstrap({},'session')):
        client.call('exec',code=code)
        assert not owner.raw.call_args.kwargs


def test_payload_streams_inputs_and_reserves_before_staging(modules,tmp_path,monkeypatch):
    import tarfile
    g=modules[1];root=tmp_path/'repo';root.mkdir();out=tmp_path/'out';out.mkdir()
    (root/'chatcarlo').mkdir();(root/'chatcarlo/runtime.py').write_text('runtime')
    inputs={}
    for key in ('old_diagnostic','p7','p8g2','therapy_bundle','g2_reference'):
        d=tmp_path/key;d.mkdir();(d/'sealed.npz').write_bytes(key.encode());inputs[key]=str(d)
    (out/'H5_jobs/gate1/jobs').mkdir(parents=True)
    (out/'H5_jobs/gate1/jobs/False_s1_00.json').write_text('{}')
    generated=out/'generated';(generated/'diagnostic/new').mkdir(parents=True)
    (generated/'diagnostic/new/job.json').write_text('{}')
    for c in 'ABC':(generated/f'water_{c}.npz').write_bytes(b'water')
    monkeypatch.setattr(g,'ROOT',root);monkeypatch.setattr(g,'verify_old_diagnostic',lambda p:None)
    small=g.tree_bytes(generated);reservations=[]
    def reserve(path,extra):
        assert not (out/'payload').exists() and not (out/'payload.tar').exists()
        reservations.append(extra)
    monkeypatch.setattr(g,'reserve',reserve)
    # A large source need not be copied to stage. Only small generated inputs may be copied.
    real=g.shutil.copy2
    def copied(src,dst):
        assert generated in Path(src).parents
        return real(src,dst)
    monkeypatch.setattr(g.shutil,'copy2',copied)
    archive=g.payload(out,dict(inputs=inputs))
    assert reservations==[archive.stat().st_size+small+2*1024**3]
    assert {str(p.relative_to(out/'payload')) for p in (out/'payload').rglob('*') if p.is_file()}=={
        'diagnostic/new/job.json',*{f'water_{c}.npz' for c in 'ABC'}}
    with tarfile.open(archive) as t:
        for name,key in [('diagnostic/old','old_diagnostic'),('bundles/p7','p7'),('bundles/p8g2','p8g2'),('therapy_bundle','therapy_bundle')]:
            assert t.extractfile('inputs/'+name+'/sealed.npz').read()==key.encode()
        assert t.extractfile('inputs/h5/gate1/jobs/False_s1_00.json').read()==b'{}'
        assert all(m.isfile() for m in t.getmembers())


def test_low_space_leaves_no_payload_directory_or_vm_marker(modules,tmp_path,monkeypatch):
    g=modules[1];out=tmp_path
    inputs={key:str(out) for key in ('old_diagnostic','p7','p8g2','therapy_bundle')}
    (out/'H5_jobs').mkdir();(out/'generated/diagnostic/new').mkdir(parents=True)
    for c in 'ABC':(out/'generated'/f'water_{c}.npz').write_bytes(b'small')
    monkeypatch.setattr(g,'ROOT',out/'empty_repo');monkeypatch.setattr(g,'verify_old_diagnostic',lambda p:None)
    monkeypatch.setattr(g.shutil,'disk_usage',lambda p:SimpleNamespace(free=3*1024**3))
    with pytest.raises(RuntimeError,match='3 GiB'):g.payload(out,dict(inputs=inputs))
    assert not (out/'payload').exists() and not (out/'payload.tar').exists() and not (out/'STARTED.json').exists()


def test_pax_tar_estimate_matches_real_archive(modules,tmp_path):
    import tarfile
    g=modules[1];p=tmp_path/('x'*150);p.write_bytes(b'x'*513)
    files={'inputs/'+p.name:p};expected=g.tar_bytes(files)
    archive=tmp_path/'actual.tar'
    with tarfile.open(archive,'w') as t:
        for name,path in files.items():t.add(path,arcname=name,recursive=False)
    assert archive.stat().st_size==expected


def test_upload_reuses_one_local_fragment_with_numbered_remote_names(modules,tmp_path):
    g=modules[1];archive=tmp_path/'payload.tar'
    block=20*1024*1024;archive.write_bytes(b'a'*block+b'b'*block+b'last')
    transfers=[]
    class Owner:
        def json(self,code):return dict(done=True,alive=False,log='')
        def raw(self,code):pass
        def call(self,action,local,remote):
            transfers.append((Path(local),remote,g.sha(local),Path(local).stat().st_size))
    g.upload_payload(Owner(),archive,'/fake')
    assert len({p for p,*_ in transfers})==1
    assert [r for _,r,*_ in transfers]==['/fake/upload_0000','/fake/upload_0001','/fake/upload_0002']
    assert [size for *_,size in transfers]==[block,block,4]
    assert len(list(tmp_path.glob('upload_*')))==1


def test_host_h5_uses_existing_remote_judges_and_output_scratch(modules,tmp_path,monkeypatch):
    import a1_gate1,a1_g2vox,tempfile
    g=modules[1];calls=[]
    def one(out,summary,**kw):
        calls.append(('g1',out,summary,kw));assert Path(tempfile.gettempdir())==tmp_path/'H5/tmp'
        return dict(status='pass',production_conditions=True)
    def two(ref,out,summary,**kw):
        calls.append(('g2',out,summary,kw));assert ref==tmp_path/'reference'
        return dict(status='pass')
    monkeypatch.setattr(a1_gate1,'run',one);monkeypatch.setattr(a1_g2vox,'run',two)
    assert g.judge_h5(tmp_path,dict(inputs={'g2_reference':str(tmp_path/'reference')}))['status']=='pass'
    assert [c[3] for c in calls]==[dict(device='cuda',remote=tmp_path/'H5/gate1'),dict(device='cuda',remote=tmp_path/'H5/g2vox')]
    assert all(tmp_path in out.parents and tmp_path in summary.parents for _,out,summary,_ in calls)


# Each case starts a fresh Python process so prior host-side imports cannot hide
# a forbidden dependency. Only hardware/private-input operations are replaced.
VM_BLOCKED_SCRIPT = r'''
import importlib.abc,sys,json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in ('xraylib','spekpy'):raise AssertionError('forbidden VM import: '+name)
sys.meta_path.insert(0,Block())
sys.path.insert(0,sys.argv[1])
import a3_t4_worker as w
from chatcarlo.gpu import therapy_inputs,therapy,therapy_checks,therapy_cuda,therapy_observation as obs
import a1_speed,a2_gates,a2_speed,a0_speed
from prototypes.mv_electron.gpu import p8g2_colab
base=Path(sys.argv[2]);base.mkdir();out=base/'out';out.mkdir()
task=sys.argv[3]
if task=='h2':
 cp=NS()
 def kernel(*a,options=(),**kw):
  if '-DG5_SET_ID=99' in options or '-DG5_SET_TAG=0ULL' in options:raise ValueError('mismatch')
  return lambda *a,**kw:None
 cp.RawKernel=kernel;sys.modules['cupy']=cp
 class GPU:
  def __init__(self,*a,**kw):
   self.kernel=cp.RawKernel(options=('-DG5_SET_ID=1','-DG5_SET_TAG=1ULL'));self.declared={}
  def run(self,*a):self.kernel(None,None,[0]*22)
 with patch.object(therapy_inputs,'load',return_value=([],{},None,None,None)),patch.object(therapy,'CPU'),patch.object(therapy_checks,'run',return_value={}),patch.object(therapy_cuda,'T4',GPU),patch.object(obs,'descriptor',return_value={'half':[5.,5.]}),patch.object(obs,'gpu_observation',side_effect=lambda gpu,args:{'half':[float(args[19]),5.]}),patch.object(p8g2_colab,'load',return_value=(None,[],{})),patch.object(a2_gates,'checks',return_value=None):
  assert w.h2(base,out)['status']=='pass'
elif task=='h5':
 from chatcarlo.gpu import bundle,core,g2_core
 from chatcarlo.gpu import source_host
 # Real worker_batches writes its NPZ/receipts in the unchanged A1 format.
 for gate,name,index,grid in [('gate1','s1','00',False),('g2vox','water','0',True)]:
  d=base/'h5'/gate/'jobs';d.mkdir(parents=True)
  for retry in ('False','True'):
   job=dict(bundle_manifest={},n_histories=4,batch_size=2,seed=42,source_seed=43)
   if grid:job['grid']=dict(origin_cm=[0,0,0],shape=[1,1,1],voxel_size_cm=1.,layout_version='g2-grid-v1')
   (d/f'{retry}_{name}_{index}.json').write_text(json.dumps(job))
 settings=dict(physics_set={},fluorescence=True)
 class Core:
  def __init__(self,*a,**kw):pass
 with patch.object(bundle,'load',return_value=({}, {},None,settings,{})),patch.object(core,'Core',Core),patch.object(g2_core,'Core',Core),patch.object(source_host,'sample_source_photons',return_value=(None,None,None)),patch.object(a1_speed,'checked_run',return_value=({'values':np.ones(2)},{'status':'pass','histories':2})):
  assert w.h5(base,out)['status']=='pass'
 for gate,target in [('gate1','False_s1_00'),('g2vox','False_water/0')]:
  assert json.loads((out/gate/target/'done.json').read_text())['status']=='complete'
  assert len(list((out/gate/target).glob('batch_*.npz')))==2
elif task=='speed':
 # Exercise the real orchestration and diagnostic construct imports. Native
 # Core and sealed bundle loading are hardware/private-data boundaries.
 realsha=w.sha
 for cid in ('g2b','g4'):
  d=base/'diagnostic/old'/cid;d.mkdir(parents=True)
  (d/'manifest.json').write_text(json.dumps(dict(format='chatcarlo-g1-v1',files={})))
 # ROOT's old package install is redirected to this test output, never repository.
 w.ROOT=base
 cfg=json.loads((w.HERE/'a1_speed_configs.json').read_text())
 old_summary=dict(status='pass')
 def diagnostic(args):
  # Exercise construct() with a tiny fake driver (no exported private tables).
  from chatcarlo.gpu import bundle,core,g2_core,source_host
  class Driver:
   def __init__(self,*a,**kw):self.native=self.fn=NS();self.cp=NS()
  for c in cfg['configurations']:
   detector=c['detector'];packed=np.array([]) if detector is None else np.r_[detector['center_cm'],detector['normal'],detector['u_axis'],np.cross(detector['normal'],detector['u_axis']),detector['size_cm'],detector['shape']]
   load=({}, {},None,dict(scene_sha256=c['scene_model_sha256'],fluorescence=c['flags']['fluorescence'],physics_set={}),{})
   load[0]['detector']=packed
   with patch.object(bundle,'load',return_value=load),patch.object(core,'Core',Driver),patch.object(g2_core,'Core',Driver):
    a1_speed.construct(c,'new',base/'diagnostic',device='cpu')
  w.write_json(args.output/'measurements.json',dict(summary=old_summary))
 with patch.object(w,'sha',side_effect=lambda p:json.loads((w.HERE/'a1_run/t4_speed/job.json').read_text())['bundles']['old/'+Path(p).parent.name]['manifest.json']['sha256'] if Path(p).name=='manifest.json' else realsha(p)),patch.object(a1_speed,'speed',side_effect=diagnostic),patch.object(p8g2_colab,'load',return_value=(None,[],{})),patch.object(w,'therapy_speed',return_value=dict(ratio={'p7A':1.})),patch.object(w,'oldnew',return_value={}):
  sys.modules['cupy']=NS()
  assert w.speed(base,out)['status']=='pass'
elif task=='therapy_speed':
 import a2_common
 class Final:
  kernel=object()
  def run(self,n,seed):return dict(kernel_s=1.)
 def build(cp,extra,cid):
  assert a2_speed.NewCore is Final
  return Final()
 with patch.object(therapy_cuda,'T4',Final),patch.object(a2_speed,'build_new',side_effect=build),patch.object(a0_speed,'construct',return_value=dict(kernel=object())),patch.object(a0_speed,'resources',return_value={}),patch.object(a0_speed,'measure',return_value=dict(kernel_s=1.)),patch.object(a2_gates,'batch_health',return_value={}),patch.object(a2_gates,'load_water',return_value=dict(arrays={})):
  result=w.therapy_speed(None,base,[],{},out)
  assert len(result['rows'])==16
  assert set(result['ratio'])=={'p7A','p7B','p7C','stack'}
elif task=='oldnew':
 import a2_judge as J
 from prototypes.mv_electron import analysis_p7
 from prototypes.mv_electron.gpu import p7_colab,p8f_colab,p8f_core
 arrays=dict(sizes=np.array([1,1,1,1,1]))
 class Old:
  def run(self,*a,**kw):return dict(p8_batches=np.ones((100,1))),None,np.ones(2),None
  def measure(self,*a,**kw):return dict(regions=np.ones((100,3420))),None
 class Final:
  def __init__(self,*a,**kw):pass
  def run(self,*a,**kw):return dict(groups=np.ones((100,1)),field=np.ones(3420))
 with patch.object(therapy_cuda,'T4',Final),patch.object(a0_speed,'construct',return_value=dict(obj=Old())),patch.object(a2_gates,'load_water',return_value=dict(arrays=arrays,fg=None,half=(5.,5.),ssd=100.)),patch.object(a2_gates,'batch_health',return_value={}),patch.object(p8f_colab,'run_health',return_value=True),patch.object(p7_colab.h,'health',return_value={}),patch.object(p7_colab.core,'require'),patch.object(J,'stack_dose',return_value={}),patch.object(J,'p7_expand',return_value={}),patch.object(analysis_p7,'dose',return_value={}),patch.object(analysis_p7,'compare',return_value={}),patch.object(p8f_core,'judge',return_value={}):
  result=w.oldnew(None,base,[],arrays,out)
  assert set(result)=={'p7A','p7B','p7C','p8g2'}
elif task=='diagnostic_vm':
 from chatcarlo.gpu import bundle,core,colab_driver,physics_sets
 source=dict(position=[0,0,0],direction=[0,0,1],field=dict(shape='parallel',size_cm=[1,1]))
 settings=dict(physics_set=physics_sets.declaration(),fluorescence=True)
 job=dict(mode='batches',device='cuda',seed=12,source_seed=13,n_histories=4,batch_size=2)
 path=base/'job.json';path.write_text(json.dumps(job))
 class Core:
  def __init__(self,*a,**kw):pass
  def run(self,*a,**kw):return dict(values=np.ones(2)),{}
 with patch.object(bundle,'load',return_value=({},source,(np.array([60.]),np.array([1.])),settings,{})),patch.object(core,'Core',Core),patch.object(colab_driver.subprocess,'run'):
  colab_driver.worker(base,path,out)
  colab_driver.source_reference(base,path,out)
 assert (out/'done.json').exists() and (out/'source_reference.json').exists()
elif task=='therapy_vm':
 from chatcarlo.gpu import therapy_colab,therapy_entry,contracts,therapy_run
 sys.modules['cupy']=NS()
 cond=dict(spectrum='mohan10',seed=1)
 with patch.object(therapy_entry,'verify',return_value=dict(condition=cond)),patch.object(contracts,'require'),patch.object(therapy_inputs,'load',return_value=([],{},None,None,None)),patch.object(therapy,'CPU'),patch.object(therapy_cuda,'T4'),patch.object(therapy_checks,'run'),patch.object(therapy_run,'run',side_effect=lambda path,*a:Path(path).mkdir()) as production:
  therapy_colab.worker(base)
  production.assert_called_once()
 assert (base/'results/done.json').exists()
elif task=='old_entry_vm':
 from prototypes.mv_electron.entry import vm_driver,gpu_adapter
 from prototypes.mv_electron.gpu import p5b_colab
 sys.modules['cupy']=NS()
 cond=dict(engine='colab_t4',batch_histories=2,seed=1)
 class ReachedGPU(Exception):pass
 def hardware(*a,**kw):raise ReachedGPU()
 with patch.object(vm_driver.bundle,'load',return_value=({},dict(condition_hash='c',materials_hash='m'))),patch.object(vm_driver.bundle,'read_json',side_effect=lambda p:cond if p.name=='condition.json' else {}),patch.object(vm_driver.bundle,'existing',return_value=set()),patch.object(vm_driver,'reject_cupy12'),patch.object(p5b_colab,'environment',return_value=dict(gpu='T4')),patch.object(gpu_adapter,'GPU',side_effect=hardware):
  try:vm_driver.run(base,out)
  except ReachedGPU:pass
  else:raise AssertionError('GPU boundary not reached')
elif task=='imports':
 # Additional VM worker/bootstrap routes; no host physics bakes/imports.
 from prototypes.mv_electron.entry import vm_driver
 from chatcarlo.gpu import therapy_colab,colab_driver,therapy_entry
 from prototypes.mv_electron import run_p8e,analysis_p7
 from prototypes.mv_electron.gpu import p7_colab,p8f_core
 from chatcarlo.gpu import source_host
 source_host.sample_source_photons(dict(position=[0,0,0],direction=[0,0,1],field=dict(shape='parallel',size_cm=[1,1])),2,np.random.default_rng(1),(np.array([60.]),np.array([1.])))
else:raise AssertionError(task)
assert not any(n.split('.')[0] in ('xraylib','spekpy') for n in sys.modules)
'''


@pytest.mark.parametrize('task',['h2','h5','speed','therapy_speed','oldnew','diagnostic_vm','therapy_vm','old_entry_vm','imports'])
def test_vm_routes_without_xraylib_or_spekpy(modules,tmp_path,task):
    import os
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(tmp_path/'mpl'),NUMBA_CACHE_DIR=str(tmp_path/'numba'))
    result=subprocess.run([sys.executable,'-c',VM_BLOCKED_SCRIPT,str(HERE),str(tmp_path/'vm'),task],
                          env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize('status,attempts',[(False,1),(True,6)])
def test_g4_frozen_outer_retry_cannot_multiply_owner_attempts(modules,tmp_path,monkeypatch,status,attempts):
    import chatcarlo.__main__ as cli
    from chatcarlo.gpu import colab_driver
    g=modules[1];calls=[];sleeps=[]
    def transport(*args,**kw):
        calls.append(args)
        raise subprocess.CalledProcessError(1,'colab',stderr='disconnected')
    monkeypatch.setattr(g.subprocess,'run',transport)
    monkeypatch.setattr(g.time,'sleep',sleeps.append)
    def main():
        code=colab_driver._listing() if status else 'import subprocess\nsubprocess.Popen(["worker"])'
        # Model the frozen driver's outer loop: it must not catch and replay an
        # exhausted Owner call, especially one that may have launched a worker.
        for attempt in range(3):
            try:return subprocess.run(['colab','exec','-s','frozen'],input=code)
            except (subprocess.CalledProcessError,subprocess.TimeoutExpired):
                if attempt==2:raise
    monkeypatch.setattr(cli,'main',main)
    with pytest.raises(RuntimeError,match='transport exhausted'):
        g.g4(g.Owner('fake',tmp_path),'/vm',tmp_path)
    assert len(calls)==attempts and sleeps==[60]*(attempts-1)


def test_payload_failure_does_not_claim_vm_started(modules,tmp_path,monkeypatch):
    g=modules[1];owner=Mock()
    monkeypatch.setattr(g,'require_prepared',lambda p:{})
    monkeypatch.setattr(g,'Owner',owner)
    def fail(*args):raise RuntimeError('no disk')
    monkeypatch.setattr(g,'payload',fail)
    with pytest.raises(RuntimeError,match='no disk'):g.run(SimpleNamespace(output=tmp_path))
    owner.assert_not_called()
    assert not (tmp_path/'STARTED.json').exists()


def test_prepare_exports_h5_and_small_inputs_on_host(modules,tmp_path,monkeypatch):
    import a1_gate1,a1_g2vox,a1_launch,a1_speed,a2_t4,a2_common
    g=modules[1];calls=[]
    monkeypatch.setattr(a1_gate1,'export',lambda path:calls.append(('g1',path)))
    monkeypatch.setattr(a1_g2vox,'export',lambda ref,path:calls.append(('g2',ref,path)))
    monkeypatch.setattr(a1_speed,'load_configs',lambda path:({'configurations':[]},None))
    monkeypatch.setattr(a1_launch,'create_new',lambda path,cfg:calls.append(('new',path)))
    monkeypatch.setattr(a2_t4,'water',lambda path,c:calls.append(('water',path,c)))
    owner=Mock();monkeypatch.setattr(g,'Owner',owner)
    g.prepare_jobs(tmp_path,SimpleNamespace(g2_reference=tmp_path/'ref',p8g2=tmp_path/'stack'))
    assert calls[:3]==[('g1',tmp_path/'H5_jobs/gate1'),('g2',tmp_path/'ref',tmp_path/'H5_jobs/g2vox'),('new',tmp_path/'generated/diagnostic/new')]
    assert calls[3:]==[('water',tmp_path/'generated'/f'water_{c}.npz',c) for c in 'ABC']
    owner.assert_not_called()
