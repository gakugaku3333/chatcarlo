"""A1 driver smoke tests, fault power, health rejection and fake-VM lifecycle."""
import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
import pytest
from chatcarlo.gpu import bundle,core
from chatcarlo.scene import load_scene

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/validation/gpu/g5'
# Drivers are deliberately outside the distributable package.
sys.path.insert(0,str(DOC))
import a1_health as health
import a1_speed as speed
import a1_launch as launch
import a1_g2vox as vox
import a1_gate1 as gate1
import a1_faults as faults
import a1_catalog_p8 as p8


def test_checked_batch_and_lost_deposit(tmp_path):
    path=tmp_path/'bundle';scene=load_scene(str(ROOT/'examples/water_phantom_pdd_ocr.yaml'))
    bundle.create(scene,None,path,g2_tally=True);a,src,spec,settings,_=bundle.load(path)
    from chatcarlo.gpu.g2_core import Core
    from chatcarlo.tally import VoxelGrid
    from chatcarlo.gpu.source_host import sample_source_photons
    driver=Core(a,grid=VoxelGrid(np.array([-2.,-2.,0.]),(2,2,2),2.))
    inputs=sample_source_photons(src,64,np.random.default_rng(42),spec)
    measured,h=health.checked_run(driver,inputs,seed=42)
    assert h['status']=='pass'
    replay=driver.run(*inputs,seed=42,record=True)
    # Finite, nonnegative, correct count, and matching measured aggregates can
    # still be physically invalid: deliberately lose one keV in an absorbed row.
    fake=copy.deepcopy(replay);j=int(np.argmax(fake['d'][:,1]))
    fake['d'][j,1]*=.5
    with pytest.raises(ValueError,match='energy balance'):health.diagnostic_health(measured,fake,inputs[2])
    assert measured['endings'].sum()==64 and 'kerma' in measured


def test_small_gate1_aggregate(tmp_path):
    scene,plane,_=gate1.legacy.inputs('s1')
    r,reports=gate1.twin(scene,plane,24,8,6000,7000,'cpu')
    assert len(reports)==3 and all(h['histories']==8 for h in reports)
    q=gate1.legacy.collect(r,'s1',24)
    assert q and all(np.isfinite(v) for v in q.values())


def test_small_g2_batches_and_retry_rules():
    from chatcarlo.tally import VoxelGrid
    scene=load_scene(str(ROOT/'examples/water_phantom_pdd_ocr.yaml'))
    values,reports=vox.batches(scene,VoxelGrid(np.array([-2.,-2.,0.]),(2,2,2),2.),16,2,221100,'cpu')
    assert values['kerma'].shape==(2,2,2,2) and len(reports)==2
    r={'water':{'kerma':{'checks':dict(outliers=False,spread=True,nonempty=True)}}}
    assert vox.retry_allowed(r)
    r['water']['kerma']['checks']['all_total']=False;assert not vox.retry_allowed(r)
    r['water']['kerma']['checks'].pop('all_total');r['water']['h10']=copy.deepcopy(r['water']['kerma']);assert not vox.retry_allowed(r)
    with pytest.raises((FileNotFoundError,ValueError)):vox.verify_reference(Path('/nonexistent/a1'))


def test_faults_philox_all_four(tmp_path):
    ref=tmp_path/'ref.npz';faults.reference(ref)
    result=faults.run(ref,tmp_path/'faults.json')
    assert result['status']=='pass' and len(result['ks'])==4
    assert all(r['normal_p']>=1e-6 and r['fault_p']<1e-6 for r in result['ks'])


def test_p8_driver_small(tmp_path):
    from chatcarlo.gpu import catalog
    m=SimpleNamespace(**{f:np.array(1.) for f in catalog.P8_DEFINITIONS})
    for f in ('stop_T','gs_T','photo_E','pair_E','brems_E'):setattr(m,f,np.array([.01,10.]))
    m.rates=np.array([[.01,1.],[10.,2.]])
    m.gs_y=np.zeros((2,3,4));m.lines=np.zeros((2,4,3,4))
    cat,blob=p8.roundtrip(m,'water')
    assert len(cat['fields'])==33


def test_config_order_and_hash_rows():
    cfg,digest=speed.load_configs(DOC/'a1_speed_configs.json')
    rows=[]
    for repeat,order in [(-1,cfg['warmup_order']),*enumerate(cfg['round_orders'])]:
        for task in order:rows.append(dict(task=task,repeat=repeat,config_sha256=digest,kernel_s=1.,health=dict(status='pass')))
    assert speed.summarize(rows,cfg,digest)['status']=='pass'
    rows[0]['config_sha256']='wrong'
    with pytest.raises(ValueError,match='config identity'):speed.summarize(rows,cfg,digest)


def test_ncu_kills_process_group(tmp_path):
    heartbeat=tmp_path/'child.heartbeat'
    child_code="import pathlib,sys,time; p=pathlib.Path(sys.argv[1]);\nwhile True:p.write_text(str(time.monotonic()));time.sleep(.02)"
    code='import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",sys.argv[2],sys.argv[1]]);time.sleep(60)'
    with pytest.raises(subprocess.TimeoutExpired):
        speed.group_call([sys.executable,'-c',code,str(heartbeat),child_code],2.)
    assert heartbeat.exists()
    first=heartbeat.read_bytes();time.sleep(.2)
    assert heartbeat.read_bytes()==first


def test_fake_colab_timeout_retry(tmp_path,monkeypatch):
    exe=tmp_path/'colab';calls=tmp_path/'calls'
    exe.write_text('''#!/bin/sh
printf '%s\\n' "$*" >> "$A1_CALLS"
exec sleep 2
''')
    exe.chmod(0o755);monkeypatch.setenv('PATH',str(tmp_path)+os.pathsep+os.environ['PATH']);monkeypatch.setenv('A1_CALLS',str(calls))
    with pytest.raises(subprocess.TimeoutExpired):launch.call(['exec','-s','fake'],tmp_path/'hb.json',timeout=.7)
    rows=[v.split() for v in calls.read_text().splitlines()];assert len(rows)==3
    assert all('--timeout' in r for r in rows)
    with pytest.raises(subprocess.TimeoutExpired):launch.call(['new','-s','fake'],tmp_path/'hb.json',timeout=.7)
    assert len(calls.read_text().splitlines())==4
    with pytest.raises(ValueError,match='unbounded'):launch.call(['unknown'],tmp_path/'hb.json')


def test_transfer_assembly_and_escape_rejection(tmp_path):
    from chatcarlo.gpu.colab_driver import split_bundle
    source=tmp_path/'source';source.mkdir();(source/'file').write_bytes(b'payload')
    parts=tmp_path/'parts/new/g2b';files=split_bundle(source,parts)
    (tmp_path/'job.json').write_text(json.dumps(dict(controls={})))
    digest=bundle.sha(tmp_path/'job.json')
    code=launch.assembly_code(str(tmp_path),{'new/g2b':files},digest)
    # Execute the exact standalone verifier/assembly, stopping before uploaded
    # imports; the installation path is exercised by fake-VM test below.
    prefix=code[:code.index('sys.path.insert')]
    exec(prefix,{})
    assert (tmp_path/'bundles/new/g2b/file').read_bytes()==b'payload'
    part=next(parts.iterdir());part.write_bytes(b'corrupt')
    with pytest.raises(AssertionError,match='part SHA256'):exec(prefix,{})
    bad=copy.deepcopy(files);bad['../escape']=bad.pop('file')
    with pytest.raises(ValueError,match='unsafe'):launch.assembly_code(str(tmp_path),{'new/g2b':bad},digest)


def test_fake_vm_old_new_isolation(tmp_path):
    # Construct both runtimes from real bundles. Old source is from git show,
    # never the working tree; bundle creation itself is tested independently.
    path=tmp_path/'bundles/new/g2b'
    scene=load_scene(str(ROOT/'docs/validation/gpu/g1/scenes/s1.yaml'));bundle.create(scene,None,path)
    import shutil
    shutil.copytree(path,tmp_path/'bundles/new/g4')
    for cid in ('g2b','g4'):
        dst=tmp_path/'bundles/old'/cid;dst.mkdir(parents=True)
        names=subprocess.check_output(['git','ls-tree','--name-only','19c281e:chatcarlo/gpu'],text=True).splitlines()
        for name in names:
            if Path(name).suffix in ('.py','.h','.cpp','.cu'):(dst/name).write_bytes(subprocess.check_output(['git','show','19c281e:chatcarlo/gpu/'+name]))
        files={p.name:bundle.sha(p) for p in dst.iterdir()}
        (dst/'manifest.json').write_text(json.dumps(dict(format='chatcarlo-g1-v1',files=files)))
    speed.install_runtimes(tmp_path)
    code='''import sys
from pathlib import Path
import chatcarlo.gpu.core as new
assert not any(n.startswith('chatcarlo_g5_old') for n in sys.modules)
import chatcarlo_g5_old.core as old
assert Path(new.__file__)!=Path(old.__file__)
assert new.Core is not old.Core
assert old.Core.__module__=='chatcarlo_g5_old.core'
'''
    subprocess.run([sys.executable,'-c',code],cwd=tmp_path,env=dict(os.environ,PYTHONPATH=str(tmp_path)),check=True,capture_output=True)


def test_cuda_single_module_and_fault_options(tmp_path,monkeypatch):
    scene=load_scene(str(ROOT/'examples/water_phantom_pdd_ocr.yaml'));bundle.create(scene,None,tmp_path,g2_tally=True)
    a,*_=bundle.load(tmp_path)
    from chatcarlo.gpu import physics_sets
    modules=[]
    class Function:
        def __init__(self,name):self.name=name
        def compile(self):pass
        def __call__(self,*args):
            if self.name=='g5_physics_set_kernel':args[2][0][:]=[1,physics_sets.declaration()['tag']]
    class Module:
        def __init__(self,**kwargs):self.kwargs=kwargs;self.names=[];modules.append(self)
        def get_function(self,name):self.names.append(name);return Function(name)
    fake=SimpleNamespace(RawModule=Module,asarray=np.asarray,asnumpy=np.asarray,zeros=np.zeros,uint64=np.uint64,
                         cuda=SimpleNamespace(Stream=SimpleNamespace(null=SimpleNamespace(synchronize=lambda:None))))
    monkeypatch.setitem(sys.modules,'cupy',fake);monkeypatch.setattr(core,'CUDA_INPUTS',{})
    from chatcarlo.gpu.g2_core import Core
    driver=Core(a,'cuda',fault=2)
    assert len(modules)==1 and {'g1_histories_kernel','g2_histories_kernel','g5_physics_set_kernel'}<=set(modules[0].names)
    assert '-DP5_FAULT=2' in modules[0].kwargs['options'] and modules[0].kwargs['options']==driver.cuda_options


def test_fake_colab_launcher_through_speed(tmp_path,monkeypatch):
    """Real split/upload/assembly/install/order/report/download/stop; fake kernels."""
    old=tmp_path/'old';scene=load_scene(str(ROOT/'docs/validation/gpu/g1/scenes/s1.yaml'))
    # A minimal historical manifest/runtime suffices for transfer here; old
    # numeric loading and arithmetic are separately checked in catalogue tests.
    for cid in ('g2b','g4'):
        dst=old/cid;dst.mkdir(parents=True)
        for name in ('__init__.py','core.py','bundle.py','source_host.py','g2_core.py'):
            (dst/name).write_bytes(subprocess.check_output(['git','show','19c281e:chatcarlo/gpu/'+name]))
        (dst/'manifest.json').write_text(json.dumps(dict(format='chatcarlo-g1-v1',files={p.name:bundle.sha(p) for p in dst.iterdir()})))
    vm=tmp_path/'vm';log=tmp_path/'cli_calls.jsonl';exe=tmp_path/'colab'
    exe.write_text('#!'+sys.executable+'\n'+r'''
import json,os,pathlib,shutil,sys,types
args=sys.argv[1:];op=args[0];vm=pathlib.Path(os.environ['A1_VM']);vm.mkdir(exist_ok=True)
with open(os.environ['A1_LOG'],'a') as f:f.write(json.dumps(args)+'\n')
def local(path):return str(path).replace('/content/g5a1-fake',str(vm))
if op=='upload':shutil.copyfile(args[3],local(args[4]))
elif op=='download':shutil.copyfile(local(args[3]),args[4])
elif op=='exec':
 code=sys.stdin.read().replace('/content/g5a1-fake',str(vm))
 if 'from a1_speed import install_runtimes' in code:
  exec(code[:code.index("with (base/'worker.log')")],{})
  sys.path.insert(0,str(vm))
  import a1_speed as speed
  cfg,digest=speed.load_configs(vm/'a1_speed_configs.json')
  fakecp=types.SimpleNamespace(cuda=types.SimpleNamespace(runtime=types.SimpleNamespace(getDeviceProperties=lambda n:dict(name='Tesla T4',major=7,minor=5)),Device=lambda:types.SimpleNamespace(id=0)))
  sys.modules['cupy']=fakecp
  speed.construct=lambda *a,**k:dict(kernel=None)
  speed.resources=lambda *a,**k:dict(attributes=dict(num_regs=120),manual=dict(blocks_per_sm=4),occupancy_api_blocks_per_sm=4)
  speed.measure=lambda *a,**k:dict(kernel_s=.01,call_s=.02,health=dict(status='pass'))
  speed.group_call=lambda *a,**k:types.SimpleNamespace(returncode=0)
  speed.speed(types.SimpleNamespace(configs=vm/'a1_speed_configs.json',bundles=vm/'bundles',output=vm/'results'))
  (vm/'worker.pid').write_text(str(os.getpid()))
 else:exec(code,{})
''')
    exe.chmod(0o755);monkeypatch.setenv('PATH',str(tmp_path)+os.pathsep+os.environ['PATH'])
    monkeypatch.setenv('A1_VM',str(vm));monkeypatch.setenv('A1_LOG',str(log))
    args=SimpleNamespace(configs=DOC/'a1_speed_configs.json',old_bundles=old,session='fake',output=tmp_path/'output',poll_seconds=0)
    launch.run(args)
    result=json.loads((args.output/'measurements.json').read_text());cfg,digest=speed.load_configs(args.configs)
    assert result['status']=='complete' and result['summary']['status']=='pass'
    assert all(r['config_sha256']==digest for r in result['measurements'])
    assert [r['task'] for r in result['measurements']]==cfg['warmup_order']+sum(cfg['round_orders'],[])
    calls=[json.loads(s) for s in log.read_text().splitlines()]
    assert calls[0][0]=='new' and calls[-1][0]=='stop'
    assert all('--timeout' in r for r in calls if r[0]=='exec')


@pytest.mark.parametrize('mode',['gate1','g2vox'])
def test_worker_modes_small_input(tmp_path,mode):
    path=tmp_path/'bundle';scene=load_scene(str(ROOT/'examples/water_phantom_pdd_ocr.yaml'))
    manifest=bundle.create(scene,None,path,g2_tally=mode=='g2vox')
    job=dict(bundle_manifest=manifest,n_histories=12,batch_size=4,source_seed=7000,seed=6000)
    if mode=='g2vox':job['grid']=dict(origin_cm=[-2.,-2.,0.],shape=[2,2,2],voxel_size_cm=2.,layout_version='g2-grid-v1')
    j=tmp_path/'job.json';j.write_text(json.dumps(job));out=tmp_path/'out'
    speed.worker_batches(SimpleNamespace(job=j,bundle=path,output=out,device='cpu'))
    assert len(list(out.glob('batch_*.npz')))==3
    assert all(json.loads(p.read_text())['health']['status']=='pass' for p in out.glob('batch_*.json'))
    job['seed']+=1;job['bundle_manifest']={};j.write_text(json.dumps(job))
    with pytest.raises(ValueError,match='identity mismatch'):speed.worker_batches(SimpleNamespace(job=j,bundle=path,output=out,device='cpu'))


def test_g2gate3_prepare_and_identity_reject(tmp_path,monkeypatch):
    import a1_g2gate3 as g3
    # prepare only creates diagnostic bundles/metadata; EGS5 is never run.
    g3.prepare(tmp_path)
    for name,(seed,n) in g3.CONDITIONS.items():
        bundle.verify(tmp_path/name)
        with np.load(tmp_path/name/'g2_tables.npz') as z:assert (int(z['seed']),int(z['n_histories']))==(seed,n)
    monkeypatch.setattr(g3.bundle,'sha',lambda p:'wrong')
    with pytest.raises(ValueError,match='EGS5 SHA256 mismatch'):g3.verify_inputs(tmp_path)


def test_remote_collection_identity_and_health(tmp_path):
    from a1_remote import export_job,collect
    scene=load_scene(str(ROOT/'examples/water_phantom_pdd_ocr.yaml'));path=tmp_path/'bundle'
    bundle.create(scene,None,path);arrays,*_=bundle.load(path)
    job=export_job(path,tmp_path/'job.json',n=8,batch=4,seed=6000,source_seed=7000)
    args=SimpleNamespace(job=tmp_path/'job.json',bundle=path,output=tmp_path/'out',device='cpu')
    speed.worker_batches(args)
    rows=list(collect(args.output,job,arrays));assert len(rows)==2
    file=args.output/'batch_000000000.json';receipt=json.loads(file.read_text());receipt['health']['status']='fail'
    file.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='health'):list(collect(args.output,job,arrays))


def test_t4_gates_payload_layout_and_retries(tmp_path,monkeypatch):
    """a1_t4_gates: the VM layout keeps a1_common.ROOT valid; launcher CLI errors are retried."""
    import tarfile
    import a1_t4_gates as gates
    payload=tmp_path/'payload';bundle.create(load_scene(str(ROOT/'docs/validation/gpu/g1/scenes/s1.yaml')),None,payload/'gate1/bundles/s1')
    gates.pack_payload(payload,tmp_path/'p.tar')
    with tarfile.open(tmp_path/'p.tar') as t:names=set(t.getnames())
    assert 'chatcarlo/gpu/core.py' in names and 'chatcarlo/__init__.py' in names
    assert 'docs/validation/gpu/g5/a1_common.py' in names and 'a1_common.py' not in names
    calls=[]
    def flaky(args,heartbeat,**kw):
        calls.append(args[0])
        if len(calls)<3:raise subprocess.CalledProcessError(1,args,stderr='ReadTimeout')
        return 'ok'
    monkeypatch.setattr(launch,'_call',flaky);monkeypatch.setattr(launch.time,'sleep',lambda s:None)
    assert launch.call(['exec','-s','x'],tmp_path/'h.json')=='ok' and calls==['exec']*3
    calls.clear()
    with pytest.raises(subprocess.CalledProcessError):launch.call(['new','-s','x'],tmp_path/'h.json')
    assert calls==['new']
