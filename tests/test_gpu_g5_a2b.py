"""A2b offline acceptance contracts; no cached inputs or private spectra."""
import copy
import ctypes as ct
import importlib.abc
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from chatcarlo.gpu import seed_geometry as g,seed_analysis as a,seed_checks,seed_condition,seed_scene,seed_run,seed_core,contracts,physics_sets,therapy,seed_colab
from chatcarlo.gpu.seed_stats import PairStore,Moments,Checkpoint
from test_gpu_g5_a3_native import artificial_material


def test_common_masks_and_constant_density():
    masks=a.load_masks();assert masks['near'].sum()==25400
    for energy_density in (1.,.03,200.):
        old=np.full(g.OLD_FINE.shape,energy_density*g.OLD_FINE.volume)
        new=np.full(g.FINE.shape,energy_density*g.FINE.volume)
        np.testing.assert_allclose(a.old_densities(old,masks),a.densities(new,1,masks),rtol=1e-14)
    assert not np.array_equal(masks['water'],g.water_mask(g.FINE))


def test_geometry_local(tmp_path):
    r=seed_checks.local(tmp_path)
    assert all(v['status']=='pass' for v in r.values())


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    coarse=g.Grid((-20.,)*3,(10.,)*3,(4,)*3)
    return seed_core.CPU(tmp_path_factory.mktemp('a2b-native'),[artificial_material()]*5,constants=np.ones(6),strength=np.arange(17,dtype=float),densities=[1]*5,energy=.004,grids=(coarse,coarse))


def test_native_split_and_faults(native,tmp_path):
    r=native.run(8,9502001,start=7);parts=[native.run(4,9502001,start=s) for s in (7,11)]
    np.testing.assert_array_equal(r['d'],np.concatenate([p['d'] for p in parts]))
    np.testing.assert_array_equal(r['source_counts'],np.concatenate([p['source_counts'] for p in parts]))
    from chatcarlo.gpu.seed_t4_checks import faults
    assert faults(lambda fault,**kw:seed_core.CPU(tmp_path,[artificial_material()]*5,constants=np.ones(6),strength=np.arange(17,dtype=float),densities=[1]*5,fault=fault,**kw))['status']=='pass'


def test_actual_argument_tamper(native,monkeypatch):
    native.run(3,9502001,start=4);expected=native.last_observation
    real=native._invoke
    def invoke(args):
        args=list(args);args[4]+=8;real(tuple(args))
    monkeypatch.setattr(native,'_invoke',invoke)
    native.run(3,9502001,start=4)
    with pytest.raises(ValueError,match='seed'):contracts.check_observation(native.last_observation,expected,'ABI')


def test_table_pointer_tamper(native):
    native.run(1,9502001);before=native.last_observation
    offset=int(native.args[1][4,0]);old=native.args[0][offset]
    try:
        native.args[0][offset]+=1;native.run(1,9502001)
        with pytest.raises(ValueError,match='material_data'):contracts.check_observation(native.last_observation,before,'material ID4')
    finally:native.args[0][offset]=old


def test_49_incomplete_50_complete_and_resume(tmp_path,monkeypatch):
    shapes={'fine':(1,),'coarse':(1,),'density':(3,)};cond=dict(batches=50,seed=9502001,per_batch_per_field=71429,histories_per_side=25000150,physics_set=physics_sets.declaration(therapy.NAME))
    cond['native_templates']=[dict(field=f,with_seeds=s,observed=dict(start=0,count=71429,seed=cond['seed']*8+f,physics_set=cond['physics_set'])) for f in range(7) for s in (True,False)]
    monkeypatch.setattr(seed_run,'shapes',lambda:shapes)
    path=tmp_path/'prostate.npz';doors={(f,s) for f in range(7) for s in (True,False)}
    report={k:dict(status='pass') for k in ('B0','B1','V0','V1a','V1b','V1c','V2a','V2b','V3','V4_rectangle','V4_prostate','V5','V5_GPU','V6','V6b','G3_grazing','G3','G5')}
    contracts.write(tmp_path/'gates.json',dict(condition_sha256=seed_run.sha256(cond),gates=report))
    p=PairStore(path,cond,shapes);pairs={k:(np.ones(s),np.ones(s)) for k,s in shapes.items()}
    for b in range(50):
        key=seed_condition.identity(cond,b)
        receipts=[dict(field=f,with_seeds=s,health='pass',observed=dict(start=key['start'],count=key['count'],seed=cond['seed']*8+f,physics_set=cond['physics_set'])) for f,s in sorted(doors)]
        digest=seed_run.sha(path) if path.exists() else None
        with pytest.raises(ValueError,match='incomplete'):p.commit(b,pairs,doors-{(6,False)},np.ones((2,3)),receipts)
        if digest:assert seed_run.sha(path)==digest
        p.commit(b,pairs,doors,np.ones((2,3)),receipts)
        p=PairStore(path,cond,shapes)
        if b==48:
            with pytest.raises(ValueError,match='50 complete'):seed_run.collect(tmp_path,cond)
    assert seed_run.collect(tmp_path,cond)['batches']==50
    for b in (-1,50,False):
        with pytest.raises(ValueError):seed_condition.identity(cond,b)
    with pytest.raises(ValueError,match='replay'):p.commit(49,pairs,doors,np.ones((2,3)),receipts)
    assert [seed_condition.identity(cond,b)['start'] for b in range(50)]==list(range(0,3571450,71429))


def test_scene_conditions():
    import yaml
    raw=yaml.safe_load(Path('examples/therapy/prostate_seed_7field.yaml').read_text());raw['run'].update(histories_per_field=3571450,seed=9502001)
    assert seed_scene.validate(raw)==raw
    for key,val in [('seed',940100),('histories_per_field',3571428)]:
        bad=copy.deepcopy(raw);bad['run'][key]=val
        with pytest.raises(ValueError):seed_scene.validate(bad)
    bad=dict(raw,physics={'set':'diagnostic-photon-kerma-v1'})
    with pytest.raises(ValueError):seed_scene.validate(bad)


def test_snapshot_safety(tmp_path):
    import tarfile,io
    p=tmp_path/'bad.tar'
    with tarfile.open(p,'w') as f:
        member=tarfile.TarInfo('../escape');member.size=1;f.addfile(member,io.BytesIO(b'x'))
    with pytest.raises(ValueError):seed_colab.unpack(p,tmp_path)


def test_no_prototypes_import_validate_preview_stats(tmp_path,monkeypatch):
    forbid_prototype_imports(monkeypatch)
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname=='prototypes' or fullname.startswith('prototypes.'):raise AssertionError(fullname)
    guard=Block();sys.meta_path.insert(0,guard)
    try:
        from chatcarlo.gpu import seed_preview
        import yaml
        raw=yaml.safe_load(Path('examples/therapy/prostate_seed_7field.yaml').read_text());raw['run'].update(histories_per_field=3571450,seed=9502001)
        seed_scene.validate(raw);assert len(seed_preview.geometry_json(raw)['sources'])==7
        m=Moments((3,));m.add(np.ones(3),np.ones(3));assert m.n==1
    finally:sys.meta_path.remove(guard)


def test_cuda_flattened_and_slab_boundary():
    src=seed_core.cuda_source();assert '#include "' not in src
    assert 't2_seed_fixed_kernel' in src and 'source_counts[3]=1' in src
    t=Path('chatcarlo/gpu/t2_kernel.h').read_text()
    assert t.startswith('#if defined(T2_SEED) && T2_SEED')

@pytest.mark.parametrize('mode',['normal','failure','interrupt-resume'])
def test_fake_vm_entry_modes_without_prototypes(tmp_path,monkeypatch,mode):
    import yaml,types,shutil,re,contextlib,io,subprocess,zipfile
    from chatcarlo.gpu import seed_entry,seed_bundle,seed_preview
    tiny=g.Grid((-1.,)*3,(2.,)*3,(1,)*3)
    monkeypatch.setattr(g,'FINE',tiny);monkeypatch.setattr(g,'COARSE',tiny)
    monkeypatch.setattr(seed_condition,'PER_BATCH_PER_FIELD',1)
    monkeypatch.setattr(a,'load_masks',lambda *args:{k:np.ones((1,1,1),bool) for k in ('water','prostate','rectum','near')})
    def expected(cond,materials,inputs,b,f,s):
        return dict(physics_set=cond['physics_set'],seed=cond['seed']*8+f,start=b,count=1,field=f,with_seeds=s)
    monkeypatch.setattr(seed_run,'expected',expected)
    manifest=dict(format='synthetic-only')
    bundle=tmp_path/'bundle';bundle.mkdir()
    for name in ('manifest.json','settings.json','catalog.json','inputs.npz','tables.npz'):(bundle/name).write_text(json.dumps(manifest))
    monkeypatch.setattr(seed_bundle,'load',lambda *args:([],{},manifest))
    raw=yaml.safe_load(Path('examples/therapy/prostate_seed_7field.yaml').read_text());raw['run'].update(histories_per_field=3571450,seed=9502001)
    scene=tmp_path/'scene.yaml';scene.write_text(yaml.safe_dump(raw));out=tmp_path/'run'
    argv=[str(scene),'--bundle',str(bundle)]
    assert seed_entry.main(['validate',*argv])==0
    assert seed_entry.main(['preview',*argv,'--out',str(out)])==0
    cond=json.loads((out/'condition.json').read_text())
    with pytest.raises(ValueError,match='approval'):seed_entry.main(['run',*argv,'--out',str(out)])
    assert seed_entry.main(['approve',*argv,'--preview',str(out)])==0
    gate={k:dict(status='pass') for k in ('B0','B1','V0','V1a','V1b','V1c','V2a','V2b','V3','V4_rectangle','V4_prostate','V5','V5_GPU','V6','V6b','G3_grazing','G3','G5')}
    class VM:
        def __init__(self,root,limit=None):self.root=root;self.limit=limit;self.stopped=False;self.calls=0;self.launches=0
        def new(self):pass
        def stop(self):self.stopped=True
        def mapped(self,path):return self.root.joinpath(*Path(path).parts[3:])
        def upload(self,src,dst):
            dest=self.mapped(dst);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
        def download(self,src,dst):shutil.copyfile(self.mapped(src),dst)
        def driver(self,b,f,s,start,n):
            if self.limit is not None and b>=self.limit:raise RuntimeError('synthetic interruption')
            self.calls+=1;d=np.zeros((n,37));d[:,0:2]=1;d[:,21:23]=1
            counts=np.column_stack((np.zeros((n,2),np.int64),np.arange(start,start+n),np.ones(n,np.int64)))
            return dict(d=d,i=np.zeros((n,79),np.int64),ledger=np.tile([1,0,0,0,0,0],(n,1)),source_counts=counts,grids=[np.ones((1,1,1))*n]*2,observed=expected(cond,[],{},b,f,s),seconds=.01)
        def execute(self,code):
            code=re.sub(r'/content/g5-seed-[a-f0-9]+',str(self.root),code).replace("pathlib.Path('/proc')","(root/'proc')")
            def popen(*args,**kw):
                self.launches+=1;proc=self.root/'proc/99999999';proc.mkdir(parents=True,exist_ok=True);(proc/'stat').write_text('99999999 (fake) Z')
                try:
                    seed_run.run(self.root/'results',cond,[],{},self.driver,lambda:gate,progress=lambda:seed_colab.publish(self.root/'results'))
                    seed_colab.publish(self.root/'results');contracts.write(self.root/'results/done.json',dict(status='complete'))
                except RuntimeError:pass
                return types.SimpleNamespace(pid=99999999)
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(subprocess,'Popen',popen);patch.setattr(sys,'path',list(sys.path));text=io.StringIO()
                with contextlib.redirect_stdout(text):exec(compile(code,'<seed-fake-vm>','exec'),{})
            for line in text.getvalue().splitlines():
                if line.startswith('G5_ENTRY '):return json.loads(line[9:])
            raise AssertionError('missing VM status')
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname=='prototypes' or fullname.startswith('prototypes.'):raise AssertionError(fullname)
    forbid_prototype_imports(monkeypatch)
    block=Block();sys.meta_path.insert(0,block)
    try:
        real=seed_colab.run;vm=VM(tmp_path/'vm',2 if mode!='normal' else None)
        monkeypatch.setattr(seed_colab,'run',lambda *args,**kw:real(*args,**kw,client=vm,poll_seconds=0))
        if mode=='normal':assert seed_entry.main(['run',*argv,'--out',str(out)])==0
        else:
            with pytest.raises(RuntimeError,match='failed'):seed_entry.main(['run',*argv,'--out',str(out)])
            assert seed_run.preflight(out/'results',cond).completed==[0,1]
            if mode=='interrupt-resume':
                resumed=VM(tmp_path/'resumed');monkeypatch.setattr(seed_colab,'run',lambda *args,**kw:real(*args,**kw,client=resumed,poll_seconds=0))
                assert seed_entry.main(['run',*argv,'--out',str(out)])==0
                assert resumed.stopped and resumed.calls==48*14
        assert vm.stopped
        if mode!='failure':assert seed_entry.main(['analyse',*argv,'--out',str(out)])==0
        with zipfile.ZipFile(out/'transfer/source.zip') as z:assert not any(n.startswith('prototypes/') for n in z.namelist())
    finally:sys.meta_path.remove(block)


def test_physics_helpers_and_notebook(tmp_path):
    import importlib.util
    def module(name):
        spec=importlib.util.spec_from_file_location(name,Path('docs/validation/gpu/g5')/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    from docs.validation.gpu.g5.a4u_d2_update import check_physics
    assert check_physics()['status']=='pass'
    result=module('a2b_notebook').create(tmp_path)
    nb=json.loads(Path(result['notebook']).read_text());code=''.join(nb['cells'][1]['source']);compile(code,'<notebook>','exec')
    import zipfile
    with zipfile.ZipFile(tmp_path/'a2b_source.zip') as z:assert 'chatcarlo/gpu/seed_colab.py' in z.namelist()


def test_legacy_adapter_fixed_grid_and_labels(monkeypatch):
    # B5 may import prototypes, but the adapter must always override .25 mm.
    import importlib.util,types
    folder=Path('docs/validation/gpu/g5');spec=importlib.util.spec_from_file_location('a2b_legacy_test',folder/'a2b_legacy_adapter.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    seen={}
    class Old:
        def __init__(self,cp,materials,**kw):seen.update(kw)
    fake=types.ModuleType('prototypes.mv_electron.gpu.p9c_colab');fake.GPU=Old
    monkeypatch.setitem(sys.modules,'prototypes.mv_electron.gpu.p9c_colab',fake)
    m.GPU(None,[artificial_material()]*5,constants=np.ones(6),strength=np.arange(17),densities=[1]*5)
    assert seen['grids'][0].step==(.05,)*3 and seen['grids'][0].shape==(120,100,150)


def test_kernel_speed_worker_script(tmp_path,monkeypatch):
    import importlib,types,subprocess
    monkeypatch.syspath_prepend(str(Path('docs/validation/gpu/g5').resolve()))
    m=importlib.import_module('a2b_launch');root=tmp_path/'worker';root.mkdir();(root/'source').mkdir()
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**kw:types.SimpleNamespace(pid=7))
    class Owner:
        def raw(self,code):exec(compile(code,'<kernel-launch>','exec'),{})
        def json(self,code):return dict(done=True)
        def call(self,action,src,dst):Path(dst).write_text('{"status":"pass"}')
    assert m.measure_kernel(Owner(),str(root),tmp_path)['status']=='pass'
    compile((root/'kernel_measure.py').read_text(),'<kernel-worker>','exec')


@pytest.mark.parametrize('fault',['none','array','material-id','missing','pin','density'])
def test_seed_bundle_pins_and_id4(tmp_path,monkeypatch,fault):
    from chatcarlo.gpu import seed_bundle,catalog
    materials=[artificial_material() for _ in range(5)]
    for j,m in enumerate(materials):m.ne=np.array(j+1.)
    gens={v['generator_path']:dict(path=v['generator_path'],sha256='a'*64,options={}) for v in catalog.P8_DEFINITIONS.values()}
    pins={n+'.'+f:dict(sha256=catalog.digest(getattr(m,f)),shape=list(np.shape(getattr(m,f)))) for n,m in zip(seed_bundle.NAMES,materials) for f in catalog.P8_DEFINITIONS}
    pins['seed_input.densities']=dict(sha256=catalog.digest(np.arange(1.,6)),shape=[5])
    pinpath=tmp_path/'pins.json';contracts.write(pinpath,pins);monkeypatch.setattr(seed_bundle,'PINS',pinpath)
    p=tmp_path/'bundle';seed_bundle.create(p,materials,{},np.arange(1.,6),gens)
    if fault=='none':
        m,i,_=seed_bundle.load(p);assert m[4].ne==5 and list(i['densities'])==[1,2,3,4,5];return
    if fault=='pin':
        pins['seed_agti.ne']['sha256']='0'*64;contracts.write(pinpath,pins)
    else:
        member={'array':'tables.npz','material-id':'settings.json','missing':'catalog.json','density':'inputs.npz'}[fault]
        path=p/member
        if path.suffix=='.json':
            r=json.loads(path.read_text())
            if fault=='material-id':r['material_names'][0],r['material_names'][4]=r['material_names'][4],r['material_names'][0]
            else:r['fields'].pop()
            contracts.write(path,r)
        else:
            with np.load(path,allow_pickle=False) as z:r={k:z[k] for k in z.files}
            if fault=='array':r['p8_seed_agti'][0]+=1
            else:r['densities']=r['densities'][::-1]
            np.savez(path,**r)
        manifest=json.loads((p/'manifest.json').read_text());manifest['files'][member]=seed_run.sha(path);contracts.write(p/'manifest.json',manifest)
    with pytest.raises(ValueError):seed_bundle.load(p)


@pytest.mark.parametrize('case',['pass','fail','insufficient-statistics'])
def test_b3_judge_fixed_thresholds(tmp_path,monkeypatch,case):
    import importlib.util,types
    path=Path('docs/validation/gpu/g5/a2b_judge.py');spec=importlib.util.spec_from_file_location('a2b_judge_test',path);j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)
    runs=[]
    for seed in range(6):
        p=tmp_path/str(seed);p.write_text(str(seed));runs.append(dict(path=str(p),seed=seed,sha256=seed_run.sha(p),batches=list(range(100))))
    ref=dict(runs=runs,batches=600,deltas=j.DELTAS,masks_sha256=seed_run.sha(a.MASK_PATH),values={k:1. for k in j.DELTAS},sem={k:0. for k in j.DELTAS})
    rp=tmp_path/'reference.json';contracts.write(rp,ref)
    rows=np.ones((50,2,3))
    if case=='fail':rows[:,0,0]+=.002
    if case=='insufficient-statistics':rows[:,0,0]+=np.tile([-.1,.1],25)
    cp=tmp_path/'prostate.npz';cp.write_bytes(b'checkpoint-test-only')
    store=types.SimpleNamespace(completed=list(range(50)),path=cp,rows=rows)
    monkeypatch.setattr(seed_condition,'verify',lambda path:dict(condition={'seed':9502001}))
    monkeypatch.setattr(contracts,'require',lambda *args:None)
    monkeypatch.setattr(seed_run,'preflight',lambda *args:store)
    summary=dict(batches=50,checkpoint_sha256=seed_run.sha(cp),statistics=a.analyse(rows));sp=tmp_path/'summary.json';contracts.write(sp,summary)
    result=j.judge(rp,sp,tmp_path/'B3.json');assert result['status']==case
    assert all(row['delta']==j.DELTAS[k] for k,row in result['rows'].items())


def test_missing_energy_record_rejected():
    with pytest.raises(ValueError,match='source energy'):seed_checks.raw(np.zeros((1,37)),np.zeros((1,79),np.int64),np.zeros((1,6)),np.array([[0,0,0,1]]),0)


def forbid_prototype_imports(monkeypatch):
    import builtins
    original=builtins.__import__
    def checked(name,*args,**kwargs):
        if name=='prototypes' or name.startswith('prototypes.'):raise AssertionError('forbidden import: '+name)
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',checked)
