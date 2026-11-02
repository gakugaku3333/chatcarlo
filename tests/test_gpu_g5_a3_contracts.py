"""Approval/batch tampering and fake-VM lifecycle, using synthetic scores only."""
import copy
import json
import sys
from pathlib import Path
import types
import numpy as np
import pytest
from chatcarlo.gpu import contracts,therapy_entry as entry,therapy_run as run,therapy_layout as layout,therapy_colab as colab,therapy_inputs
from chatcarlo.gpu import therapy_scene,physics_sets
from chatcarlo.gpu.therapy import FieldGeometry
from chatcarlo.gpu.therapy_observation import descriptor


def fixture(tmp_path,monkeypatch):
    s=dict(domain='therapy_mv',source=dict(spectrum='mohan4',field_cm=[10,10],ssd_cm=100),phantom=dict(material='water',thickness_cm=30),scoring='pdd_ocr',run=dict(histories=100,seed=7,engine='colab_t4'))
    a={k:np.zeros(1) for k in ('faces','rho','bins','strength','edges','cum','ids','owner','constants')};a['sizes']=np.array([1,300,150,600,0],np.int64)
    fg=FieldGeometry();fg.sizes[:]=layout.SIZES;fg.offsets[:]=np.r_[0,np.cumsum(layout.SIZES)[:-1]]
    m=dict(format='synthetic-test-only',physics_set=physics_sets.declaration('therapy-electron-v1'))
    inputs=([],a,fg,layout.VOLUMES,m)
    monkeypatch.setattr(therapy_inputs,'load',lambda *args:inputs)
    bundle=tmp_path/'inputs';bundle.mkdir();contracts.write(bundle/'manifest.json',m)
    for name in ('catalog.json','settings.json','tables.npz','inputs.npz'):(bundle/name).write_text('synthetic-only')
    cond=entry.condition(s,m);out=tmp_path/'run';entry.prepare(out,cond,bundle)
    return out,cond,a,fg,inputs


class Driver:
    def __init__(self,cond,a,fg,limit=None):self.cond,self.a,self.fg=cond,a,fg;self.limit=limit;self.calls=0
    def __call__(self,n,seed,start):
        if self.limit is not None and self.calls>=self.limit:raise RuntimeError('synthetic interruption')
        self.calls+=1
        self.obs=descriptor(self.a,self.fg,(5.,5.),100.,seed,start,n,self.cond['physics_set'])
        groups=np.zeros((100,1087));groups[0,1050]=n;groups[0,1052]=n
        return dict(visits=np.ones(n,np.int64),groups=groups,group_ints=np.zeros((100,379),np.uint64),field=np.zeros(3420),ledger=np.zeros(15),field_additions=np.zeros(3420,np.uint64),hits=np.zeros(80,np.uint64))
    def observation(self):return self.obs


@pytest.mark.parametrize('fault',['no-approval','seed','condition','set','bundle-set','native-id','native-tag','native-half','receipt-condition','missing','duplicate','range','no-gates'])
def test_tampering(tmp_path,monkeypatch,fault):
    out,c,a,fg,_=fixture(tmp_path,monkeypatch)
    if fault=='no-approval':
        with pytest.raises(ValueError,match='approval'):contracts.require(out,c,entry.verify)
        return
    contracts.approve(out,entry.verify)
    if fault in ('seed','set'):
        expected=copy.deepcopy(c)
        if fault=='seed':expected['scene']['run']['seed']+=1
        else:expected['physics_set']=physics_sets.declaration()
        with pytest.raises(ValueError):contracts.require(out,expected,entry.verify)
        return
    if fault in ('condition','bundle-set'):
        path=out/('condition.json' if fault=='condition' else 'bundle/manifest.json');r=json.loads(path.read_text())
        if fault=='condition':r['seed']+=1
        else:r['physics_set']=physics_sets.declaration()
        contracts.write(path,r)
        with pytest.raises(ValueError):contracts.require(out,c,entry.verify)
        return
    d=Driver(c,a,fg,limit=1)
    if fault.startswith('native-'):
        original=d.observation
        def changed():
            actual=copy.deepcopy(original())
            if fault=='native-half':actual['half'][0]=6.
            else:actual['physics_set'][fault[7:]]+=1
            return actual
        d.observation=changed
        with pytest.raises(ValueError,match='observation'):run.run(out/'results',c,a,fg,d,lambda:None)
        return
    with pytest.raises(RuntimeError,match='interruption'):run.run(out/'results',c,a,fg,d,lambda:None)
    results=out/'results';p=results/'batch_000.json';r=json.loads(p.read_text())
    if fault=='receipt-condition':r['condition_sha256']='a'*64
    elif fault=='range':r['identity']['start']=100
    elif fault=='duplicate':(results/'batch_100.json').write_text(p.read_text())
    elif fault=='missing':
        # Move aside without deleting any file.
        (results/'batch_000.npz').rename(results/'missing.saved')
    elif fault=='no-gates':(results/'gates.json').rename(results/'gates.saved')
    if fault in ('receipt-condition','range'):contracts.write(p,r)
    with pytest.raises(ValueError):run.preflight(results,c,a,fg)


def test_final_artifacts_readback(tmp_path,monkeypatch):
    out,c,a,fg,_=fixture(tmp_path,monkeypatch);contracts.approve(out,entry.verify)
    saved=run.run(out/'results',c,a,fg,Driver(c,a,fg),lambda:None)
    assert saved['metadata']['arrays']['absorbed_dose_Gy']['unit']=='Gy/history'
    with np.load(out/'results/dose.npz') as z:assert str(z['physics_set'])=='therapy-electron-v1'


class FakeVM:
    """Run the actual bootstrap and listing Python locally; replace only Popen."""
    def __init__(self,out,c,a,fg,tmp_path,limit=None):
        self.out,self.c,self.a,self.fg=out,c,a,fg;self.root=tmp_path/'vm';self.limit=limit;self.stopped=False;self.launches=0;self.still_running=False
    def mapped(self,path):
        parts=Path(path).parts
        return self.root.joinpath(*parts[3:])  # /content/<unique-root>/...
    def new(self):pass
    def stop(self):self.stopped=True
    def upload(self,src,dst):
        import shutil
        p=self.mapped(dst);p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,p)
    def download(self,src,dst):
        import shutil
        p=Path(dst);p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(self.mapped(src),p)
    def execute(self,code):
        import re,contextlib,io,subprocess
        code=re.sub(r'/content/g5-entry-[a-f0-9]+',str(self.root),code)
        code=code.replace("pathlib.Path('/proc')","(root/'proc')")
        def popen(*args,**kwargs):
            self.launches+=1
            proc=self.root/'proc/99999999';proc.mkdir(parents=True,exist_ok=True)
            (proc/'stat').write_text('99999999 (fake) '+('R' if self.still_running else 'Z'))
            if self.still_running:return types.SimpleNamespace(pid=99999999)
            try:
                run.run(self.root/'results',self.c,self.a,self.fg,Driver(self.c,self.a,self.fg,self.limit),lambda:None)
                contracts.write(self.root/'results/done.json',{'status':'complete'})
            except RuntimeError:pass
            return types.SimpleNamespace(pid=99999999)
        with pytest.MonkeyPatch.context() as m:
            m.setattr(subprocess,'Popen',popen)
            m.setattr(sys,'path',list(sys.path))
            text=io.StringIO()
            with contextlib.redirect_stdout(text):
                try:exec(compile(code,'<fake-vm>','exec'),{})
                except SystemExit as exc:
                    if exc.code:raise
        for line in text.getvalue().splitlines():
            if line.startswith('G5_ENTRY '):return json.loads(line[9:])
        raise AssertionError('missing fake status')


@pytest.mark.parametrize('mode',['normal','interrupt-resume','failure'])
def test_fake_vm_modes(tmp_path,monkeypatch,mode):
    out,c,a,fg,_=fixture(tmp_path,monkeypatch);contracts.approve(out,entry.verify)
    vm=FakeVM(out,c,a,fg,tmp_path,limit=2 if mode!='normal' else None)
    if mode=='normal':colab.run(out,c,a,fg,client=vm,poll_seconds=0)
    else:
        with pytest.raises(RuntimeError,match='failed'):colab.run(out,c,a,fg,client=vm,poll_seconds=0)
        assert (out/'results/batch_001.npz').exists()
        if mode=='interrupt-resume':
            second=FakeVM(out,c,a,fg,tmp_path/'second')
            colab.run(out,c,a,fg,client=second,poll_seconds=0)
            assert second.stopped and (out/'results/dose.npz').exists()
    assert vm.stopped
    import zipfile
    with zipfile.ZipFile(out/'transfer/source.zip') as z:
        assert all(not n.startswith('prototypes/') for n in z.namelist())
        assert 'chatcarlo/gpu/therapy_cuda.py' in z.namelist()


def test_cli_modes_with_fake_vm(tmp_path,monkeypatch):
    out,c,a,fg,inputs=fixture(tmp_path,monkeypatch)
    import yaml
    path=tmp_path/'scene.yaml';path.write_text(yaml.safe_dump(c['scene']))
    monkeypatch.setattr(therapy_inputs,'DEFAULT_BUNDLE',tmp_path/'inputs')
    target=tmp_path/'cli-run'
    assert entry.main(['validate',str(path)])==0
    assert entry.main(['preview',str(path),'--out',str(target)])==0
    original=entry.execute
    vm=FakeVM(target,c,a,fg,tmp_path/'cli-vm')
    monkeypatch.setattr(entry,'execute',lambda out,cond,data:original(out,cond,data,client=vm))
    assert entry.main(['run',str(path),'--out',str(target)])==1
    assert entry.main(['approve',str(path),'--preview',str(target)])==0
    assert entry.main(['run',str(path),'--out',str(target)])==0
    assert vm.stopped and (target/'results/dose.npz').exists()


def test_complete_collection_detects_missing_pair(tmp_path,monkeypatch):
    out,c,a,fg,_=fixture(tmp_path,monkeypatch)
    run.run(out/'results',c,a,fg,Driver(c,a,fg),lambda:None)
    for ext in ('.npz','.json'):
        p=out/'results'/('batch_009'+ext);p.rename(out/('missing_009'+ext+'.saved'))
    with pytest.raises(ValueError,match='missing completed'):run.collect(out/'results',c,a,fg)


def test_bootstrap_retry_does_not_duplicate_worker(tmp_path,monkeypatch):
    out,c,a,fg,_=fixture(tmp_path,monkeypatch);contracts.approve(out,entry.verify)
    transfer,files=colab.build(out);vm=FakeVM(out,c,a,fg,tmp_path/'retry');vm.still_running=True
    remote='/content/g5-entry-abcdef'
    for p in (transfer/'parts').iterdir():vm.upload(p,remote+'/parts/'+p.name)
    code=colab.colab_driver._bootstrap(files,None,therapy_root=remote)
    assert vm.execute(code)['status']=='launched'
    assert vm.execute(code)['status']=='launched'
    assert vm.launches==1
