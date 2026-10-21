"""Local G4 acceptance: declarations, native identity, approval and image output."""
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import yaml
from chatcarlo import __main__ as cli
from chatcarlo.scene import load_scene, validate_scene, detector_from_scene
from chatcarlo.detector import DetectorPlane
from chatcarlo.transport import run_transport
from chatcarlo.gpu import approval, bundle, condition, core, colab_driver, outputs
from chatcarlo.gpu.aggregate import aggregate

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT/'docs/validation/gpu/g4'
SCENE = DOC/'scene_portrait_detector.yaml'
DET = dict(center_cm=[0,0,0], normal=[0,0,1], u_axis=[1,0,0], size_cm=[15,18], shape=[100,120])


def options(**kw):
    return dict(engine='gpu', gpu_device='colab', n_histories=41, seed=2040, batch_size=20, no_uncertainty=False, **kw)


def invoke(monkeypatch, *argv):
    monkeypatch.setattr(sys, 'argv', ['chatcarlo', *map(str, argv)])
    return cli.main()


def fingerprint(v):
    if isinstance(v, np.ndarray): return dict(shape=list(v.shape), dtype=str(v.dtype), sha256=hashlib.sha256(v.tobytes()).hexdigest())
    if dataclasses.is_dataclass(v): return {f.name:fingerprint(getattr(v,f.name)) for f in dataclasses.fields(v)}
    if isinstance(v, dict): return {k:fingerprint(x) for k,x in v.items()}
    if isinstance(v, (tuple,list)): return [fingerprint(x) for x in v]
    if isinstance(v, np.generic): return v.item()
    if isinstance(v, float) and np.isnan(v): return 'NaN'
    return v


def assert_bytes(a, b):
    a,b=np.asarray(a),np.asarray(b)
    assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


@pytest.mark.parametrize('key,value', [
    ('extra',0), ('normal',[0,0,2]), ('u_axis',[0,0,1]), ('shape',[100.,120]),
    ('shape',[True,120]), ('roi',[[0,101],[0,120]]), ('center_cm',[0,np.inf,0]),
    ('normal',[0,np.nan,1]), ('size_cm',[0,2]), ('shape',[1]), ('center_cm',[1,2]),
    ('roi',[[False,2],[0,2]]), ('roi',None), ('u_axis','bad')])
def test_gpu_g4_detector_errors(key,value):
    raw=copy.deepcopy(load_scene(str(SCENE)).raw);raw['detector'][key]=value
    s=validate_scene(raw)
    assert any(e.path=='detector.'+key for e in s.errors)


@pytest.mark.parametrize('value', [None, [], 'bad', {}, {'center_cm':[0,0,0]}])
def test_gpu_g4_detector_malformed(value):
    raw=copy.deepcopy(load_scene(str(SCENE)).raw);raw['detector']=value
    s=validate_scene(raw)
    assert not s.ok and all(e.path.startswith('detector') for e in s.errors)


def test_gpu_g4_scene_copy_and_declaration():
    original=yaml.safe_load((ROOT/'docs/jsmp133/request_d_history_noise/scene_portrait.yaml').read_text())
    copied=yaml.safe_load(SCENE.read_text())
    assert copied.pop('detector')==DET and copied==original
    s=load_scene(str(SCENE));assert s.ok
    p,roi=detector_from_scene(s)
    assert p.shape==(100,120) and roi is None


@pytest.mark.parametrize('number', [1,2])
def test_gpu_g4_frozen_engine(number):
    fixed=json.loads((DOC/'local_baseline.json').read_text())
    r=run_transport(load_scene(str(ROOT/f'docs/validation/gpu/g1/scenes/s{number}.yaml')),n_histories=1000,seed=42,batch_size=400,engine='gpu',detector=DetectorPlane([0,0,0],[0,0,1],[1,0,0],(30,30),(60,60)))
    expected=json.loads(json.dumps(fixed[f'gpu_s{number}']).replace('NaN','"NaN"'))
    assert fingerprint(r)==expected


def test_gpu_g4_frozen_kernel():
    fixed=json.loads((DOC/'local_baseline.json').read_text())
    r=run_transport(load_scene(str(ROOT/'examples/water_phantom_pdd_ocr.yaml')),n_histories=10000,seed=42,engine='kernel')
    assert fingerprint(r)==json.loads(json.dumps(fixed['kernel']).replace('NaN','"NaN"'))


@pytest.mark.parametrize('engine,n', [('numpy',400),('gpu',10000)])
def test_gpu_g4_cli_direct_identical(engine,n,tmp_path,monkeypatch):
    path=tmp_path/'image.npz'
    assert invoke(monkeypatch,'run',SCENE,'--engine',engine,'-n',n,'--seed',2040,'--batch-size',200,'--detector-out',path)==0
    scene=load_scene(str(SCENE));plane,roi=detector_from_scene(scene)
    direct=run_transport(scene,n_histories=n,seed=2040,batch_size=200,engine=engine,detector=plane,detector_roi=roi)
    with np.load(path,allow_pickle=False) as z:
        for key in ('category_fluence','photon_count','energy_sum_keV','energy_sum2_keV2','n_batches_hit'):
            assert_bytes(z[key],getattr(direct.detector,key))
        assert_bytes(z['total_rel_err'],direct.detector.total_relative_error())
    meta=json.loads(path.with_suffix('.json').read_text())
    assert meta['energy_deposited_MeV']==direct.energy_deposited_MeV
    assert len(meta['condition_sha256'])==64 and path.with_suffix('.png').is_file()


@pytest.fixture
def prepared(tmp_path):
    scene=load_scene(str(SCENE));args=options()
    path=approval.prepare(scene,args,tmp_path/'run')
    return scene,args,path


def test_gpu_g4_approval_tamper(prepared):
    scene,args,path=prepared
    cond=json.loads((path/'generation.json').read_text())['condition']
    with pytest.raises(ValueError,match='approval.json missing'): approval.require(path,cond)
    approval.approve(path);approval.require(path,cond)
    with pytest.raises(ValueError,match='approved run directory'): approval.prepare(scene,args,path)
    changed=copy.deepcopy(cond);changed['track_uncertainty']=False
    with pytest.raises(ValueError,match='condition mismatch'): approval.require(path,changed)
    from chatcarlo.gpu.cli_run import expected_run
    for key in ('seed','n_histories'):
        changed_args=dict(args);changed_args[key]+=1
        with pytest.raises(ValueError,match='condition mismatch'): approval.require(path,expected_run(scene,condition.resolve(changed_args)))
    changed=copy.deepcopy(scene);changed.raw['geometry'][0]['size_cm'][0]+=.1
    with pytest.raises(ValueError,match='condition mismatch'): approval.require(path,expected_run(changed,condition.resolve(args)))
    member=path/'bundle/source.json';member.write_text(member.read_text()+' ')
    with pytest.raises(ValueError,match='bundle SHA256'): approval.require(path,cond)


def test_gpu_g4_html_tamper(prepared):
    _,_,path=prepared
    html=path/'preview.html';text=html.read_text();digest=json.loads((path/'generation.json').read_text())['condition_sha256']
    # Keep the visible digest, modify detector drawing coordinates only.
    assert digest in text
    text=text.replace('"label_at": [0.0, 0.0, 0.0]', '"label_at": [1.0, 0.0, 0.0]')
    assert text!=html.read_text();html.write_text(text)
    with pytest.raises(ValueError,match='HTML SHA256'):approval.approve(path)
    assert not (path/'approval.json').exists()


@pytest.mark.parametrize('fault', ['seed','start'])
def test_gpu_g4_local_fault(fault,tmp_path,monkeypatch):
    monkeypatch.setenv('CHATCARLO_G4_FAULT',fault)
    assert invoke(monkeypatch,'run',SCENE,'--engine','gpu','-n',20,'--seed',2040,'--detector-out',tmp_path/'image.npz')==1
    assert not (tmp_path/'image.npz').exists()


@pytest.mark.parametrize('tracked', [False,True])
def test_gpu_g4_cpu_worker_approval_and_timing(tracked,tmp_path):
    scene=load_scene(str(SCENE));args=options();args['no_uncertainty']=not tracked
    path=approval.prepare(scene,args,tmp_path/'run');approval.approve(path)
    cond=json.loads((path/'generation.json').read_text())['condition'];approval.require(path,cond)
    job=json.loads((path/'job.json').read_text());job['device']='cpu'
    jobpath=tmp_path/'cpu_job.json';jobpath.write_text(json.dumps(job))
    results=tmp_path/'results';colab_driver.worker(path/'bundle',jobpath,results)
    batches,timing=approval.verify_batches(path/'bundle',job,results)
    assert timing['new_batches']==3 and timing['reused_batches']==0
    names=[f'batch_{b:06d}.npz' for b in range(3)]
    _,all_reused=approval.verify_batches(path/'bundle',job,results,names)
    _,partial=approval.verify_batches(path/'bundle',job,results,names[:1])
    assert all_reused['elapsed_s']==timing['elapsed_s']==partial['elapsed_s']
    assert all_reused['new_elapsed_s']==0 and partial['new_batches']==2 and partial['reused_batches']==1
    r=aggregate(batches,['air','aluminum'],detector_from_scene(scene)[0],track_uncertainty=tracked)
    outputs.save(r,tmp_path/'image.npz',{})
    with np.load(tmp_path/'image.npz',allow_pickle=False) as z:
        if not tracked:
            assert np.isnan(z['total_rel_err']).all() and (z['n_batches_hit']==-1).all()
    receipt=results/(names[0]+'.receipt.json');old=json.loads(receipt.read_text());new=copy.deepcopy(old);new.pop('observed');receipt.write_text(json.dumps(new))
    with pytest.raises(ValueError,match='observation mismatch'):approval.verify_batches(path/'bundle',job,results,names)
    new=copy.deepcopy(old);new['identity']['job']['seed']+=1;receipt.write_text(json.dumps(new))
    with pytest.raises(ValueError,match='result identity mismatch'):approval.verify_batches(path/'bundle',job,results)


@pytest.mark.parametrize('fault', ['seed','start'])
def test_gpu_g4_worker_fault_observation(fault,prepared,tmp_path,monkeypatch):
    _,_,path=prepared;job=json.loads((path/'job.json').read_text());job['device']='cpu'
    jp=tmp_path/'cpu_job.json';jp.write_text(json.dumps(job));(tmp_path/'fault.json').write_text(json.dumps({'fault':fault}))
    # worker reads fault.json on the VM side, separate from approved job.
    monkeypatch.setenv('CHATCARLO_G4_FAULT','')
    colab_driver.worker(path/'bundle',jp,tmp_path/'results')
    with pytest.raises(ValueError,match='observation mismatch: batch_000000.*'+fault):approval.verify_batches(path/'bundle',job,tmp_path/'results')
    monkeypatch.setenv('CHATCARLO_G4_FAULT','')


def test_gpu_g4_png_orientation():
    from chatcarlo.detector import DetectorTally
    t=DetectorTally(DetectorPlane([0,0,0],[0,0,1],[1,0,0],(6,8),(2,3)))
    t.photon_count[:]=[[1,2,3],[4,5,9]]
    fig=outputs.count_figure(t);ax=fig.axes[0];im=ax.images[0]
    assert_bytes(np.asarray(im.get_array()),t.photon_count.T)
    assert im.origin=='lower' and ax.get_xlabel()=='u pixel index' and ax.get_ylabel()=='v pixel index'
    assert [x.get_xlabel() or x.get_ylabel() for x in ax.child_axes]==['u [cm]','v [cm]']
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_gpu_g4_preview_bounds():
    from chatcarlo.preview import scene_to_json
    scene=load_scene(str(SCENE));scene.raw['detector'].update(center_cm=[1000,0,0],normal=[0,1,0])
    data=scene_to_json(scene);p,_=detector_from_scene(scene)
    obj=data['objects'][-1]
    np.testing.assert_array_equal(obj['faces'][0],p.corners_cm()[[0,2,3,1]])
    for corner in p.corners_cm(): assert np.linalg.norm(corner-data['center'])<=data['radius']


@pytest.mark.parametrize('argv', [[], ['--gpu-device','cpu','--engine','numpy'], ['--engine','gpu','--workers','2','--seed','1'], ['--engine','gpu','--dose-grid','--seed','1'], ['--engine','gpu','--kernel-chunks','0','--seed','1']])
def test_gpu_g4_cli_invalid(argv,monkeypatch,tmp_path):
    if not argv:argv=['--engine','gpu']
    assert invoke(monkeypatch,'run',SCENE,*argv,'-n',10,'--detector-out',tmp_path/'image.npz')==1


@pytest.mark.parametrize('prefix', [[],['--preview','somewhere']])
def test_gpu_g4_mv_approve_routing(prefix,monkeypatch):
    from prototypes.mv_electron.entry import cli as mv
    scene=str(ROOT/'examples/therapy/water_10mv_10x10.yaml')
    argv=['approve',*prefix,scene]+([] if prefix else ['--preview','somewhere'])
    monkeypatch.setattr(mv,'main',lambda got: 19 if got==argv else 0)
    assert cli._mv_dispatch(argv)==19


@pytest.mark.parametrize('opt,val', [('--gpu-device','cpu'),('--source-seed','12'),('--detector-out','image.npz'),('--run-dir','run')])
def test_gpu_g4_mv_new_options(opt,val,monkeypatch):
    scene=ROOT/'examples/therapy/water_10mv_10x10.yaml'
    with pytest.raises(SystemExit) as e:invoke(monkeypatch,'run',opt,val,scene)
    assert e.value.code==2


@pytest.mark.parametrize('track', [False,True])
@pytest.mark.parametrize('with_plane', [False,True])
@pytest.mark.parametrize('sizes', [(5,), (5,5,2)])
def test_gpu_g4_aggregate_independent(track,with_plane,sizes):
    """Independent loops and explicit moment equations, including zero-hit batches."""
    names=['water','air','lead']
    plane=DetectorPlane([0,0,0],[0,0,1],[1,0,0],(2,3),(2,3)) if with_plane else None
    roi=((0,2),(0,3)) if plane else None
    batches=[]
    for b,n in enumerate(sizes):
        images=np.zeros((7,2,3))
        if b!=1:
            images[0]=[[1,0,2],[3,1,0]]
            images[1]=[[0,1,1],[0,2,0]]
            images[4]=images[:4].sum(axis=0)
            images[5]=images[4]*40;images[6]=images[4]*1600
        batches.append((n,dict(deposits=np.array([1.1+b,0.,3.2+b]),endings=np.array([2.,n-2,0]),scatter_sum=2.3+b,n_fluorescence=b,images=images)))
    r=aggregate(batches,names,plane,roi,track)
    N=sum(sizes);M=len(sizes)
    sums=np.zeros(3);Q=np.zeros(3);ends=np.zeros(3);scatter=fluor=0.
    image=np.zeros((7,2,3));catQ=np.zeros((4,2,3));totalQ=np.zeros((2,3));hit=np.zeros((2,3),np.int32)
    prev=np.zeros((4,2,3));p=s=qp=qs=cps=0.
    for n,b in batches:
        for i in range(3):sums[i]+=float(b['deposits'][i]);Q[i]+=float(b['deposits'][i])**2/n
        ends+=b['endings'];scatter+=b['scatter_sum'];fluor+=b['n_fluorescence']
        image+=b['images']
        delta=image[:4]-prev;prev=image[:4].copy()
        catQ+=delta**2/n;totalQ+=delta.sum(axis=0)**2/n;hit+=np.any(delta!=0,axis=0).astype(np.int32)
        pb=delta[0].sum();sb=delta[1:].sum();p+=pb;s+=sb;qp+=pb**2/n;qs+=sb**2/n;cps+=pb*sb/n
    assert r.n_histories==N and r.n_batches==(M if track else 0)
    assert r.fraction_absorbed==ends[0]/N and r.fraction_escaped==ends[1]/N and r.mean_scatter_events==scatter/N and r.n_fluorescence==fluor
    for i,name in enumerate(names):
        assert r.energy_deposited_MeV[name]==sums[i]/1000
        if track:
            sem=np.sqrt(max(0.,(Q[i]-sums[i]**2/N)/(M-1))/N) if M>=2 else np.nan
            assert_bytes(r.energy_deposited_sem_MeV[name],sem/1000)
            expected=sem/(sums[i]/N) if sums[i] else np.nan
            assert_bytes(r.energy_deposited_rel_err[name],expected)
    if plane:
        t=r.detector
        for name,i in [('photon_count',4),('energy_sum_keV',5),('energy_sum2_keV2',6)]:assert_bytes(getattr(t,name),image[i])
        assert_bytes(t.category_fluence,image[:4])
        if track:
            assert_bytes(t.category_sum2,catQ);assert_bytes(t.total_sum2,totalQ);assert_bytes(t.n_batches_hit,hit)
            assert (t.roi_P,t.roi_S,t.roi_QP,t.roi_QS,t.roi_CPS)==(p,s,qp,qs,cps)
            total=image[:4].sum(axis=0)
            with np.errstate(invalid='ignore',divide='ignore'):
                sem=np.sqrt(np.clip((totalQ-total**2/N)/(M-1),0,None)/N) if M>=2 else np.full_like(total,np.nan)
                rel=np.where(total!=0,sem/(total/N),np.nan)
            np.testing.assert_array_equal(t.total_relative_error(),rel)
            if M>=2:
                vp=max(0.,(qp-p*p/N)/(M-1))/N;vs=max(0.,(qs-s*s/N)/(M-1))/N;cov=(cps-p*s/N)/(M-1)/N
                expected=np.sqrt(max(0.,(s/p)**2*(vs/(s/N)**2+vp/(p/N)**2-2*cov/((s/N)*(p/N)))))
                assert_bytes(t.stpr_sem(),expected)
        else:assert t.total_sum2 is None and t.n_batches_hit is None
    else:assert r.detector is None


def test_gpu_g4_image_regression():
    from chatcarlo.gpu.validation import image_regression
    critical=set(core.SOURCES)|{'tables.npz','geometry.npz','spectrum.npz','source.json'}
    meta=dict(seed=2040,source_seed=3712336832425968660,batch_size=100000,n_histories=1000000,bundle={'files':{k:'hash' for k in critical|{'core.py','settings.json'}}})
    a=dict(photon_count=np.array([[0.,1.],[2.,10.]]), category_fluence=np.ones((4,2,2)),energy_sum_keV=np.ones((2,2)),energy_sum2_keV2=np.ones((2,2)))
    b={k:v.copy() for k,v in a.items()}
    b['energy_sum_keV'][0,0]=np.nextafter(1.,2.)
    assert image_regression(a,b,meta,meta)['status']=='pass'
    b['energy_sum_keV'][0,0]+=1e-9
    assert image_regression(a,b,meta,meta)['status']=='fail'
    b={k:v.copy() for k,v in a.items()};a['energy_sum_keV'][0,0]=0.;b['energy_sum_keV'][0,0]=1e-300
    assert image_regression(a,b,meta,meta)['status']=='fail'
    b['energy_sum_keV'][0,0]=0.
    assert image_regression(a,b,meta,meta)['status']=='pass'
    b['photon_count'][0,0]+=1
    assert image_regression(a,b,meta,meta)['status']=='fail'
    changed=copy.deepcopy(meta);changed['bundle']['files']['core.py']='different'
    assert image_regression(a,a,meta,changed)['status']=='pass'
    assert image_regression(a,a,meta,changed)['runtime_differences']==['core.py']
    changed['bundle']['files']['tables.npz']='different'
    assert image_regression(a,a,meta,changed)['status']=='inapplicable'


def test_gpu_g4_observe_default_and_positions(prepared):
    from chatcarlo.gpu.source_host import sample_source_photons
    _,_,path=prepared;a,src,spec,settings,_=bundle.load(path/'bundle')
    o,d,e=sample_source_photons(src,10,np.random.default_rng(123),spec)
    driver=core.Core(a)
    plain=driver.run(o,d,e,seed=12)
    observed_result,observed=driver.run(o,d,e,seed=12,observe=True)
    for key in plain:assert_bytes(plain[key],observed_result[key])
    expected=core.expected_observation(a,o,d,e,12,0,settings['fluorescence'],'cpu')
    assert observed==expected
    # ABI position mixup is visible even when the caller labels stay unchanged.
    original=driver._observe
    def swapped(args,arrays):
        geometry=core.ctypes.cast(args[5],core.ctypes.POINTER(core.Geometry)).contents
        geometry.centers,geometry.halves=geometry.halves,geometry.centers
        result=original(args,arrays)
        geometry.centers,geometry.halves=geometry.halves,geometry.centers
        return result
    driver._observe=swapped
    _,actual=driver.run(o,d,e,seed=12,observe=True)
    with pytest.raises(ValueError,match='observation mismatch'):approval.check_observation(actual,expected,'batch_000000')


def test_gpu_g4_existing_scene_validation_unchanged(tmp_path):
    import subprocess
    import importlib.util
    previous=subprocess.check_output(['git','show','HEAD:chatcarlo/scene.py'],cwd=ROOT,text=True)
    path=tmp_path/'old_scene.py';path.write_text(previous)
    spec=importlib.util.spec_from_file_location('_g4_old_scene',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    try:
        for example in sorted((ROOT/'examples').glob('*.yaml')):
            raw=yaml.safe_load(example.read_text())
            if 'detector' in raw or raw.get('domain')=='therapy_mv':continue
            old=module.validate_scene(copy.deepcopy(raw));new=validate_scene(copy.deepcopy(raw))
            assert [str(e) for e in old.errors]==[str(e) for e in new.errors]
            assert [str(e) for e in old.warnings]==[str(e) for e in new.warnings]
            assert old.raw==new.raw
    finally:sys.modules.pop(spec.name,None)


def test_gpu_g4_cli_approval(prepared,monkeypatch,tmp_path):
    _,_,path=prepared
    assert invoke(monkeypatch,'approve',tmp_path)==1
    assert invoke(monkeypatch,'approve',path)==0
    html=path/'preview.html';html.write_text(html.read_text()+' ')
    assert invoke(monkeypatch,'approve',path)==1


def test_gpu_g4_cli_no_approval_stops_before_launch(tmp_path,monkeypatch):
    from chatcarlo.gpu import colab_driver
    def forbidden(*args,**kwargs):raise AssertionError('unapproved launch')
    monkeypatch.setattr(colab_driver,'launch',forbidden)
    assert invoke(monkeypatch,'run',SCENE,'--engine','gpu','--gpu-device','colab','-n',10,'--seed',12,'--run-dir',tmp_path)==1


def test_gpu_g4_no_uncertainty_approved_cpu_run(tmp_path):
    # Test fixture for the same immutable approval workflow on the CPU twin;
    # production previews create CUDA jobs, and never offer local approval.
    scene=load_scene(str(SCENE));args=options();args['no_uncertainty']=True
    path=approval.prepare(scene,args,tmp_path/'run')
    job=json.loads((path/'job.json').read_text());job['device']='cpu'
    generation=json.loads((path/'generation.json').read_text())
    cond=generation['condition'];cond['job']=job
    (path/'job.json').write_text(condition.canonical(job)+'\n')
    snapshot=json.loads((path/'preview_snapshot.json').read_text());snapshot['condition']=cond
    (path/'preview_snapshot.json').write_text(condition.canonical(snapshot)+'\n')
    # Create a CPU fixture's HTML and generation as the CPU analogue of prepare.
    from chatcarlo.preview import render_html
    (path/'preview.html').write_text(render_html(scene,approval_text=condition.sha256(cond)))
    generation['condition_sha256']=condition.sha256(cond)
    generation['files']={name:bundle.sha(path/name) for name in generation['files']}
    (path/'generation.json').write_text(condition.canonical(generation)+'\n')
    approval.approve(path);approval.require(path,cond)
    colab_driver.worker(path/'bundle',path/'job.json',path/'results')
    batches,_=approval.verify_batches(path/'bundle',job,path/'results')
    r=aggregate(batches,['air','aluminum'],detector_from_scene(scene)[0],track_uncertainty=cond['track_uncertainty'])
    assert r.n_histories==41 and r.detector.track_uncertainty is False
    changed=copy.deepcopy(cond);changed['track_uncertainty']=True
    with pytest.raises(ValueError,match='condition mismatch'):approval.require(path,changed)


@pytest.mark.parametrize('with_plane',[False,True])
def test_gpu_g4_cuda_observation_positions_without_device(prepared,with_plane):
    _,_,path=prepared;a,_,_,settings,_=bundle.load(path/'bundle')
    if not with_plane:a['detector']=np.array([])
    class DeviceArray(np.ndarray):
        @property
        def data(self):return SimpleNamespace(ptr=self.ctypes.data)
    driver=object.__new__(core.Core);driver.a=a;driver.device='cuda';driver._hash_cache={}
    driver.cp=SimpleNamespace(ndarray=DeviceArray,asnumpy=np.asarray)
    arrays={k:v.view(DeviceArray) for k,v in a.items()}
    o=np.zeros((2,3));d=np.ones((2,3));e=np.full(2,60.)
    arrays.update(origins=o.view(DeviceArray),directions=d.view(DeviceArray),energies=e.view(DeviceArray))
    scalars=dict(me=int(a['me']),ng=int(a['ng']),nm=int(a['nm']),nb=len(a['materials']),bg=int(a['background']),seed=12,start=0,n=2,fluorescence=int(settings['fluorescence']),record=0,trace_capacity=0)
    args=[None]*25
    for pos,name in core.CUDA_INPUTS.items():args[pos]=arrays[name] if name in core.ARRAY_INPUTS else scalars[name]
    if not with_plane:args[11]=np.uint64(0)
    expected=core.expected_observation(a,o,d,e,12,0,settings['fluorescence'],'cuda')
    assert driver._observe(args,arrays)==expected
    # Cache must not hide a swap of actual pointer positions.
    args[5],args[6]=args[6],args[5]
    with pytest.raises(ValueError,match='observation mismatch'):approval.check_observation(driver._observe(args,arrays),expected,'batch_000000')


def test_gpu_g4_colab_output_copy(tmp_path):
    from chatcarlo.gpu.cli_run import save_detector
    scene=load_scene(str(SCENE));plane,_=detector_from_scene(scene)
    r=aggregate([(1,dict(deposits=np.array([1.,2.]),endings=np.array([1.,0.,0.]),scatter_sum=0.,n_fluorescence=0.,images=np.zeros((7,100,120))))],['air','aluminum'],plane)
    args=SimpleNamespace(**options(),run_dir=tmp_path/'run',detector_out=tmp_path/'copy/image.npz')
    r.gpu_metadata=dict(condition=condition.local_condition(scene,condition.resolve(args)))
    save_detector(scene,args,r)
    for suffix in ('.npz','.json','.png'):
        assert (tmp_path/'run'/('detector'+suffix)).read_bytes()==(tmp_path/'copy'/('image'+suffix)).read_bytes()


def test_gpu_g4_existing_preview_bytes_unchanged(tmp_path):
    import subprocess
    import importlib.util
    from chatcarlo.preview import render_html
    previous=subprocess.check_output(['git','show','HEAD:chatcarlo/preview.py'],cwd=ROOT,text=True)
    path=tmp_path/'old_preview.py';path.write_text(previous)
    spec=importlib.util.spec_from_file_location('chatcarlo._g4_old_preview',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for example in sorted((ROOT/'examples').glob('*.yaml')):
        scene=load_scene(str(example))
        if scene.ok and 'detector' not in scene.raw:
            assert render_html(scene).encode()==module.render_html(scene).encode()
