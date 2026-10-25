"""CPU-only P9c entry contracts; no Colab calls and no external cache writes."""
from copy import deepcopy
import json
from pathlib import Path
import re
import numpy as np
import pytest
import yaml
from prototypes.mv_electron.entry import scene, seed_scene, seed_condition as condition
from prototypes.mv_electron.entry import seed_preview as preview, seed_run as runner, seed_analysis as analysis
from prototypes.mv_electron.gpu import p9c_core as core, p9c_io as io, p9c_colab as colab
from prototypes.mv_electron.gpu.p5b_core import write_json, Stop

REJECTED = (ValueError, Stop)
from prototypes.mv_electron.photon_transport_p8e import Material


@pytest.fixture
def s():
    return yaml.safe_load(Path('examples/therapy/prostate_seed_7field.yaml').read_text())


@pytest.fixture
def assets(tmp_path, monkeypatch):
    export=tmp_path/'export'; export.mkdir(); t0=tmp_path/'t0'; t0.mkdir()
    monkeypatch.setenv('CHATCARLO_P9C_EXPORT',str(export)); monkeypatch.setenv('CHATCARLO_P9C_T0',str(t0))
    # A structurally valid, tiny material bundle exercises the real read-only loader.
    for j in range(5):
        np.savez(export/f'material_{j}.npz',**{k:np.array([1.]) for k in Material._fields})
    (export/'source.zip').write_bytes(b'fake source'); np.savez(export/'cpu_reference.npz',x=[1])
    local={g:dict(status='pass') for g in condition.LOCAL_GATES}; local['V5']['cpu_pair']='exact test'
    sources=core.fingerprint()
    write_json(export/'input.json',dict(sources=sources,bundles={'test':'materials'},local=local))
    refresh(export)
    gates={g:dict(status='pass') for g in colab.INITIAL_GATES}
    write_json(t0/'job.json',dict(mode='t0',sources=sources,status='partial-accepted',reason='test accepted rectangle'))
    write_json(t0/'gates.json',gates); write_json(t0/'rectangle_summary.json',dict(batches=100))
    monkeypatch.setattr(core,'FINE',core.Grid((-3.,-2.5,-2.5),(.5,)*3,(12,10,15)))
    monkeypatch.setattr(core,'COARSE',core.Grid((-20.,)*3,(10.,)*3,(4,)*3))
    def forbidden(*args,**kwargs):
        pytest.fail('real Colab call forbidden')
    monkeypatch.setattr(runner.client,'call',forbidden)
    return export,t0


def refresh(export):
    write_json(export/'manifest.json',{p.name:io.sha(p) for p in export.iterdir() if p.name!='manifest.json'})


def approved(s,tmp_path):
    out=tmp_path/'approval'; preview.write(s,out); preview.approve(s,out)
    return out


def fake_result(args, cond):
    out=args.output; out.mkdir(exist_ok=True)
    identity=condition.checkpoint_identity(cond)
    payload={k:np.zeros((5,*v)) for k,v in runner.shapes().items()}
    # Constant paired batch values, producing a defined ratio and zero SEM.
    for a in payload.values():
        a[:]=100
    np.savez(out/'prostate.npz',identity=json.dumps(identity,sort_keys=True),completed=np.arange(100),**payload)
    summary=dict(batches=100,checkpoint_sha256=io.sha(out/'prostate.npz'),V3_CUDA_acceptance=dict(status='pass'))
    gates={g:dict(status='pass') for g in condition.GPU_GATES}
    gates['V2b']=dict(status='insufficient-statistics',fallback='V2a')
    write_json(out/'prostate_summary.json',summary)
    write_json(out/'job.json',dict(status='complete',mode='production',sources=cond['sources'],beams=['prostate'],gates=gates,results={'prostate':summary}))
    return 0


def mutate_cp(out, edit):
    p=out/'prostate.npz'
    with np.load(p) as f:data={k:f[k].copy() for k in f.files}
    edit(data); np.savez(p,**data)


def test_allowlist(s):
    assert scene.validate(s)==s
    for n in (100,7142900):
        c=deepcopy(s); c['run']['histories_per_field']=n; scene.validate(c)
    for seed in (0,2**58-1):
        c=deepcopy(s); c['run']['seed']=seed; scene.validate(c)


@pytest.mark.parametrize('path,value',[
    ('source.gantry_deg',[0,45]),('source.gantry_deg',[0,45,90,135,225,270,270]),
    ('source.gantry_deg',[45,0,90,135,225,270,315]),('source.field','rectangle_7x7'),
    ('source.sad_cm',True),('source.sad_cm','100'),('body.ellipse_cm',[36,True]),
    ('target.prostate_cm',[4.5,3.,None]),('seeds.material',None),('run.engine','cpu_local'),
    ('run.histories_per_field',99),('run.histories_per_field',101),('run.histories_per_field',7143000),
    ('run.histories_per_field',100.),('run.histories_per_field',True),('run.histories_per_field',None),
    ('run.seed',-1),('run.seed',2**58),('run.seed',True),('run.seed',1.),('run.seed',None),
    ('source',None),('body',[]),('run',None),
])
def test_reject(s,path,value):
    c=deepcopy(s); target=c; keys=path.split('.')
    for k in keys[:-1]:target=target[k]
    target[keys[-1]]=value
    with pytest.raises(scene.SceneError,match=keys[0]):scene.validate(c)


@pytest.mark.parametrize('path',['','source','body','target','seeds','run'])
def test_unknown_and_missing(s,path):
    for missing in (False,True):
        c=deepcopy(s); target=c[path] if path else c
        if missing:del target[next(iter(target))]
        else:target['unknown']=1
        with pytest.raises(scene.SceneError):scene.validate(c)


@pytest.mark.parametrize('raw',[None,[],5,'seed_ratio'])
def test_non_mapping(raw):
    with pytest.raises(scene.SceneError,match='mapping'):scene.validate(raw)


def test_geometry(s):
    g=preview.geometry_json(s); objects=g['objects']
    assert seed_scene.prostate_diameters(s)==[4.5,3.5,3.]
    body,prostate,rectum=objects[:3]
    assert body['radii_cm']==[18,12] and body['length_cm']==30 and len(body['faces'])>=64
    pts=np.array(body['edges']).reshape(-1,3)
    np.testing.assert_allclose(pts.min(0),[-18,-15,-12]); np.testing.assert_allclose(pts.max(0),[18,15,12])
    pts=np.array(prostate['edges']).reshape(-1,3)
    np.testing.assert_allclose(pts.max(0),[2.25,1.75,1.5])
    assert rectum['label_at']==[0,0,3] and rectum['length_cm']==3.5 and rectum['radius_cm']==1.5
    for obj,seed in zip(objects[3:61],core.seeds()):
        assert obj['label_at']==list(seed[:3]) and obj['radius_cm']==.05 and obj['half_length_cm']==.25 and obj['axis']=='y'
    assert len(objects[3:61])==58
    for beam,angle in zip(g['sources'],core.ANGLES):
        np.testing.assert_allclose(beam['position'],core.frame(angle)[0]); assert np.linalg.norm(beam['position'])==pytest.approx(100)
    assert g['sources'][0]['distance_to_surface_cm']==pytest.approx(88)
    assert g['sources'][2]['distance_to_surface_cm']==pytest.approx(82)


def test_no_approval(s,assets,tmp_path):
    out=tmp_path/'missing'
    with pytest.raises(REJECTED,match='approval'):runner.run(s,out)
    assert not out.exists()


def test_visible_and_approval(s,assets,tmp_path):
    out=approved(s,tmp_path); cond=condition.identities(s)
    text=(out/'preview.html').read_text(); visible=re.sub(r'<script.*?</script>','',text,flags=re.S)
    assert all(preview.snapshot(cond)[k] in visible for k in ('condition_hash','bundle_hash','t0_hash'))
    assert '58 本' in visible and 'SAD 100' in visible
    assert ' @ SID ' not in text and ' @ SAD ' in text
    assert '${DATA.beam.field_label} cm' not in text
    assert preview.require_approval(out,cond)['time']


@pytest.mark.parametrize('changed',['seed','histories_per_field','t0'])
def test_changed_condition(s,assets,tmp_path,changed):
    out=approved(s,tmp_path)
    if changed=='t0':
        p=assets[1]/'rectangle_summary.json'; write_json(p,dict(batches=99))
    else:s['run'][changed]+=100
    cond=condition.identities(s)
    with pytest.raises(REJECTED,match='approval condition_hash: mismatch'):runner.run(s,out)
    with pytest.raises(REJECTED,match='approval.json'):runner.run(s)
    assert not (out/'results').exists() and not condition.default_out(cond).exists()


@pytest.mark.parametrize('file',['preview.html','preview_snapshot.json','visible'])
def test_preview_tamper(s,assets,tmp_path,file):
    out=approved(s,tmp_path); cond=condition.identities(s)
    p=out/('preview.html' if file=='visible' else file)
    if file=='visible':
        text=p.read_text(); hash=condition.digest(cond); text=text.replace('条件ハッシュ: '+hash,'条件ハッシュ: removed'); p.write_text(text)
        with pytest.raises(REJECTED,match='visible hash missing'):preview.approve(s,out)
    else:p.write_text(p.read_text()+' ')
    with pytest.raises(REJECTED):runner.run(s,out)
    assert not (out/'results').exists()


@pytest.mark.parametrize('file',condition.T0_FILES)
@pytest.mark.parametrize('operation',['change','remove'])
def test_t0_files(s,assets,tmp_path,file,operation):
    out=approved(s,tmp_path); p=assets[1]/file
    if operation=='remove':p.unlink()
    else:p.write_text(p.read_text()+' ')
    with pytest.raises(REJECTED):runner.run(s,out)


@pytest.mark.parametrize('problem',['bundle_byte','fingerprint','t0_sources','reason','t0_mode','t0_status','local_fail','local_missing','gates'])
def test_assets_rejected(s,assets,problem):
    export,t0=assets
    if problem=='bundle_byte':
        p=export/'source.zip'; p.write_bytes(p.read_bytes()+b'x')
    elif problem in ('fingerprint','local_fail','local_missing'):
        p=export/'input.json'; obj=condition.read(p)
        if problem=='fingerprint':obj['sources']['wrong']='hash'
        elif problem=='local_fail':obj['local']['V6']['status']='fail'
        else:del obj['local']['V6b']
        write_json(p,obj); refresh(export)
    elif problem=='gates':write_json(t0/'gates.json',{})
    else:
        p=t0/'job.json'; obj=condition.read(p)
        if problem=='t0_sources':obj['sources']={}
        elif problem=='reason':del obj['reason']
        elif problem=='t0_mode':obj['mode']='production'
        else:obj['status']='partial'
        write_json(p,obj)
    with pytest.raises(REJECTED,match='P9c 束/T0'):condition.identities(s)


def test_fake_complete_resume_and_locks(s,assets,tmp_path,monkeypatch):
    out=approved(s,tmp_path); cond=condition.identities(s)
    def fake(args):
        assert args.session=='p9c'+condition.digest(cond)[:8]
        assert args.job==f'--mode production --histories-per-field 200000 --t0 /content/p9c_t0 --beams prostate --seed 941000'
        assert args.output==out/'results' and args.interval==600
        assert condition.read(args.output/'condition_hash.json')==condition.digest(cond)
        return fake_result(args,cond)
    monkeypatch.setattr(runner.client,'run',fake)
    record=runner.run(s,out)
    assert (out/'seed_analysis.json').is_file() and (out/'seed_transverse.png').stat().st_size>0
    assert condition.read(out/'run_record.json')==record and 'not independently observed' in record['provenance']
    assert record['gates']['gpu']['V2b']['fallback']=='V2a'
    # Same condition partial data can resume. The supervisor receives the existing checkpoint.
    mutate_cp(out/'results',lambda d:d.update(completed=np.arange(10)))
    runner.run(s,out)
    write_json(out/'results/condition_hash.json','other')
    with pytest.raises(REJECTED,match='existing output condition_hash'):runner.run(s,out)
    (out/'execution.lock').write_text('occupied')
    with pytest.raises(REJECTED,match='二重実行'):runner.run(s,out)
    (out/'execution.lock').unlink()
    (out/'results/condition_hash.json').unlink()
    with pytest.raises(REJECTED,match='condition hash missing'):runner.run(s,out)


@pytest.mark.parametrize('field,value',[
    ('mode','t0'),('beams',['rectangle']),('status','partial'),('sources',{}),
    ('gates',{}),('fail',None),('fallback',None),('batches',99),('v3','fail'),('sha','stale'),
])
def test_bad_job(s,assets,tmp_path,field,value):
    cond=condition.identities(s); out=tmp_path/'result'; fake_result(type('Args',(),dict(output=out)),cond)
    job=condition.read(out/'job.json')
    if field=='fail':job['gates']['V1c']['status']='fail'
    elif field=='fallback':job['gates']['V2b']['fallback']='wrong'
    elif field=='batches':job['results']['prostate']['batches']=value
    elif field=='v3':job['results']['prostate']['V3_CUDA_acceptance']['status']=value
    elif field=='sha':job['results']['prostate']['checkpoint_sha256']=value
    else:job[field]=value
    write_json(out/'job.json',job)
    with pytest.raises(REJECTED):runner.verify(out,cond)


@pytest.mark.parametrize('field',['n','seed','angles','grids','beam','sources','bundles','lines_seed_ids','extra','missing','duplicate','noncontiguous','summary_sha','summary_batches','summary_gate'])
def test_bad_checkpoint(s,assets,tmp_path,field):
    cond=condition.identities(s); out=tmp_path/'result'; fake_result(type('Args',(),dict(output=out)),cond)
    def change(data):
        if field in ('missing','duplicate','noncontiguous'):
            data['completed']=np.arange(99) if field=='missing' else np.array([*range(99),98 if field=='duplicate' else 100])
        else:
            identity=json.loads(str(data['identity'])); identity[field]='changed'
            data['identity']=np.array(json.dumps(identity,sort_keys=True))
    if field.startswith('summary_'):
        p=out/'prostate_summary.json'; summary=condition.read(p)
        if field=='summary_sha':summary['checkpoint_sha256']='old'
        elif field=='summary_batches':summary['batches']=99
        else:del summary['V3_CUDA_acceptance']
        write_json(p,summary)
    else:mutate_cp(out,change)
    with pytest.raises(REJECTED):runner.verify(out,cond)


@pytest.mark.parametrize('kind',['zero','denominator','no_cells'])
def test_undefined_analysis(s,assets,tmp_path,kind):
    cond=condition.identities(s); out=tmp_path/'result'; fake_result(type('Args',(),dict(output=out)),cond)
    def change(d):
        d['fine'][:]=0; d['aggregate'][:]=0
        if kind=='denominator':d['fine'][0]=100
    mutate_cp(out,change)
    grid=core.FINE
    if kind=='no_cells':grid=core.Grid((100.,)*3,(.5,)*3,grid.shape)
    report=analysis.analyse(out/'prostate.npz',tmp_path/'arbitrary_output',grid)
    assert report['prostate']['ratio'] is None and report['prostate']['reason']
    assert all(row['ratio'] is None and row['reason'] for row in report['radial'])
    assert (tmp_path/'arbitrary_output/seed_transverse.png').is_file()


def test_radial_and_slab_formulas(assets):
    grid=core.FINE; rng=np.random.default_rng(13); shape=grid.shape
    r=dict(mean_with=rng.random(shape),mean_without=rng.random(shape)+1,ratio_sem=rng.random(shape)*.01)
    report,mask,pros=analysis.tables(r,grid)
    x,y,z=grid.centers(); X,Y,Z=np.meshgrid(x,y,z,indexing='ij'); distance=np.full(shape,np.inf)
    for sx,sy,sz,rad,h,_ in core.seeds():
        distance=np.minimum(distance,np.where(abs(Y-sy)<=h-.025,np.hypot(X-sx,Z-sz),np.inf))
    for row in report['radial']:
        lo,hi=row['interval_cm']; sel=pros&np.isfinite(distance)&(distance>=lo)&(distance<hi)
        assert row['cells']==int(sel.sum())
        if sel.any():
            assert row['ratio']==pytest.approx(r['mean_with'][sel].sum()/r['mean_without'][sel].sum())
            assert row['sem']==pytest.approx(np.sqrt(np.sum(r['ratio_sem'][sel]**2))/sel.sum())
    ys=(y>=-.45)&(y<.05); ref=r['mean_without'][pros].mean()
    slab=analysis.slab_arrays(r,mask,pros,grid)
    np.testing.assert_allclose(slab[1],r['mean_without'][:,ys,:].mean(1)/ref)


@pytest.mark.parametrize('gate',condition.GPU_GATES)
@pytest.mark.parametrize('action',['missing','fail'])
def test_each_required_gpu_gate(s,assets,tmp_path,gate,action):
    cond=condition.identities(s); out=tmp_path/'result'; fake_result(type('Args',(),dict(output=out)),cond)
    job=condition.read(out/'job.json')
    if action=='missing':del job['gates'][gate]
    else:job['gates'][gate]=dict(status='fail')
    write_json(out/'job.json',job)
    with pytest.raises(REJECTED):runner.verify(out,cond)


@pytest.mark.parametrize('gate',condition.LOCAL_GATES)
def test_each_local_gate(s,assets,gate):
    export,t0=assets; record=condition.read(export/'input.json'); del record['local'][gate]
    write_json(export/'input.json',record); refresh(export)
    with pytest.raises(REJECTED,match='P9c 束/T0'):condition.identities(s)


def test_cli_fake_flow(s,assets,tmp_path,monkeypatch,capsys):
    from prototypes.mv_electron.entry.cli import main
    path=tmp_path/'scene.yaml'; path.write_text(yaml.safe_dump(s)); out=tmp_path/'out'
    assert main(['validate',str(path)])==0
    assert 'seed_ratio' in capsys.readouterr().out
    assert main(['run',str(path),'--out',str(out)])==1
    assert not out.exists()
    assert main(['preview',str(path),'--out',str(out)])==0
    assert main(['approve',str(path),'--preview',str(out)])==0
    cond=condition.identities(s)
    monkeypatch.setattr(runner.client,'run',lambda args:fake_result(args,cond))
    assert main(['run',str(path),'--out',str(out)])==0
    assert (out/'run_record.json').is_file()


def test_prelaunch_recheck(s,assets,tmp_path,monkeypatch):
    out=approved(s,tmp_path); real=condition.identities; count=0
    def changed(raw):
        nonlocal count
        count+=1
        cond=real(raw)
        if count==2:cond['job']+=' altered'
        return cond
    monkeypatch.setattr(condition,'identities',changed)
    monkeypatch.setattr(runner.client,'run',lambda args:pytest.fail('changed condition launched'))
    with pytest.raises(REJECTED,match='pre-launch conditions'):runner.run(s,out)
    assert not (out/'execution.lock').exists()


def test_renderer_initialization_and_views(s,assets,tmp_path):
    # Execute the actual embedded renderer, including draw/resize and every camera.
    # This catches runtime failures that a JavaScript syntax check cannot detect.
    import shutil
    import subprocess
    node=shutil.which('node')
    if node is None:pytest.skip('optional Node renderer smoke check')
    out=tmp_path/'preview'; preview.write(s,out)
    script=re.search(r'<script\b[^>]*>(.*?)</script>',(out/'preview.html').read_text(),re.S)[1]
    stub='''
const context = new Proxy({}, {get: (o,k) => o[k] || (()=>{})});
const elements = {};
const element = id => elements[id] || (elements[id] = {
  checked:true,style:{},clientWidth:1000,clientHeight:700,
  parentElement:{getBoundingClientRect:()=>({width:1000,height:700})},
  getContext:()=>context,addEventListener:()=>{}
});
globalThis.document = {getElementById:element,querySelectorAll:()=>[]};
globalThis.addEventListener = ()=>{}; globalThis.devicePixelRatio=1;
for (const id of ['ckFaces','ckBeam','ckLabels','ckTraj']) globalThis[id]=element(id);
'''
    smoke=tmp_path/'smoke.js'
    smoke.write_text(stub+script+'''
for (const pair of Object.values(views)) { [yaw,pitch]=pair; resize(); }
for (const id of ['ckFaces','ckBeam','ckLabels','ckTraj']) { element(id).checked=false;draw(); }
if (!element('legend').innerHTML.includes('SAD 100')) throw Error('SAD legend missing');
''')
    subprocess.run([node,str(smoke)],check=True,capture_output=True,text=True)
