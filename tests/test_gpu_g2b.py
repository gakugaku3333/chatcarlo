"""G2b local acceptance; no Colab or CuPy is used."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from chatcarlo import __main__ as cli
from chatcarlo.scene import load_scene, detector_from_scene, validate_scene
from chatcarlo.geometry import Geometry
from chatcarlo.transport import run_transport
from chatcarlo.tally import VoxelGrid
from chatcarlo.diagnostics import dose_map_Gy
from chatcarlo.gpu import condition, approval, bundle, colab_driver, core, g2_core
from chatcarlo.gpu.aggregate import aggregate
from chatcarlo.gpu.g2_tables import config

ROOT = Path(__file__).resolve().parents[1]
S2 = ROOT/'docs/validation/gpu/g2/scene_lead_slab.yaml'


def opts(**kw):
    return dict(dict(engine='gpu', gpu_device='colab', n_histories=41, seed=2401,
                     batch_size=20, dose_grid=True, resolution=2., no_uncertainty=False), **kw)


def invoke(monkeypatch, *argv):
    monkeypatch.setattr(sys, 'argv', ['chatcarlo', *map(str, argv)])
    return cli.main()


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())


def saved(r, scene, uncertainty=True):
    g = r.grid
    d = dose_map_Gy(g, Geometry(scene.raw['geometry'])) / r.n_histories
    h = g.h10_map_pSv() / r.n_histories
    out = dict(dose_per_history_Gy=d, h10_per_history_pSv=h,
               kerma_keV=g.kerma_keV, h10_track_pSv_cm3=g.h10_track_pSv_cm3,
               origin_cm=g.origin_cm, voxel_size_cm=g.voxel_size_cm, shape=np.array(g.shape))
    if uncertainty:
        dr, hr = g.kerma_relative_error(), g.h10_relative_error()
        out.update(rel_err_dose=dr, rel_err_h10=hr, sem_dose_per_history_Gy=dr*d,
                   sem_h10_per_history_pSv=hr*h, n_batches=np.array(g.n_batches),
                   n_batches_hit=g.n_batches_hit)
    return out


@pytest.mark.parametrize('path', [S2, ROOT/'examples/water_phantom_pdd_ocr.yaml'])
@pytest.mark.parametrize('resolution', [2., 5.])
def test_t1_grid_spec(path, resolution):
    scene = load_scene(str(path))
    geometry = Geometry(scene.raw['geometry'])
    reference = VoxelGrid.from_bbox(geometry.tally_bbox_min, geometry.tally_bbox_max, resolution)
    spec = condition.grid_spec(scene, None, resolution)
    condition.check_grid(reference, spec)
    assert spec['layout_version'] == 'g2-grid-v1'


def test_t2_no_grid_legacy(tmp_path):
    scene = load_scene(str(S2))
    options = condition.resolve(opts(dose_grid=False))
    job = condition.scene_job(scene, None, options)
    legacy = dict(mode='batches', device='cuda', **{k: options[k] for k in
                  ('n_histories', 'batch_size', 'seed', 'source_seed')})
    assert condition.canonical(job) == condition.canonical(legacy)
    path = approval.prepare(scene, opts(dose_grid=False), tmp_path/'run')
    generation = json.loads((path/'generation.json').read_text())
    assert set(generation['condition']) == {'format', 'bundle_sha256', 'job', 'detector', 'roi', 'track_uncertainty'}
    from chatcarlo.preview import render_html
    cond = generation['condition']
    text = f"condition SHA256: {condition.sha256(cond)}; histories: {job['n_histories']}; seed: {job['seed']}; source_seed: {job['source_seed']}; detector: {cond['detector']}"
    assert (path/'preview.html').read_text() == render_html(scene, approval_text=text)


def test_t3_gate_a(tmp_path, monkeypatch, capsys):
    path = tmp_path/'dose.npz'
    args = ['run', S2, '--engine', 'gpu', '--gpu-device', 'cpu', '--seed', 2401,
            '-n', '2e5', '--batch-size', 50000, '--dose-grid', '--resolution', 2, '--dose-out', path]
    assert invoke(monkeypatch, *args) == 0
    first = capsys.readouterr().out
    scene = load_scene(str(S2))
    direct = run_transport(scene, engine='gpu', gpu_device='cpu', n_histories=200000,
                           seed=2401, batch_size=50000, dose_grid=True, grid_resolution_cm=2)
    expected = saved(direct, scene)
    with np.load(path) as z:
        assert set(z.files) == set(expected)
        for key in z.files: same(z[key], expected[key])
    from chatcarlo.gpu import cli_run
    direct.gpu_metadata.update(condition_sha256='api', timing={})
    monkeypatch.setattr(cli_run, 'run', lambda *a: direct)
    assert invoke(monkeypatch, *args) == 0
    second = capsys.readouterr().out
    # All dose/statistics/warning lines use the existing CLI presentation.
    select = lambda text: [line for line in text.splitlines()
                          if not line.startswith(('engine:', '計算時間・バッチ:'))]
    assert select(first) == select(second)


@pytest.fixture
def prepared(tmp_path):
    scene = load_scene(str(S2))
    path = approval.prepare(scene, opts(), tmp_path/'run')
    approval.approve(path)
    job = json.loads((path/'job.json').read_text())
    return scene, path, job


@pytest.mark.parametrize('fault', ['origin', 'resolution', 'shape', 'same_size_shape', 'layout', 'remove', 'tables'])
def test_t4_approval_faults(prepared, fault):
    scene, path, job = prepared
    expected = json.loads((path/'generation.json').read_text())['condition']
    if fault == 'tables':
        with (path/'bundle/g2_tables.npz').open('ab') as f: f.write(b'changed')
    else:
        if fault == 'origin': job['grid']['origin_cm'][0] += 1
        if fault == 'resolution': job['grid']['voxel_size_cm'] *= 2
        if fault == 'shape': job['grid']['shape'][2] -= 1
        if fault == 'same_size_shape': job['grid']['shape'].reverse()
        if fault == 'layout': job['grid']['layout_version'] = 'bad'
        if fault == 'remove': job.pop('grid')
        (path/'job.json').write_text(json.dumps(job))
    with pytest.raises(ValueError, match='SHA256 mismatch|condition mismatch'):
        approval.require(path, expected)


@pytest.mark.parametrize('fault, message', [('g2_cfg', ':cfg'), ('g2_cfg_inplace', ':cfg'),
                                           ('g2_table', ':g2_table'), ('g2_swap', 'output mapping mismatch')])
def test_t4_native_faults(monkeypatch, fault, message):
    monkeypatch.setenv('CHATCARLO_G4_FAULT', fault)
    with pytest.raises(ValueError, match=message):
        run_transport(load_scene(str(S2)), engine='gpu', seed=2401, n_histories=41,
                      batch_size=20, dose_grid=True, grid_resolution_cm=2)


@pytest.mark.parametrize('counts, uncertainty', [([100000, 100000, 50000], True),
                                               ([100000], True), ([100000, 100000, 50000], False)])
def test_t5_independent_statistics(counts, uncertainty):
    spec = dict(origin_cm=[0.,0.,0.], shape=[1,1,2], voxel_size_cm=2., layout_version='g2-grid-v1')
    batches = [(n, dict(deposits=np.array([0.]), endings=np.array([float(n),0.,0.]),
                       scatter_sum=0., n_fluorescence=0.,
                       kerma=np.array([v, 0.]), h10=np.array([v*3, 0.])))
               for n,v in zip(counts, [.1,.2,.3])]
    g = condition.grid_for(spec, uncertainty)
    r = aggregate(batches, ['air'], track_uncertainty=uncertainty, grid=g)
    totals = [np.zeros(g.shape), np.zeros(g.shape)]
    qs = [np.zeros(g.shape), np.zeros(g.shape)]
    hits = np.zeros(g.shape, np.int32)
    for n, batch in batches:
        deltas = []
        for i, key in enumerate(('kerma', 'h10')):
            prev = totals[i].copy()
            totals[i] += batch[key].reshape(g.shape)
            delta = totals[i]-prev
            qs[i] += delta**2/n
            deltas.append(delta)
        hits += ((deltas[0] != 0) | (deltas[1] != 0)).astype(np.int32)
    same(g.kerma_keV, totals[0]); same(g.h10_track_pSv_cm3, totals[1])
    assert g.n_batches == (len(counts) if uncertainty else 0)
    assert g.n_histories == (sum(counts) if uncertainty else 0)
    if uncertainty:
        same(g.kerma_sum2, qs[0]); same(g.h10_sum2, qs[1]); same(g.n_batches_hit, hits)
        for i, key in enumerate(('kerma', 'h10')):
            direct = sum(batch[key].reshape(g.shape)**2/n for n, batch in batches)
            np.testing.assert_allclose(qs[i], direct, rtol=1e-12)
        data = saved(r, load_scene(str(S2)), uncertainty)
        for total, q, name, value_name, sem_name in (
            (totals[0], qs[0], 'rel_err_dose', 'dose_per_history_Gy', 'sem_dose_per_history_Gy'),
            (totals[1], qs[1], 'rel_err_h10', 'h10_per_history_pSv', 'sem_h10_per_history_pSv')):
            if len(counts) == 1:
                rel = np.full(g.shape, np.nan)
            else:
                sem = np.sqrt(np.clip((q-total**2/sum(counts))/(len(counts)-1), 0., None)/sum(counts))
                rel = np.divide(sem, total/sum(counts), out=np.full(g.shape,np.nan), where=total != 0)
            same(data[name], rel)
            same(data[sem_name], rel*data[value_name])
    else:
        assert 'rel_err_dose' not in saved(r, load_scene(str(S2)), False)


@pytest.mark.parametrize('device, offset', [('cpu',19), ('cuda',25)])
def test_t6_abi(device, offset):
    base = list(range(offset))
    outputs = [np.zeros(2) for _ in range(6)]
    if device == 'cpu':
        extra = [123,456] + [a.ctypes.data for a in outputs]
        expected_outputs = outputs
    else:
        class Buffer:
            def __init__(self, ptr): self.data = SimpleNamespace(ptr=ptr)
        expected_outputs = [Buffer(i+100) for i in range(6)]
        extra = [123,456] + expected_outputs
    args = g2_core.final_arguments(base, extra)
    assert args[offset:offset+2] == (123,456)
    assert g2_core.OUTPUT_NAMES[:2] == ('kerma', 'h10')
    g2_core.check_outputs(args, expected_outputs, device)
    with pytest.raises(ValueError, match='output mapping mismatch'):
        g2_core.check_outputs(g2_core.final_arguments(base, extra, 'g2_swap'), expected_outputs, device)


def test_t6_inplace_after_observation(prepared, monkeypatch):
    _, path, job = prepared
    a,src,spec,settings,_ = bundle.load(path/'bundle')
    from chatcarlo.gpu.source_host import sample_source_photons
    o,d,e = sample_source_photons(src, 20, np.random.default_rng(job['source_seed']), spec)
    driver = g2_core.Core(a, grid=g2_core.grid_from(job['grid']))
    _, before = driver.run(o,d,e,seed=2401,observe=True)
    addr = driver.cfg.ctypes.data
    monkeypatch.setenv('CHATCARLO_G4_FAULT', 'g2_cfg_inplace')
    _, after = driver.run(o,d,e,seed=2401,observe=True)
    assert driver.cfg.ctypes.data == addr
    assert before['20:cfg'] != after['20:cfg']


def test_t7_worker_aggregate_api(prepared):
    scene, path, job = prepared
    job['device'] = 'cpu'
    (path/'job.json').write_text(json.dumps(job))
    colab_driver.worker(path/'bundle', path/'job.json', path/'results')
    batches, timing = approval.verify_batches(path/'bundle', job, path/'results')
    g = condition.grid_for(job['grid'], True)
    a,_,_,settings,_ = bundle.load(path/'bundle')
    result = aggregate(batches, settings['material_names'], grid=g)
    direct = run_transport(scene, engine='gpu', seed=2401, n_histories=41, batch_size=20,
                           dose_grid=True, grid_resolution_cm=2)
    for key, value in saved(result, scene).items(): same(value, saved(direct,scene)[key])
    assert timing['new_batches'] == 3
    colab_driver.worker(path/'bundle', path/'job.json', path/'results')
    _, timing = approval.verify_batches(path/'bundle', job, path/'results',
                                       [f'batch_{i:06d}.npz' for i in range(3)])
    assert timing['reused_batches'] == 3


def test_t9_preview(tmp_path, monkeypatch):
    path = tmp_path/'preview'
    assert invoke(monkeypatch, 'preview', S2, '--engine', 'gpu', '--gpu-device', 'colab',
                  '--seed', 2401, '--dose-grid', '--resolution', 2, '--run-dir', path) == 0
    job = json.loads((path/'job.json').read_text())
    text = (path/'preview.html').read_text()
    assert 'resolution=2.0 cm' in text
    assert str(job['grid']['origin_cm']) in text and str(job['grid']['shape']) in text
    assert 'voxels='+str(np.prod(job['grid']['shape'])) in text
    assert json.loads((path/'preview_snapshot.json').read_text())['condition']['job']['grid'] == job['grid']


@pytest.mark.parametrize('field', ['kerma', 'h10'])
@pytest.mark.parametrize('fault', ['missing', 'float32', 'length', 'dimension', 'nan', 'inf', 'negative'])
def test_t10_result_validation(prepared, field, fault):
    _,path,job = prepared
    job['device']='cpu'
    (path/'job.json').write_text(json.dumps(job))
    colab_driver.worker(path/'bundle', path/'job.json', path/'results')
    manifest = bundle.verify(path/'bundle')
    name = 'batch_000000.npz'
    key = colab_driver.expected_identity(manifest, job, name)
    r = colab_driver.read_result(path/'results', name, key)
    receipt = json.loads((path/'results'/(name+'.receipt.json')).read_text())
    if fault == 'missing': r.pop(field)
    elif fault == 'float32': r[field] = r[field].astype(np.float32)
    elif fault == 'length': r[field] = r[field][:-1]
    elif fault == 'dimension': r[field] = r[field].reshape(-1,1)
    else: r[field][0] = dict(nan=np.nan, inf=np.inf, negative=-1.)[fault]
    colab_driver.write_result(path/'results',name,r,key,observed=receipt['observed'],elapsed_s=0.)
    with pytest.raises(ValueError, match='invalid G2 output: '+field):
        approval.verify_batches(path/'bundle',job,path/'results')


@pytest.mark.parametrize('mode', ['timeout', 'running', 'failed', 'new_timeout'])
def test_t8_launch(prepared, tmp_path, monkeypatch, mode):
    _,path,job = prepared
    gate = tmp_path/'gate.json'; gate.write_text('{"status":"pass"}')
    commands = []
    starts = polls = 0
    beats = []
    out = path/'results'
    def fake_run(argv, **kwargs):
        nonlocal starts, polls
        assert kwargs['timeout'] == 300
        cmd = argv[1]
        text = kwargs.get('input') or ''
        commands.append(cmd)
        if cmd == 'new' and mode == 'new_timeout':
            raise subprocess.TimeoutExpired(argv,300)
        if cmd == 'exec' and 'with (base/\'worker.log\')' in text:
            starts += 1
            if mode in ('running', 'new_timeout'):
                raise subprocess.TimeoutExpired(argv,300)
            return SimpleNamespace(stdout='G1_LAUNCHED', stderr='')
        if cmd == 'exec' and 'worker.job.sha256' in text:
            return SimpleNamespace(stdout='G1_LAUNCHED', stderr='')
        if cmd == 'exec' and 'G1_RESULT ' in text:
            polls += 1
            if mode == 'timeout':
                raise subprocess.TimeoutExpired(argv,300)
            if (out/'heartbeat.json').exists():
                beats.append(json.loads((out/'heartbeat.json').read_text()))
            return SimpleNamespace(stdout='G1_RUNNING' if polls == 1 else 'G1_FAILED',stderr='')
        return SimpleNamespace(stdout='', stderr='')
    monkeypatch.setattr(colab_driver.subprocess, 'run', fake_run)
    monkeypatch.setattr(colab_driver.time, 'sleep', lambda n: None)
    monkeypatch.setattr(colab_driver, 'split_bundle', lambda *a, **kw:
                        ((Path(a[1]).mkdir()), {})[1])
    with pytest.raises((subprocess.TimeoutExpired, RuntimeError)):
        colab_driver.launch(path/'bundle',job,out,gate1=gate,gate2=gate,poll_seconds=0)
    assert starts == 1
    assert commands[-1] == 'stop'
    if mode == 'timeout': assert polls == 3
    else:
        assert polls == 2
        assert beats[0]['polls'] == 1
        assert json.loads((out/'heartbeat.json').read_text())['polls'] == 2
    if mode == 'new_timeout': assert commands.count('new') == 1


@pytest.mark.parametrize('state', ['Z', 'missing'])
def test_t8_listing(tmp_path, monkeypatch, state, capsys):
    base = tmp_path/'remote'; (base/'results').mkdir(parents=True)
    (base/'worker.pid').write_text('123456')
    monkeypatch.setattr(colab_driver, 'REMOTE', str(base))
    original = Path.read_text
    def read(path, *args, **kwargs):
        if str(path) == '/proc/123456/stat':
            if state == 'missing': raise FileNotFoundError()
            return '123456 (a name with spaces) Z 1 2 3'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    exec(colab_driver._listing(), {})
    assert 'G1_FAILED' in capsys.readouterr().out


def test_t8_failure_collects_before_stop(prepared, tmp_path, monkeypatch):
    import shutil
    _, path, job = prepared
    job['device'] = 'cpu'
    (path/'job.json').write_text(json.dumps(job))
    remote = tmp_path/'remote_results'
    colab_driver.worker(path/'bundle',path/'job.json',remote)
    name='batch_000000.npz'
    gate=tmp_path/'gate.json';gate.write_text('{"status":"pass"}')
    out=path/'collected'
    def fake(argv, **kw):
        assert kw['timeout']==300
        if argv[1]=='download':
            shutil.copyfile(remote/Path(argv[-2]).name, argv[-1])
        if argv[1]=='stop':
            assert (out/name).is_file() and (out/(name+'.receipt.json')).is_file()
        text=kw.get('input') or ''
        answer='G1_RESULT '+name+'\nG1_FAILED' if 'G1_RESULT ' in text else 'G1_LAUNCHED'
        return SimpleNamespace(stdout=answer,stderr='')
    monkeypatch.setattr(colab_driver.subprocess,'run',fake)
    monkeypatch.setattr(colab_driver,'split_bundle',lambda *a,**k: ((Path(a[1]).mkdir()),{})[1])
    with pytest.raises(RuntimeError,match='G1_FAILED'):
        colab_driver.launch(path/'bundle',job,out,gate1=gate,gate2=gate)
    assert json.loads((out/'heartbeat.json').read_text())['collected_batches']==1


def test_t11_grid_detector(tmp_path,monkeypatch):
    import yaml
    raw=copy.deepcopy(load_scene(str(S2)).raw)
    raw['detector']=dict(center_cm=[30,0,0], normal=[0,0,1],u_axis=[1,0,0],
                         size_cm=[60,60],shape=[6,7])
    scene=validate_scene(raw); assert scene.ok
    path=tmp_path/'scene.yaml';path.write_text(yaml.safe_dump(raw))
    dose=tmp_path/'dose.npz';det=tmp_path/'detector.npz'
    assert invoke(monkeypatch,'run',path,'--engine','gpu','--gpu-device','cpu','--seed',2401,
                  '-n',41,'--batch-size',20,'--dose-grid','--resolution',5,
                  '--dose-out',dose,'--detector-out',det)==0
    plane,roi=detector_from_scene(scene)
    r=run_transport(scene,engine='gpu',seed=2401,n_histories=41,batch_size=20,
                    dose_grid=True,grid_resolution_cm=5,detector=plane,detector_roi=roi)
    with np.load(dose) as z:
        for key,value in saved(r,scene).items(): same(z[key],value)
    with np.load(det) as z:
        for key in ('category_fluence','photon_count','energy_sum_keV','energy_sum2_keV2','n_batches_hit'):
            same(z[key],getattr(r.detector,key))
    spec=condition.grid_spec(scene,plane,5)
    corners = plane.corners_cm()
    assert np.all(corners.min(axis=0) >= np.array(spec['origin_cm']))
    assert np.all(corners.max(axis=0) <= np.array(spec['origin_cm']) + np.array(spec['shape'])*5)
    assert corners[:,0].max() > max(g['center'][0]+g['size_cm'][0]/2 for g in raw['geometry'])
    condition.check_grid(r.grid,spec)


def test_t6_table_inplace_after_observation(prepared):
    _,path,job=prepared
    a,src,spec,settings,_=bundle.load(path/'bundle')
    from chatcarlo.gpu.source_host import sample_source_photons
    o,d,e=sample_source_photons(src,20,np.random.default_rng(job['source_seed']),spec)
    driver=g2_core.Core(a,grid=g2_core.grid_from(job['grid']))
    _,before=driver.run(o,d,e,seed=2401,observe=True)
    address=driver.a['g2_table'].ctypes.data
    driver.a['g2_table'][0]=np.nextafter(driver.a['g2_table'][0],np.inf)
    _,after=driver.run(o,d,e,seed=2401,observe=True)
    assert address==driver.a['g2_table'].ctypes.data
    with pytest.raises(ValueError,match='observation mismatch: batch_000000 19:g2_table'):
        approval.check_observation(after,before,'batch_000000')


@pytest.mark.parametrize('device,offset',[('cpu',19),('cuda',25)])
def test_t6_expected_positions(prepared,device,offset):
    _,path,job=prepared
    a,_,_,_,_=bundle.load(path/'bundle')
    cfg=config(a,g2_core.grid_from(job['grid']))
    source=(np.zeros((1,3)),np.ones((1,3)),np.array([60.]))
    expected=g2_core.expected_observation(a,cfg,*source,2401,0,True,device)
    legacy=core.expected_observation(a,*source,2401,0,True,device)
    assert set(expected)-set(legacy)=={f'{offset}:g2_table',f'{offset+1}:cfg'}
    assert expected[f'{offset}:g2_table']==core.array_sha(a['g2_table'])
    assert expected[f'{offset+1}:cfg']==core.array_sha(cfg)
    assert {k:expected[k] for k in legacy}==legacy


@pytest.mark.parametrize('uncertainty',[True,False])
def test_cli_single_batch_and_no_uncertainty(tmp_path,monkeypatch,uncertainty):
    path=tmp_path/'dose'
    args=['run',S2,'--engine','gpu','--gpu-device','cpu','--seed',2401,'-n',41,
          '--batch-size',100000,'--dose-grid','--resolution',2,'--dose-out',path]
    if not uncertainty: args.append('--no-uncertainty')
    assert invoke(monkeypatch,*args)==0
    scene=load_scene(str(S2))
    r=run_transport(scene,engine='gpu',seed=2401,n_histories=41,batch_size=100000,
                    dose_grid=True,grid_resolution_cm=2,track_uncertainty=uncertainty)
    expected=saved(r,scene,uncertainty)
    with np.load(str(path)+'.npz') as z:
        assert set(z.files)==set(expected)
        for key,value in expected.items(): same(z[key],value)


@pytest.mark.parametrize('n,batch_size,uncertainty',
                         [(250000,100000,True),(100000,100000,True),(250000,100000,False)])
def test_gate_b_verified_batch_statistics(tmp_path,n,batch_size,uncertainty):
    scene=load_scene(str(S2))
    path=approval.prepare(scene,opts(n_histories=n,batch_size=batch_size,
                                    no_uncertainty=not uncertainty),tmp_path/'run')
    job=json.loads((path/'job.json').read_text());job['device']='cpu'
    (path/'job.json').write_text(json.dumps(job))
    colab_driver.worker(path/'bundle',path/'job.json',path/'results')
    batches,_=approval.verify_batches(path/'bundle',job,path/'results')
    _,_,_,settings,_=bundle.load(path/'bundle')
    g=condition.grid_for(job['grid'],uncertainty)
    result=aggregate(batches,settings['material_names'],track_uncertainty=uncertainty,grid=g)
    total=[np.zeros(g.shape),np.zeros(g.shape)]
    q=[np.zeros(g.shape),np.zeros(g.shape)]
    direct_q=[np.zeros(g.shape),np.zeros(g.shape)]
    hits=np.zeros(g.shape,np.int32)
    for count,r in batches:
        delta=[]
        for i,key in enumerate(('kerma','h10')):
            previous=total[i].copy()
            total[i]+=r[key].reshape(g.shape)
            delta.append(total[i]-previous)
            q[i]+=delta[i]**2/count
            direct_q[i]+=r[key].reshape(g.shape)**2/count
        hits+=((delta[0]!=0)|(delta[1]!=0)).astype(np.int32)
    same(g.kerma_keV,total[0]);same(g.h10_track_pSv_cm3,total[1])
    assert g.n_histories==(n if uncertainty else 0)
    assert g.n_batches==(len(batches) if uncertainty else 0)
    # Use only the immutable production unit conversion on independently
    # reconstructed totals. No end_batch or moment helpers form this reference.
    reference=condition.grid_for(job['grid'],False)
    reference.kerma_keV[:]=total[0]
    reference.h10_track_pSv_cm3[:]=total[1]
    expected=dict(kerma_keV=total[0],h10_track_pSv_cm3=total[1],
                  origin_cm=reference.origin_cm,shape=np.array(reference.shape),
                  voxel_size_cm=reference.voxel_size_cm,
                  dose_per_history_Gy=dose_map_Gy(reference,Geometry(scene.raw['geometry']))/n,
                  h10_per_history_pSv=reference.h10_map_pSv()/n)
    if uncertainty:
        same(g.kerma_sum2,q[0]);same(g.h10_sum2,q[1]);same(g.n_batches_hit,hits)
        assert np.any(hits==0)
        for i in range(2): np.testing.assert_allclose(q[i],direct_q[i],rtol=1e-12,atol=0.)
        expected.update(n_batches=np.array(len(batches)),n_batches_hit=hits)
        for i,quantity,mean_key,sem_key in (
            (0,'dose','dose_per_history_Gy','sem_dose_per_history_Gy'),
            (1,'h10','h10_per_history_pSv','sem_h10_per_history_pSv')):
            if len(batches)<2: rel=np.full(g.shape,np.nan)
            else:
                sem=np.sqrt(np.clip((q[i]-total[i]**2/n)/(len(batches)-1),0.,None)/n)
                rel=np.divide(sem,total[i]/n,out=np.full(g.shape,np.nan),where=total[i]!=0)
            expected['rel_err_'+quantity]=rel
            expected[sem_key]=rel*expected[mean_key]
    actual=saved(result,scene,uncertainty)
    assert set(actual)==set(expected)
    for key in expected: same(actual[key],expected[key])
    output=tmp_path/'independent_acceptance.npz'
    np.savez(output,**actual)
    with np.load(output) as z:
        for key in expected: same(z[key],expected[key])
