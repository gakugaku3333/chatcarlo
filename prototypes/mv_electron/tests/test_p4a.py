"""P4a planned synthetic checks and small ELSEPA transport regressions.

No checks/pilot/production/analyse entry point is executed by this suite.
All output-writing tests use tmp_path; table access is read-only.
"""
import ast
from pathlib import Path
import json
import numpy as np
import pytest
from prototypes.mv_electron import photon_transport_p4 as t
from prototypes.mv_electron import photon_transport_p3b as old
from prototypes.mv_electron import analysis_p4a as a, run_p4a as r, run_p3s as s


@pytest.fixture(scope='module')
def inputs():
    from prototypes.mv_electron.p2_common import TABLES, ELSEPA
    from prototypes.mv_electron.photon_data_p2 import TABLE
    files = [TABLES['elsepa'], TABLE]+[ELSEPA/'tables'/f'z{z}_q-1.npz' for z in (1,8)]
    try:
        for path in files:
            with path.open('rb') as f:f.read(1)
    except (FileNotFoundError, PermissionError, OSError) as exc:
        pytest.skip('required ELSEPA/photon tables unreadable: '+str(exc))
    # Hash/energy errors are test failures, not skips.
    return s.inputs('elsepa')


def kw(**options):
    result = dict(annihilate_in_flight=True,score_kerma=False,score_pair=False,**s.rule_kwargs('own'))
    result.update(options)
    return result


def execute(inputs, n=200, E=1., seed=20291781, **options):
    thickness = options.pop('thickness',30.)
    return t.run_coupled(E,n,seed,.05,.025,thickness,*inputs,**kw(**options))


def fano(inputs,system='A',n=200,seed=20292881,**options):
    opts = r.fano_options(1,system,history=True)
    opts.update(options)
    return execute(inputs,n=n,seed=seed,**opts)


def same(left,right):
    assert len(left)==len(right)
    assert r.byte_equal(left,right)


def ledger(values):
    assert not np.any(values[7])
    assert not np.any(values[8][5:7])
    assert values[27]==0


@pytest.mark.parametrize('options',[
    dict(layer_faces=[0.,30.]),dict(layer_rho=[1.]),
    dict(layer_faces=[0.,30.],layer_rho=[0.]),
    dict(layer_faces=[0.,30.],layer_rho=[-1.]),
    dict(layer_faces=[0.,30.],layer_rho=[np.nan]),
    dict(layer_faces=[0.,30.],layer_rho=[np.inf]),
    dict(layer_faces=[0.,30.],layer_rho=[1.,1.]),
    dict(layer_faces=[[0.,30.]],layer_rho=[1.]),
    dict(layer_faces=[1.,30.],layer_rho=[1.]),
    dict(layer_faces=[0.,29.],layer_rho=[1.]),
    dict(layer_faces=[0.,30.,30.],layer_rho=[1.,1.]),
    dict(layer_faces=[0.,20.,10.,30.],layer_rho=[1.,1.,1.]),
    dict(layer_faces=[0.,np.nan,30.],layer_rho=[1.,1.]),
    dict(mass_bin=0.),dict(mass_bin=-1.),dict(mass_bin=np.inf),dict(mass_bin=np.nan),
    dict(mass_bin=7.),dict(mass_bin=1e12),dict(fault=-1),dict(fault=3),
    dict(fano_source_T=0.,mass_bin=1.),dict(fano_source_T=-1.,mass_bin=1.),
    dict(fano_source_T=np.nan,mass_bin=1.),dict(fano_source_T=np.inf,mass_bin=1.),
    dict(fano_source_T=10.01,mass_bin=1.),dict(fano_source_T=1.),
    dict(fano_source_charge=0),dict(record_histories=True),
]+[dict(fano_source_T=1.,mass_bin=1.,**{key:value}) for key,value in (
    ('source_state',np.zeros(23)),('charged_source_T',1.),('positron_source_T',1.),
    ('spectrum_edges',np.array([0.,1.])),('spectrum_cum',np.array([1.])))]
+[dict(layer_faces=[0.,30.],layer_rho=[1.],**{key:value}) for key,value in (
    ('regions',r.REGIONS),('record_deposits',1),('field_half',5.),('infinite_medium',True),
    ('charged_source_T',1.),('positron_source_T',1.),('source_state',np.zeros(23)),
    ('spectrum_edges',np.array([0.,1.])),('spectrum_cum',np.array([1.])),('score_kerma',True))])
def test_argument_validation(options):
    # Invalid arguments must fail before table loading or compiling transport.
    with pytest.raises(ValueError):
        t.run_coupled(1.,100,1,.05,.025,30.,**dict(dict(score_kerma=False),**options))


def test_geometry_and_synthetic_scoring():
    faces,rho,mf = t.mass_geometry(3.,[0.,1.,3.],[2.,.5],.5)
    np.testing.assert_array_equal(mf,[0.,2.,3.])
    assert t.mass_coordinate(2.,1,faces,rho,mf)==2.5
    score = np.zeros(6)
    t.mass_segment(score,.25,.75,0,4.,.5,faces,rho,mf)
    np.testing.assert_array_equal(score,[0.,2.,2.,0.,0.,0.])
    score[:] = 0
    t.mass_segment(score,3.,1.,1,4.,.5,faces,rho,mf)
    np.testing.assert_array_equal(score,[0.,0.,0.,0.,2.,2.])
    t.mass_local(score,2.,1,3.,.5,faces,rho,mf)
    assert score[5]==5.
    t.mass_segment(score,1.,1.,0,2.,.5,faces,rho,mf)
    assert score[4]==4.
    with pytest.raises(ValueError):t.mass_geometry(3.,[0.,1.,3.],[2.,.5],.7)


@pytest.mark.parametrize('index',[0,1,2])
@pytest.mark.parametrize('system',['A','B'])
def test_fixed_geometry(index,system):
    g = a.geometry(index,system);h = a.geometry(index,'H_'+system)
    np.testing.assert_allclose(g['mass_faces'],h['layer_faces'],rtol=1e-15)
    np.testing.assert_allclose(np.diff(g['layer_faces'])*g['layer_rho'],np.diff(h['layer_faces']),rtol=1e-14)
    assert g['mass_faces'][-1] == pytest.approx((5.5 if system=='A' else 5.025)*a.RANGES[index])
    t.mass_geometry(g['thickness'],g['layer_faces'],g['layer_rho'],g['mass_bin'])


@pytest.mark.parametrize('diff,sem,status',[
    (.01,1/300,'合格'),(-.01,0.,'合格'),(.0100001,0.,'不合格'),
    (0.,1/300+1e-12,'統計不足'),(.1,.1,'統計不足'),(0.,0.,'合格')])
def test_classification(diff,sem,status):
    assert a.classify(diff,sem)==status
    assert a.classify(diff,sem,False)=='判定できない'


@pytest.mark.parametrize('n',[100,200000,1000000])
def test_batch_expectation_and_sem(n):
    mf = np.array([0.,2.,3.]);width = 1.;T = 2.
    rates = np.linspace(.98,1.02,100)
    batches = np.repeat((n/100*T/3*rates)[:,None],3,axis=1)
    rows = a.layer_statistics(batches,n,T,mf,width)
    for j,row in enumerate(rows):
        assert row['expected']==pytest.approx(n*T*(mf[j+1]-mf[j])/3)
        assert row['difference']==pytest.approx(np.mean(rates-1),abs=1e-15)
        assert row['sem']==pytest.approx(np.std(rates-1,ddof=1)/10)
    with pytest.raises(ValueError):a.layer_statistics(batches,n+1,T,mf,width)
    zero = a.layer_statistics(np.ones((100,3))*(n/100*T/3),n,T,mf,width)
    assert zero[0]['sem']==0.


def test_sizing_rounding_and_cap():
    assert a.required_histories([0.,0.])==200000
    assert a.required_histories([1/300])==400000
    assert a.required_histories([.01])==2800000
    assert a.required_histories([a.SEM_LIMIT*np.sqrt(100/1.5)])==20000000
    with pytest.raises(ValueError,match='20000000'):
        a.required_histories([a.SEM_LIMIT*np.sqrt(101/1.5)])


def test_history_counting():
    left = np.ones((20000,3));right = left.copy()
    right[:20,0]+=1e-7
    row = a.compare_histories(left,right,1.)
    assert row['matched']==19980 and row['status']=='合格'
    assert len(row['mismatches'])==20 and row['mismatches'][0]['history']==0
    right[20,0]+=1e-7
    assert a.compare_histories(left,right,1.)['status']=='不合格'
    # Each history matches, but the aggregate threshold remains independent.
    row = a.compare_histories(np.zeros((2,3)),np.full((2,3),1e-10),1.)
    assert row['matched']==2 and not row['aggregate_match']
    boundary = a.compare_histories(np.zeros((1,1)),np.array([[1e-9]]),1.)
    assert boundary['matched']==1


def lock(current):
    return dict(provenance=current,rules=r.RULES,runs={f'{i}_{system}':dict(n=200000,seed=20291783+1000*i+100*j)
        for i in range(3) for j,system in enumerate(('A','B'),1)})


def test_lock_hash_and_production_refusal(tmp_path,monkeypatch):
    current = dict(source_hashes={'kernel':'current'},table_hashes={},versions={})
    frozen = lock(current);r.validate_lock(frozen,current)
    changed = dict(current,source_hashes={'kernel':'changed'})
    with pytest.raises(RuntimeError,match='mismatch'):r.validate_lock(frozen,changed)
    r.write_json(tmp_path/'lock_p4a.json',frozen)
    monkeypatch.setattr(r,'OUT',tmp_path)
    monkeypatch.setattr(s,'inputs',lambda backend:())
    monkeypatch.setattr(r,'provenance',lambda inputs=None:changed)
    with pytest.raises(RuntimeError,match='mismatch'):r.production()
    checks = dict(provenance=current,A0={'status':'合格'},A1={'status':'判定できない'},A4={'status':'合格'})
    r.write_json(tmp_path/'checks_p4a.json',checks)
    with pytest.raises(RuntimeError,match='A1'):r.require_checks(current)
    with pytest.raises(RuntimeError,match='provenance'):r.require_checks(changed)


def test_outputs_never_overwritten_and_reuse(tmp_path,monkeypatch):
    path = tmp_path/'result.npz';config = dict(n=100,provenance={'hash':'a'})
    r.save_result(path,config,{'totals':np.arange(6)})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):r.save_result(path,config,{'totals':np.zeros(6)})
    assert path.read_bytes()==original
    np.testing.assert_array_equal(r.load_result(path,config)['totals'],np.arange(6))
    with pytest.raises(RuntimeError):r.load_result(path,dict(config,n=200))
    r.write_json(tmp_path/'out.json',config)
    with pytest.raises(FileExistsError):r.write_json(tmp_path/'out.json',{})
    r.write_text(tmp_path/'out.md','original')
    with pytest.raises(FileExistsError):r.write_text(tmp_path/'out.md','changed')
    monkeypatch.setattr(r,'CACHE',tmp_path)
    current = {'hash':'synthetic'}
    monkeypatch.setattr(r,'provenance',lambda inputs=None:current)
    calls = []
    def fake(*args,**kwargs):
        calls.append(1)
        return tuple(np.array([0.]) for _ in t.OUTPUTS)
    monkeypatch.setattr(t,'run_coupled',fake)
    r.saved('saved',1.,100,1,current=current,inputs=(),thickness=3.)
    r.saved('saved',1.,100,1,current=current,inputs=(),thickness=3.)
    assert calls==[1]
    with pytest.raises(RuntimeError):r.saved('saved',1.,200,1,current=current,inputs=(),thickness=3.)


def test_new_sources_have_no_deletion_code():
    for name in ('photon_transport_p4.py','analysis_p4a.py','run_p4a.py'):
        tree = ast.parse((Path(t.__file__).parent/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                assert node.func.attr not in ('unlink','rmtree','remove')
    assert len(t.OUTPUTS)==29
    assert t.OUTPUTS[-4:]==('mass_batches','history_mass','mass_ledger','path_total')


@pytest.mark.parametrize('case',[{}, {'charged_source_T':2.,'brems_mode':2},{'positron_source_T':5.}])
def test_a1_legacy(inputs,case):
    args = (6.,200,20291780,.05,.025,30.,*inputs)
    original = old.run_coupled(*args,**kw(**case))
    result = t.run_coupled(*args,**kw(**case))
    same(original,result[:25]);ledger(result)
    if not case:
        explicit = t.run_coupled(*args,layer_faces=[0.,30.],layer_rho=[1.],**kw())
        same([original[5],original[8]],[explicit[5],explicit[8]])


def test_a2_fano_and_fault(inputs):
    hetero = fano(inputs);homogeneous = fano(inputs,'H_A')
    ledger(hetero);ledger(homogeneous)
    row = a.compare_histories(hetero[26],homogeneous[26],1.)
    assert row['matched']==200 and row['aggregate_match']
    faulty = fano(inputs,fault=1)
    assert a.compare_histories(faulty[26],homogeneous[26],1.)['matched']<200


@pytest.mark.parametrize('E',[1.,6.])
def test_multilayer_photon_history(inputs,E):
    values = []
    for system in ('A','H_A'):
        g = a.geometry(1,system)
        out = execute(inputs,n=500,E=E,seed=20292881,thickness=g['thickness'],
                      layer_faces=g['layer_faces'],layer_rho=g['layer_rho'],mass_bin=g['mass_bin'],
                      history_rng=True,record_histories=True)
        ledger(out);values.append(out)
    row = a.compare_histories(values[0][26],values[1][26],E)
    assert row['matched']==500 and row['aggregate_match']


def test_fano_positron_ledger(inputs):
    out = fano(inputs,n=1,fano_source_charge=1,brems_mode=2)
    from prototypes.mv_electron.physics import M
    assert out[5][0]==1.+2*M
    ledger(out)


def test_history_recording_no_physics_change(inputs):
    recorded = fano(inputs);plain = fano(inputs,record_histories=False)
    same(recorded[:26],plain[:26]);same(recorded[27:],plain[27:])
    assert plain[26].shape==(0,200+20)


def test_field_regions_all_25_outputs(inputs):
    args = (6.,500,20291780,.05,.025,30.,*inputs)
    options = kw(field_half=5.,ssd=100.,regions=r.REGIONS)
    same(old.run_coupled(*args,**options),t.run_coupled(*args,**options)[:25])
