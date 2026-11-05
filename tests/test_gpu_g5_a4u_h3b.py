"""H3b uses saved EGS5 data and synthetic/fake CPU results here; no MC runs."""
import json
import sys
import numpy as np
import pytest
from docs.validation.gpu.g5 import a4u_h3b_gate as h3


@pytest.fixture
def reference():return h3.read_reference(h3.REFERENCE)


def synthetic(reference,arm):
    n=h3.ARMS[arm]['histories'];batches=n//h3.ARMS[arm]['batch_size']
    total=(np.asarray(reference['counts'])*n/h3.REFERENCE_N).astype(int)
    h=np.zeros((batches,10001))
    for i,count in enumerate(total):
        q,r=divmod(int(count),batches);h[:,i]=q;h[:r,i]+=1
    return h


def test_saved_reference_and_bin_edges(reference,tmp_path,monkeypatch):
    c=reference['counts']
    assert [sum(c[lo:hi]) for lo,hi in h3.BANDS]==[89,2152,42,3078,422,5883,236194]
    assert reference['escaped']==309706 and reference['histories']==1000000
    bad=tmp_path/'reference.out';bad.write_bytes(h3.REFERENCE.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='SHA256'):h3.read_reference(bad)
    text=h3.REFERENCE.read_text().replace('    0.0110        2152.0\n','')
    bad.write_text(text)
    import hashlib
    monkeypatch.setattr(h3,'REFERENCE_SHA256',hashlib.sha256(bad.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match='120'):h3.read_reference(bad)


def test_production_means_sem_and_K_observation_B(reference):
    r=h3.compare(synthetic(reference,'production'),'production',reference)
    assert r['status']=='pass' and len(r['bands'])==7
    a=r['bands']['10-11']
    assert a['ChatCarlo']==pytest.approx(2152/1000000)
    assert a['EGS5_SEM']==pytest.approx(np.sqrt(2152)/1000000)
    assert a['threshold']==pytest.approx(max(2*a['SEM'],.05*a['EGS5']))
    k=r['bands']['72-86'];p=236194/309706
    assert k['ChatCarlo']==pytest.approx(p) and k['EGS5']==pytest.approx(p)
    assert k['EGS5_SEM']==pytest.approx(np.sqrt(p*(1-p)/309706))
    assert 'fraction of all escaped' in k['quantity']


@pytest.mark.parametrize('arm,factor',[('CK_off',.84),('f13_zero',.90)])
def test_fault_identification_is_Lalpha_not_L_total(reference,arm,factor):
    h=synthetic(reference,arm);before=h[:,10].copy();h[:,10]=np.floor(factor*h[:,10])
    h[:,12]+=before-h[:,10]  # Synthetic redistribution, not a physical model.
    r=h3.compare(h,arm,reference)
    assert r['status']=='pass' and r['spectrum_comparison_status']=='fail'
    assert not r['bands']['10-11']['pass_'] and r['bands']['9-16']['pass_']
    assert r['detected_10_11'] and r['L_total_pass'] and r['precision_sufficient']
    assert r['self_2SEM_relative_10_11']<.05


def test_fault_precision_is_not_repaired_or_adapted(reference):
    h=synthetic(reference,'CK_off');h[:,10]=np.tile([0,200],10)
    r=h3.compare(h,'CK_off',reference)
    assert r['status']=='insufficient' and not r['precision_sufficient']
    assert r['histories']==1000000


def test_native_batch_resume_is_lazy_and_checks_counts(tmp_path,monkeypatch):
    spec=dict(fault=2,histories=20,batch_size=10,seed=1)
    monkeypatch.setitem(h3.ARMS,'f13_zero',spec);calls=[];builds=[]
    class FakeCPU:
        def __init__(self,arrays,device,*,build_dir,_test_photon_fault):
            builds.append((device,_test_photon_fault))
        def run(self,o,d,e,*,seed,start,fluorescence):
            np.testing.assert_array_equal(o,np.tile(h3.BEAM['origin'],(10,1)))
            assert np.all(e==100) and fluorescence
            calls.append((seed,start));s=np.zeros(10001);s[10]=1
            return dict(escaped_spectrum=s,endings=[0,1,0],completed=len(e))
    monkeypatch.setattr(h3.core,'Core',FakeCPU)
    h=h3.run_arm(tmp_path,{},'f13_zero','seal')
    assert h.shape==(2,10001) and builds==[('cpu',2)] and calls==[(1,0),(1,10)]
    calls.clear();builds.clear()
    np.testing.assert_array_equal(h3.run_arm(tmp_path,{},'f13_zero','seal'),h)
    assert not calls and not builds
    with pytest.raises(ValueError,match='resume identity'):h3.run_arm(tmp_path,{},'f13_zero','different-seal')
    with pytest.raises(ValueError,match='10001'):h3.valid_spectrum(np.zeros(120))


def test_gate_stops_on_disagreement_and_rejects_changed_condition(reference,tmp_path,monkeypatch):
    monkeypatch.setattr(h3,'RESULT',tmp_path/'result.json')
    monkeypatch.setattr(h3,'make_arrays',lambda:dict(data=np.array([1.])))
    calls=[]
    def fake_run(out,arrays,arm,sha):
        calls.append(arm);h=synthetic(reference,arm);h[:,10]*=2
        return h
    monkeypatch.setattr(h3,'run_arm',fake_run)
    r=h3.gate(tmp_path/'run')
    assert r['status']=='fail' and calls==['production']
    assert r['pending_arms']==['CK_off','f13_zero'] and not r['arms']['production']['bands']['10-11']['pass_']
    saved=json.loads(h3.RESULT.read_text());assert saved['status']=='fail'
    seal=tmp_path/'run/condition.json';s=json.loads(seal.read_text());s['condition']['arms']['production']['seed']=99;seal.write_text(json.dumps(s))
    calls.clear()
    with pytest.raises(ValueError,match='condition mismatch'):h3.gate(tmp_path/'run')
    assert not calls


def test_cli_does_not_measure_without_out(monkeypatch,capsys):
    monkeypatch.setattr(h3,'gate',lambda *a:pytest.fail('must not measure'))
    for args,code in (([],2),(['--help'],0)):
        monkeypatch.setattr(sys,'argv',['a4u_h3b_gate',*args])
        with pytest.raises(SystemExit) as e:h3.main()
        assert e.value.code==code
    capsys.readouterr()


def test_complete_gate_preserves_fault_failures(reference,tmp_path,monkeypatch):
    monkeypatch.setattr(h3,'RESULT',tmp_path/'result.json')
    monkeypatch.setattr(h3,'make_arrays',lambda:dict(data=np.array([1.])))
    calls=[]
    def fake_run(out,arrays,arm,sha):
        calls.append(arm);h=synthetic(reference,arm)
        if arm!='production':
            before=h[:,10].copy();h[:,10]=np.floor(.84*h[:,10]);h[:,12]+=before-h[:,10]
        return h
    monkeypatch.setattr(h3,'run_arm',fake_run)
    r=h3.gate(tmp_path/'run')
    assert calls==['production','CK_off','f13_zero'] and r['status']=='pass'
    assert r['condition']['reference']['sha256']==h3.REFERENCE_SHA256
    assert [r['condition']['arms'][arm]['fault'] for arm in calls]==[0,1,2]
    assert [r['condition']['arms'][arm]['histories'] for arm in calls]==[4000000,1000000,1000000]
    for arm in ('CK_off','f13_zero'):
        assert not r['arms'][arm]['bands']['10-11']['pass_']
        assert r['arms'][arm]['bands']['9-16']['pass_']


def test_geometry_matches_existing_slab_without_transport(monkeypatch):
    from chatcarlo import kernel
    from chatcarlo.geometry import Geometry
    monkeypatch.setattr(kernel,'bake_scene_materials',lambda names:None)
    # make_arrays zeroes the background density (vacuum); give it that slot.
    monkeypatch.setattr(h3.core,'pack',lambda tables:dict(data=np.zeros(200000),offsets=np.full(22,155346,np.int64)))
    a=h3.make_arrays()
    size=[.05,100.,100.]
    geom=Geometry([dict(name='slab',shape='box',material='lead',center=[0.,0.,0.],size_cm=size)])
    np.testing.assert_array_equal(a['centers'],[[0,0,0]])
    np.testing.assert_array_equal(a['halves'],[np.array(size)/2])
    np.testing.assert_array_equal(a['world'],np.r_[(geom.bbox_min+geom.bbox_max)/2,(geom.bbox_max-geom.bbox_min)/2])
    assert h3.BEAM['origin']==[-.05/2-5,0,0] and h3.BEAM['direction']==[1,0,0]
