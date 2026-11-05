"""H4 tool tests use artificial outer batches and fake native drivers only."""
import json
import math
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from docs.validation.gpu.g5 import a4u_h4_gate as h4


def synthetic(relative=.2):
    pattern=np.tile([-1.,1.],50)
    fine=(1+relative*pattern[:,None])*np.ones((100,600))*1e-12
    profile=(1+relative*pattern[:,None,None,None])*np.ones((100,2,3,320))*1e-12
    values,meta=h4.candidate_samples(fine,profile)
    return dict(samples=values,meta=meta,N=1000000)


def test_D7_registration_and_separate_N_allocation():
    ref=synthetic();reg=h4.registration(ref);twin=ref['samples']*2-ref['samples'].mean(0)
    report=h4.analyse(ref,twin,reg)
    # Registration approved by the user 2026-11-04: the judged set is the depth
    # curve only and stops at 15 cm, so K drops from 608 (600 PDD + 8 OCR) to 300
    # on this 0-30 cm synthetic grid; OCR and the deep tail are measured, never judged.
    assert reg['K']==300 and reg['valid']
    assert reg['mandatory_counts']=={'fine':10}
    assert reg['minimum_evaluable']==math.ceil(.8*300)
    assert reg['z_critical']==pytest.approx(h4.NormalDist().inv_cdf(1-.05/(2*300)))
    assert report['status']=='pilot_complete' and report['pilot_D7']['status']=='insufficient'
    first=report['bins'][0];r=first['EGS5'];sr=first['EGS5_SEM'];st=first['CPU_SEM']
    assert st==pytest.approx(2*sr)
    assert first['SEM_difference']==pytest.approx(np.hypot(sr,st))
    for name,s in [('EGS5',sr),('CPU',st)]:
        needed=math.ceil(1000000*(s/(.01*r/np.sqrt(2)))**2)
        assert first['required_histories'][name]==needed
        assert report['main_histories'][name]==100*math.ceil(1.5*report['required_max_before_margin'][name]/100)
    assert report['main_histories']['CPU']>report['main_histories']['EGS5']


def test_cutoff_keeps_all_failed_bins_and_suppresses_main_N():
    ref=synthetic(.02);reg=h4.registration(ref)
    r=h4.analyse(ref,ref['samples']*2,reg)
    assert r['status']=='stop_cutoff' and len(r['cutoff_bins'])==300
    assert r['main_histories'] is None
    assert all(row['required_histories']==dict(EGS5=None,CPU=None) for row in r['bins'])
    row=r['bins'][0]
    assert row['D7_tolerance']==pytest.approx(max(reg['z_critical']*row['SEM_difference'],.02*row['EGS5']))
    assert row['lower_abs_difference']==pytest.approx(abs(row['difference'])-2*row['SEM_difference'])


def test_known_failure_is_not_removed_and_historical_rule_is_reported():
    ref=synthetic(.02);reg=h4.registration(ref)
    r=h4.analyse(ref,ref['samples']+.025e-12,reg)
    assert not r['cutoff_bins'] and r['pilot_D7']['status']=='fail'
    assert r['pilot_D7']['failed']==300 and r['pilot_D7']['historical_2sigma_failed']==300
    assert len(r['bins'])==300 and len(r['known_failure_bins'])==300


def test_zero_unestimable_and_budget_stops():
    # The OCR bins used to carry this case. The approved registration judges the
    # depth curve only, and a zeroed depth bin simply drops out of the 5%-of-peak
    # selection, so register first and zero selected bins afterwards: that is the
    # real shape of the branch (a registered bin whose reference value is 0).
    ref=synthetic();reg=h4.registration(ref);ref['samples'][:,290:300]=0
    r=h4.analyse(ref,ref['samples'],reg)
    assert r['status']=='stop_unestimable' and len(r['unestimable_bins'])==10
    assert r['main_histories'] is None
    noisy=synthetic(.95);r=h4.analyse(noisy,noisy['samples'],h4.registration(noisy))
    assert r['status']=='stop_budget' and max(r['main_histories'].values())>h4.MAX_MAIN_HISTORIES


def reference_fixture(root,energy=20):
    from docs.validation.gpu.g5 import a4_egs
    case=h4.CASES[energy];p=root/case['label'];p.mkdir(parents=True)
    edges,cum=a4_egs.mono_edges(energy)
    config=dict(N=case['N'],seed=case['egs_seed'],batches=100,field=[5.,100.],regions=h4.REGIONS,thickness=30.,nb=1,E=10.,spectrum_arrays_sha256=h4.array_hash(edges,cum))
    summary=dict(status='合格',tests=dict(health=True),effective_flags=a4_egs.WANT_ON,wall_seconds=12.,CPU_seconds=10.)
    (p/'config.json').write_text(json.dumps(config));(p/'summary.json').write_text(json.dumps(summary))
    np.savez(p/'result.npz',fine=np.ones((100,600)),profile_fine=np.full((100,2,3,320),10.),N=case['N'],seed=case['egs_seed'],unit='Gy cm2',seconds=12.)
    return p


def test_reference_mono_normalization_timing_and_flags(tmp_path,monkeypatch):
    from docs.validation.gpu.g5 import a4_egs
    monkeypatch.setattr(a4_egs,'run',lambda *a,**kw:pytest.fail('never start EGS5'))
    p=reference_fixture(tmp_path);r=h4.read_reference(20,tmp_path)
    assert r['edges']==[.0199,.0201] and r['cum']==[1.]
    assert np.all(r['samples'][:,:600]==1)
    assert np.all(r['samples'][:,600:]==10*h4.GY_PER_MEV_G/(10000*.25))
    assert r['timing']['wall_seconds_per_history']==12/1000000
    d=json.loads((p/'config.json').read_text());d['spectrum_arrays_sha256']='bad';(p/'config.json').write_text(json.dumps(d))
    with pytest.raises(ValueError,match='spectrum'):h4.read_reference(20,tmp_path)


def test_both_references_required_before_CPU(tmp_path,monkeypatch):
    reference_fixture(tmp_path);monkeypatch.setattr(h4,'REFERENCE_ROOT',tmp_path)
    monkeypatch.setattr(h4,'prepare_cpu',lambda *a:pytest.fail('CPU must wait for both completed references'))
    with pytest.raises(ValueError,match='incomplete'):h4.gate(tmp_path/'out')


def test_prepare_regenerates_photon_even_when_export_contains_old_table(monkeypatch):
    from docs.validation.gpu.g5 import a2_common
    from chatcarlo.gpu import catalog,generate_photon_tables
    old=SimpleNamespace(x=np.array([1.]),photon=np.array([-999.]))
    monkeypatch.setattr(a2_common,'inputs',lambda:({},[old],dict(strength=np.array([1.]),constants=np.array([1.]))))
    monkeypatch.setattr(a2_common,'field_geometry',lambda:None)
    monkeypatch.setattr(catalog,'P8_DEFINITIONS',dict(x={},photon={}))
    # This test isolates regeneration; the real-material test below covers
    # the full production catalog encoding and native table hashes.
    monkeypatch.setattr(h4,'catalog_table_hashes',lambda materials:{})
    def fresh(m):
        assert not hasattr(m,'photon');return np.array([42.])
    monkeypatch.setattr(generate_photon_tables,'ensure',fresh)
    m,a,_=h4.prepare_cpu(20)
    np.testing.assert_array_equal(m[0].photon,[42.])
    np.testing.assert_array_equal(a['edges'],[.0199,.0201])
    assert a['sizes'][-1]==1


def test_CPU_cache_is_lazy_and_bound_to_identity(tmp_path,monkeypatch):
    from chatcarlo.gpu import therapy,physics_sets,therapy_health,therapy_observation
    decl=physics_sets.declaration(physics_sets.NAME,specialization=physics_sets.ELECTRON)
    cond=dict(N=100,cpu_seed=20011,physics_set=decl)
    builds=[];calls=[]
    class FakeCPU:
        def __init__(self,work,m,a,**kw):
            assert '_test_legacy_physics' not in kw
            assert kw['half']==(5.,5.) and kw['ssd']==100.
            builds.append(kw);self.observed=decl;self.last_observation=dict(marker=1)
        def histories(self,n,seed,start,slots):
            assert n==1 and slots==32
            calls.append((seed,start));g=np.zeros((100,1087));g[0,450:1050]=1
            return dict(groups=g,field=np.ones(3420))
    monkeypatch.setattr(therapy,'CPU',FakeCPU)
    monkeypatch.setattr(therapy_health,'check',lambda *a:None)
    monkeypatch.setattr(therapy_observation,'descriptor',lambda *a:dict(marker=1))
    x,t=h4.run_cpu(tmp_path,cond,[],{},None)
    assert len(builds)==1 and len(calls)==100 and x.shape==(100,608)
    assert calls[0]==(20011,0) and calls[-1]==(20011,99)
    builds.clear();calls.clear()
    y,t2=h4.run_cpu(tmp_path,cond,[],{},None)
    np.testing.assert_array_equal(x,y);assert t==t2 and not builds and not calls
    with pytest.raises(ValueError,match='condition mismatch'):h4.run_cpu(tmp_path,{**cond,'cpu_seed':3},[],{},None)
    p=tmp_path/'batch_000.npz';p.write_bytes(p.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='SHA256'):h4.run_cpu(tmp_path,cond,[],{},None)


def test_gate_stops_before_second_CPU_after_cutoff(tmp_path,monkeypatch):
    ref=synthetic(.02);ref.update(files={},edges=[.0199,.0201],cum=[1.],timing=dict(wall_seconds_per_history=1e-4))
    monkeypatch.setattr(h4,'read_reference',lambda energy:ref)
    monkeypatch.setattr(h4,'prepare_cpu',lambda energy:([],{},None))
    monkeypatch.setattr(h4,'condition',lambda *a:dict(dummy=True))
    calls=[]
    def fake_cpu(*args):calls.append(1);return ref['samples']*2,dict(wall_seconds_per_history=1e-5)
    monkeypatch.setattr(h4,'run_cpu',fake_cpu);monkeypatch.setattr(h4,'RESULT',tmp_path/'result.json')
    r=h4.gate(tmp_path/'run')
    assert r['status']=='stop_cutoff' and r['pending_energies']==[60] and len(calls)==1
    assert r['cases']['20']['main_histories'] is None and not r['H5_executed']


def test_no_build_or_transport_without_explicit_out(monkeypatch,capsys):
    monkeypatch.setattr(h4,'gate',lambda *a:pytest.fail('must not run pilot'))
    for argv,code in (([],2),(['--help'],0)):
        monkeypatch.setattr(sys,'argv',['a4u_h4_gate',*argv])
        with pytest.raises(SystemExit) as exc:h4.main()
        assert exc.value.code==code
    capsys.readouterr()


def test_changed_second_case_stops_before_first_CPU(tmp_path,monkeypatch):
    ref=synthetic();ref.update(files={},edges=[.0199,.0201],cum=[1.])
    monkeypatch.setattr(h4,'read_reference',lambda energy:ref)
    monkeypatch.setattr(h4,'prepare_cpu',lambda energy:([],{},None))
    monkeypatch.setattr(h4,'condition',lambda energy,*a:dict(energy=energy))
    monkeypatch.setattr(h4,'run_cpu',lambda *a:pytest.fail('preflight must precede CPU'))
    h4.check_condition(tmp_path/'60',dict(energy=60,old=True),write=True)
    with pytest.raises(ValueError,match='condition mismatch'):h4.gate(tmp_path)


def test_real_prepare_table_hashes_match_all_packed_fields():
    from chatcarlo.gpu import catalog,therapy,therapy_observation
    materials,arrays,_=h4.prepare_cpu(20)
    data,meta=therapy.pack(materials)
    actual=therapy_observation.table_hashes(data,meta)
    expected=arrays['_table_hashes']
    names={str(i)+'.'+name for i in range(4) for name in catalog.P8_DEFINITIONS}
    assert set(actual)==set(expected)==names
    mismatches=[name for name in sorted(names) if actual[name]!=expected[name]]
    assert not mismatches,mismatches
    # The raw export uses integer line counts. Native and production catalog
    # storage use float64, preserving each material's own (nz,4) shape.
    for i,m in enumerate(materials):
        assert m.line_counts.shape==(len(m.z),4)
        assert m.line_counts.dtype==np.dtype('int64')
        assert actual[str(i)+'.line_counts']==catalog.digest(np.asarray(m.line_counts,np.float64))
