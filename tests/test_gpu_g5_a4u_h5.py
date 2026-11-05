"""H5 uses artificial scores and fake CPU drivers; no main transport runs."""
import copy
import json
import sys
import numpy as np
import pytest
from docs.validation.gpu.g5 import a4u_h4_gate as h4,a4u_h5_gate as h5


def data(N=1000000):
    pattern=1+np.tile([-.01,.01],50)
    f=pattern[:,None]*np.ones((100,600))*1e-12
    p=pattern[:,None,None,None]*np.ones((100,2,3,320))*1e-12
    s,m=h4.candidate_samples(f,p)
    return dict(samples=s,meta=m,fine=f,profile_fine=p,N=N,timing={})


def cond():
    return dict(N=20127000,cpu_seed=20111,retry_cpu_seed=20121,EGS5_N=18438400,EGS5_seed=20101,attempt='initial',registration={'sealed':'same'},reference_files={'main/result.npz':'same'})


def report_with_failure(index=0,attempt='initial'):
    ref=data(18438400);cpu=data(20127000);cpu['samples'][:,index]+=.03e-12
    c=cond()
    if attempt=='retry':c.update(attempt='retry',cpu_seed=c['retry_cpu_seed'])
    return h5.evaluate_case(ref,cpu,h4.registration(ref),c)


def test_rules_are_exact_H4_function_objects_and_approved():
    for name in ('registration','analyse','moments','candidate_samples'):
        assert getattr(h5,name) is getattr(h4,name)
    ref=data();reg=h5.registration(ref)
    assert reg['proposal']['status']==h5.APPROVAL and reg['K']==300
    assert all(b['kind']=='fine' and b['z_cm']<=15 for b in reg['bins'])


def test_shared_analysis_called_with_actual_unequal_N(monkeypatch):
    calls=[]
    def spy(ref,twin,reg):calls.append(ref['N']);return h4.analyse(ref,twin,reg)
    monkeypatch.setattr(h5,'analyse',spy)
    ref=data(18438400);cpu=data(20127000)
    r=h5.evaluate_case(ref,cpu,h4.registration(ref),cond())
    assert calls==[18438400] and r['status']=='pass'
    assert r['D7']['insufficient']==0 and r['D7']['mandatory_evaluable']
    assert 'main_histories' not in r and 'required_histories' not in r['bins'][0]


def test_OCR_and_deep_tail_measured_but_never_judged():
    ref=data();cpu=data();cpu['samples'][:,300:]*=1000
    cpu['fine'][:,300:]*=1000;cpu['profile_fine']*=1000
    r=h5.evaluate_case(ref,cpu,h5.registration(ref),cond())
    assert r['status']=='pass' and not r['OCR_judged']
    assert np.array(r['measured']['profile_fine']['CPU_mean']).shape==(2,3,320)
    assert r['measured']['fine']['CPU_mean'][400]>100*ref['fine'][:,400].mean()
    assert len(r['bins'])==300


def test_failures_retained_retry_is_plan_only_and_historical_counts_present():
    r=report_with_failure()
    assert r['status']=='retry_required' and r['known_failure_bins']==['fine:0']
    assert r['D7']['failed']==1 and r['D7']['historical_2sigma_failed']==1
    assert r['retry']['CPU_seed']==20121 and r['retry']['automatic'] is False
    assert r['retry']['EGS5_reference_reused']


def test_insufficient_mandatory_statistics_does_not_pass():
    ref=data();cpu=data();cpu['samples'][:,:10]=(1+np.tile([-.95,.95],50))[:,None]*1e-12
    r=h5.evaluate_case(ref,cpu,h5.registration(ref),cond())
    assert r['status']=='insufficient' and not r['D7']['mandatory_evaluable']
    assert r['D7']['insufficient']==10 and not r['retry']['automatic']


def test_repeat_same_failure_new_failure_and_clean_repeat():
    first=report_with_failure()
    assert h5.judge_retry(first,report_with_failure(attempt='retry'))['status']=='fail'
    distinct=h5.judge_retry(first,report_with_failure(index=1,attempt='retry'))
    assert distinct['status']=='review_required' and not distinct['further_retry_allowed']
    ref=data();c={**cond(),'attempt':'retry','cpu_seed':20121}
    clean=h5.evaluate_case(ref,ref,h5.registration(ref),c)
    assert h5.judge_retry(first,clean)['status']=='pass'
    wrong=copy.deepcopy(clean);wrong['condition']['reference_files']['main/result.npz']='changed'
    with pytest.raises(ValueError,match='mismatch'):h5.judge_retry(first,wrong)


def make_ref_files(root,energy=20):
    from docs.validation.gpu.g5.a4_egs import WANT_ON
    c=h5.CASES[energy];base=root/c['label'];base.mkdir()
    edges,cum=h4.mono(energy)
    config=dict(N=c['EGS5_N'],seed=c['EGS5_seed'],batches=100,field=[5.,100.],regions=h4.REGIONS,thickness=30.,nb=1,spectrum_arrays_sha256=h4.array_hash(edges,cum))
    main_wall=83+6e-6*c['EGS5_N']
    summary=dict(status='合格',tests={str(i):True for i in range(19)},effective_flags=WANT_ON,wall_seconds=main_wall,CPU_seconds=6e-6*c['EGS5_N'])
    (base/'config.json').write_text(json.dumps(config));(base/'summary.json').write_text(json.dumps(summary))
    np.savez(base/'result.npz',N=c['EGS5_N'],seed=c['EGS5_seed'],unit='Gy cm2',seconds=main_wall,fine=np.ones((100,600)),profile_fine=np.ones((100,2,3,320)))
    pilot=root/h4.CASES[energy]['label'];pilot.mkdir()
    pc=dict(N=1000000,seed=h4.CASES[energy]['egs_seed'],spectrum_arrays_sha256=config['spectrum_arrays_sha256'])
    (pilot/'config.json').write_text(json.dumps(pc));(pilot/'summary.json').write_text(json.dumps(dict(wall_seconds=89.)))
    return base


def test_saved_main_reference_normalization_seal_and_startup_model(tmp_path,monkeypatch):
    base=make_ref_files(tmp_path);monkeypatch.setattr(h5,'REFERENCE_ROOT',tmp_path)
    r=h5.read_reference(20);c=h5.CASES[20]
    assert r['N']==18438400 and r['seed']==20101 and np.all(r['fine']==1)
    assert np.all(r['profile_fine']==h4.GY_PER_MEV_G/(c['EGS5_N']/100*.25))
    assert r['timing']['fixed_startup_seconds_estimate']==pytest.approx(83.)
    assert r['timing']['marginal_seconds_per_history_estimate']==pytest.approx(6e-6)
    assert 'wall_seconds_per_history' not in r['timing'] and len(r['files'])==5
    p=base/'config.json';d=json.loads(p.read_text());d['seed']=20001;p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='condition'):h5.read_reference(20)


def test_main_source_seed_and_N_constants_differ_from_pilot():
    for e in (20,60):
        c=h5.CASES[e]
        assert len({c['CPU_seed'],c['retry_CPU_seed'],h4.CASES[e]['cpu_seed']})==3
        assert c['CPU_N']%100==c['EGS5_N']%100==0
        spans=h5.chunks(0,c['CPU_N']//100,h5.CHUNK_SIZE)
        assert sum(k for _,k in spans)==c['CPU_N']//100
        assert max(k for _,k in spans)<=10000
    assert [h5.CASES[e]['CPU_N'] for e in (20,60)]==[20127000,23164000]


def test_chunked_CPU_receipts_and_resume_without_build(tmp_path,monkeypatch):
    from chatcarlo.gpu import therapy,therapy_health,therapy_observation,physics_sets
    decl=physics_sets.declaration(physics_sets.NAME,specialization=physics_sets.ELECTRON)
    c=dict(N=300,cpu_seed=20111,retry_cpu_seed=20121,chunk_size=2,attempt='initial',physics_set=decl)
    calls=[];builds=[]
    def descriptor(a,fg,half,ssd,seed,start,n,ps):return dict(seed=seed,start=start,count=n,physics_set=ps)
    class FakeCPU:
        def __init__(self,work,m,a,**kw):
            assert '_test_legacy_physics' not in kw
            self.observed=decl;builds.append(1)
        def histories(self,n,seed,start,slots):
            assert n<=2 and slots==h4.SLOTS
            calls.append((start,n));self.last_observation=descriptor(None,None,None,None,seed,start,n,decl)
            g=np.zeros((100,1087));g[0,450:1050]=n
            return dict(groups=g,field=np.full(3420,float(n)))
    monkeypatch.setattr(therapy,'CPU',FakeCPU);monkeypatch.setattr(therapy_health,'check',lambda *a:None)
    monkeypatch.setattr(therapy_observation,'descriptor',descriptor)
    r=h5.run_cpu(tmp_path,c,[],{},None)
    assert len(calls)==200 and calls[:2]==[(0,2),(2,1)] and calls[-1]==(299,1)
    assert np.all(r['fine']==h4.GY_PER_MEV_G/.05)
    assert np.all(r['profile_fine']==h4.GY_PER_MEV_G/.25)
    p=tmp_path/'batch_000.json';receipt=json.loads(p.read_text())
    assert len(receipt['chunks'])==2 and receipt['physics_set']==decl
    calls.clear();builds.clear();r2=h5.run_cpu(tmp_path,c,[],{},None)
    assert not calls and not builds and r['timing']==r2['timing']
    with pytest.raises(ValueError,match='retry execution'):h5.run_cpu(tmp_path,{**c,'attempt':'retry'},[],{},None)
    receipt['chunks'][0]['start']=9;p.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='receipt'):h5.run_cpu(tmp_path,c,[],{},None)


def test_gate_never_auto_retries_or_runs_second_case_after_failure(tmp_path,monkeypatch):
    ref=data();monkeypatch.setattr(h5,'read_reference',lambda e:ref)
    monkeypatch.setattr(h4,'prepare_cpu',lambda e:([],{},None))
    monkeypatch.setattr(h5,'make_condition',lambda *a:cond())
    calls=[]
    def fake_run(*a):
        calls.append(1);cpu=data();cpu['samples'][:,0]+=.03e-12;return cpu
    monkeypatch.setattr(h5,'run_cpu',fake_run);monkeypatch.setattr(h5,'RESULT',tmp_path/'result.json')
    r=h5.gate(tmp_path/'out')
    assert r['status']=='retry_required' and r['pending_energies']==[60]
    assert len(calls)==1 and not r['retry_executed']


def test_both_reference_conditions_checked_before_CPU(tmp_path,monkeypatch):
    ref=data();monkeypatch.setattr(h5,'read_reference',lambda e:ref)
    monkeypatch.setattr(h4,'prepare_cpu',lambda e:([],{},None))
    monkeypatch.setattr(h5,'make_condition',lambda e,*a:dict(energy=e))
    h4.check_condition(tmp_path/'60',dict(energy=60,old=True),write=True)
    monkeypatch.setattr(h5,'run_cpu',lambda *a:pytest.fail('must preflight all cases'))
    with pytest.raises(ValueError,match='condition mismatch'):h5.gate(tmp_path)


def test_import_and_cli_guard_do_not_measure(monkeypatch,capsys):
    monkeypatch.setattr(h5,'gate',lambda *a:pytest.fail('must not measure'))
    for argv,code in (([],2),(['--help'],0)):
        monkeypatch.setattr(sys,'argv',['a4u_h5_gate',*argv])
        with pytest.raises(SystemExit) as exc:h5.main()
        assert exc.value.code==code
    capsys.readouterr()


def test_condition_uses_main_N_and_reserves_new_retry_seed(monkeypatch):
    calls=[]
    def original(e,ref,reg,m,a,fg):
        calls.append(e);return dict(N=1000000,cpu_seed=20011,legacy=False,registration=reg,reference_files=ref['files'])
    monkeypatch.setattr(h4,'condition',original)
    reg={'approved':'sealed'}
    for energy in (20,60):
        c=h5.make_condition(energy,dict(files={'main':'sha'}),reg,[],{},None)
        assert c['N']==h5.CASES[energy]['CPU_N'] and c['cpu_seed']==h5.CASES[energy]['CPU_seed']
        assert c['retry_cpu_seed']==h5.CASES[energy]['retry_CPU_seed']
        assert c['registration'] is reg and c['legacy'] is False and c['attempt']=='initial'
    assert calls==[20,60]


def test_reference_with_failed_health_never_accepted(tmp_path,monkeypatch):
    base=make_ref_files(tmp_path);monkeypatch.setattr(h5,'REFERENCE_ROOT',tmp_path)
    p=base/'summary.json';d=json.loads(p.read_text());d['tests']['0']=False;p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='health'):h5.read_reference(20)
