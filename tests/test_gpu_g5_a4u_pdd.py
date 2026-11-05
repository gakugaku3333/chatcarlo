"""Artificial PDD scores and fake CPU calls only; no transport runs."""
import json
import hashlib
import sys
import numpy as np
import pytest
from docs.validation.gpu.g5 import a4u_h4_gate as h4, a4u_h5_gate as h5
from docs.validation.gpu.g5 import a4u_pdd_gate as gate, a4u_pdd_run as run
from docs.validation.gpu.g5 import a4u_pdd_cases as pdd


def data():
    pattern=1+np.tile([-.01,.01],50)
    samples,meta=gate.candidate_samples(pattern[:,None]*np.ones((100,120))*1e-12)
    return dict(samples=samples,meta=meta,N=h5.CASES[20]['EGS5_N'],files={})


def cpu(ref):
    return dict(samples=ref['samples'].copy(),fine=np.ones((100,600)),condition={})


def save_h5(tmp_path,monkeypatch,fine):
    path=tmp_path/'sealed_h5.json'
    path.write_text(json.dumps(dict(cases={'20':dict(measured=dict(fine=dict(CPU_mean=fine.mean(0).tolist())))})))
    monkeypatch.setattr(h5,'RESULT',path)


def test_registration_120_bins_depth_peak_and_mandatory():
    ref=data();reg=gate.registration(ref)
    assert reg['indices']==list(range(60)) and reg['K']==60
    assert reg['mandatory']==[True]*4+[False]*56
    assert reg['mandatory_counts']=={'fine':4} and reg['valid']
    assert [b['z_cm'] for b in reg['bins'][:5]]==[.125,.375,.625,.875,1.125]
    assert reg['minimum_evaluable']==48
    assert reg['z_critical']==pytest.approx(h4.NormalDist().inv_cdf(1-h4.ALPHA/120))
    ref['samples'][:,20]=.049e-12
    assert 20 not in gate.registration(ref)['indices']
    ref['samples'][:]=20.
    ref['samples'][:,20]=1.
    assert 20 in gate.registration(ref)['indices']
    ref['samples'][:,0]=.001e-12
    reg=gate.registration(ref)
    assert reg['mandatory_counts']=={'fine':3} and not reg['valid']
    assert h4.MANDATORY_PDD_END_CM==.5  # sealed H4 is never patched


def test_rule_imports_and_shared_analyse():
    assert gate.analyse is h4.analyse
    assert run.CASES is h5.CASES and gate.CASES is h5.CASES
    for name in ('PDD_FRACTION','MAX_JUDGED_DEPTH_CM','ALPHA','SEM_LIMIT',
                 'RELATIVE_LIMIT','CUTOFF_SIGMA','MIN_EVALUABLE_COUNT','MIN_EVALUABLE_FRACTION'):
        assert getattr(gate,name) is getattr(h4,name)
    ref=data();reg=gate.registration(ref)
    assert gate.analyse(ref,ref['samples'],reg)['pilot_D7']['status']=='pass'


def test_evaluate_calls_shared_analysis_and_ignores_deep_bins(tmp_path,monkeypatch):
    ref=data();c=cpu(ref);save_h5(tmp_path,monkeypatch,c['fine']);calls=[]
    def spy(*args):
        calls.append(args)
        return h4.analyse(*args)
    monkeypatch.setattr(gate,'analyse',spy)
    c['samples'][:,60:]*=1000
    result=gate.evaluate_case(20,ref,c)
    assert len(calls)==1 and calls[0][0] is ref and calls[0][1] is c['samples']
    assert result['status']=='pass' and result['fine_matches_H5'] and not result['OCR_judged']
    assert len(result['measured']['axis_coarse']['CPU_mean'])==120
    assert 'main_histories' not in result and 'required_histories' not in result['bins'][0]
    assert result['D7']['historical_2sigma_failed']==0


@pytest.mark.parametrize('index',[0,1,2,3])
def test_every_mandatory_bin_must_be_evaluable(tmp_path,monkeypatch,index):
    ref=data();c=cpu(ref);save_h5(tmp_path,monkeypatch,c['fine'])
    c['samples'][:,index]=(1+np.tile([-.95,.95],50))*1e-12
    result=gate.evaluate_case(20,ref,c)
    assert result['status']=='insufficient' and not result['D7']['mandatory_evaluable']
    assert result['D7']['insufficient']==1


def test_minimum_coverage_and_cutoff_stop(tmp_path,monkeypatch):
    ref=data();c=cpu(ref);save_h5(tmp_path,monkeypatch,c['fine'])
    c['samples'][:,4:17]=(1+np.tile([-.95,.95],50))[:,None]*1e-12
    result=gate.evaluate_case(20,ref,c)
    assert result['status']=='insufficient' and result['D7']['mandatory_evaluable']
    c=cpu(ref);c['samples'][:,5]*=1.1
    result=gate.evaluate_case(20,ref,c)
    assert result['status']=='stop_cutoff' and result['cutoff_bins']==['axis_coarse:5']


def test_fine_exact_mismatch_stops_before_analysis(tmp_path,monkeypatch):
    ref=data();c=cpu(ref);save_h5(tmp_path,monkeypatch,c['fine'])
    gate.check_fine(20,c['fine'])
    c['fine'][:,10]=np.nextafter(1.,2.)
    monkeypatch.setattr(gate,'analyse',lambda *a:pytest.fail('fine must stop before analyse'))
    with pytest.raises(ValueError,match='exactly match'):gate.evaluate_case(20,ref,c)


@pytest.mark.parametrize('energy',[20,60])
@pytest.mark.parametrize('bad',[None,'N','seed'])
@pytest.mark.parametrize('cases',[None,pdd.CASES])
def test_reference_identity_and_normalization(tmp_path,monkeypatch,energy,bad,cases):
    c=(h5.CASES if cases is None else cases)[energy];base=tmp_path/c['label'];base.mkdir()
    values=dict(N=c['EGS5_N'],seed=c['EGS5_seed'])
    if bad:values[bad]+=1
    # No scoring array for mismatched identity: it must fail before reading it.
    if not bad:values['axis_coarse']=np.ones((100,120))
    np.savez(base/'result.npz',**values)
    monkeypatch.setattr(h5,'REFERENCE_ROOT',tmp_path)
    if bad:
        with pytest.raises(ValueError,match='identity'):gate.read_reference(energy,cases)
    else:
        result=gate.read_reference(energy,cases)
        assert np.all(result['samples']==h4.GY_PER_MEV_G/(c['EGS5_N']/100))


def test_run_adds_raw_fields_preserves_H5_and_resumes(tmp_path,monkeypatch):
    from chatcarlo.gpu import therapy, therapy_health, therapy_observation, physics_sets
    decl=physics_sets.declaration(physics_sets.NAME,specialization=physics_sets.ELECTRON)
    cond=dict(N=300,cpu_seed=20111,chunk_size=2,batches=100,attempt='initial',physics_set=decl)
    calls=[];builds=[];health=[]
    def descriptor(a,fg,half,ssd,seed,start,n,ps):return dict(seed=seed,start=start,count=n,physics_set=ps)
    class FakeCPU:
        def __init__(self,*a,**kw):self.observed=decl;builds.append(1)
        def histories(self,n,seed,start,slots):
            calls.append((start,n,seed,slots))
            self.last_observation=descriptor(None,None,None,None,seed,start,n,decl)
            g=np.zeros((100,1087));g[0,450:1050]=n
            return dict(groups=g,field=np.arange(3420,dtype=float)*n)
    monkeypatch.setattr(therapy,'CPU',FakeCPU)
    monkeypatch.setattr(therapy_health,'check',lambda *a:health.append(1))
    monkeypatch.setattr(therapy_observation,'descriptor',descriptor)
    result=run.run_cpu(tmp_path/'pdd',cond,[],{},None)
    assert len(calls)==len(health)==200 and len(builds)==1
    assert calls[0]==(0,2,20111,h4.SLOTS) and calls[-1][:2]==(299,1)
    assert np.array_equal(result['raw_field'],np.tile(np.arange(540)*3.,(100,1)))
    baseline=h5.run_cpu(tmp_path/'h5',cond,[],{},None)
    assert np.array_equal(result['fine'],baseline['fine'])
    assert np.array_equal(result['profile_fine'],baseline['profile_fine'])
    receipt=json.loads((tmp_path/'pdd/batch_000.json').read_text())
    old=json.loads((tmp_path/'h5/batch_000.json').read_text())
    assert receipt.keys()==old.keys()
    for key in ('identity','physics_set','chunks'):assert receipt[key]==old[key]
    monkeypatch.setattr(gate,'CASES',{20:dict(CPU_N=300,CPU_seed=20111)})
    monkeypatch.setattr(gate,'CHUNK_SIZE',2)
    loaded=gate.read_cpu(tmp_path/'pdd',20)
    assert np.array_equal(loaded['samples'],result['raw_field'][:,:120]*h4.GY_PER_MEV_G/3)
    assert np.array_equal(loaded['fine'],result['fine'])
    calls.clear();builds.clear();again=run.run_cpu(tmp_path/'pdd',cond,[],{},None)
    assert not calls and not builds and np.array_equal(again['raw_field'],result['raw_field'])
    assert again['timing']==result['timing']
    with pytest.raises(ValueError,match='retry execution'):
        run.run_cpu(tmp_path/'pdd',{**cond,'attempt':'retry'},[],{},None)
    receipt['chunks'][0]['start']=9
    (tmp_path/'pdd/batch_000.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='receipt'):run.run_cpu(tmp_path/'pdd',cond,[],{},None)
    with pytest.raises(ValueError,match='receipt'):gate.read_cpu(tmp_path/'pdd',20)


@pytest.mark.parametrize('module',[run,gate])
def test_help_and_required_out_never_run(module,monkeypatch,capsys):
    monkeypatch.setattr(module,'run' if module is run else 'gate',lambda *a:pytest.fail('no execution'))
    for argv,code in ((['--help'],0),([],2)):
        monkeypatch.setattr(sys,'argv',['pdd',*argv])
        with pytest.raises(SystemExit) as exc:module.main()
        assert exc.value.code==code
    capsys.readouterr()


def test_pdd_cases_and_fixed_rules():
    assert pdd.CASES=={
        20:dict(label='pdd_20kev',EGS5_N=52727700,EGS5_seed=20201,CPU_N=47953300,CPU_seed=20211),
        60:dict(label='pdd_60kev',EGS5_N=150767400,EGS5_seed=60201,CPU_N=146722900,CPU_seed=60211)}
    for name,value in dict(PDD_FRACTION=.05,MAX_JUDGED_DEPTH_CM=15.,ALPHA=.05,
            SEM_LIMIT=.01,RELATIVE_LIMIT=.02,CUTOFF_SIGMA=2,MIN_EVALUABLE_COUNT=8,
            MIN_EVALUABLE_FRACTION=.8,MANDATORY_PDD_END_CM=1.0).items():
        assert getattr(gate,name)==value
    assert gate.MIN_MANDATORY_BINS=={'fine':4}
    assert gate.analyse is h4.analyse


@pytest.mark.parametrize('selection',[None,'h5','pdd'])
@pytest.mark.parametrize('energy',[None,20,60])
def test_run_cli_selects_reference_and_condition(tmp_path,monkeypatch,selection,energy):
    selected=pdd.CASES if selection=='pdd' else h5.CASES
    energies=list(selected) if energy is None else [energy]
    calls=[];conditions=[];registrations=[]
    def reference(energy,cases=None):
        calls.append((energy,cases))
        return {**data(),'edges':[0.,1.],'cum':[0.,1.]}
    def forbidden(*a):pytest.fail('wrong reference reader')
    monkeypatch.setattr(h5,'read_reference',forbidden if selection=='pdd' else reference)
    monkeypatch.setattr(gate,'read_reference',reference if selection=='pdd' else forbidden)
    def register(ref):
        registrations.append(ref)
        return {'fixed':'registration'}
    monkeypatch.setattr(gate if selection=='pdd' else h5,'registration',register)
    monkeypatch.setattr(h4,'prepare_cpu',lambda e:([],{},None))
    # Exercise the real H5 make_condition, replacing only the table/hash builder.
    monkeypatch.setattr(h4,'condition',lambda *a:dict(batches=100,registration=a[2]))
    monkeypatch.setattr(h4,'check_condition',lambda *a:None)
    monkeypatch.setattr(run,'run_cpu',lambda out,cond,*a:conditions.append((out,cond)))
    argv=['pdd','--out',str(tmp_path)]
    if selection is not None:argv+=['--cases',selection]
    if energy is not None:argv+=['--energy',str(energy)]
    monkeypatch.setattr(sys,'argv',argv)
    assert run.main()==0
    assert calls==[(e,selected if selection=='pdd' else None) for e in energies]
    assert len(registrations)==len(conditions)==len(energies)
    for energy,(out,cond) in zip(energies,conditions):
        c=selected[energy]
        assert out==tmp_path/str(energy)
        assert (cond['N'],cond['cpu_seed'],cond['EGS5_N'],cond['EGS5_seed'])==(
            c['CPU_N'],c['CPU_seed'],c['EGS5_N'],c['EGS5_seed'])
        assert cond['attempt']=='initial' and cond['chunk_size']==10000
        if selection=='pdd':
            assert cond['retry_cpu_seed']==cond['retry_policy']=='not applicable: see A4U_PDD_PREREG.md stop condition'
        else:
            assert cond['retry_cpu_seed']==c['retry_CPU_seed']
            assert cond==h5.make_condition(energy,{},cond['registration'],[],{},None)


def test_run_energy_20_with_missing_60_reference(tmp_path,monkeypatch):
    c=pdd.CASES[20];base=tmp_path/c['label'];base.mkdir()
    np.savez(base/'result.npz',N=c['EGS5_N'],seed=c['EGS5_seed'],
             axis_coarse=np.ones((100,120)))
    monkeypatch.setattr(h5,'REFERENCE_ROOT',tmp_path)
    assert not (tmp_path/pdd.CASES[60]['label']/'result.npz').exists()
    monkeypatch.setattr(h4,'prepare_cpu',lambda e:([],{},None))
    monkeypatch.setattr(h4,'condition',lambda *a:dict(batches=100,registration=a[2]))
    monkeypatch.setattr(h4,'check_condition',lambda *a:None)
    calls=[]
    monkeypatch.setattr(run,'run_cpu',lambda out,*a:calls.append(out))
    monkeypatch.setattr(sys,'argv',['pdd','--out',str(tmp_path/'cpu'),
                                  '--cases','pdd','--energy','20'])
    assert run.main()==0
    assert calls==[tmp_path/'cpu/20']


@pytest.mark.parametrize('energy',['40','20,60'])
def test_run_cli_rejects_invalid_energy(monkeypatch,energy):
    monkeypatch.setattr(run,'run',lambda *a:pytest.fail('invalid energy must not run'))
    monkeypatch.setattr(sys,'argv',['pdd','--out','unused','--energy',energy])
    with pytest.raises(SystemExit) as exc:run.main()
    assert exc.value.code==2


@pytest.mark.parametrize('selection',[None,'h5','pdd'])
def test_gate_cli_selects_cases_and_fine_check(tmp_path,monkeypatch,selection):
    selected=pdd.CASES if selection=='pdd' else h5.CASES
    reads=[];checks=[]
    def reference(energy,cases):
        reads.append(('ref',energy,cases))
        return data()
    def read_cpu(path,energy,cases):
        reads.append(('cpu',energy,cases))
        return cpu(data())
    def check(energy,fine):
        if selection=='pdd':pytest.fail('pdd must never check H5 fine')
        checks.append(energy)
    monkeypatch.setattr(gate,'read_reference',reference)
    monkeypatch.setattr(gate,'read_cpu',read_cpu)
    monkeypatch.setattr(gate,'check_fine',check)
    monkeypatch.setattr(h5,'RESULT',tmp_path/'H5-record-must-not-be-read.json')
    argv=['pdd','--run',str(tmp_path/'cpu'),'--out',str(tmp_path/'report')]
    if selection is not None:argv+=['--cases',selection]
    monkeypatch.setattr(sys,'argv',argv)
    assert gate.main()==0
    assert reads==[(kind,e,selected) for kind in ('ref','cpu') for e in selected]
    assert checks==([] if selection=='pdd' else [20,60,20,60])
    report=json.loads((tmp_path/'report/a4u_pdd.json').read_text())
    assert report['status']=='pass' and report['pending_energies']==[]
    for result in report['cases'].values():
        assert result['fine_matches_H5']==(selection!='pdd')
        if selection=='pdd':
            assert result['fine_matches_H5_reason']=='別の N・seed なので H5 との一致検査は適用外'
        else:assert 'fine_matches_H5_reason' not in result


@pytest.mark.parametrize('energy',[20,60])
def test_pdd_cpu_identity_and_normalization(tmp_path,monkeypatch,energy):
    c=pdd.CASES[energy];decl={'test':'artificial'}
    cond=dict(N=c['CPU_N'],cpu_seed=c['CPU_seed'],attempt='initial',
              chunk_size=gate.CHUNK_SIZE,batches=100,physics_set=decl)
    seal=dict(condition=cond,sha256=h4.digest(cond))
    (tmp_path/'condition.json').write_text(json.dumps(seal))
    n=cond['N']//100
    for b in range(100):
        path=tmp_path/f'batch_{b:03d}.npz'
        np.savez(path,fine=np.ones(600),profile_fine=np.ones((2,3,320)),raw_field=np.ones(540))
        receipt=dict(identity=dict(condition_sha256=seal['sha256'],batch=b,start=b*n,count=n,seed=cond['cpu_seed']),
            sha256=h4.sha(path),physics_set=decl,
            chunks=[dict(start=s,count=k) for s,k in h5.chunks(b*n,n,gate.CHUNK_SIZE)])
        path.with_suffix('.json').write_text(json.dumps(receipt))
    result=gate.read_cpu(tmp_path,energy,pdd.CASES)
    assert np.all(result['samples']==h4.GY_PER_MEV_G/n)
    assert result['condition']['N']==c['CPU_N'] and result['condition']['cpu_seed']==c['CPU_seed']
    with pytest.raises(ValueError,match='fixed condition'):gate.read_cpu(tmp_path,energy)


def test_group_four_sums_each_batch_and_sets_depths():
    axis=np.arange(100*120,dtype=float).reshape(100,120)
    before=axis.copy()
    samples,meta=gate.candidate_samples(axis,group=4)
    expected=np.column_stack([axis[:,j:j+4].sum(1) for j in range(0,120,4)])
    assert samples.shape==(100,30)
    np.testing.assert_array_equal(samples,expected)
    np.testing.assert_array_equal(samples[0,:3],[6.,22.,38.])  # sums, not means
    np.testing.assert_array_equal(axis,before)
    assert [r['z_cm'] for r in meta]==[(j+.5)*1.0 for j in range(30)]
    assert [r['name'] for r in meta]==[f'axis_1cm:{j}' for j in range(30)]
    original,original_meta=gate.candidate_samples(axis)
    explicit,explicit_meta=gate.candidate_samples(axis,group=1)
    assert original.tobytes()==explicit.tobytes()==axis.tobytes()
    assert original_meta==explicit_meta


@pytest.mark.parametrize('value',['0','2','8','-1','1.0'])
def test_group_cli_rejects_other_values(monkeypatch,value):
    monkeypatch.setattr(gate,'gate',lambda *a:pytest.fail('invalid group must not read data'))
    monkeypatch.setattr(sys,'argv',['pdd','--run','unused','--out','unused','--group',value])
    with pytest.raises(SystemExit) as exc:gate.main()
    assert exc.value.code==2


def test_group_four_cli_groups_both_sides_and_marks_post_hoc(tmp_path,monkeypatch):
    reads=[];checks=[]
    ref=data();c=cpu(ref)
    # Different depth profiles expose asymmetric grouping or averaging.
    ref['samples']*=np.arange(11,131)[None,:]
    c['samples']*=np.arange(121,241)[None,:]
    def reference(energy,cases):
        reads.append(('ref',energy,cases))
        return {**ref}
    def read_cpu(path,energy,cases):
        reads.append(('cpu',energy,cases))
        return {**c}
    monkeypatch.setattr(gate,'read_reference',reference)
    monkeypatch.setattr(gate,'read_cpu',read_cpu)
    monkeypatch.setattr(gate,'check_fine',lambda e,f:checks.append(e))
    monkeypatch.setattr(sys,'argv',['pdd','--run','unused','--out',str(tmp_path),'--group','4'])
    assert gate.main()==1  # deliberately different profiles trigger the unchanged cutoff
    report=json.loads((tmp_path/'a4u_pdd.json').read_text())
    assert report['status']=='stop_cutoff'
    assert reads==[(kind,60,{60:h5.CASES[60]}) for kind in ('ref','cpu')]
    assert checks==[60,60]
    assert set(report['cases'])=={'60'}
    assert report['group']==4 and report['bin_width_cm']==1.0 and report['post_hoc'] is True
    assert report['post_hoc_reason']=='0.25 cm では統計不足だったのを見てから粗くした後出しの判定。A4U_PDD_PREREG.md の追補 2'
    result=report['cases']['60'];reg=result['registration']
    assert reg['K']==15 and reg['minimum_evaluable']==12
    assert reg['valid'] and reg['mandatory_counts']=={'fine':1}
    assert reg['mandatory']==[True]+[False]*14
    assert [r['z_cm'] for r in reg['bins']]==[(j+.5)*1.0 for j in range(15)]
    for side,original in (('EGS5',ref['samples']),('CPU',c['samples'])):
        expected=np.column_stack([original[:,j:j+4].sum(1) for j in range(0,120,4)])
        mean,sem=gate.moments(expected)
        measured=result['measured']['axis_coarse']
        np.testing.assert_array_equal(measured[side+'_mean'],mean)
        np.testing.assert_array_equal(measured[side+'_SEM'],sem)
        assert len(measured[side+'_mean'])==30


@pytest.mark.parametrize('group',[1,4])
def test_surface_coverage_uses_same_one_cm_and_all_mandatory_bins(group):
    ref=data()
    ref['samples'],ref['meta']=gate.candidate_samples(ref['samples'],group)
    c=cpu(ref)
    reg=gate.registration(ref,group)
    assert reg['valid'] and reg['mandatory_counts']=={'fine':4//group}
    assert sum(reg['mandatory'])==4//group
    assert gate.evaluate_case(60,ref,c,cases='pdd',group=group)['status']=='pass'
    # Losing any required surface bin must still prevent coverage.
    for index in range(4//group):
        noisy=cpu(ref)
        noisy['samples'][:,index]*=1+np.tile([-.95,.95],50)
        result=gate.evaluate_case(60,ref,noisy,cases='pdd',group=group)
        assert result['status']=='insufficient'
        assert not result['D7']['mandatory_evaluable']
    missing={**ref,'samples':ref['samples'].copy()}
    missing['samples'][:,0]=0
    assert not gate.registration(missing,group)['valid']
    assert gate.MIN_MANDATORY_BINS==h4.MIN_MANDATORY_BINS=={'fine':4}
    assert gate.MIN_MANDATORY_BINS is h4.MIN_MANDATORY_BINS
    assert gate.MANDATORY_PDD_END_CM==1.0


@pytest.mark.parametrize('selection,expected_hash',[
    ('h5','93295c88919f2fe2d958981a8ecad65a8d02737b014804adfe17ed9ffcc135de'),
    ('pdd','ffe0d25e76260483a527b95ae246e8c8359e87cabd1cd2c3ef4167d2e7195f59'),
])
def test_default_report_bytes_match_before_group_option(tmp_path,monkeypatch,capsys,selection,expected_hash):
    # SHA256 captured from the unmodified gate with these artificial inputs.
    monkeypatch.setattr(gate,'read_reference',lambda e,c:data())
    monkeypatch.setattr(gate,'read_cpu',lambda path,e,c:cpu(data()))
    monkeypatch.setattr(gate,'check_fine',lambda *a:None)
    argv=['pdd','--run','unused','--out',str(tmp_path),'--cases',selection]
    outputs=[]
    for extra in ([],['--group','1']):
        monkeypatch.setattr(sys,'argv',argv+extra)
        assert gate.main()==0
        raw=(tmp_path/'a4u_pdd.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest()==expected_hash
        outputs.append(capsys.readouterr().out)
    assert outputs==['{\n  "status": "pass",\n  "pending_energies": []\n}\n']*2


@pytest.mark.parametrize('selection',[None,'h5','pdd'])
@pytest.mark.parametrize('energy',[20,60])
def test_gate_cli_energy_reads_only_selected_energy(tmp_path,monkeypatch,selection,energy):
    selected=pdd.CASES if selection=='pdd' else h5.CASES
    reads=[]
    def reference(e,cases):
        assert e==energy and cases=={energy:selected[energy]}
        reads.append(('ref',e))
        return data()
    def read_cpu(path,e,cases):
        assert e==energy and cases=={energy:selected[energy]}
        assert path==tmp_path/'cpu'/str(energy)
        reads.append(('cpu',e))
        return cpu(data())
    monkeypatch.setattr(gate,'read_reference',reference)
    monkeypatch.setattr(gate,'read_cpu',read_cpu)
    monkeypatch.setattr(gate,'check_fine',lambda *a:None)
    argv=['pdd','--run',str(tmp_path/'cpu'),'--out',str(tmp_path/'report'),
          '--energy',str(energy)]
    if selection is not None:argv+=['--cases',selection]
    monkeypatch.setattr(sys,'argv',argv)
    assert gate.main()==0
    assert reads==[('ref',energy),('cpu',energy)]
    report=json.loads((tmp_path/'report/a4u_pdd.json').read_text())
    assert set(report['cases'])=={str(energy)} and report['pending_energies']==[]


def test_gate_cli_energy_20_group_four_rejected_before_reading(tmp_path,monkeypatch):
    monkeypatch.setattr(gate,'read_reference',lambda *a:pytest.fail('must reject before reading'))
    monkeypatch.setattr(gate,'read_cpu',lambda *a:pytest.fail('must reject before reading'))
    out=tmp_path/'report'
    monkeypatch.setattr(sys,'argv',['pdd','--run','unused','--out',str(out),
                                  '--energy','20','--group','4'])
    with pytest.raises(ValueError,match='--energy 20.*--group 4'):gate.main()
    assert not out.exists()


@pytest.mark.parametrize('energy',['40','20,60'])
def test_gate_cli_rejects_invalid_energy(monkeypatch,energy):
    monkeypatch.setattr(gate,'gate',lambda *a:pytest.fail('invalid energy must not read data'))
    monkeypatch.setattr(sys,'argv',['pdd','--run','unused','--out','unused','--energy',energy])
    with pytest.raises(SystemExit) as exc:gate.main()
    assert exc.value.code==2


def precision_data(exceeded):
    ref=data()
    # 100 batches alternating +/-8% give a relative SEM of ~0.804%,
    # above 1%/sqrt(2) but below 1%: the split budget must be enforced.
    noisy=(1+np.tile([-.08,.08],50))[:,None]*1e-12
    ref['samples'][:,:exceeded]=noisy
    # Deep bins and a below-peak bin must not count toward the budget or K.
    ref['samples'][:,60:]=noisy
    ref['samples'][:,59]=noisy[:,0]*.01
    return ref


@pytest.mark.parametrize('exceeded',[0,11,12])
def test_reference_precision_within_budget_passes(exceeded):
    ref=precision_data(exceeded)
    # Restore bin 59: K=60, so exactly 12/60 (20%) is permitted.
    ref['samples'][:,59]=1e-12
    reg=gate.registration(ref)
    assert reg['K']==60
    gate.check_reference_precision(ref,reg)


def test_reference_precision_over_budget_raises():
    ref=precision_data(12);reg=gate.registration(ref)
    assert reg['K']==59  # 12/59 exceeds 20%; unselected noisy bins are ignored.
    with pytest.raises(ValueError,match='参照側だけで予算を使い切っている.*bin 幅か参照の履歴数を見直す.*12/59'):
        gate.check_reference_precision(ref,reg)


@pytest.mark.parametrize('selection',[None,'h5','pdd'])
def test_reference_precision_stops_cli_before_analyse(tmp_path,monkeypatch,selection):
    ref=precision_data(12)
    monkeypatch.setattr(gate,'read_reference',lambda *a:ref)
    monkeypatch.setattr(gate,'read_cpu',lambda *a:cpu(ref))
    monkeypatch.setattr(gate,'check_fine',lambda *a:None)
    monkeypatch.setattr(gate,'analyse',lambda *a:pytest.fail('precision must stop before analyse'))
    out=tmp_path/'report'
    argv=['pdd','--run','unused','--out',str(out)]
    if selection is not None:argv+=['--cases',selection]
    monkeypatch.setattr(sys,'argv',argv)
    with pytest.raises(ValueError,match='12/59'):gate.main()
    assert not out.exists()
