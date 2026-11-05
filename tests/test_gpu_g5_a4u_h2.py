"""H2 launcher checks use synthetic outer runs and a fake CPU; never measure H2."""
import json
import sys
import numpy as np
import pytest
from docs.validation.gpu.g5 import a4u_h2_gate as h2


def samples():
    names=[f'q{i}' for i in range(57)]+list(h2.gates.DIVERGENT)
    x=np.arange(10,dtype=float)+100
    old={name:x.copy() for name in names};new={name:x.copy() for name in names}
    old['q0']=np.zeros(10);new['q0']=np.zeros(10)
    return old,new


def test_all_61_rows_and_expected_divergence():
    old,new=samples()
    for name in h2.gates.DIVERGENT:new[name]+=100
    r=h2.judge(old,new)
    assert r['status']=='pass' and not r['unexpected']
    assert len(r['quantities'])==61
    assert r['quantities']['q0']['z']==0 and r['quantities']['q0']['SEM']==0
    row=r['quantities'][h2.gates.DIVERGENT[0]]
    sem=np.std(np.arange(10),ddof=1)/np.sqrt(10)
    assert row['K_only_SEM']==pytest.approx(sem)
    assert row['all_shell_SEM']==pytest.approx(sem)
    assert row['SEM']==pytest.approx(np.sqrt(2)*sem)
    assert row['difference']==100
    assert row['z']==pytest.approx(100/(np.sqrt(2)*sem))
    assert row['detectable_difference']==pytest.approx(3*np.sqrt(2)*sem)
    assert r['divergence_sha256']==h2.gates.DIVERGENCE_SHA256


def test_unexpected_difference_retry_is_bounded_and_never_forced_pass():
    old,new=samples();sem=np.sqrt(2)*old['q1'].std(ddof=1)/np.sqrt(10)
    new['q1']+=3.5*sem
    first=h2.judge(old,new)
    assert first['status']=='retry' and first['branch_status']=='stop'
    assert first['unexpected']==['q1']
    assert h2.judge(old,new,retry=True)['status']=='fail'
    new['q1']+=sem
    assert h2.judge(old,new)['status']=='fail'
    del new['q2']
    with pytest.raises(ValueError,match='61'):h2.judge(old,new)


def test_cpu_builds_seeds_resume_and_preserved_judgement(tmp_path,monkeypatch):
    builds=[];calls=[]
    class FakeCPU:
        def __init__(self,arrays,device,*,build_dir,_test_legacy_physics):
            assert device=='cpu'
            self.legacy=_test_legacy_physics
            self.set_options=(f'-DG5_TEST_LEGACY_LOCAL={int(self.legacy)}',)
            builds.append((device,self.legacy))
    monkeypatch.setattr(h2.core,'Core',FakeCPU)
    monkeypatch.setattr(h2,'inputs',lambda sid:(None,None,sid+'-scene-sha'))
    monkeypatch.setattr(h2.bundle,'create',lambda *args:None)
    monkeypatch.setattr(h2.bundle,'load',lambda path:({}, {}, (),dict(material_names=[],fluorescence=True),dict(files={})))
    def fake_sample(driver,source,spectrum,names,sid,seed,source_seed,fluorescence):
        calls.append((sid,driver.legacy,seed,source_seed))
        q={f'metric{i}':1. for i in range(19)}
        if sid=='s2':q.update({name.split(':',1)[1]:1. for name in h2.gates.DIVERGENT})
        return q
    monkeypatch.setattr(h2,'sample',fake_sample)
    r=h2.gate(tmp_path)
    assert builds==[('cpu',True),('cpu',False)]*3
    assert len(calls)==60 and len(r['quantities'])==61 and len(r['attempts'])==1
    assert {c[2] for c in calls if c[1]}==set(range(5000,5010))
    assert {c[2] for c in calls if not c[1]}==set(range(6000,6010))
    assert {c[3] for c in calls}==set(range(7000,7010))
    calls.clear()
    assert h2.gate(tmp_path)['status']==r['status']
    assert not calls
    path=next(tmp_path.glob('h2_initial_s1_legacy_*.json'))
    saved=json.loads(path.read_text());saved['identity']['build_options']=[]
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError,match='resume identity'):h2.gate(tmp_path)


def test_cli_never_starts_without_explicit_out(monkeypatch,capsys):
    def forbidden(*args):pytest.fail('H2 must not start')
    monkeypatch.setattr(h2,'gate',forbidden)
    for args,code in (([],2),(['--help'],0)):
        monkeypatch.setattr(sys,'argv',['a4u_h2_gate',*args])
        with pytest.raises(SystemExit) as exc:h2.main()
        assert exc.value.code==code
    capsys.readouterr()
