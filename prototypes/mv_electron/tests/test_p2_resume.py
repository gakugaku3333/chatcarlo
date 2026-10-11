"""Resume/control tests do not change any Numba kernel globals."""
import json
import re
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pytest
from prototypes.mv_electron import workflow_p2 as w
from prototypes.mv_electron.compare_photon_p2 import run_requirement


def test_upper_limit_batch_formula_without_transport(tmp_path):
    source=(w.ROOT/'prototypes/mv_electron/p2_egs.f').read_text()
    expression=re.search(r'^\s*b=(min\(100,1\+.*\))\s*$',source,re.M).group(1)
    fc=shutil.which('gfortran')
    if not fc:pytest.skip('gfortran absent; upper-limit check not evaluated')
    code=f'''program upper
implicit none
integer i,ncase,b,j
integer ids(8)
ncase=200000000
ids=(/1,2000000,2000001,21470000,21474837,198000000,198000001,200000000/)
do j=1,8
 i=ids(j)
 b={expression}
 print *,i,b
enddo
end program
'''
    (tmp_path/'upper.f90').write_text(code)
    subprocess.run([fc,'-fcheck=all','-o',str(tmp_path/'upper'),str(tmp_path/'upper.f90')],check=True,capture_output=True)
    rows=subprocess.check_output([str(tmp_path/'upper')],text=True).splitlines()
    for row in rows:
        i,b=map(int,row.split());assert b==min(100,1+(i-1)//2000000)


def test_requirement_includes_peak_and_combined_half_step_sem():
    z=(np.arange(150)+.5)*.2;d=10-.1*((z-3)**2+.2**2/12)
    rng=np.random.default_rng(719)
    a=d[None,:]+rng.normal(0,.015,(100,150));f=np.repeat(a,4,axis=1)
    half=a+rng.normal(0,2,(100,150));hf=np.repeat(half,4,axis=1)
    lock=dict(status='合格',Dref=10.,window=[8,21],deep=[25],near=list(range(8,22)),build=[0,1])
    r=run_requirement(dict(coarse=a,fine=f),dict(coarse=a,fine=f),[dict(coarse=half,fine=hf)],lock,1000000,891)
    assert r['Q7_N']>r['dose_N']
    assert r['required_N']==max(1000000,r['dose_N'],r['peak_N'],r['Q7_N'])
    r=run_requirement(dict(coarse=-a,fine=-f),dict(coarse=a,fine=f),[],lock,1000000,891)
    assert r['peak_status']=='未評価'


def test_cap_is_per_run_and_does_not_change_seed():
    c=dict(backend='mf',E=10.,seed=77,fe=.05,k=.025)
    config,est=w.capped_run(c,300000000,10000.)
    assert config['N']==8640000 and config['seed']==77
    assert est['projected_statistics']=='統計不足'
    assert est['predicted_seconds']<=86400
    config,est=w.capped_run(c,300000000,1.)
    assert config['N']==200000000


def test_full_route_reuses_completed_pilots_lock_and_production(monkeypatch,tmp_path):
    """Exercise pilot -> lock -> explicit freeze -> main -> analysis, then resume."""
    out=tmp_path/'out';out.mkdir();root=tmp_path/'root';code=root/'prototypes/mv_electron';code.mkdir(parents=True)
    monkeypatch.setattr(w,'OUT',out);monkeypatch.setattr(w,'ROOT',root)
    def write(path,value):
        p=Path(path);p=p if p.is_absolute() else out/p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
    monkeypatch.setattr(w,'write',write)
    monkeypatch.setattr(w,'portable',lambda p:str(Path(p).relative_to(tmp_path)))
    from prototypes.mv_electron import p2_common
    monkeypatch.setattr(p2_common,'resolve',lambda p:tmp_path/p)
    groups=[]
    def verify(group='protected'):
        groups.append(group)
        m=json.loads((out/'manifest.json').read_text())
        if group=='production_frozen':
            assert m[group]
            for p,h in m[group].items():assert w.sha(tmp_path/p)==h
        return dict(status='合格')
    monkeypatch.setattr(w,'verify',verify)
    monkeypatch.setattr(w,'checkpoint',lambda *a,**k:None)
    monkeypatch.setattr(w,'mark_main',lambda *a,**k:verify('production_frozen'))
    for name in w.PRODUCTION_CODE:(code/name).write_text('test input '+name)
    assert all(not name.endswith('.md') for name in w.PRODUCTION_CODE)
    for name in ('q1_amendment1.json','q2.json','q3.json','q4.json','t5_cutoffs_amendment1.json','t5_stopping_amendment2.json','t5_selfcheck_d2.json','d2_verification.json','d2_reference_restart.json'):write(name,dict(status='合格'))
    write('t5_stopping_amendment2.json',dict(status='合格',affected_source_energies=[]))
    write('q0_regression.json',dict(status='合格'));write('manifest.json',dict(protected={},production_frozen={}))
    tables={b:tmp_path/f'{b}.npz' for b in w.BACKENDS}
    for p in tables.values():p.write_bytes(b'table')
    monkeypatch.setattr(w,'TABLES',tables);monkeypatch.setattr(w,'TABLE',tmp_path/'photon.npz');w.TABLE.write_bytes(b'photons')
    z=(np.arange(150)+.5)*.2;d=np.maximum(.01,10-.05*((z-3)**2+.2**2/12));a=np.tile(d,(100,1))
    counts={};production_calls=[]
    def fake(name,reference=False,production=False,**c):
        import inspect
        from prototypes.mv_electron.reference_p2 import run as actual_reference
        from prototypes.mv_electron.run_coupled_p2 import run_saved as actual_prototype
        inspect.signature(actual_reference if reference else actual_prototype).bind(name,production=production,**c)
        assert 'N' not in c and 'n' in c
        folder=out/('egs5' if reference else 'runs')/name;folder.mkdir(parents=True,exist_ok=True)
        if production:verify('production_frozen')
        if not (folder/'result.npz').exists():
            counts[name]=counts.get(name,0)+1
            np.savez(folder/'result.npz',coarse=a,fine=np.repeat(a,4,axis=1),pair=np.zeros((100,600)),kerma=np.repeat(a,4,axis=1),collision_kerma=np.repeat(a,4,axis=1),violations=np.zeros(3,dtype=int),counts=np.zeros(14,dtype=int),seconds=1.)
            if production:production_calls.append(name)
        if reference:
            write(folder/'summary.json',dict(status='合格',wall_seconds=1.))
            return dict(status='合格')
        return folder/'result.npz'
    monkeypatch.setattr(w,'reference_run',lambda name,**kw:fake(name,reference=True,**kw))
    monkeypatch.setattr(w,'run_saved',fake);monkeypatch.setattr(w,'run_budgeted',lambda name,**kw:fake(name,production=True,**kw))
    analysed=[];real_analyse=w.analyse
    monkeypatch.setattr(w,'plots',lambda lock:None)
    def analyse(lock):
        analysed.append(lock);real_analyse(lock)
    monkeypatch.setattr(w,'analyse',analyse)
    w.continue_calculations();first=json.loads((out/'pilot_lock.json').read_text())
    w.continue_calculations();assert first==json.loads((out/'pilot_lock.json').read_text())
    assert all(n==1 for n in counts.values()) and len(analysed)==2
    assert len(production_calls)==len(first['runs'])+len(first['reference_runs'])==32
    assert json.loads((out/'q6.json').read_text())['status']=='合格'
    assert json.loads((out/'q7.json').read_text())['status']=='合格'
    # Invalid reference selfchecks must not be promoted by matching dose curves.
    write(out/'egs5/main_ref_0_0/summary.json',dict(status='不合格'))
    real_analyse(first)
    bad=[r for r in json.loads((out/'q6.json').read_text())['rows'] if r['E']==1.25]
    assert all(r['status']=='未評価' for r in bad)
    frozen=json.loads((out/'manifest.json').read_text())['production_frozen']
    assert not any(p.endswith('.md') or 'plans/' in p for p in frozen)
    # A third command cannot accept a changed frozen code or lock.
    (code/'p2_egs.f').write_text('changed')
    with pytest.raises(AssertionError):w.continue_calculations()


def test_process_wall_budget_and_surviving_supervisor(tmp_path):
    import sys
    import time
    from prototypes.mv_electron.process_p2 import run_process,alive
    timer=tmp_path/'timer.json';log=tmp_path/'run.log'
    r=run_process(timer,[sys.executable,'-c','import time; time.sleep(2)'],tmp_path,log,.1)
    assert r['timed_out'] and r['total_seconds']>=.1 and r['attempts']==1
    r=run_process(timer,[sys.executable,'-c','raise Exception("should not run")'],tmp_path,log,.1)
    assert r['attempts']==1
    timer2=tmp_path/'attach.json';timer2.write_text(json.dumps(dict(total_seconds=0.,attempts=0)))
    module=Path(__import__('prototypes.mv_electron.process_p2',fromlist=['x']).__file__)
    child=subprocess.Popen([sys.executable,str(module),str(timer2),json.dumps([sys.executable,'-c','import time; time.sleep(.2)']),str(tmp_path),str(log),'none'])
    deadline=time.monotonic()+10
    while not json.loads(timer2.read_text()).get('active'):
        assert time.monotonic()<deadline;time.sleep(.01)
    r=run_process(timer2,[sys.executable,'-c','raise Exception("duplicate")'],tmp_path,log)
    child.wait()
    assert r['returncode']==0 and r['attempts']==1 and not r['active']


def test_completed_prototype_file_hash_is_stable_on_resume(monkeypatch,tmp_path):
    from prototypes.mv_electron import run_coupled_p2 as entry
    calls=[]
    monkeypatch.setattr(entry,'OUT',tmp_path)
    monkeypatch.setattr(entry,'verify',lambda *a:None)
    monkeypatch.setattr(entry,'inputs',lambda backend:())
    def kernel(*a):
        calls.append(a[2])
        return (np.zeros((100,150)),*(np.zeros((100,600)) for _ in range(4)),
                np.zeros(6),np.zeros(3),np.zeros(3,dtype=int),np.zeros(14,dtype=int),0)
    monkeypatch.setattr(entry,'run_coupled',kernel)
    p=entry.run_saved('stable','mf',2.,100,19);before=entry.sha(p)
    p2=entry.run_saved('stable','mf',2.,100,19)
    assert len(calls)==100 and p2==p and entry.sha(p)==before
    with pytest.raises(RuntimeError,match='config mismatch'):
        entry.run_saved('stable','mf',2.,100,20)
