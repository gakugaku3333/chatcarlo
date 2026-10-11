"""Small P3s regression and rule tests; full S1–S7 use --checks."""
import json
import numpy as np
import pytest
from prototypes.mv_electron import run_p3s as p
from prototypes.mv_electron.photon_transport_p3s import hinge_transition, rule_value
from prototypes.mv_electron.physics import M


@pytest.fixture(scope='module')
def spectrum():
    if not p.base.SPECTRUM.is_file():pytest.skip('private Mohan spectrum absent')
    return p.base.load_spectrum()


@pytest.mark.parametrize('backend,E', [('mf',6.),('elsepa',10.)])
def test_exact_regression(spectrum,backend,E):
    edges,_,cum=spectrum
    ed,cu=(None,None) if backend=='mf' else (edges,cum)
    a=p.execute(E,100,20291501,backend,ed,cu,original=True)
    b=p.execute(E,100,20291501,backend,ed,cu)
    assert all(p.equality(a,b,p.base.KEYS).values())
    assert p.base.healthy(b)


@pytest.mark.parametrize('kerma,pair',[(False,False),(False,True),(True,False)])
def test_scoring_flags(spectrum,kerma,pair):
    edges,_,cum=spectrum
    a=p.execute(10.,100,20291502,'elsepa',edges,cum)
    b=p.execute(10.,100,20291502,'elsepa',edges,cum,score_kerma=kerma,score_pair=pair)
    assert all(p.equality(a,b,p.PHYSICS+('steps',)).values())
    if not kerma:assert not np.any(b['kerma']) and not np.any(b['collision_kerma'])
    if not pair:assert not np.any(b['pair'])


def test_step_boundaries():
    points=[0.,.02,np.nextafter(.02,1.),1.,np.nextafter(1.,2.),100.]
    assert [p.value('own',x) for x in points]==[.1,.1,.05,.05,.025,.025]


def test_linear_rule_and_clamping():
    e=np.array([.01,1.,10.]);k=np.array([.1,.05,.025])
    for T,expected in [(0.,.1),(.505,.075),(5.5,.0375),(100.,.025)]:
        assert abs(rule_value(T,.025,2,e,k)-expected)<1e-15


def test_egs_runtime_rule():
    if not p.KFILE.is_file():pytest.skip('K1.dat absent')
    _,e,k=p.rules('egs')
    assert np.all(np.diff(e)>0)
    assert p.value('egs',0.)==k[0] and p.value('egs',100.)==k[-1]
    for T,expected in zip((.04,1.5,6.,8.),(.077556,.042736,.017484,.010557)):
        assert abs(p.value('egs',T)/expected-1)<=1e-4


@pytest.mark.parametrize('kind',[1,2])
def test_flat_rule_exact(spectrum,kind):
    edges,_,cum=spectrum
    e,k=(np.array([.02,1.]),np.full(3,.025)) if kind==1 else (np.array([.01,10.]),np.full(2,.025))
    a=p.execute(10.,100,20291502,'elsepa',edges,cum)
    b=dict(zip(p.KEYS,p.run_coupled(10.,100,20291502,.05,.025,30.,*p.inputs('elsepa'),
               annihilate_in_flight=True,spectrum_edges=edges,spectrum_cum=cum,
               rule_kind=kind,rule_energies=e,rule_values=k)))
    assert all(p.equality(a,b,p.KEYS).values())


@pytest.mark.parametrize('charge',[-1,1])
def test_charged_source_balance(charge):
    r=p.execute(2.,100,20291510,'mf',charged_source_T=2.,charged_source_charge=charge,
                score_lateral=True)
    assert p.base.healthy(r)
    expected=100*(2.+(2*M if charge==1 else 0))
    assert np.isclose(r['totals'][0],expected,rtol=1e-14)
    assert r['lateral'].sum()>0
    if charge==-1:assert r['totals'][4]==r['totals'][5]==0


def test_lateral_does_not_change_rng_or_dose():
    a=p.execute(2.,100,20291510,'mf',charged_source_T=2.)
    b=p.execute(2.,100,20291510,'mf',charged_source_T=2.,score_lateral=True)
    assert all(p.equality(a,b,p.base.KEYS+('steps',)).values())
    assert not np.any(a['lateral'])


@pytest.mark.parametrize('old,tail,new,expected',[(.025,.015,.1,[.045,.07,.1]),(.1,.07,.025,[.0775,.0175,.025])])
def test_hinge_transition(old,tail,new,expected):
    a=np.zeros(22);a[12]=99.;a[13]=tail;a[14]=old
    assert hinge_transition(a,new,.3)==old
    np.testing.assert_allclose(a[12:15],expected,rtol=0,atol=1e-15)


def test_real_hinge_uses_old_K():
    ok,values=p.hinge_checks()
    assert ok,values


def test_saved_resume(spectrum,tmp_path,monkeypatch):
    monkeypatch.setattr(p,'CACHE',tmp_path/'cache');monkeypatch.setattr(p,'OUT',tmp_path/'out')
    edges,_,cum=spectrum
    kw=dict(N=200,seed=20291520,edges=edges,cum=cum,rule='own',backend='mf',check=True)
    assert p.run_saved('resume',**kw,stop_after=3) is None
    assert len(list((tmp_path/'cache/prototype/resume').glob('batch_*.npz')))==3
    a=p.load(p.run_saved('resume',**kw));b=p.load(p.run_saved('whole',**kw))
    assert all(p.equality(a,b,[k for k in a if k!='seconds']).values())
    with pytest.raises(RuntimeError,match='config mismatch'):p.run_saved('resume',**{**kw,'seed':20291521})


def test_gate_rejects_stale_checks(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'OUT',tmp_path)
    p.write(tmp_path/'checks.json',dict(status='合格',source_hashes={},checks={}))
    with pytest.raises(RuntimeError,match='current hashes'):p.require_checks('own')
