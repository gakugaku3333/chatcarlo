"""P2 tests do not patch any global used by an existing Numba dispatcher."""
import json
import numpy as np
import pytest
from scipy.integrate import quad
from prototypes.mv_electron.physics import M
from prototypes.mv_electron.p2_common import OUT,ROOT
from prototypes.mv_electron.photon_data_p2 import TABLE,coefficients,kn_transfer,kn_total
from prototypes.mv_electron.check_sampling_p2 import kn_density,chi
from prototypes.mv_electron.photon_sampling_p2 import pair_parameters,samples,annihilation_samples

@pytest.fixture(scope='module')
def photon_table():
    if not TABLE.exists():pytest.skip('P2 XCOM table not generated; not a pass')
    return dict(np.load(TABLE))

def test_table_bounds_and_strict_thresholds(photon_table):
    p=photon_table
    for e in (.000999,20.000001):
        with pytest.raises(ValueError,match='outside'):coefficients(e,p['E'],p['water'],p['thresholds'])
    for j,t in enumerate(p['thresholds']):
        assert coefficients(float(t),p['E'],p['water'],p['thresholds'])[2+j]==0
        assert coefficients(float(t+1e-8),p['E'],p['water'],p['thresholds'])[2+j]>0

def test_closed_kn_transfer_and_total_against_quadrature():
    for e in (.01,.1,.511,1.25,2.,6.,10.,20.):
        total=quad(lambda mu:kn_density(mu,e),-1,1,epsabs=1e-30,epsrel=1e-10)[0]
        transfer=quad(lambda mu:(1-1/(1+e/M*(1-mu)))*kn_density(mu,e),-1,1,epsabs=1e-30,epsrel=1e-10)[0]
        assert kn_total(e)==pytest.approx(total,rel=2e-9)
        assert kn_transfer(e)==pytest.approx(transfer,rel=2e-8)

def test_sampler_reproducible_and_momentum():
    args=pair_parameters()
    a,b=samples(6.,2000,117,*args);c,d=samples(6.,2000,117,*args)
    assert np.array_equal(a,c) and np.array_equal(b,d)
    p=a[:,0,None]*a[:,1:4]+np.sqrt(a[:,4]*(a[:,4]+2*M))[:,None]*a[:,5:8]
    assert np.allclose(p,[0,0,6.],rtol=0,atol=1e-12)
    assert np.max(abs(b[:,0]+b[:,5]+2*M-6.))<1e-12
    assert np.max(np.linalg.norm(b[:,1:3]+b[:,6:8],axis=1))<1e-12
    with pytest.raises(ValueError,match='below'):samples(2*M-1e-8,1,117,*args)
    a=annihilation_samples(2000,117)
    assert np.allclose(np.sum(a*a,axis=1),1.,atol=1e-14)

def test_chi_merges_small_expected_bins():
    values=np.repeat(np.arange(4)+.5,[10,40,50,1000])
    a=chi(values,np.arange(5),np.r_[0,np.cumsum([10,40,50,1000])/1100])
    assert a['min_expected']>=100 and a['merged_bins']==2 and a['p']==pytest.approx(1.)

def test_freeze_is_explicit_and_excludes_documents():
    if not (OUT/'manifest.json').exists():pytest.skip('T0 not saved')
    m=json.loads((OUT/'manifest.json').read_text())
    assert not any(p.endswith('.md') or p.startswith('docs/ai/plans/') for p in m['protected'])
    assert 'prototypes/mv_electron/transport.py' in m['mutable_before']

def test_actual_coupled_history_balance_and_limits(photon_table):
    from prototypes.mv_electron.run_coupled_p2 import execute
    r=execute('mf',2.,100,117)
    assert r['totals'][0]==pytest.approx(200.,rel=1e-14)
    assert not np.any(r['violations']) and max(r['maxima'])<1e-9
    assert not r['counts'][5] and not r['counts'][6]
    assert np.allclose(r['fine'].reshape(100,150,4).sum(axis=2),r['coarse'],rtol=0,atol=1e-10)
    r=execute('mf',2.,100,117,maxsteps=1)
    assert r['counts'][5]>0 and r['totals'][3]>0
    assert not r['violations'][0]
    r=execute('mf',10.,100,117,maxstack=1)
    assert r['counts'][6]>0 and r['totals'][3]>0
    assert not r['violations'][0]

def test_bin_averaged_peak_and_invalid_curvature():
    from prototypes.mv_electron.compare_photon_p2 import peak
    z=(np.arange(150)+.5)*.2
    d=10-((z-3.)**2+.2**2/12)
    b=np.tile(d,(100,1))
    assert peak(b,[8,21])==pytest.approx(3.,abs=1e-12)
    assert peak(-b,[8,21]) is None
    assert peak(b,[30,35]) is None

def test_q6_gates_and_reference_width_precedence():
    from prototypes.mv_electron.compare_photon_p2 import compare
    z=(np.arange(150)+.5)*.2
    d=10-((z-3.)**2+.2**2/12)
    coarse=np.tile(d,(100,1));fine=np.tile(np.repeat(d,4),(100,1))
    ref=dict(coarse=coarse,fine=fine);proto=dict(coarse=coarse.copy(),fine=fine.copy())
    lock=dict(status='合格',Dref=10.,window=[8,21],near=list(range(8,22)),deep=[22,23],build=list(range(32)))
    assert compare(proto,ref,ref,lock,19,False,True)['status']=='未評価'
    assert compare(proto,ref,ref,lock,19,True,False)['status']=='未評価'
    out=compare(proto,ref,ref,lock,19,True,True)
    assert out['engineering']=='工学的基準に合格' and out['statistical']=='統計的にも一致'
    sensitivity=dict(coarse=coarse+.11,fine=fine+.11)
    assert compare(proto,ref,sensitivity,lock,19,True,True)['status']=='判定保留'
