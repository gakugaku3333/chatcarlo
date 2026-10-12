"""Independent formula quadrature and deterministic P3r transport checks.

Full statistical counts are run by run_p3r checks; these unit tests never build
EGS5 or write ELSEPA/cache tables. No private reference data are embedded.
"""
import inspect
import numpy as np
import pytest
from scipy.integrate import quad
from numba import njit
from prototypes.mv_electron import bremsstrahlung_p3r as b
from prototypes.mv_electron import photon_transport_p3r as t
from prototypes.mv_electron import run_p3s as old
from prototypes.mv_electron.physics import M, RE, ALPHA, Z, NJ, ER, LOGET, collision, hard


def independent_shape(T, k):
    # Rewritten SLAC-R-730 §2.7 Born formula, not a call to the table builder.
    incident=T+M
    remaining=incident-k
    value=0.
    for charge,density in zip(Z,NJ):
        a2=(charge*ALPHA)**2
        coulomb=a2*(1/(1+a2)+.20206-.0369*a2+.0083*a2*a2-.002*a2**3)
        if charge==1:xi=6.144/(5.31-coulomb)
        else:xi=np.log(1194/charge**(2/3))/(np.log(184.15/charge**(1/3))-coulomb)
        d=136*M*k/(charge**(1/3)*incident*remaining)
        shift=4*np.log(charge)/3
        if d>=np.exp((21.12-shift)/4.184)-.952:continue
        if d<=1:
            phi1=20.867-3.242*d+.625*d*d
            phi2=20.209-1.930*d-.086*d*d
        else:phi1=phi2=21.12-4.184*np.log(d+.952)
        ratio=remaining/incident
        value+=density*RE**2*ALPHA*charge*(charge+xi)*((1+ratio**2)*(phi1-shift)-2*ratio*(phi2-shift)/3)
    return value


def independent_integral(T,lo,hi,log=False):
    # Independent analytic subdivision of each screening branch.
    E=T+M; splits=[]
    for z in Z:
        for d in (1.,np.exp((21.12-4*np.log(z)/3)/4.184)-.952):
            edge=E*d/(d+136*M/(E*z**(1/3)))
            if lo<edge<hi:splits.append(np.log(edge) if log else edge)
    fun=(lambda u:independent_shape(T,np.exp(u))) if log else (lambda x:independent_shape(T,x))
    low,high=(np.log(lo),np.log(hi)) if log else (lo,hi)
    return quad(fun,low,high,points=sorted(splits),epsabs=1e-12,epsrel=1e-10,limit=150)[0]


def independent_quantities(T):
    rad=np.interp(np.log(T),LOGET,ER)
    norm=rad/independent_integral(T,0.,T)
    if T<=.01:return 0.,0.,rad
    lam=norm*independent_integral(T,.01,T,True)
    discrete=norm*independent_integral(T,.01,T)
    soft=norm*independent_integral(T,0.,.01)
    return lam,discrete,soft


def table_check():
    e,v=b.tables()
    points=[.01,np.nextafter(.01,np.inf),(.01+e[1])/2,.0101,.0105,.011,
            .015,.02,.03,.05,.07,.1,.15,.3,.5,.7,1.,1.5,3.,5.,7.,10.,15.]
    rows=[]
    for j,T in enumerate(points):
        actual=np.array(b.quantities(T,e,v)[:3]);expected=np.array(independent_quantities(T))
        error=abs(actual-expected)
        if j==0:passed=actual[0]==actual[1]==0 and error[2]<=expected[2]*1e-3
        elif j in (1,2):passed=error[0]<=1e-6 and error[1]<=1e-8 and error[2]<=expected[2]*1e-3
        else:passed=np.all(error<=expected*1e-3)
        rad=np.interp(np.log(T),LOGET,ER)
        closure=abs((actual[1]+actual[2])/rad-1)
        stopping=b.restricted_stopping(T,e,v)
        stopping_ok=stopping==collision(T)-hard(T)[1]+actual[2]
        rows.append(dict(T=T,actual=actual.tolist(),independent=expected.tolist(),
                         absolute_error=error.tolist(),closure=closure,passed=bool(passed and closure<=1e-3 and stopping_ok)))
    node_error=max(abs((v[:,1]+v[:,2])/np.interp(np.log(e),LOGET,ER)-1))
    return all(r['passed'] for r in rows) and node_error<=1e-6,dict(points=rows,node_closure_max=float(node_error))


def independent_photon_coefficients(E,args):
    # Integrate free-electron differential Klein–Nishina; never XCOM incoherent.
    from prototypes.mv_electron.physics import NE
    def dmu(mu):
        q=1/(1+E/M*(1-mu))
        return np.pi*RE**2*NE*q*q*(q+1/q-(1-mu*mu))
    compton=quad(dmu,-1,1,epsabs=1e-12,epsrel=1e-11)[0]
    es,water=args[6:8]
    photo=np.exp(np.interp(np.log(E),np.log(es),np.log(water[:,2])))
    return np.array([compton,photo,0.,0.])


def test_independent_table():
    passed,details=table_check()
    assert passed,details


@pytest.mark.parametrize('T',[.001,.01,.02,.1,1.,10.,20.])
def test_stopping_disabled_exact(T):
    from prototypes.mv_electron.physics import stopping
    e,v=b.tables()
    assert b.restricted_stopping(T,e,v,True)==stopping(T)


def test_above_range_rejected():
    e,v=b.tables()
    with pytest.raises(ValueError):b.quantities(20.01,e,v)
    with pytest.raises(ValueError):b.shape(20.01,.02)


@njit
def fixed_distances(n,seed,rate):
    np.random.seed(seed);a=np.empty(n)
    for i in range(n):a[i]=b.new_remaining(True)/rate
    return a


def test_fixed_distance_exponential():
    from scipy.stats import kstest
    x=fixed_distances(100000,20291607,3.)
    assert kstest(x,'expon',args=(0,1/3)).pvalue>.01


@pytest.mark.parametrize('event',['boundary','energy','scattering','hard','stack'])
def test_remaining_preserved(event):
    a=t.initial(2.,.1,0.,0.,1.,0,0,.05,.025)
    a[22]=1.7; a[22]=b.advance_remaining(a[22],.2,.3)
    expected=1.7-.2*.3
    if event=='energy' or event=='hard':t.energy_init(a,.05)
    elif event=='scattering':old.hinge_transition(a,.05,.2)
    elif event=='boundary':a[16]=1
    else:
        stack=np.zeros((2,23));stack[0]=a;a=stack[0].copy()
    assert a[22]==expected
    assert b.advance_remaining(a[22],1.,.3,False)==expected


@pytest.mark.parametrize('winner',range(6))
def test_tie_order(winner):
    ds=np.ones(6);ds[:winner]=2.
    assert b.event_order(*ds)==winner


def test_all_birth_sites_initialize():
    # Check actual kernel call sites in addition to testing the initializer.
    source=inspect.getsource(t._run_coupled.py_func)
    assert source.count('=initial(')==7  # two sources, Compton, PE, two pair, Moller
    assert source.count('not disable_brems)')>=8
    for primary in (0,1):
        a=t.initial(1.,.1,0.,0.,1.,0,primary,.05,.025)
        assert np.isfinite(a[22]) and a[22]>0
        disabled=t.initial(1.,.1,0.,0.,1.,0,primary,.05,.025,False)
        assert disabled[22]==np.inf


@pytest.mark.parametrize('parent',[(0.,0.,1.),(0.,0.,-1.),(.3,.4,np.sqrt(.75))])
def test_fixed_direction(parent):
    u=np.array(parent);unchanged=u.copy()
    for phi in np.linspace(0,2*np.pi,31):
        out=np.array(b.direction(1.,*u,phi))
        assert abs(np.linalg.norm(out)-1)<1e-12
        assert abs(np.arccos(np.clip(u@out,-1,1))-M/(1+M))<1e-12
    np.testing.assert_array_equal(u,unchanged)


@pytest.fixture(scope='module')
def mf_inputs():
    return old.inputs('mf')


@pytest.mark.parametrize('charge',[-1,1])
@pytest.mark.parametrize('mode',[1,2])
def test_small_transport_balance(mf_inputs,charge,mode):
    r=dict(zip(old.KEYS+('brems','first_record'),t.run_coupled(10.,100,20291608,.05,.025,30.,*mf_inputs,
        annihilate_in_flight=True,charged_source_T=10.,charged_source_charge=charge,
        brems_mode=mode,**old.rule_kwargs('own'))))
    assert old.base.healthy(r)
    assert r['totals'][3]==0
    assert r['brems'][0]>0 and r['brems'][1]>0
    assert np.all(r['brems'][2:4]==0)


def test_mode_zero_and_disabled_loop(mf_inputs):
    args=(10.,100,20291609,.05,.025,30.,*mf_inputs)
    kw=dict(charged_source_T=10.,annihilate_in_flight=True,**old.rule_kwargs('own'))
    a=old.run_coupled(*args,**kw)
    delegated=t.run_coupled(*args,brems_mode=0,**kw)
    disabled=t.run_coupled(*args,disable_brems=True,**kw)
    for x,y,z in zip(a,delegated,disabled):
        np.testing.assert_array_equal(x,y)
        np.testing.assert_allclose(x,z,rtol=1e-12,atol=0)


def test_infinite_water(mf_inputs):
    r=dict(zip(old.KEYS+('brems','first_record'),t.run_coupled(2.,100,20291610,.05,.025,30.,*mf_inputs,
        charged_source_T=2.,brems_mode=1,infinite_medium=True,**old.rule_kwargs('own'))))
    assert not np.any(r['coarse']) and not np.any(r['fine'])
    assert r['totals'][2]==r['totals'][3]==0
    assert not np.any(r['violations'])
    assert r['brems'][8]>=r['brems'][1]


@pytest.mark.parametrize('charge',[-1,1])
def test_large_k_cutoff_closes_ledger(mf_inputs,monkeypatch,charge):
    # Force a large-k event on the real copied loop; eliminate competing hazards.
    e,v=b.tables()
    monkeypatch.setattr(t,'quantities',lambda T,*args:(1000.,0.,0.,1.))
    monkeypatch.setattr(t,'restricted_stopping',lambda *args:1.)
    monkeypatch.setattr(t,'hard',lambda T:(0.,0.))
    monkeypatch.setattr(t,'table_rates',lambda *args:(0.,0.))
    monkeypatch.setattr(t,'sample_k',lambda T,*args:(T-.005,0))
    r=dict(zip(old.KEYS+('brems','first_record'),t._run_coupled.py_func(1.,1,20291611,.05,.025,30.,*mf_inputs,
        charged_source_T=1.,charged_source_charge=charge,brems_mode=1,
        brems_energies=e,brems_values=v)))
    assert old.base.healthy(r)
    assert r['brems'][0]>=1
    if charge==1:assert r['counts'][12]==1 and r['totals'][4]==0


def test_subcutoff_compton_at_scatter_point(mf_inputs,monkeypatch):
    e,v=b.tables();z=.1;energy=.0101;remainder=.0099
    # Record the real photon branch and ensure the child photon deposits in its bin.
    monkeypatch.setattr(t,'coefficients',lambda *args:np.array([100.,0.,0.,0.]))
    monkeypatch.setattr(t,'compton',lambda E,*args:(remainder,0.,0.,1.,E-remainder,0.,0.,1.))
    calls=[];original=t.tracked_local
    def record(score,zz,loss,dz,touched,j):
        calls.append((zz,loss,j));return original(score,zz,loss,dz,touched,j)
    monkeypatch.setattr(t,'tracked_local',record)
    source=t.photon_state(energy,z,0.,0.,1.,0)
    r=t._run_coupled.py_func(energy,1,20291612,.05,.025,30.,*mf_inputs,
        source_state=source,brems_energies=e,brems_values=v,record_first=True,score_kerma=False)
    locations=[zz for zz,loss,j in calls if loss==remainder and j==0]
    assert len(locations)==1
    assert locations[0]==z+r[-1][5]
    assert r[0][0,int(locations[0]/.2)]>=remainder
    assert r[5][1]==pytest.approx(energy,rel=1e-12)
