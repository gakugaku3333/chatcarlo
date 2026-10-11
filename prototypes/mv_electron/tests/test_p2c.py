"""Small P2c tests; the full preregistered samples are --checks only."""
import numpy as np
import pytest
from scipy.integrate import quad
from prototypes.mv_electron.physics import M, NE
from prototypes.mv_electron.flight_annihilation_p2c import (
    sigma, differential, zeta_min, annihilation_rate, samples)
from prototypes.mv_electron.run_p2c import execute, balance, healthy, initial
from prototypes.mv_electron.run_coupled_p2 import inputs, KEYS
from prototypes.mv_electron.photon_transport_p2 import run_coupled as original


@pytest.mark.parametrize('T,reference',[(.01,1.27912842995e-24),(1.,1.71929574266e-25),(10.,3.82961894215e-26)])
def test_cross_section_normalization_and_independent_values(T,reference):
    integral=quad(lambda z:differential(T,z)/sigma(T),zeta_min(T),.5,epsabs=0,epsrel=1e-12)[0]
    assert abs(integral-1)<=1e-9
    assert abs(sigma(T)/reference-1)<=1e-9
    assert abs(annihilation_rate(1.)/(NE*1.71929574266e-25)-1)<=1e-9


@pytest.mark.parametrize('T',[.05,.5,2.,5.,9.])
def test_three_dimensional_conservation(T):
    z,phi,maxima=samples(T,2000,177)
    assert z.min()>=zeta_min(T) and z.max()<=.5
    assert phi.min()>=0 and phi.max()<2*np.pi
    assert max(maxima[:2])<=(T+2*M)*1e-9
    assert maxima[2]<=1e-12


@pytest.mark.parametrize('E',[6.,10.])
def test_flight_off_exact_regression(E):
    before=dict(zip(KEYS,original(E,100,117,.05,.025,30.,*inputs('mf'))))
    after=execute(E,100,117,annihilate_in_flight=False)
    for key in KEYS:
        assert np.array_equal(before[key],after[key][:14] if key=='counts' else after[key])
    assert not np.any(after['counts'][14:])


def test_positron_balance_without_double_annihilation():
    r=execute(2.,1000,117,annihilate_in_flight=True,positron_source_T=2.,annihilation_scale=1000.)
    assert healthy(r)
    assert balance(r,1000)['equal']
    assert r['counts'][15]>=990


def test_electron_has_no_flight_annihilation():
    source=initial(1.,15.,0.,0.,1.,0,1,.05,.025)
    r=execute(1.,100,117,annihilate_in_flight=True,source_state=source)
    assert healthy(r) and r['counts'][14]==0


def test_stack_overflow_keeps_unresolved_energy_and_rest():
    r=execute(2.,100,117,annihilate_in_flight=True,positron_source_T=2.,annihilation_scale=1000.,maxstack=1)
    assert r['counts'][6]>0 and r['totals'][3]>0 and r['totals'][4]>0
    assert not np.any(r['violations']) and max(r['maxima'])<=1e-9
    assert not r['counts'][12] and not r['counts'][14]
