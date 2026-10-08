import json
import numpy as np
import pytest
from scipy.integrate import quad
from prototypes.mv_electron.physics import *
from prototypes.mv_electron.gs import coefficients,distribution
from prototypes.mv_electron.component_checks import OUT

def test_k1_all_fetched_points():
    assert max(abs(np.array([collision(t) for t in ET])/estar['collision']-1))<.01
@pytest.mark.parametrize('t',[.05,.1,1,2,10,np.nextafter(.02,0),.02,np.nextafter(.02,np.inf)])
def test_k2_independent_integral(t):
    num=quad(lambda w:w*dcs_moller(t,w),DELTA,t/2,epsabs=1e-16)[0] if t>2*DELTA else 0.
    assert hard(t)[1]==pytest.approx(num,rel=1e-6,abs=1e-30)
def test_k2_saved_large_sampling():
    a=json.loads((OUT/'components.json').read_text())
    assert a['K2']['status']=='pass'
    for row in a['K2']['rows']:
        if 'N' in row:
            assert row['N']==1000000 and row['min_expected']>=100 and row['chi_p']>.001
def test_k3_series_normalization_and_monotonicity():
    y,cdf,p0,G,total,d=distribution(.02,.001,L=2048)
    assert abs(d['norm']-1)<=1e-6 and np.all(np.diff(cdf)>=-1e-12)
    assert p0==pytest.approx(np.exp(-.001/G[1]*total),abs=1e-15)
def test_k3_full_sampling_grid():
    a=json.loads((OUT/'gs_checks.json').read_text())['K3']
    assert a['status']=='pass'
    assert len(a['rows'])==20 and a['cdf_monotone'] and a['normalization_max_error']<=1e-6
def test_k9i_sequential_collision_reference():
    a=json.loads((OUT/'fixed_checks.json').read_text())['K9i']
    assert a['status']=='pass'
