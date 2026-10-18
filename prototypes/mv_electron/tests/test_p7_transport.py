"""P7 C++ builds vs the P3b Numba kernel (small runs), scorer, sinks and the field-half fault."""
import numpy as np
import pytest
from prototypes.mv_electron.gpu import p7, p7_core as h


@pytest.fixture(scope='module')
def data():
    from prototypes.mv_electron.gpu.p5a import load_inputs
    return load_inputs(full=True)


@pytest.fixture(scope='module')
def build(tmp_path_factory):
    return tmp_path_factory.mktemp('p7_build')


@pytest.mark.parametrize('name',['A','B','C'])
def test_p7_mt_matches_numba(data,build,name):
    d=h.condition_inputs(data,name);c=h.condition(name);mt=p7.CPU(build,d,'mt')
    expected,_=p7.numba_reference(300,h.SEED,name);actual,_=mt.mt(300,h.SEED,c['field_half'])
    assert h.exact(h.reduced(actual),expected)['status']=='pass'


def test_p7_field_half_fault_detected(data,build):
    d=h.condition_inputs(data,'A');expected,_=p7.numba_reference(300,h.SEED,'A')
    bad=p7.CPU(build,d,'mt',p7_fault=8);actual,_=bad.mt(300,h.SEED,2.5)
    assert h.exact(h._reduced(actual),expected,tuple(expected))['status']=='fail'


def test_p7_scorer_and_sinks(data,build):
    mt=p7.CPU(build,h.condition_inputs(data,'B'),'mt')
    assert p7.check_scorer(mt,2000)['status']=='pass'
    assert p7.check_sinks(mt)['status']=='pass'


def test_p7_no_field_matches_p5j(data,build):
    from prototypes.mv_electron.gpu.p5k import CPU as P5CPU
    a=p7.CPU(build,data).histories(500,h.SEED,5.,field=False,regions=False)
    b=P5CPU(build/'p5j',data).histories(500,'P5j',start=0)
    assert h.exact(a,b,p7.core.HISTORY_ORDER)['status']=='pass'
