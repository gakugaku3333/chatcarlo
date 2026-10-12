"""P3t must reproduce P3r bit for bit (and P3s when the discrete events are off)."""
import numpy as np
import pytest
from prototypes.mv_electron import photon_transport_p3r as r
from prototypes.mv_electron import photon_transport_p3t as t
from prototypes.mv_electron import run_p3s as old


@pytest.fixture(scope='module')
def mf_inputs():
    return old.inputs('mf')


def same(a,b):
    # Byte comparison: array_equal alone would accept +0.0 against -0.0.
    assert len(a)==len(b)
    for x,y in zip(a,b):
        x=np.asarray(x);y=np.asarray(y)
        assert x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()


CASES=[
    dict(brems_mode=1),
    dict(brems_mode=2),
    dict(brems_mode=2,disable_brems=True),
    dict(brems_mode=2,charged_source_T=10.,charged_source_charge=1),
    dict(brems_mode=1,charged_source_T=10.,charged_source_charge=-1),
    dict(brems_mode=1,charged_source_T=2.,infinite_medium=True),
    dict(brems_mode=2,score_kerma=False,score_pair=False),
    dict(brems_mode=2,score_lateral=True),
    dict(brems_mode=2,record_first=True),
]


@pytest.mark.parametrize('case',CASES)
def test_identical_to_p3r(mf_inputs,case):
    args=(10.,200,20291630,.05,.025,30.,*mf_inputs)
    kw=dict(annihilate_in_flight=True,**old.rule_kwargs('own'),**case)
    same(r.run_coupled(*args,**kw),t.run_coupled(*args,**kw))


@pytest.mark.parametrize('rule',['egs','const'])
def test_identical_other_step_rules(mf_inputs,rule):
    if rule=='egs' and not old.KFILE.is_file():pytest.skip('K1.dat absent')
    args=(10.,200,20291633,.05,.025,30.,*mf_inputs)
    kw=dict(annihilate_in_flight=True,brems_mode=2,**old.rule_kwargs(rule))
    same(r.run_coupled(*args,**kw),t.run_coupled(*args,**kw))


def test_disabled_identical_to_p3s(mf_inputs):
    args=(10.,200,20291631,.05,.025,30.,*mf_inputs)
    kw=dict(annihilate_in_flight=True,**old.rule_kwargs('own'))
    a=old.run_coupled(*args,**kw)
    b=t.run_coupled(*args,disable_brems=True,**kw)
    same(a,b[:len(a)])
    assert not np.any(b[13][:4])  # no discrete events; the soft radiative loss is still tallied


def test_mode_zero_delegates(mf_inputs):
    args=(10.,50,20291632,.05,.025,30.,*mf_inputs)
    kw=dict(annihilate_in_flight=True,**old.rule_kwargs('own'))
    same(old.run_coupled(*args,**kw),t.run_coupled(*args,brems_mode=0,**kw))
