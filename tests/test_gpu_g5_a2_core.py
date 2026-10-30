"""A2 local contracts: standalone native twin and real sealed transport inputs."""
import ctypes as ct
import json
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'docs/validation/gpu/g5'))
import a2_common as common
import a2_cases,a2_health,a2_faults,a2_scorer
from chatcarlo.gpu import therapy,physics_sets


@pytest.fixture(scope='module')
def inputs():
    return common.inputs()[1:]


def test_declaration():
    d=physics_sets.declaration(therapy.NAME)
    assert d['id']==2 and d['tag']!=0
    assert physics_sets.REGISTRY[therapy.NAME]['descriptor']['energy_band_keV']==[10.,10000.]
    with pytest.raises(ValueError,match='unsupported'):physics_sets.require_quantity(therapy.NAME,'H*(10)')


def test_cuda_source():
    src=therapy.cuda_source()
    assert '#include "' not in src
    assert 'atomicAdd(next,1)' in src and 'unsigned masks[4]' in src
    assert 'atomicAdd(target,amount)' in src
    assert 'atomicAdd(additions+i,1ULL)' in src
    assert 'field_scores[3420]' not in src and 'field_additions[3420]' not in src
    assert 'g5_base' not in src or 'philox' in src


def test_cases(tmp_path):
    r=a2_cases.run(tmp_path)
    assert len(r['cases'])==16 and len(r['paths'])==18 and len(r['transitions'])==9


def test_faults(tmp_path):
    assert len(a2_faults.run(tmp_path)['faults'])==10


def test_identity_and_refill(tmp_path,inputs):
    m,a=inputs;a=common.no_spectrum(a)
    g=common.field_geometry();cpu=therapy.CPU(tmp_path,m,a,field_geometry=g)
    assert cpu.observed==physics_sets.declaration(therapy.NAME)
    assert '-DT2_FIELD=1' in cpu.build['flags']
    with pytest.raises(ValueError,match='mismatch'):
        therapy.CPU(tmp_path,m,a,physics_declaration=physics_sets.declaration())
    a2_scorer.check(cpu)
    # Actual hard/brems suspend and resume: xyz assertions inside the native twin.
    from prototypes.mv_electron.gpu import p8g2_core as core
    total=np.zeros(80,np.uint64)
    for name in ('hard','brems'):
        case=next(v for v in core.fixed_cases() if v['name']==name)
        r=cpu.histories(1,core.SEED2,start=case['start'],injected=np.array([case['state']]))
        total+=r['hits']
    assert total[70:73].min()>0
    assert not total[60:62].any()
    low=next(v for v in core.fixed_cases() if v['name']=='low-photon')
    cpu.g.energy=.007
    pair=cpu.histories(2,core.SEED2,injected=np.tile(low['state'],(2,1)))
    single=cpu.histories(1,core.SEED2,start=1,injected=np.array([low['state']]))
    assert np.allclose(pair['field'],2*single['field'],rtol=1e-14,atol=0)
    a2_health.check(pair,cpu.g,a);a2_health.ledger(pair,g)


def test_depth_only_scoring_specialization(tmp_path,inputs):
    m,a=inputs
    a=common.no_spectrum(a)
    depth=therapy.CPU(tmp_path,m,a)
    field=therapy.CPU(tmp_path,m,a,field_geometry=common.field_geometry())
    assert '-DT2_FIELD=0' in depth.build['flags']
    # Both builds transport identical histories, including slot reuse. Only the
    # optional lateral scoring differs; its disabled outputs retain zero values.
    r=depth.histories(32,20301831,slots=3)
    scored=field.histories(32,20301831,slots=3)
    for key in r:
        if key in ('field','ledger','field_additions'):
            assert not r[key].any()
        else:
            np.testing.assert_array_equal(r[key],scored[key])
    assert scored['field'].sum()>0
    a2_health.check(r,depth.g,a)
    a2_health.ledger(scored,common.field_geometry())
