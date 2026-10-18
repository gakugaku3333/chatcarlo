"""Small CPU transport smoke checks; full numerical gates remain CLI commands."""
import numpy as np
import pytest
from prototypes.mv_electron.gpu import p6, p6_core as h, p5b_core as core

@pytest.fixture(scope='module')
def inputs():
    from prototypes.mv_electron.gpu.p5a import load_inputs
    return load_inputs(full=True)


def test_p6_mt_small_field_batch_and_deposits(inputs,tmp_path):
    n=64;seed=h.SEED
    expected,values=p6.numba_reference(n,seed,True)
    cpu=p6.CPU(tmp_path,inputs,'mt');actual,records=cpu.mt(n,seed,record_capacity=2000000,source_capacity=n)
    assert h.exact(h.reduced(actual),expected)['status']=='pass'
    assert core.bits(records['deposits'],values[21])
    assert core.bits(records['sources'],values[23])
    assert records['meta'][1:].tolist()==[0,n,0]
    expected_counts=np.bincount(records['sites'],minlength=6)[1:]
    assert np.array_equal(records['site_counts'],expected_counts)
    # Gate1 counts every site without allocating the (large) deposit stream.
    for capacity in (0,1):
        unrecorded,rec=cpu.mt(n,seed,record_capacity=capacity)
        assert h.exact(actual,unrecorded)['status']=='pass'
        assert np.array_equal(rec['site_counts'],expected_counts)
        assert rec['meta'][1]==(len(records['deposits'])-1 if capacity else 0)
        assert p6.site_coverage(rec)=={str(s):int(expected_counts[s-1]) for s in range(1,6)}


def test_p6_fixed_actual_energy_endpoint(inputs,tmp_path):
    cpu=p6.CPU(tmp_path,inputs,'mt')
    check=p6.check_endpoint(cpu,tmp_path)
    assert check['status']=='pass'
    assert check['coverage']=={'1':0,'2':0,'3':1,'4':0,'5':1}
    assert check['cpp_meta']==[2,0,1,0]
    with np.load(tmp_path/'endpoint_addendum2.npz') as rec:
        assert rec['cpp_sites'].tolist()==[3,5]
        assert rec['numba_deposits'].shape==(2,9)
        assert core.bits(rec['numba_deposits'],rec['cpp_deposits'])
        # One continuous loss followed by the actual-energy endpoint, at a
        # non-origin position inside a profile layer. No synthetic deposits.
        assert rec['cpp_deposits'][:,1].tolist()==[0.,1.]
        assert rec['cpp_deposits'][0,8]>0.
        assert core.bits(rec['cpp_deposits'][0,5:8],rec['cpp_deposits'][1,2:5])
        source=rec['source_state']
    with pytest.raises(ValueError,match='without field'):
        cpu.mt(1,h.SEED,source_state=source)


def test_p6_no_field_histories_and_trace(inputs,tmp_path):
    from prototypes.mv_electron.gpu.p5k import CPU as P5CPU
    cpu=p6.CPU(tmp_path/'p6',inputs);old=P5CPU(tmp_path/'p5j',inputs)
    actual,tr,meta=cpu.histories(32,h.SEED,field=False,regions=False,capacity=8192)
    expected,rt,rm=old.histories(32,'P5j',start=0,capacity=8192)
    assert h.exact(actual,expected,core.HISTORY_ORDER)['status']=='pass'
    assert h.k.traces(tr,meta,rt,rm)['status']=='pass'
    assert not np.any(actual['regions']) and not np.any(actual['region_ledger'])


def test_p6_local_reset_failure_and_resume_range(inputs,tmp_path):
    cpu=p6.CPU(tmp_path,inputs)
    for steps in (1,2):
        a,tr,meta=cpu.histories(33,h.SEED,start=7000,maxsteps=steps,capacity=32)
        b,rt,rm=cpu.histories(33,h.SEED,start=7000,maxsteps=steps,fresh=True,capacity=32)
        assert h.exact(a,b)['status']=='pass'
        assert h.k.traces(tr,meta,rt,rm)['status']=='pass'
        assert np.any(a['counts'][:,5])
    a=cpu.histories(100,h.SEED,start=7000)
    b=cpu.histories(20,h.SEED,start=7080)
    assert h.exact({k:v[80:] for k,v in a.items()},b)['status']=='pass'
    plain=cpu.run(100,h.SEED,start=7000)
    run=cpu.run(100,h.SEED,start=7000,count_additions=True)
    assert h.exact(plain,run)['status']=='pass'
    assert cpu.region_additions.dtype==np.uint64 and cpu.region_additions.any()
    assert h.c3(a,run,region_additions=cpu.region_additions)['status']=='pass'
