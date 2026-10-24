"""P8g gate 0: real CPU functions and fault builds, read-only frozen inputs."""
import ctypes as ct
from pathlib import Path
import numpy as np
import pytest
from prototypes.mv_electron.gpu import p8g as p,p8f,p8g_core as core

@pytest.fixture(scope='module')
def data():return p8f.inputs()

@pytest.fixture(scope='module')
def builds(data,tmp_path_factory):
    m,i,e,c=data;work=tmp_path_factory.mktemp('p8g')
    kw=dict(densities=i['densities'],edges=e,cum=c,rng='philox',cap=64)
    normal=p.CPU(work/'normal',m,**kw)
    checked=p.CPU(work/'check',m,check=True,**kw)
    old=p8f.CPU(work/'p8f',m,**kw)
    return normal,checked,old,work,kw

def check_values(cpu):
    for mid in range(len(cpu.materials)):
        actual,reference=cpu.values(mid)
        assert np.array_equal(np.isnan(actual),np.isnan(reference[:,0]))
        good=np.isfinite(reference[:,0]);core.bitwise(actual[good],reference[good,0],f'0b material {mid}')

from prototypes.mv_electron.gpu.p8g_core import nodes,component_cases

def check_components(cpu,old):
    for which,mid,x in component_cases(cpu.materials):
        for j,(a,b) in enumerate(zip(cpu.probe(which,x,mid),old.probe(which,x,mid))):
            core.bitwise(a,b,f'0e function {which} material {mid} record {j}')

def test_p8g_0b_initialization(builds):
    check_values(builds[0]);check_values(builds[1])

def test_p8g_0e_components(builds):
    check_components(builds[0],builds[2]);check_components(builds[1],builds[2])

def test_p8g_0e_synthetic_rounded_log_and_duplicate(builds):
    cpu=builds[1];fn=cpu.lib.p8g_interp_probe;fn.argtypes=[p.D,p.D,ct.c_int,p.D,ct.c_int,p.D];fn.restype=None
    a=10.;b=np.nextafter(a,np.inf)
    assert np.log(a)==np.log(b)
    x=np.array([1.,a,b,b,20.]);y=np.array([2.,3.,4.,5.,6.]);e=nodes(x);out=np.zeros((len(e),4))
    fn(x,y,len(x),e,len(e),out);core.bitwise(out[:,0].copy(),out[:,1].copy(),'0e synthetic right')
    core.bitwise(out[:,2].copy(),out[:,3].copy(),'0e synthetic left')


def test_p8g_0c_histories(builds):
    normal,checked,old,_,_=builds
    expected=old.histories(2000,core.SEED2,7000)
    for cpu in (normal,checked):core.exact_histories(cpu.histories(2000,core.SEED2,7000),expected,'0c')
    counts=checked.counters();assert not counts[:,1:36:2].any()
    # Capture only the registered histories' counters, separate from component probes.
    checked.lib.p8g_init(checked.data,checked.meta,checked.constants,4)
    core.exact_histories(checked.histories(2000,core.SEED2,7000),expected,'0c reset')
    counts=checked.counters();assert not counts[:,1:36:2].any()
    p.write_json(p.OUT/'gate0_reads.json',dict(n=2000,start=7000,seed=core.SEED2,sites=core.SITES,counts=counts.tolist()))


def test_p8g_0d_faults(builds):
    normal,checked,old,work,kw=builds
    check_values(normal);check_components(normal,old)
    rows=[]
    for fault in (1,2,3):
        cpu=p.CPU(work/f'f{fault}',normal.materials,check=True,fault=fault,**kw)
        row=dict(fault=fault,build='success',library=cpu.lib._name,normal_pass=['0b','0e','0c'],identity=cpu.identity,gate1='deferred to Claude',gate2='deferred to Claude')
        if fault==1:
            with pytest.raises(AssertionError,match='0b'):check_values(cpu)
            with pytest.raises(AssertionError,match='0e'):check_components(cpu,old)
            row['rejected_by']=['0b','0e']
        elif fault==2:
            with pytest.raises(AssertionError,match='0e'):check_components(cpu,old)
            row['rejected_by']=['0e']
        else:
            # f3 checked read returns NaN; a shape rejection sampler would otherwise
            # spin forever. Its checked-read guard aborts the history immediately.
            cpu.histories(2000,core.SEED2,7000)
            assert cpu.counters()[:,1:36:2].sum()>0
            row['rejected_by']=['0c invalid reads']
        row['target_reads']=int(cpu.counters()[:,36].sum());assert row['target_reads']>0
        row['invalid_reads']=int(cpu.counters()[:,1:36:2].sum());rows.append(row)
    p.write_json(p.OUT/'faults_gate0.json',dict(status='pass',faults=rows))


def test_p8g_identity_and_source():
    fp=core.fingerprint();assert 'p8g_logs.h' in fp and not any('colab' in k for k in fp)
    src=core.cuda_source();assert '#include "' not in src
    assert 'p8g_init(' in src and 'atomicAdd(next,1)' in src

def test_p8g_0e_raw_quantities(builds):
    cpu=builds[1];fn=cpu.lib.p8g_parts;fn.argtypes=[p.D,p.I,p.D,ct.c_int,p.D,ct.c_int,p.D];fn.restype=None
    for mid,m in enumerate(cpu.materials):
        x=nodes(m.brems_E);out=np.zeros((len(x),40))
        fn(cpu.data,cpu.meta,cpu.constants,mid,x,len(x),out)
        core.bitwise(out[:,:20].copy(),out[:,20:].copy(),'raw quantities')
