"""Gate 0 exercises real transport; frozen inputs are read only."""
import numpy as np
import pytest
from prototypes.mv_electron.gpu import p8g2 as p,p8g2_core as core

@pytest.fixture(scope='module')
def builds(tmp_path_factory):
    m,i,e,c=p.inputs();work=tmp_path_factory.mktemp('p8g2')
    kw=dict(densities=i['densities'],edges=e,cum=c,cap=18)
    return p.CPU(work/'normal',m,**kw),m,work,kw

@pytest.mark.parametrize('n,start',[(20000,0),(2000,7000)])
def test_p8g2_histories_and_slots(builds,n,start):
    cpu,_,_,_=builds
    from prototypes.mv_electron.gpu import p8g
    _,m,work,kw=builds
    old=p8g.CPU(work/'old-histories',m,**kw,rng='philox')
    expected=old.histories(n,core.SEED2,start)
    rows=[]
    for slots in (1,32,64):
        h,v,hits,ag=cpu.histories(n,core.SEED2,start,slots=slots,details=True)
        core.check_visits(v);core.exact_histories(h,expected,f'0b/c slots={slots} start={start}');core.check_group_records(ag,h,n)
        rows.append(dict(slots=slots,status='pass',outputs=int(v.sum()),fields=list(h),build=cpu.build))
    p.write_json(p.OUT/f'histories_{n}_{start}.json',dict(status='pass',seed=core.SEED2,n=n,start=start,reference='frozen P8g-1 CPU',comparisons=rows))


def test_p8g2_direct_aggregate(builds):
    from prototypes.mv_electron.gpu import p8g
    cpu,m,work,kw=builds;old=p8g.CPU(work/'old',m,**kw,rng='philox')
    # This exercises the exact direct accumulation order used by gate 1.
    actual=cpu.run(200,core.SEED2);expected=old.run(200,core.SEED2)
    for key in expected:
        if key!='seconds':core.bitwise(actual[key],expected[key],'direct '+key)


def test_p8g2_source_contract():
    fp=core.fingerprint()
    for name in ('p8g2_kernel.h','p8g2_cpu.cpp','p8g2_gpu.cu','p8g2.py','p8g2_core.py'):assert name in fp
    assert not any('colab' in n or '_cli' in n for n in fp)
    src=core.cuda_source();assert '#include "' not in src
    assert 'unsigned masks[4]' in src and 'atomicAdd(next,1)' in src
    assert 'atomicAdd(hits+j,1ULL)' in src
    assert (p.OUT/'state_variables.md').exists()


_fixed_cache={}

def fixed_cpu(builds,case,fault=0):
    _,m,work,kw=builds;kw=dict(kw,cap=case['cap'],edges=None,cum=None)
    key=(str(work),case['cap'],fault)
    if key not in _fixed_cache:_fixed_cache[key]=p.CPU(work/(case['name']+str(fault)),m,fault=fault,**kw)
    return _fixed_cache[key]


def test_p8g2_required_paths(builds):
    rows=[];total=np.zeros(80,np.uint64)
    for case in core.fixed_cases():
        cpu=fixed_cpu(builds,case);kw=dict(start=case['start'],maxsteps=case['maxsteps'],injected=np.array([case['state']]))
        h,v,hits,_=cpu.histories(1,core.SEED2,details=True,**kw)
        expected=cpu.histories(1,core.SEED2,reference=True,**kw)
        core.check_visits(v);core.exact_histories(h,expected,case['name'])
        for idx in case['targets']:assert hits[idx]>0,(case['name'],core.PATHS[idx])
        total+=hits;rows.append(dict(case,counts=hits.tolist(),status='pass'))
    for name,idx in core.TRANSITIONS.items():assert total[idx]>0,name
    p.write_json(p.OUT/'paths_gate0.json',dict(status='pass',seed=core.SEED2,cases=rows,transitions={k:int(total[v]) for k,v in core.TRANSITIONS.items()}))


def test_p8g2_faults(builds):
    normal,m,work,kw=builds;rows=[]
    # g1 is the fixed in-flight annihilation witness, not uninitialized storage.
    case=next(c for c in core.fixed_cases() if c['name']=='flight')
    good=fixed_cpu(builds,case);bad=fixed_cpu(builds,case,1)
    args=dict(start=case['start'],injected=np.array([case['state']]))
    expected=good.histories(1,core.SEED2,reference=True,**args)
    h,v,hits,_=good.histories(1,core.SEED2,details=True,**args);core.exact_histories(h,expected,'g1 normal');assert hits[6]>0
    h,v,hits,_=bad.histories(1,core.SEED2,details=True,**args);assert hits[30]>0
    with pytest.raises(AssertionError) as error:core.exact_histories(h,expected,'g1 0b')
    rows.append(dict(fault='g1',build='success',target=int(hits[30]),normal='pass',rejected_by=str(error.value),build_identity=bad.identity,build_configuration=bad.build))
    # Two fixed low photons reuse one actual slot. The second inherits bins in g2.
    low=next(c for c in core.fixed_cases() if c['name']=='low-photon')
    states=np.tile(low['state'],(2,1));good=fixed_cpu(builds,low)
    expected=good.histories(2,core.SEED2,injected=states,reference=True)
    gh,gv,_,ga=good.histories(2,core.SEED2,injected=states,details=True)
    core.exact_histories(gh,expected,'g2 normal');core.check_visits(gv);core.check_group_records(ga,gh,2)
    for label,fault,target in (('g2',2,31),('g3a',3,32),('g3b',6,33),('g4',4,34),('g5',5,35)):
        bad=fixed_cpu(builds,low,fault)
        h,v,hits,ag=bad.histories(2,core.SEED2,injected=states,details=True);assert hits[target]>0
        with pytest.raises(AssertionError) as error:
            if fault==2:core.exact_histories(h,expected,'g2 0c')
            elif fault in (3,6):core.check_visits(v)
            else:core.check_group_records(ag,h,2)
        rows.append(dict(fault=label,build='success',target=int(hits[target]),normal='pass',rejected_by=str(error.value),build_identity=bad.identity,build_configuration=bad.build))
    p.write_json(p.OUT/'faults_gate0.json',dict(status='pass',faults=rows))


@pytest.mark.parametrize('cap,maxsteps',[(18,1),(3,10000000)])
def test_p8g2_small_limits(builds,cap,maxsteps):
    _,m,work,kw=builds
    cpu=p.CPU(work/f'limits-{cap}-{maxsteps}',m,**dict(kw,cap=cap))
    expected=cpu.histories(2000,core.SEED2,7000,reference=True,maxsteps=maxsteps)
    h,v,hits,_=cpu.histories(2000,core.SEED2,7000,slots=32,maxsteps=maxsteps,details=True)
    core.check_visits(v);core.exact_histories(h,expected,'small limits')
    assert h['counts'][:,5 if maxsteps==1 else 6].sum()>0


def test_p8g2_normal_vs_check(builds):
    cpu,m,work,kw=builds;normal=p.CPU(work/'unchecked',m,check=False,**kw)
    core.exact_histories(normal.histories(2000,core.SEED2,7000),cpu.histories(2000,core.SEED2,7000),'check vs normal')
