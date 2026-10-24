"""P9b: geometry transitions, capacity, parent scoring and frozen z regression."""
import json
from pathlib import Path
import numpy as np
import pytest
from prototypes.mv_electron.gpu import p9b,p9b_core as core,p9b_cases as cases
from prototypes.mv_electron.gpu.p9b_io import known_scores
from prototypes.mv_electron.gpu.p9b_colab import g5_check

@pytest.fixture(scope='module')
def inputs():
    return p9b.p9a_data.load_bundles()

@pytest.fixture(scope='module')
def build(tmp_path_factory):return tmp_path_factory.mktemp('p9b')

def test_g0_bitwise(build,inputs):
    materials,identity=inputs
    from prototypes.mv_electron.run_p3a import load_spectrum
    e,_,c=load_spectrum();kw=dict(layers=p9b.reference.reference.LAYERS,densities=identity['densities'],edges=e,cum=c)
    old=p9b.reference.CPU(build/'old',materials,**kw);new=p9b.CPU(build/'new',materials,**kw)
    core.core.exact_histories(new.histories(2000,core.core.SEED2),old.histories(2000,core.core.SEED2,reference=False),'G0')


def test_g3_cpu(build,inputs):
    materials,identity=inputs;r=cases.g3(build/'g3',materials,identity['densities'])
    assert r['status']=='pass' and r['cases']==161 and r['photon_recomputes']==2


def test_known_seed_scores(build,inputs):
    materials,identity=inputs
    cpu=p9b.CPU(build/'score',materials,layers=((1,30.,.2),),densities=identity['densities'],seeds=((0,0,10.,.05,.25,4),),field=True)
    assert known_scores(cpu)['point_and_segment']=='exact'
    h=cpu.histories(100,42,field_history=True)
    assert core.core.history_health(h)
    assert np.allclose(cpu.last['ledger'].sum(1),h['dep'],rtol=0,atol=1e-12)
    assert cpu.last['field'].shape==(100,3420)


def test_capacity_and_inputs(inputs):
    _,identity=inputs;rho=identity['densities'];assert cases.rejections(rho)==7
    g,_,_=core.geometry(((1,30.,.2),),rho)
    assert (g.nb,g.nc,g.nf)==(150,150,600)
    for kw in (dict(axis=3),dict(half=(0,1,1)),dict(field=True,axis=0),dict(edges=[0,1],cum=[.9])):
        with pytest.raises(ValueError):core.geometry(((1,10.,.2),),rho,**kw)


def test_g5_rejects_faults():
    # One nonzero score in each batch; exact synthetic production then corruption.
    n=100;values=np.zeros((n,3420));values[:,0]=1.;values[:,420]=6.
    h=dict(field=values,sites=np.ones((n,6),np.uint64),ledger=np.ones((n,6)),dep=np.full(n,6.),sources=np.full(n,6.),totals=np.zeros((n,6)))
    ag=dict(field=values.copy(),additions_field=np.ones((100,3420),np.uint64),sites=h['sites'].sum(0),ledger=h['ledger'].sum(0))
    assert g5_check(ag,h)['status']=='pass'
    ag['field'][0,0]+=.01
    with pytest.raises(AssertionError,match='bound'):g5_check(ag,h)
    ag['field']=values.copy();ag['sites'][0]+=1
    with pytest.raises(AssertionError,match='site counts'):g5_check(ag,h)


@pytest.mark.parametrize('failure',('G3','G4','G5'))
def test_failed_gate_stops_production(tmp_path,inputs,monkeypatch,failure):
    import sys
    from prototypes.mv_electron.gpu import p9b_colab as colab,p9b_io as io
    _,identity=inputs;record=dict(densities=identity['densities'],edges=[0.,1.],cum=[1.])
    monkeypatch.setitem(sys.modules,'cupy',object())
    monkeypatch.setattr(io,'load',lambda _: (record,[]))
    monkeypatch.setattr(colab,'g3_gpu',lambda *a: dict(status='fail' if failure=='G3' else 'pass'))
    calls=[];references={}
    for name,kw in io.transport_cases(record).items():
        g=core.geometry(**kw)[0]
        h=core.core.unpack(np.zeros((1,g.nb+g.nc+g.nf+37)),np.zeros((1,79+g.nb),np.int64),g.nb,g.nc,g.nf)
        references.update({name+'_'+k:v for k,v in h.items()});references[name+'_field']=np.zeros((1,3420))
    np.savez(tmp_path/'cpu_reference.npz',**references)
    class FakeGPU:
        def __init__(self,cp,m,**kw):self.g=core.geometry(**kw)[0]
        def run(self,n,*a,**kw):calls.append(n);return {},{'field':np.zeros((1,3420))}
    monkeypatch.setattr(colab,'GPU',FakeGPU)
    monkeypatch.setattr(colab.core.core,'compare_histories',lambda *a: dict(status='fail' if failure=='G4' else 'pass',report_only={}))
    def reject(*a):raise AssertionError('G5 injected failure')
    monkeypatch.setattr(colab,'g5_check',reject)
    with pytest.raises(AssertionError,match=failure):colab.run(tmp_path,tmp_path/'results')
    assert all(n==2000 for n in calls)
    assert not (tmp_path/'results/G1_x').exists()
    assert not (tmp_path/'results/G2').exists()
