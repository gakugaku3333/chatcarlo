"""Native harness with artificial tables and sub-cutoff photons, no cache data.

This tests argument observation/refill/scoring, never therapy physics accuracy.
"""
import ctypes as ct
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from chatcarlo.gpu import therapy,catalog,therapy_checks,therapy_health,physics_sets,contracts
from chatcarlo.gpu.therapy_observation import descriptor


def artificial_material():
    m=SimpleNamespace(**{k:np.array(1.) for k in catalog.P8_DEFINITIONS})
    for k in ('stop_T','gs_T','gs_K','gs_u','brems_E','photo_E','pair_E'):setattr(m,k,np.array([.001,10.]))
    for k in ('stop_delta','stop_collision','stop_rad','photo','pair_n','pair_e'):setattr(m,k,np.ones(2))
    m.log_stop_T=np.log(m.stop_T);m.rates=np.array([[.001,1.,1.],[10.,1.,1.]])
    m.z=np.array([2.]);m.nj=np.ones(1);m.gs_y=np.ones((2,2,2));m.brems_V=np.ones((2,5));m.dl=np.ones((6,6))
    m.thresholds=np.array([1.,2.]);m.bindings=np.ones((1,5));m.pm=np.ones((1,5,4));m.shell_nj=np.ones(1)
    m.yields=np.zeros((1,4));m.lines=np.zeros((1,4,1,4));m.line_counts=np.ones((1,4))
    from chatcarlo.gpu.generate_photon_tables import material
    m.photon=material([2],[1.])[0]
    return m


def geometry():
    g=therapy.FieldGeometry();lo=[[-1.,-1.,0.],[-1.,-1.,0.],[-np.inf,-np.inf,0.]];hi=[[1.,1.,30.],[1.,1.,30.],[np.inf,np.inf,30.]]
    axes=[2]*3;origins=[0.]*3;widths=[.25,.1,.25];sizes=[120,300,120]
    for width in (.25,.125):
        for axis in (0,1):
            for depth in (5,10,20):
                l=[-1.,-1.,depth-.5];h=[1.,1.,depth+.5];l[axis]=-20.;h[axis]=20.
                lo.append(l);hi.append(h);axes.append(axis);origins.append(-20.);widths.append(width);sizes.append(int(40/width))
    for k,v in dict(lo=np.array(lo).ravel(),hi=np.array(hi).ravel(),axes=axes,origins=origins,widths=widths,sizes=sizes,offsets=np.r_[0,np.cumsum(sizes)[:-1]]).items():getattr(g,k)[:]=v
    return g


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    g=geometry();a=dict(faces=np.array([0.,30.]),rho=np.ones(1),bins=np.arange(301)*.1,strength=np.arange(17,dtype=float),
        edges=np.empty(0),cum=np.empty(0),ids=np.zeros(1,np.int64),owner=np.zeros(300,np.int64),constants=np.ones(6),sizes=np.array([1,300,150,600,0],np.int64))
    a['_table_hashes']={'0.'+k:catalog.digest(getattr(artificial_material(),k)) for k in catalog.P8_DEFINITIONS}
    cpu=therapy.CPU(tmp_path_factory.mktemp('native'),[artificial_material()],a,field_geometry=g,half=(5.,5.));cpu.g.energy=.004
    return cpu,a,g


def test_native_split_and_scorer(native):
    cpu,a,g=native
    assert therapy_checks.run(cpu,a,g,123)['status']=='pass'


def test_native_argument_observation(native):
    cpu,a,g=native
    cpu.histories(8,123,start=7)
    expected=descriptor(a,g,(5.,5.),100.,123,7,8,physics_sets.declaration(therapy.NAME))
    contracts.check_observation(cpu.last_observation,expected,'normal')
    original=cpu.half;cpu.half=(6.,5.)
    try:
        cpu.histories(8,123,start=7)
        with pytest.raises(ValueError,match='half'):contracts.check_observation(cpu.last_observation,expected,'field fault')
    finally:cpu.half=original


@pytest.mark.parametrize('key',['id','tag'])
def test_native_only_id_tag(native,monkeypatch,key):
    cpu,a,g=native
    name='t2_set_'+key;fn=getattr(cpu.lib,name);value=fn()
    monkeypatch.setattr(cpu.lib,name,lambda:value+1)
    cpu.histories(1,7)
    with pytest.raises(ValueError,match='mismatch'):physics_sets.check(cpu.last_observation['physics_set'],cpu.physics_set)


def test_inspection_build_is_conditional():
    source=therapy.cuda_source()
    assert '#if T2_SOURCE_CHECK' in source and 'sources[h*6+j]=q->source_state[j]' in source
    assert 'sources[h*6+j]' not in source.split('P5_FN void p8g2_gpu_begin')[0]


def test_native_table_mutation(native):
    cpu,a,g=native
    offset=int(cpu.meta[0,0]);before=cpu.data[offset]
    try:
        cpu.data[offset]+=1
        cpu.histories(1,9)
        want=descriptor(a,g,(5.,5.),100.,9,0,1,cpu.physics_set)
        with pytest.raises(ValueError,match='tables'):contracts.check_observation(cpu.last_observation,want,'native table')
    finally:cpu.data[offset]=before


def test_actual_native_argument_tuple_tamper(native,monkeypatch):
    cpu,a,g=native;original=cpu._invoke
    def changed(args):
        values=list(args);values[12]=6.;return original(tuple(values))
    monkeypatch.setattr(cpu,'_invoke',changed)
    cpu.histories(4,17)
    want=descriptor(a,g,(5.,5.),100.,17,0,4,cpu.physics_set)
    with pytest.raises(ValueError,match='half'):contracts.check_observation(cpu.last_observation,want,'actual native arguments')
