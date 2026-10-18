"""P6 deterministic scorer and sink acceptance, using temporary build paths."""
import ctypes as ct
import numpy as np
import pytest
from prototypes.mv_electron.gpu import p6, p6_core as h, p5b_core as core

@pytest.fixture(scope='module')
def scorer(tmp_path_factory):
    return p6.CPU(tmp_path_factory.mktemp('p6_scorer'))


def test_p6_geometry_matches_canonical_boxes():
    from prototypes.mv_electron.photon_transport_p3b import expand_regions
    expected=expand_regions(h.PARAMS['regions'],30.,.25,.1);g,arrays=h.geometry()
    for name in ('lo','hi','origins','widths','axes','sizes'):
        actual=arrays[name].reshape(expected['region_'+name].shape)
        assert np.array_equal(actual,expected['region_'+name])
    assert sum(h.SIZES)==1836 and h.OFFSETS[-1]+h.SIZES[-1]==1836
    assert ct.sizeof(g)==1144


def test_p6_scorer_100000_and_edges(scorer):
    assert p6.check_scorer(scorer)['status']=='pass'


def test_p6_group_and_local_sinks(scorer,tmp_path):
    # Same fixed input; only the Addendum 1 per-bin addition bound changes.
    assert p6.check_sinks(scorer,tmp_path)['status']=='pass'


def test_p6_addition_counts_are_per_group_and_bin(scorer):
    records=np.array([[0,0,0,0,0,0,0,1,1],
                      [0,1,0,0,.5,0,0,.5,2],
                      [0,1,0,0,.5,0,0,.5,0],
                      [0,1,0,0,30,0,0,30,1],
                      [1,1,0,0,.5,0,0,.5,1],
                      [2,1,0,0,.5,0,0,.5,1]],float)
    histories=records[:,0].astype(np.int64);nh=200
    groups=np.zeros((100,h.BINS));local=np.zeros((nh,h.BINS));sg=np.zeros((nh,15));sl=sg.copy()
    additions=np.zeros(groups.shape,np.uint64)
    scorer.lib.p6_sinks(ct.byref(scorer.geometry),records,histories,len(records),nh,groups,local,sg,sl,additions)
    # The segment ends exactly on bin 4: += 0 counts too; z=30 is outside.
    assert additions[0,:5].tolist()==[1,1,4,1,1]
    assert additions[1,:5].tolist()==[0,0,1,0,0]
    assert not additions[2:].any() and not additions[:,540:].any()
    assert np.array_equal(additions[:,:120],additions[:,420:540])
    assert core.bits(sg,sl)


def test_p6_deterministic_replay_fixture(scorer):
    fixture=p6.scorer_fixture(scorer)
    assert fixture['regions'].shape==(74,1836)
    assert fixture['sums'].shape==(74,15)
    assert np.any(fixture['regions'][:,540:])
    assert np.allclose(fixture['regions'][:,420:540].sum(1),fixture['sums'][:,2],rtol=0,atol=1e-14)


def test_p6_scorer_fault_is_numeric(scorer,tmp_path):
    x=p6.scorer_vectors(0);good=scorer.score(x)
    bad=p6.CPU(tmp_path,fault=1).score(x)
    assert not core.bits(good[0],bad[0])
    assert core.bits(good[2],bad[2])
