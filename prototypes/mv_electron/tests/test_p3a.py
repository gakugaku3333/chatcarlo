"""P3a source, saving and analysis tests; full D1–D10 are the explicit --checks."""
import json
from pathlib import Path
import numpy as np
import pytest
from prototypes.mv_electron import run_p3a as p
from prototypes.mv_electron.photon_transport_p2c import run_coupled as original
from prototypes.mv_electron.run_coupled_p2 import inputs, KEYS as OLD_KEYS


@pytest.fixture(scope='module')
def spectrum():
    if not p.SPECTRUM.is_file():
        pytest.skip('private Mohan spectrum file is absent')
    return p.load_spectrum()


def test_spectrum_loading(spectrum):
    edges,prob,cum = spectrum
    assert edges.shape==(21,) and prob.shape==cum.shape==(20,)
    assert abs(prob.sum()-1)<=1e-12 and cum[-1]==1.
    mean,variance = p.spectrum_stats(edges,prob)
    assert abs(mean/p.MEAN-1)<=1e-9 and variance>0


def test_spectrum_sha_guard(spectrum,tmp_path):
    path = tmp_path/'altered.spectrum'
    path.write_bytes(p.SPECTRUM.read_bytes()+b'\n')
    with pytest.raises(RuntimeError,match='SHA256'):
        p.load_spectrum(path)


def test_actual_source_sampling_counts(spectrum):
    edges,prob,cum = spectrum
    r = p.execute(10.,10000,173,'mf',edges,cum)
    passed,v = p.sampling_result(r,edges,prob,10000)
    assert passed, v
    assert r['nspec'].shape==(20,10) and r['nspec'].dtype==np.int64
    assert p.healthy(r)


def test_no_spectrum_exact_regression(spectrum):
    args = inputs('mf')
    before = dict(zip(OLD_KEYS,original(6.,100,173,.05,.025,30.,*args,annihilate_in_flight=True)))
    after = p.execute(6.,100,173)
    assert all(np.array_equal(before[k],after[k]) for k in OLD_KEYS)
    assert after['nspec'].shape==(0,10)


def test_subcutoff_deposition_and_saved_normalization(spectrum,tmp_path,monkeypatch):
    monkeypatch.setattr(p,'CACHE',tmp_path/'cache')
    monkeypatch.setattr(p,'OUT',tmp_path/'out')
    edges = np.array([0.,.01]); cum = np.array([1.])
    r = p.load(p.run_saved('cutoff',1000,173,edges,cum,backend='mf',E=.01,spectrum_sha='synthetic'))
    passed,v = p.cutoff_result(r,1000)
    assert passed, v
    assert r['nspec'].sum()==1000
    assert np.array_equal(r['batch_totals'].sum(axis=0),r['totals'])
    assert np.array_equal(r['batch_counts'].sum(axis=0),r['counts'])
    target = tmp_path/'out/runs/cutoff/result.npz'; before = p.sha(target)
    p.run_saved('cutoff',1000,173,edges,cum,backend='mf',E=.01,spectrum_sha='synthetic')
    assert p.sha(target)==before


def test_saved_resume_and_changed_conditions(spectrum,tmp_path,monkeypatch):
    monkeypatch.setattr(p,'CACHE',tmp_path/'cache')
    monkeypatch.setattr(p,'OUT',tmp_path/'out')
    edges = np.array([0.,.01]); cum = np.array([1.])
    args = dict(N=200,seed=173,edges=edges,cum=cum,backend='mf',E=.01,spectrum_sha='synthetic')
    assert p.run_saved('partial',**args,stop_after=3) is None
    assert len(list((tmp_path/'cache/prototype/partial').glob('batch_*.npz')))==3
    partial = p.load(p.run_saved('partial',**args)); whole = p.load(p.run_saved('whole',**args))
    assert all(p.equality(partial,whole,[k for k in partial if k!='seconds']).values())
    with pytest.raises(RuntimeError,match='config mismatch'):
        p.run_saved('partial',**{**args,'seed':174})


def test_spectrum_rejects_other_sources(spectrum):
    edges,prob,cum = spectrum
    with pytest.raises(ValueError,match='photon source'):
        p.run_coupled(10.,100,173,.05,.025,30.,*inputs('mf'),
                      spectrum_edges=edges,spectrum_cum=cum,positron_source_T=1.)


def test_analysis_check_without_spectrum(monkeypatch,tmp_path):
    monkeypatch.setattr(p,'SPECTRUM',tmp_path/'absent')
    passed,values = p.analysis_check()
    assert passed,values


def test_define_regions_and_empty_guards():
    z = (np.arange(150)+.5)*.2
    curve = np.maximum(.01,10-np.abs(z-3)*.3)
    r = dict(coarse=np.tile(curve,(100,1)),fine=np.tile(np.repeat(curve,4),(100,1)))
    lock = p.define_regions(r)
    assert lock['window'][0]<=np.argmax(curve)<=lock['window'][1]
    assert lock['physical_blocks'][-1][-1]==149
    assert lock['build'][-1]<4*lock['window'][0]
    assert len(p.compare(r,r,lock)['blocks'])==len(lock['blocks'])
    for key in ('coarse','fine'):
        bad = dict(r); bad[key] = np.zeros_like(bad[key])
        with pytest.raises(RuntimeError,match='領域を定義できない'):
            p.define_regions(bad)


def test_plain_json_handles_numpy(tmp_path):
    path = tmp_path/'values.json'
    p.write(path,dict(value=np.bool_(True),counts=np.array([1,2]),nested=[np.float64(.5)]))
    assert json.loads(path.read_text())==dict(value=True,counts=[1,2],nested=[.5])


def test_gates_reject_stale_checks(monkeypatch,tmp_path):
    monkeypatch.setattr(p,'OUT',tmp_path)
    p.write(tmp_path/'checks.json',dict(status='合格',source_hashes={},checks={}))
    with pytest.raises(RuntimeError,match='D1–D10'):
        p.require_checks()
