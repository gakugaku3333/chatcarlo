"""P7 judgement guards added after the Codex review of the analysis code."""
import json
import shutil
import numpy as np
import pytest
from prototypes.mv_electron import run_p7 as r, analysis_p7 as A
from prototypes.mv_electron.gpu import p7_core as h


def test_stop_on_missing_crossing():
    s=A.spec(2.5);nb=s['radius_bins'];x=np.ones((100,nb));x[:50,s['central'][0]:]=.3;x[50:,s['central'][0]:]=.7
    row=dict(proto=A.bootstrap_position(x,.25,s['central'][0]),egs=dict(position=1.,missing=0))
    with pytest.raises(ValueError):A.stop_on_missing(dict(J3=[row]))


@pytest.fixture
def egs_copy(tmp_path,monkeypatch):
    src=r.OUT/'egs5'/'pilot_egs_A'
    if not (src/'result.npz').exists():pytest.skip('P7 pilot EGS5 result not present')
    dst=tmp_path/'egs5'/'pilot_egs_A';shutil.copytree(src,dst);(dst/'p7_record.json').unlink()
    monkeypatch.setattr(r,'OUT',tmp_path);return dst


def test_read_egs_requires_seal(egs_copy):
    with pytest.raises(h.Stop):r.read_egs('A','pilot',r.PILOT_N)
    assert int(r.read_egs('A','pilot',r.PILOT_N,seal=True)['N'])==r.PILOT_N
    assert int(r.read_egs('A','pilot',r.PILOT_N)['N'])==r.PILOT_N


@pytest.mark.parametrize('edit',['float_seed','totals'])
def test_read_egs_rejects_result_edits(egs_copy,edit):
    v=dict(np.load(egs_copy/'result.npz'))
    if edit=='float_seed':v['seed']=np.array(float(v['seed'])+.5)
    else:v['totals']=v['totals']*1.0000001
    np.savez(egs_copy/'result.npz',**v)
    with pytest.raises(h.Stop):r.read_egs('A','pilot',r.PILOT_N,seal=True)


def test_slab_ratio_zero_denominator():
    x=np.zeros((100,120));y=np.ones((100,120))+np.random.default_rng(1).normal(0,.01,(100,120))
    rows=A.slab_ratios(dict(slab_coarse=x),dict(slab_coarse=y))
    assert rows[0]['ratio_minus_1_percent']==-100. and np.isfinite(rows[0]['SEM_percent'])
    assert A.slab_ratios(dict(slab_coarse=y),dict(slab_coarse=x))[0]['SEM_percent'] is None


@pytest.mark.parametrize('key,value',[('E',4.),('IUNRST',1),('thickness',20.),('user_code','p3r_egs'),('seed',1)])
def test_read_egs_rejects_config_edits(egs_copy,key,value):
    c=json.loads((egs_copy/'config.json').read_text());c[key]=value;(egs_copy/'config.json').write_text(json.dumps(c))
    with pytest.raises(h.Stop):r.read_egs('A','pilot',r.PILOT_N,seal=True)


def test_read_egs_rejects_change_after_fixing(egs_copy):
    r.read_egs('A','pilot',r.PILOT_N,seal=True)  # fixes the hashes
    v=dict(np.load(egs_copy/'result.npz'));v['axis_coarse']=v['axis_coarse']*1.0001;np.savez(egs_copy/'result.npz',**v)
    with pytest.raises(h.Stop):r.read_egs('A','pilot',r.PILOT_N)


def test_read_egs_rejects_wrong_N(egs_copy):
    with pytest.raises(h.Stop):r.read_egs('A','pilot',2*r.PILOT_N,seal=True)
