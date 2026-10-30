"""Therapy bundle corruption and cross-set refusal without repeated huge copies."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'docs/validation/gpu/g5'))
import a2_bundle,a2_common as common
from chatcarlo.gpu import catalog,therapy_bundle,therapy,physics_sets,bundle


def small_material():
    # Schema-only fixture (never passed to transport). Real arrays checked by CLI.
    return SimpleNamespace(**{k:np.arange(6,dtype=float).reshape(2,3)+1 for k in catalog.P8_DEFINITIONS})


def test_real_roundtrip(tmp_path):
    # Use one persisted real bundle for all fields, geometry and initialized logs.
    r=a2_bundle.run(out=tmp_path/'bundle',summary=common.SUMMARY/'bundle.json')
    assert r['fields_per_material']==33 and len(r['rows'])>132


@pytest.fixture
def fixture_bundle(tmp_path):
    m=small_material();m.rates=np.array([[1.,2.,3.],[2.,3.,4.]])
    for k in catalog.P8_GRIDS.values():
        if k!='rates[:,0]':setattr(m,k,np.array([1.,2.]))
    gens={v['generator_path']:{'path':v['generator_path'],'sha256':'a'*64,'options':{}} for v in catalog.P8_DEFINITIONS.values()}
    p=tmp_path/'fixture';therapy_bundle.create(p,[m]*4,{'sizes':np.array([1,1,1,1,0])},gens)
    return p


@pytest.mark.parametrize('fault',['swap','missing','offset','tag'])
def test_corrupt(fixture_bundle,fault):
    p=fixture_bundle
    if fault=='tag':
        m=json.loads((p/'manifest.json').read_text());m['physics_set']['tag']+=1;(p/'manifest.json').write_text(json.dumps(m))
    else:
        c=json.loads((p/'catalog.json').read_text())
        if fault=='swap':c['fields'][0]['blob']='p8_lung'
        if fault=='missing':c['fields'].pop()
        if fault=='offset':c['fields'][0]['offset']+=1
        (p/'catalog.json').write_text(json.dumps(c))
        m=json.loads((p/'manifest.json').read_text());m['files']['catalog.json']=therapy_bundle.sha(p/'catalog.json');(p/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):therapy_bundle.verify(p)


def test_cross_set(fixture_bundle):
    with pytest.raises(ValueError):therapy_bundle.verify(fixture_bundle,expected_declaration=physics_sets.declaration())
    with pytest.raises(ValueError):bundle.verify(fixture_bundle)


def test_space_and_private_output(tmp_path,monkeypatch):
    monkeypatch.setattr(therapy.shutil,'disk_usage',lambda p:SimpleNamespace(free=3*1024**3-1))
    with pytest.raises(RuntimeError,match='3 GiB'):therapy.space(tmp_path)
    with pytest.raises(ValueError,match='outside'):therapy_bundle.create(common.HERE/'a2_run/private',[],{}, {})
