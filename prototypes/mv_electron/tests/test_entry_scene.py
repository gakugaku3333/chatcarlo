import copy
import json
from pathlib import Path
import pytest
import yaml
from prototypes.mv_electron.entry import scene, condition, preview
from prototypes.mv_electron.gpu import p5b_core as core

EXAMPLE = Path(__file__).resolve().parents[3]/'examples/therapy/water_10mv_10x10.yaml'


@pytest.fixture
def raw():
    return scene.load(EXAMPLE)


@pytest.mark.parametrize('path,value', [
    ('source.field_cm', [10, 12]), ('source.ssd_cm', 90), ('source.spectrum', 'mohan4'),
    ('phantom.thickness_cm', 20), ('run.histories', 201), ('run.histories', 0),
    ('run.histories', -100), ('run.histories', 240000100), ('run.histories', 200.0),
    ('run.histories', True), ('run.seed', -1), ('run.seed', 2**64), ('run.seed', True),
    ('run.seed', 1.5), ('run.engine', 'cuda'), ('source.ssd_cm', True), ('source.field_cm', [True, 10])])
def test_entry_allowlist(raw, path, value):
    owner, key = path.split('.')
    raw[owner][key] = value
    with pytest.raises(scene.SceneError, match=path):
        scene.validate(raw)


@pytest.mark.parametrize('section,keys', [('', ('domain', 'source', 'phantom', 'scoring', 'run')),
    ('source', ('spectrum', 'field_cm', 'ssd_cm')), ('phantom', ('material', 'thickness_cm')),
    ('run', ('histories', 'seed', 'engine'))])
def test_entry_missing_unknown(raw, section, keys):
    for key in keys:
        changed = copy.deepcopy(raw)
        target = changed[section] if section else changed
        target.pop(key)
        with pytest.raises(scene.SceneError, match=key):
            scene.validate(changed)
    target = raw[section] if section else raw
    target['extra'] = 1
    with pytest.raises(scene.SceneError, match='extra'):
        scene.validate(raw)


def test_entry_cpu_limit(raw):
    raw['run'].update(engine='cpu_local', histories=200100)
    with pytest.raises(scene.SceneError, match='run.histories'):
        scene.validate(raw)
    raw['run'].update(histories=200000, seed=2**64-1)
    scene.validate(raw)


def test_entry_hash_and_order(raw, tmp_path):
    # No table files needed to test the canonical condition or YAML ordering.
    import numpy as np
    data = dict(edges=np.arange(3.), cum=np.array([.5, 1.]))
    a = condition.condition(raw, data)
    assert condition.digest(a) == condition.digest(condition.condition(raw, data))
    path = tmp_path/'scene.yaml'
    path.write_text(yaml.safe_dump(dict(reversed(list(raw.items()))), sort_keys=False))
    assert condition.digest(a) == condition.digest(condition.condition(scene.load(path), data))
    raw['run']['seed'] += 1
    assert condition.digest(a) != condition.digest(condition.condition(raw, data))


def test_entry_preview_geometry():
    data = preview.geometry_json()
    assert data['beam']['source'] == [0, 0, 130]
    assert {tuple(p) for p in data['beam']['corners']} == {(x, y, 30.) for x in (-5., 5.) for y in (-5., 5.)}
    objects = data['objects']
    assert {p[2] for edge in objects[0]['edges'] for p in edge} == {0., 30.}
    for depth in (5, 10, 20):
        bands = [o for o in objects if o['name'].startswith(f'深さ {depth} cm')]
        assert len(bands) == 2
        for band in bands:
            assert {p[2] for e in band['edges'] for p in e} == {30-depth-.5, 30-depth+.5}


def test_entry_approval_failures(raw, tmp_path, monkeypatch):
    import numpy as np
    data = dict(edges=np.arange(3.), cum=np.array([.5, 1.]))
    c = condition.condition(raw, data); m = {'test': 'materials'}
    monkeypatch.setattr(preview, 'identities', lambda s, d=None: (condition.condition(s, data), m, data))
    preview.write(raw, tmp_path, data)
    original = (tmp_path/'preview.html').read_text()
    assert condition.digest(c) in original
    raw['run']['seed'] += 1
    with pytest.raises(core.Stop, match='preview_snapshot'):
        preview.approve(raw, tmp_path, data)
    assert not (tmp_path/'approval.json').exists()
    raw['run']['seed'] -= 1
    (tmp_path/'preview.html').write_text(original.replace(condition.digest(c), 'absent'))
    with pytest.raises(core.Stop, match='visible hash'):
        preview.approve(raw, tmp_path, data)
    assert not (tmp_path/'approval.json').exists()
    (tmp_path/'preview.html').write_text(original)
    preview.approve(raw, tmp_path, data)
    preview.require_approval(tmp_path, c, m)
    with pytest.raises(core.Stop):
        preview.require_approval(tmp_path, dict(c, seed=0), m)
