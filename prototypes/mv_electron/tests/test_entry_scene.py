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
    ('source.field_cm', [10, 12]), ('source.ssd_cm', 90), ('source.spectrum', 'mohan6'),
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
    import re
    assert not re.findall(r'__[A-Z_]+__', original)  # テンプレートの目印が残ると script が壊れる（2026-10-23）
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


def test_entry_mohan4_scene_and_identity(raw):
    """4 MV is allowed with the same field/SSD/phantom; its identities differ from 10 MV (spectrum arrays, 16 bins)."""
    four = copy.deepcopy(raw)
    four['source']['spectrum'] = 'mohan4'
    scene.validate(four)
    c10, m10, _ = condition.identities(raw)
    c4, m4, d4 = condition.identities(four)
    assert c4['spectrum'] == 'mohan4' and c4['spectrum_file_sha256'] != c10['spectrum_file_sha256']
    assert m4['dimensions']['nspec'] == 16 and m10['dimensions']['nspec'] == 20
    assert condition.digest(c4) != condition.digest(c10) and condition.digest(m4) != condition.digest(m10)
    assert set(c4) == set(c10)  # no new keys: the 10 MV condition hash must not change
    snap4 = preview.snapshot(c4, m4)
    assert snap4['geometry']['beam']['energy_label'] == 'Mohan 4 MV'
    assert preview.snapshot(c10, m10)['geometry']['beam']['energy_label'] == 'Mohan 10 MV'


RECORDED = Path(__file__).resolve().parents[3]/'runs/mv_water_10mv_20261023'


@pytest.mark.skipif(not (RECORDED/'entry/approval.json').exists(), reason='recorded 10 MV run (gitignored) not present')
def test_entry_10mv_identities_unchanged_from_recorded_run():
    """The 2026-10-23 approved 10 MV run: its condition/materials hashes must still be reproduced exactly."""
    approval = json.loads((RECORDED/'entry/approval.json').read_text())
    cond, mat, _ = condition.identities(scene.load(RECORDED/'scene.yaml'))
    assert condition.digest(cond) == approval['condition_hash']
    assert condition.digest(mat) == approval['materials_hash']


def test_entry_mohan4_scene_and_identity(raw):
    """4 MV is allowed with the same field/SSD/phantom; its identities differ from 10 MV (spectrum arrays, 16 bins)."""
    four = copy.deepcopy(raw)
    four['source']['spectrum'] = 'mohan4'
    scene.validate(four)
    c10, m10, _ = condition.identities(raw)
    c4, m4, d4 = condition.identities(four)
    assert c4['spectrum'] == 'mohan4' and c4['spectrum_file_sha256'] != c10['spectrum_file_sha256']
    assert m4['dimensions']['nspec'] == 16 and m10['dimensions']['nspec'] == 20
    assert condition.digest(c4) != condition.digest(c10) and condition.digest(m4) != condition.digest(m10)
    assert set(c4) == set(c10)  # no new keys: the 10 MV condition hash must not change
    assert preview.snapshot(c4, m4)['geometry']['beam']['energy_label'] == 'Mohan 4 MV'
    assert preview.snapshot(c10, m10)['geometry']['beam']['energy_label'] == 'Mohan 10 MV'


RECORDED = Path(__file__).resolve().parents[3]/'runs/mv_water_10mv_20261023'


@pytest.mark.skipif(not (RECORDED/'entry/approval.json').exists(), reason='recorded 10 MV run (gitignored) not present')
def test_entry_10mv_identities_unchanged_from_recorded_run():
    """The 2026-10-23 approved 10 MV run: its condition/materials hashes must still be reproduced exactly."""
    approval = json.loads((RECORDED/'entry/approval.json').read_text())
    cond, mat, _ = condition.identities(scene.load(RECORDED/'scene.yaml'))
    assert condition.digest(cond) == approval['condition_hash']
    assert condition.digest(mat) == approval['materials_hash']


# Recorded 2026-10-23 approval of the 10 MV run (runs/mv_water_10mv_20261023, since moved away). Written out so the
# check does not depend on that gitignored folder.
RECORDED_10MV = dict(
    condition='bdc4554e91426b59bf782a8f3ae8decd01b77173cec50e58ebbe6fe7a8f370fa',
    materials='4721c0afca7437d2f6f2195f066d7c810d3b52bd8466ea67457e7032b34add38')


def test_entry_10mv_identities_fixed_values():
    scene_10mv = {'domain': 'therapy_mv', 'source': {'spectrum': 'mohan10', 'field_cm': [10, 10], 'ssd_cm': 100},
                  'phantom': {'material': 'water', 'thickness_cm': 30}, 'scoring': 'pdd_ocr',
                  'run': {'histories': 23000000, 'seed': 20261023, 'engine': 'colab_t4'}}
    cond, mat, _ = condition.identities(scene.validate(scene_10mv))
    assert condition.digest(cond) == RECORDED_10MV['condition']
    assert condition.digest(mat) == RECORDED_10MV['materials']
