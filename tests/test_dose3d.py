"""Approved dose3d plan T1–T12. Synthetic grids are independent of transport."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest
import yaml

from chatcarlo import diagnostics
from chatcarlo.dose3d import DoseOverlayError, build_dose_overlay
from chatcarlo.geometry import Geometry
from chatcarlo.preview import render_html, scene_to_json
from chatcarlo.scene import load_scene
from chatcarlo.tally import VoxelGrid

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def case(tmp_path):
    raw = dict(source=dict(position=[0, -40, 0], direction=[0, 1, 0],
                           spectrum=[dict(energy_keV=60, weight=1)],
                           field=dict(shape='rect', size_cm=[10, 10], sid_cm=100)),
               geometry=[dict(name='water', shape='box', material='water',
                              center=[0, 0, 0], size_cm=[10, 10, 10])])
    path = tmp_path / 'scene.yaml'
    path.write_text(yaml.safe_dump(raw))
    scene = load_scene(str(path))
    assert scene.ok, scene.errors
    g = Geometry(raw['geometry'])
    grid = VoxelGrid.from_bbox(g.tally_bbox_min, g.tally_bbox_max, 10)
    data = dict(origin_cm=grid.origin_cm.copy(), shape=np.array(grid.shape), voxel_size_cm=10.,
                h10_per_history_pSv=np.zeros(grid.shape), dose_per_history_Gy=np.zeros(grid.shape),
                rel_err_h10=np.full(grid.shape, .05), rel_err_dose=np.full(grid.shape, .05),
                n_batches=4, n_batches_hit=np.ones(grid.shape, dtype=int))
    points = grid.voxel_centers()
    near = ((g.material_at(points) == g.background) &
            (np.linalg.norm(points-raw['source']['position'], axis=1) < g.nearest_object_distance_cm(np.array(raw['source']['position']))))
    far_ids = np.flatnonzero(~near)
    near_id = int(np.flatnonzero(near)[0])
    data['h10_per_history_pSv'].flat[far_ids[0]] = 10
    data['dose_per_history_Gy'].flat[far_ids[0]] = 10
    return scene, path, data, far_ids, near_id


def build(case, tmp_path, data=None, quantity='h10'):
    scene, _, default, _, _ = case
    file = tmp_path/'dose.npz'
    np.savez(file, **(default if data is None else data))
    return build_dose_overlay(file, scene, quantity)


def cli(*args):
    return subprocess.run([sys.executable, '-m', 'chatcarlo', *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True)


def preview(case, tmp_path, data=None, *extra):
    np.savez(tmp_path/'dose.npz', **(case[2] if data is None else data))
    return cli('preview', case[1], '--dose', tmp_path/'dose.npz', '-o', tmp_path/'out.html', *extra)


@pytest.mark.parametrize('fault', ['shape', 'origin', 'missing', 'value_shape', 'partial', 'size0', 'size_nan', 'inf', 'nan', 'R_shape', 'hit_shape'])
def test_t1_invalid(case, tmp_path, fault):
    d = case[2]
    if fault == 'shape': d['shape'] = d['shape'] + [1, 0, 0]
    if fault == 'origin': d['origin_cm'][0] += 1e-6
    if fault == 'missing': del d['shape']
    if fault == 'value_shape': d['h10_per_history_pSv'] = np.zeros((1, 2, 3))
    if fault == 'partial': del d['n_batches']
    if fault == 'size0': d['voxel_size_cm'] = 0
    if fault == 'size_nan': d['voxel_size_cm'] = np.nan
    if fault in ('inf', 'nan'): d['h10_per_history_pSv'].flat[0] = getattr(np, fault)
    if fault == 'R_shape': d['rel_err_h10'] = np.zeros((2, 3, 4))
    if fault == 'hit_shape': d['n_batches_hit'] = np.zeros((2, 3, 4))
    with pytest.raises(DoseOverlayError, match='期待値.*実際の値') as exc:
        build(case, tmp_path)
    if fault in ('value_shape', 'R_shape', 'hit_shape'):
        assert str(tuple(int(x) for x in d['shape'])) in str(exc.value)
        assert str((1, 2, 3) if fault == 'value_shape' else (2, 3, 4)) in str(exc.value)


def test_t1_tolerance_unreadable(case, tmp_path):
    case[2]['origin_cm'][0] += 1e-10
    build(case, tmp_path)
    (tmp_path/'bad.npz').write_text('not npz')
    with pytest.raises(DoseOverlayError): build_dose_overlay(tmp_path/'bad.npz', case[0])


def test_t2_zero(case, tmp_path):
    case[2]['h10_per_history_pSv'][:] = 0
    result = preview(case, tmp_path)
    assert result.returncode == 1 and '[エラー]' in result.stderr
    assert not (tmp_path/'out.html').exists()


def test_t3_near(case, tmp_path):
    d, ids, near = case[2:]
    d['h10_per_history_pSv'].flat[near] = 1e7
    o = build(case, tmp_path)
    assert o['counts']['near_source_air'] == 1
    assert o['vmax'] == 10
    embedded = np.ravel_multi_index((o['main']['ix'], o['main']['iy'], o['main']['iz']), tuple(d['shape']))
    assert near not in embedded
    d['h10_per_history_pSv'].flat[ids[0]] = 0
    result = preview(case, tmp_path)
    assert result.returncode == 0 and (tmp_path/'out.html').exists()
    assert '線源のすぐ近く' in result.stdout and '-n を増やす' not in result.stdout


def test_t4_statistics(case, tmp_path, monkeypatch):
    d, ids = case[2:4]
    d['rel_err_h10'].flat[ids[0]] = .10
    o = build(case, tmp_path)
    assert len(o['main']['v']) == 1  # R=.10, hit/M=.25 inclusive
    for r, batches, hit, reason in ((.1001, 4, 1, 1), (.05, 8, 1, 2), (np.nan, 4, 1, 4)):
        d['rel_err_h10'].flat[ids[0]] = r
        d['n_batches'] = batches
        d['n_batches_hit'].flat[ids[0]] = hit
        o = build(case, tmp_path)
        assert o['unreliable']['why'] == [reason]
        assert o['vmin'] is None and o['vmax'] is None
    d['rel_err_h10'].flat[ids[0]] = .1001
    monkeypatch.setattr(diagnostics, '_UNRELIABLE_R_THRESHOLD', .2)
    assert len(build(case, tmp_path)['main']['v']) == 1
    monkeypatch.setattr(diagnostics, '_UNRELIABLE_HIT_FRACTION_THRESHOLD', .3)
    assert build(case, tmp_path)['unreliable']['why'] == [2]
    result = preview(case, tmp_path)
    assert result.returncode == 0 and '-n を増やす' in result.stdout
    d['n_batches_hit'].flat[ids[0]] = 0
    o = build(case, tmp_path)
    assert not o['main']['v'] and not o['unreliable']['v']
    result = preview(case, tmp_path)
    assert result.returncode == 0 and '表示できるボクセルが無い' in result.stdout
    assert 'zero' in result.stdout and (tmp_path/'out.html').exists()


@pytest.mark.parametrize('batches', [None, 1])
def test_t5_no_stats(case, tmp_path, batches):
    d = case[2]
    if batches is None:
        for k in ('rel_err_h10', 'n_batches', 'n_batches_hit'): del d[k]
    else:
        d['n_batches'] = batches
        d['n_batches_hit'][:] = 0
    o = build(case, tmp_path)
    assert not o['stats_available'] and len(o['main']['v']) == 1 and not o['unreliable']['v']
    assert o['main']['R'] == [None] and o['main']['hit'] == [None]
    html = render_html(case[0], dose_overlay=o)
    assert '統計の情報が無いため、どのボクセルの信頼性も確かめていない' in html
    assert 'background:#991b1b' in html


def large_case(case, tmp_path, values, unreliable=False):
    # Finer grid supplies enough eligible cells, without changing scene geometry.
    scene = case[0]
    g = Geometry(scene.raw['geometry'])
    grid = VoxelGrid.from_bbox(g.tally_bbox_min, g.tally_bbox_max, 2)
    pts = grid.voxel_centers()
    near = ((g.material_at(pts) == g.background) &
            (np.linalg.norm(pts-scene.raw['source']['position'], axis=1) < g.nearest_object_distance_cm(np.array(scene.raw['source']['position']))))
    ids = np.flatnonzero(~near)[:len(values)]
    d = dict(origin_cm=grid.origin_cm, shape=grid.shape, voxel_size_cm=2.,
             h10_per_history_pSv=np.zeros(grid.shape), rel_err_h10=np.full(grid.shape, 1. if unreliable else .05),
             n_batches=4, n_batches_hit=np.ones(grid.shape, dtype=int))
    d['h10_per_history_pSv'].flat[ids] = values
    return build(case, tmp_path, d), d, ids


def total_counts(o):
    c = o['counts']
    return c['zero'] + c['near_source_air'] + sum(sum(c[k].values()) for k in ('below_range', 'thinned', 'embedded'))


def test_t6_thinning(case, tmp_path):
    for values, unreliable, expected, thinned in (
        (np.linspace(1, 2, 30001), False, 30000, 1),
        (np.r_[np.full(10, 100), np.full(100, 10), np.ones(50000)], False, 30000, 20110),
        (np.linspace(1, 2, 20001), True, 20000, 1)):
        o, d, ids = large_case(case, tmp_path, values, unreliable)
        name = 'unreliable' if unreliable else 'main'
        assert len(o[name]['v']) == expected
        assert o['counts']['thinned'][name] == thinned
        assert max(o[name]['v']) == max(values)
        assert total_counts(o) == np.prod(d['shape'])
        assert build(case, tmp_path, d)[name] == o[name]
        if len(values) == 50110:
            assert o['decades']['main'] == {'0': {'kept': 10, 'original': 10}, '-1': {'kept': 100, 'original': 100}, '-2': {'kept': 29890, 'original': 50000}}
    o, d, ids = large_case(case, tmp_path, [1., 1e-6, .999e-6])
    assert sorted(o['main']['v']) == [1e-6, 1.]
    assert o['counts']['below_range']['main'] == 1
    assert total_counts(o) == np.prod(d['shape'])


def test_t7_units(case, tmp_path):
    scene, _, d, *_ = case
    scene.raw['source']['mas'] = 100
    o = build(case, tmp_path)
    assert '/history' in o['unit_label']
    d['h10_pSv'] = d['h10_per_history_pSv'] * 1000
    o = build(case, tmp_path)
    assert '校正済み' in o['calibration_note'] and 'mAs=100' in o['calibration_note']
    assert o['main']['v'] == [10000]
    scene.raw['source']['ctdi_vol_mGy'] = 10
    assert 'CTDIvol=10' in build(case, tmp_path)['calibration_note']
    o = build(case, tmp_path, quantity='dose')
    assert '空気のボクセルの吸収線量は、人や検出器がいる位置の評価には使えない' in render_html(scene, dose_overlay=o)


def test_t8_html(case, tmp_path):
    o = build(case, tmp_path)
    html = render_html(case[0], dose_overlay=o)
    assert all(x not in html for x in ('<script src', 'http://', 'https://'))
    embedded = json.loads(re.search(r'const DATA = (.*);', html)[1])['dose']
    assert embedded['counts'] == o['counts'] and embedded['materials']
    assert len(embedded['lut']) == 256
    assert all(re.fullmatch('#[0-9a-f]{6}', c) for c in embedded['lut'])
    assert 'dose' not in scene_to_json(case[0])
    if not shutil.which('node'): pytest.skip('node unavailable')
    script = tmp_path/'overlay.js'
    script.write_text(re.search(r'<script>(.*?)</script>', html, re.S)[1])
    result = subprocess.run(['node', '--check', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# If these example scenes change intentionally, update the approved golden hashes.
@pytest.mark.parametrize('name,digest', [
    ('abdomen_table', 'f0fcabbd61f842a4a928cbf7cb380fd674823dbfaf4d26dfaf4df346344d8c18'),
    ('angio_room', '3f14bb5e71f5ac9ec7a0d17e5781db8daa7e572d3df8044fe3f89800915f0e76'),
    ('chest_room', 'f0d0630737277be9a4c69e1449007cc27f5eea018fe694131f7cdfd82062cfab'),
    ('ct_room', '7c07a89707d189afa339de09e0d1546a9e37ceb8146707cd8bd39a63099fd332'),
    ('portable_ward', '61a4bb4bef46b7035899a00006b77d2936e8b8da2ce425e662da872a039e0b0e'),
    ('water_phantom_pdd_ocr', 'e9c8d05bfbe0adb67ffee627cd0a7c9f2be9c8498d80cf1e3bf154e5a36d74e2')])
def test_t9_golden(name, digest):
    html = render_html(load_scene(str(ROOT/'examples'/f'{name}.yaml')), title='golden')
    assert hashlib.sha256(html.encode()).hexdigest() == digest


def test_t10_cli(tmp_path):
    npz = tmp_path/'chest.npz'
    result = cli('run', 'examples/chest_room.yaml', '-n', '2e4', '--batch-size', 1000,
                 '--dose-grid', '--resolution', 10, '--seed', 1, '--dose-out', npz)
    assert result.returncode == 0, result.stderr
    for args in (['examples/chest_room.yaml', '--dose', npz],
                 ['--dose', npz, '--quantity', 'h10', 'examples/chest_room.yaml']):
        result = cli('preview', *args)
        assert result.returncode == 0, result.stderr
        assert (tmp_path/'chest_3d.html').exists()
    result = cli('preview', 'examples/water_phantom_pdd_ocr.yaml', '--dose', npz, '-o', tmp_path/'wrong.html')
    assert result.returncode == 1 and '[エラー]' in result.stderr
    assert not (tmp_path/'wrong.html').exists()


def test_t11_mv_dispatch(tmp_path, monkeypatch):
    from chatcarlo.__main__ import _mv_dispatch
    scene = tmp_path/'mv.yaml'
    scene.write_text('domain: therapy_mv\n')
    seen = []
    stub = types.ModuleType('prototypes.mv_electron.entry.cli')
    stub.main = lambda argv: seen.append(argv) or 42
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    args = ['preview', '--dose', 'x.npz', '--quantity', 'h10', str(scene)]
    assert _mv_dispatch(args) == 42 and seen == [args]


def test_t12_gpu_rejected(case, tmp_path):
    run_dir = tmp_path/'gpu_run'
    result = cli('preview', '--engine', 'gpu', '--gpu-device', 'colab', '--run-dir', run_dir,
                 '--dose', 'x.npz', case[1])
    assert result.returncode == 1 and '[エラー]' in result.stderr
    assert not run_dir.exists()
