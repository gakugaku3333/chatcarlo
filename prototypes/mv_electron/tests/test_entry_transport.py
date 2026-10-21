import copy
import ctypes as ct
from pathlib import Path
import json
import numpy as np
import pytest
from prototypes.mv_electron.entry import condition, scene, preview, bundle, vm_driver, cli
from prototypes.mv_electron.entry.gpu_adapter import CPU
from prototypes.mv_electron.gpu import p7, p7_core as h, p5b_core as core


@pytest.fixture(scope='module')
def context():
    raw = scene.load(Path(__file__).resolve().parents[3]/'examples/therapy/water_10mv_10x10.yaml')
    raw['run'].update(engine='cpu_local', histories=2000)
    c, m, d = condition.identities(raw)
    return raw, c, m, d


@pytest.fixture(scope='module')
def work(tmp_path_factory):
    return tmp_path_factory.mktemp('entry_cpu')


@pytest.mark.parametrize('history,start', [(True, 0), (True, 2048), (False, 20)])
def test_entry_p7_byte_identity(context, work, history, start):
    raw, c, m, d = context
    cpu = CPU(work, d, c, m)
    n = 2048 if history else 20
    actual = cpu.measure(n, c['seed'], start=start, history=history)
    condition.verify_observation(cpu.last_observation, c, m, n=n, start=start)
    other = p7.CPU(work, d, 'philox', layout='p6')
    expected = (other.histories if history else other.run)(n, c['seed'], 5., 100., start=start)
    assert set(actual) == set(expected)
    assert all(core.bits(actual[k], expected[k]) for k in expected)


def test_entry_fault_at_native_boundary(context, work, monkeypatch):
    raw, c, m, d = context
    cpu = CPU(work, d, c, m)
    monkeypatch.setenv('MV_ENTRY_FAULT', 'field_half')
    cpu.measure(20, c['seed'])
    with pytest.raises(core.Stop, match='field_half_cm'):
        condition.verify_observation(cpu.last_observation, c, m, n=20, start=0)


@pytest.mark.parametrize('kind', ['edges', 'cum', 'dimension', 'scalar', 'geometry'])
def test_entry_misbinding(context, work, kind):
    raw, c, m, d = context
    cpu = CPU(work, d, c, m)
    if kind in ('edges', 'cum'):
        setattr(cpu.tables, kind, d['brems_e'].ctypes.data_as(ct.POINTER(ct.c_double)))
    elif kind == 'dimension':
        cpu.tables.ne += 1
    elif kind == 'scalar':
        cpu.tables.bpar += .1
    else:
        cpu.geometry.lo[0] += .25
    with pytest.raises(core.Stop):
        cpu.measure(20, c['seed'])
        condition.verify_observation(cpu.last_observation, c, m, n=20, start=0)


@pytest.fixture(scope='module')
def local_run(context, work):
    raw, c, m, d = context
    out = work/'run'
    preview.write(raw, out, d)
    # Synthetic approval within the test only, not a user-facing simulation approval.
    preview.approve(raw, out, d)
    bundle.build(out/'bundle', c, m, d, out)
    vm_driver.run(out/'bundle', out/'results', session='local-test')
    return out


def test_entry_cpu_end_to_end(context, local_run, monkeypatch):
    raw, c, m, d = context
    rows = bundle.collect(local_run/'results', c, m)
    assert len(rows) == 100
    metas = [bundle.read_json(bundle.batch_paths(local_run/'results', b)[1]) for b in range(100)]
    assert all(x['condition_hash'] == condition.digest(c) for x in metas)
    assert sum(int(r['nspec'].sum()) for r in rows) == 2000
    # A resume must skip every existing batch, including start > 0.
    def fail(*args, **kwargs):
        pytest.fail('recomputed an existing batch')
    monkeypatch.setattr(CPU, 'measure', fail)
    vm_driver.run(local_run/'bundle', local_run/'results', session='resume-test')


@pytest.mark.parametrize('kind', ['output', 'observation', 'seed', 'condition', 'range'])
def test_entry_provenance_tamper(context, local_run, tmp_path, kind):
    import shutil
    raw, c, m, d = context
    for p in bundle.batch_paths(local_run/'results', 0):
        shutil.copyfile(p, tmp_path/p.name)
    path, meta, obs = bundle.batch_paths(tmp_path, 0)
    if kind == 'output':
        data = core.read_npz(path); data['totals'][0] += 1; core.save_npz(path, data)
    elif kind == 'observation':
        data = bundle.read_json(obs); data['observed']['field_half_cm'] = 5.5; core.write_json(obs, data)
    else:
        data = bundle.read_json(meta)
        data[{'seed': 'seed', 'condition': 'condition_hash', 'range': 'start'}[kind]] = 1
        core.write_json(meta, data)
    with pytest.raises(core.Stop):
        bundle.read_batch(tmp_path, 0, c, m)


def test_entry_bundle_tamper(context, local_run, tmp_path, monkeypatch):
    import shutil
    raw, c, m, d = context
    shutil.copytree(local_run/'bundle', tmp_path/'bundle')
    path = tmp_path/'bundle/condition.json'
    changed = bundle.read_json(path); changed['field_half_cm'] = 5.5; core.write_json(path, changed)
    with pytest.raises(core.Stop, match='condition.json: SHA256'):
        bundle.load(tmp_path/'bundle')
    # Even bypassing the file-manifest check does not bypass approval hashes.
    monkeypatch.setattr(bundle, 'verify_files', lambda *a: None)
    with pytest.raises(core.Stop, match='condition_hash'):
        bundle.load(tmp_path/'bundle')


def test_entry_no_approval_no_bundle(context, tmp_path, monkeypatch):
    import yaml
    raw, c, m, d = context
    path = tmp_path/'scene.yaml'; path.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(condition, 'load_data', lambda: d)
    assert cli.main(['run', str(path), '--out', str(tmp_path/'out')]) == 1
    assert not (tmp_path/'out/bundle').exists()
    preview.write(raw, tmp_path/'out', d); preview.approve(raw, tmp_path/'out', d)
    raw['run']['seed'] += 1; path.write_text(yaml.safe_dump(raw))
    assert cli.main(['run', str(path), '--out', str(tmp_path/'out')]) == 1
    assert not (tmp_path/'out/bundle').exists()


def test_entry_changed_condition_rejected_at_argument_boundary(context, work):
    raw, c, m, d = context
    # Bypass the bundle checks deliberately; approval still expects the original condition.
    cpu = CPU(work, d, dict(c, field_half_cm=5.5), m)
    cpu.measure(20, c['seed'])
    with pytest.raises(core.Stop, match='field_half_cm'):
        condition.verify_observation(cpu.last_observation, c, m, n=20, start=0)


def test_entry_mixed_seed_observation(context, local_run, tmp_path, work):
    raw, c, m, d = context
    other = CPU(work, d, dict(c, seed=c['seed']+1), m)
    result = other.measure(20, c['seed']+1)
    bundle.save_batch(tmp_path, 0, result, other.last_observation, c, m, 'other-seed')
    with pytest.raises(core.Stop, match='seed'):
        bundle.read_batch(tmp_path, 0, c, m)
