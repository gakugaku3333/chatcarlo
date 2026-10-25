"""Gate T: exact replay, provenance, raw-table identity and failure controls.

Raw input trees are never changed, including in the negative controls.
"""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
MV = ROOT / 'docs/validation/mv'
CACHE = Path.home() / '.cache/chatcarlo'
STAGES = ('statsstack', 'stats4mv')
MODES = [('statsstack', 'plain'), ('statsstack', 'orig'), ('statsstack', 'detail'),
         ('stats4mv', 'plain'), ('stats4mv', 'orig')]
RETIRED = {'statsstack': [CACHE / 'statsstack/p8f', CACHE / 'p8g3/colab'],
           'stats4mv': [CACHE / 'mv_entry/96782f068a81']}
# These are stable identities/order, independent of the generated JSON.
EXPECTED = {
    'statsstack': [('egs_0', 'egs_production', 60000000, 820001),
                   ('egs_1', 'egs_add_820201', 20000000, 820201),
                   ('egs_2', 'egs_add_820202', 20000000, 820202),
                   ('gpu_0', 'p8f_production', 20000000, 820101),
                   ('gpu_1', 'p8g3_production', 80000000, 820301)],
    'stats4mv': [('egs_0', 'main_egs_C', 26000000, 20291823),
                 ('egs_1', 'add_C_20292301', 37000000, 20292301),
                 ('egs_2', 'add_C_20292302', 37000000, 20292302),
                 ('gpu_0', 'runs/mv_water_4mv_20261024/entry', 26000000, 20261024),
                 ('gpu_1', str(CACHE / 'mv_entry/96782f068a81'), 74000000, 20292401)]}

# Installed before runpy or any scientific-library import in a fresh process.
# Library/code reads are exempt; all other reads must be the two named inputs.
AUDITED_RUNNER = r'''
import os, sys
from pathlib import Path
sys.dont_write_bytecode = True
script = Path(sys.argv[1]).resolve()
probe = sys.argv[2]
args = sys.argv[3:]
allowed = {script.parent / 'batch_tables.npz', script.parent / 'batch_tables.json'}
root = Path(os.environ['BATCH_TABLES_ROOT']).resolve()
home = Path.home()
forbidden = [home / '.cache/chatcarlo', root / 'runs',
             home / 'Library/CloudStorage', home / 'Google Drive', Path('/content/drive')]
libraries = [Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()]
def audit(event, args):
    if event != 'open' or isinstance(args[0], int):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    if any(path == p or path.is_relative_to(p) for p in forbidden):
        raise PermissionError('forbidden batch input: ' + str(path))
    if path in allowed:
        return
    if path.suffix in ('.py', '.pyc', '.so', '.dylib') or any(path.is_relative_to(p) for p in libraries):
        return
    raise PermissionError('unexpected batch input: ' + str(path))
sys.addaudithook(audit)
if probe != '-':
    Path(probe).read_bytes()
else:
    import runpy
    sys.argv = [str(script)] + args
    runpy.run_path(str(script), run_name='__main__')
'''


def audited(stage, mode, cwd, probe='-'):
    return subprocess.run(
        [sys.executable, '-B', '-c', AUDITED_RUNNER, str(MV / stage / 'pool_tables.py'),
         str(probe)] + ([] if mode == 'plain' else [mode]),
        cwd=cwd, env=dict(os.environ, BATCH_TABLES_ROOT=str(ROOT), PYTHONDONTWRITEBYTECODE='1'),
        check=True, capture_output=True)


@pytest.mark.parametrize('stage,mode', MODES)
def test_replay_bytes_with_input_audit(stage, mode, tmp_path):
    baseline = (MV / stage / f'baseline_{mode}.txt').read_bytes()
    assert baseline
    assert audited(stage, mode, tmp_path).stdout == baseline


@pytest.mark.parametrize('probe', [CACHE / 'missing-probe.npz', ROOT / 'runs/missing-probe.npz',
                                   Path.home() / 'Library/CloudStorage/GoogleDrive-test/probe.npz',
                                   ROOT / 'unexpected-input.npz'])
def test_audit_refuses_read_bytes(probe, tmp_path):
    with pytest.raises(subprocess.CalledProcessError) as error:
        audited('statsstack', 'plain', tmp_path, probe)
    assert b'PermissionError' in error.value.stderr
    assert str(probe).encode() in error.value.stderr


@pytest.fixture(scope='module', params=STAGES)
def saved(request):
    directory = MV / request.param
    with np.load(directory / 'batch_tables.npz', allow_pickle=False) as archive:
        arrays = dict(archive)
    return request.param, arrays, json.loads((directory / 'batch_tables.json').read_text())


def equal_array(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert np.array_equal(actual, expected)


def test_saved_statistics_and_metadata(saved):
    stage, arrays, metadata = saved
    count = 220 if stage == 'statsstack' else 228
    records = metadata['runs']
    assert [(r['id'], r['name'], r['N'], r['seed']) for r in records] == EXPECTED[stage]
    assert [(r['group'], r['order']) for r in records] == [('egs', 0), ('egs', 1), ('egs', 2), ('gpu', 0), ('gpu', 1)]
    assert metadata['orig'] == dict(egs=['egs_0'], gpu=['gpu_0'])
    assert metadata['stage'] == stage
    assert metadata['pool_sha256'] == hashlib.sha256((MV / stage / 'pool.py').read_bytes()).hexdigest()
    assert len(metadata['commit']) == 40 and int(metadata['commit'], 16) >= 0
    assert set(metadata['versions']) == {'python', 'numpy', 'scipy'}
    assert all(metadata['versions'].values())
    assert metadata['source_roots'] == dict(repo='repository root', cache='~/.cache/chatcarlo')
    assert metadata['columns']['count'] == count
    if stage == 'stats4mv':
        assert metadata['columns']['NR'] == 36
        assert metadata['columns']['units'] == 'Gy/history'
        assert metadata['columns']['PDD_bin_cm'] == metadata['columns']['OCR_bin_cm'] == 0.25
    else:
        assert metadata['columns']['units'] == 'MeV/history/(g/cm2)'
        assert arrays['edges_cm'].shape == (221,)
        assert arrays['bin_layer'].shape == arrays['mass_g_cm2'].shape == (220,)
    expected_keys = set()
    for record in records:
        ident = record['id']
        q = arrays[ident + '_q']
        assert record['shape'] == [100, count] and record['dtype'] == 'float64'
        assert q.shape == (100, count) and q.dtype == np.dtype('float64')
        equal_array(arrays[ident + '_mean'], q.mean(0))
        equal_array(arrays[ident + '_sem'], q.std(0, ddof=1) / np.sqrt(100))
        expected_keys.update(ident + '_' + key for key in ('q', 'mean', 'sem'))
        files = metadata['source_files'][ident]
        assert files and set(files) <= {'cache', 'repo'}
        for entries in files.values():
            assert entries
            for relative, sha in entries.items():
                assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
                assert len(sha) == 64 and int(sha, 16) >= 0
    if stage == 'statsstack':
        expected_keys.update(('edges_cm', 'bin_layer', 'mass_g_cm2'))
    assert set(arrays) == expected_keys
    assert set(metadata['source_files']) == {r['id'] for r in records}


def check_raw_inputs(stage, metadata, require_raw):
    """Only whole retired trees may be absent in normal mode; partial loss fails."""
    absent_trees = [p for p in RETIRED[stage] if not p.exists()]
    missing = []
    for files in metadata['source_files'].values():
        for base, entries in files.items():
            for relative in entries:
                path = (CACHE if base == 'cache' else ROOT) / relative
                if not path.is_file():
                    missing.append(path)
    if missing:
        if not require_raw and all(any(p.is_relative_to(tree) for tree in absent_trees) for p in missing):
            pytest.skip('raw inputs retired: ' + ', '.join(str(p) for p in absent_trees))
        raise FileNotFoundError('required raw input missing: ' + str(missing[0]))


def extract_raw(stage, metadata, require_raw):
    check_raw_inputs(stage, metadata, require_raw)
    extractor = runpy.run_path(str(MV / stage / 'extract_tables.py'))
    with contextlib.redirect_stdout(io.StringIO()):
        namespace = extractor['load_pool']()
    arrays, actual_metadata = extractor['tables'](namespace)
    return arrays, actual_metadata, namespace


@pytest.fixture(scope='module', params=STAGES)
def raw(request):
    stage = request.param
    metadata = json.loads((MV / stage / 'batch_tables.json').read_text())
    arrays, actual_metadata, namespace = extract_raw(
        stage, metadata, os.environ.get('BATCH_TABLES_REQUIRE_RAW') == '1')
    return stage, arrays, actual_metadata, namespace


@pytest.mark.parametrize('ident', ['egs_0', 'egs_1', 'egs_2', 'gpu_0', 'gpu_1'])
def test_raw_table_identity(raw, ident):
    stage, actual, _, _ = raw
    with np.load(MV / stage / 'batch_tables.npz', allow_pickle=False) as saved:
        for suffix in ('q', 'mean', 'sem'):
            equal_array(saved[ident + '_' + suffix], actual[ident + '_' + suffix])


def test_raw_metadata_and_grid(raw):
    stage, arrays, actual_metadata, _ = raw
    metadata = json.loads((MV / stage / 'batch_tables.json').read_text())
    for key, expected in actual_metadata.items():
        assert metadata[key] == expected
    if stage == 'statsstack':
        with np.load(MV / stage / 'batch_tables.npz', allow_pickle=False) as saved:
            for key in ('bin_layer', 'edges_cm', 'mass_g_cm2'):
                equal_array(saved[key], arrays[key])


def test_required_mode_rejects_one_missing_input(monkeypatch):
    stage = 'statsstack'
    metadata = json.loads((MV / stage / 'batch_tables.json').read_text())
    missing = ROOT / 'docs/validation/mv/p8e/egs_production.npz'
    is_file = Path.is_file
    monkeypatch.setattr(Path, 'is_file', lambda p: False if p == missing else is_file(p))
    with pytest.raises(FileNotFoundError, match='egs_production.npz'):
        extract_raw(stage, metadata, require_raw=True)


@pytest.mark.parametrize('fault', ['hash', 'condition'])
def test_reader_verification_errors_are_not_skipped(monkeypatch, fault):
    metadata = json.loads((MV / 'statsstack/batch_tables.json').read_text())
    batch = CACHE / 'statsstack/p8f/p8f/results/production/batch_000'
    read_text, read_bytes = Path.read_text, Path.read_bytes

    def faulty_metadata(path, *args, **kwargs):
        if path == batch.with_suffix('.json'):
            return json.dumps(dict(n=0 if fault == 'condition' else 200000,
                                   start=0, seed=820101, sha256='0' * 64))
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', faulty_metadata)
    monkeypatch.setattr(Path, 'read_bytes',
                        lambda p: b'corrupt batch' if p == batch.with_suffix('.npz') else read_bytes(p))
    # Exercise load_p8f's actual assertion, even after its whole tree is retired.
    monkeypatch.setattr(sys.modules[__name__], 'check_raw_inputs', lambda *a: None)
    with pytest.raises(AssertionError):
        extract_raw('statsstack', metadata, require_raw=True)


def test_row_permutation_is_rejected(raw):
    _, arrays, _, _ = raw
    q = arrays['egs_0_q']
    changed = q.copy()
    changed[[0, 1]] = changed[[1, 0]]
    # These summaries remain numerically equal while the ordered table differs.
    np.testing.assert_allclose(changed.mean(0), q.mean(0), rtol=1e-14, atol=0)
    np.testing.assert_allclose(changed.std(0, ddof=1), q.std(0, ddof=1), rtol=1e-14, atol=0)
    with pytest.raises(AssertionError):
        equal_array(changed, q)


@pytest.mark.parametrize('stage', STAGES)
def test_extractor_restores_argv_and_cwd_on_failure(stage, monkeypatch, tmp_path):
    extractor = runpy.run_path(str(MV / stage / 'extract_tables.py'))
    argv = ['caller', 'orig', 'detail']
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.chdir(tmp_path)
    calls = []

    def fail(path, run_name):
        calls.append(path)
        assert run_name == '__main__'
        assert sys.argv == [str(MV / stage / 'pool.py')]
        assert Path.cwd() == ROOT
        raise ValueError('reader failed')

    monkeypatch.setattr(runpy, 'run_path', fail)
    with pytest.raises(ValueError, match='reader failed'):
        extractor['load_pool']()
    assert len(calls) == 1
    assert sys.argv is argv and Path.cwd() == tmp_path


def test_derived_files_under_five_mb():
    assert sum((MV / stage / f'batch_tables.{ext}').stat().st_size
               for stage in STAGES for ext in ('npz', 'json')) < 5_000_000
