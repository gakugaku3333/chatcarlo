"""Sealed input bundle and provenance-checked, resumable outer batches."""
import ctypes as ct
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from ..gpu import p5b_core as core, p7_core as h
from .condition import condition, materials, digest, check_equal, verify_observation

COMPARISON = ('egs5/main_egs/result.npz', 'egs5/main_egs/summary.json',
              'egs5/main_egs/config.json', 'pilot_lock_p3b3.json')


def read_json(path):
    return json.loads(Path(path).read_text())


def safe_path(root, name):
    rel = Path(name)
    if rel.is_absolute() or '..' in rel.parts:
        raise core.Stop('unsafe bundle path')
    return Path(root)/rel


def verify_files(root, files):
    for name, value in files.items():
        path = safe_path(root, name)
        if not path.is_file() or core.file_sha(path) != value:
            raise core.Stop(name+': SHA256 mismatch')


def scorer_fixture(cpu):
    """Nine-column deposits, including p6 boundaries for both point/segment sinks."""
    from ..gpu import p7
    vectors, _ = p7.scorer_vectors(0, extent=9.)
    rows = []
    for v in vectors:
        for kind in (0, 1):
            rows.append([len(rows), kind, *v[:3], *(v[3:6] if kind == 0 else v[:3]), v[6]])
    # Every coarse/fine bin edge, both axes, every band edge, forward/reverse segments and points.
    for depth in h.DEPTHS:
        for z in (depth-.5, depth, depth+.5):
            for axis in (0, 1):
                for x in np.arange(-9., 9.0001, .125):
                    p = [0., 0., z]; p[axis] = float(x)
                    q = p.copy(); q[axis] = float(x+.25)
                    for a, b in ((p, q), (q, p), (p, p)):
                        rows.append([len(rows), int(a == b), *a, *b, 1.])
    records = np.ascontiguousarray(rows, dtype=np.float64)
    n = len(records)
    groups = np.zeros((100, h.BINS)); local = np.zeros((n, h.BINS))
    sg = np.zeros((n, 15)); sl = sg.copy()
    cpu.lib.p7_sinks(ct.byref(cpu.geometry), records, np.arange(n, dtype=np.int64), n, n,
                     groups, local, sg, sl, np.zeros(groups.shape, np.uint64))
    return dict(records=records, regions=local, sums=sl)


def build(directory, cond, mat, data, approval_directory, comparison_root=None):
    """Approval is checked by the caller before creating this directory."""
    from .preview import require_approval
    require_approval(approval_directory, cond, mat)
    directory = Path(directory)
    manifest_path = directory/'manifest.json'
    if manifest_path.exists():
        loaded, saved = load(directory)
        check_equal(saved['condition_hash'], digest(cond), 'bundle condition')
        check_equal(saved['materials_hash'], digest(mat), 'bundle materials')
        return saved
    directory.mkdir(parents=True, exist_ok=True)
    for name in ('approval.json', 'preview_snapshot.json', 'preview.html'):
        shutil.copyfile(Path(approval_directory)/name, directory/name)
    core.write_json(directory/'condition.json', cond)
    core.write_json(directory/'materials.json', mat)
    core.save_npz(directory/'inputs.npz', data)
    # Source archive has only code; spectrum values are deliberately untracked bundle data.
    with zipfile.ZipFile(directory/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for root in (h.ROOT/'prototypes',):
            for path in sorted(root.rglob('*')):
                if path.is_file() and path.suffix in ('.py', '.h', '.cpp', '.cu') and '__pycache__' not in path.parts and 'tests' not in path.parts:
                    archive.write(path, str(path.relative_to(h.ROOT)))
    manifest = dict(condition_hash=digest(cond), materials_hash=digest(mat), inputs=core.describe(data))
    if cond['engine'] == 'colab_t4':
        from .gpu_adapter import CPU
        cpu = CPU(directory/'reference_build', data, cond, mat)
        reference = cpu.measure(2048, cond['seed'], history=True)
        verify_observation(cpu.last_observation, cond, mat, n=2048, start=0)
        core.save_npz(directory/'reference.npz', reference)
        core.save_npz(directory/'scorer.npz', scorer_fixture(cpu))
        if cond['spectrum'] == 'mohan4':
            # 4 MV: no EGS5 comparison in the entry (user choice). Dref only scales the report-only gate numbers;
            # it comes from the sealed P7 condition C pilot lock, which the entry only reads.
            lock_path = h.ROOT/'docs/validation/mv/p7/pilot_lock_C.json'
            manifest['comparison_inputs'] = {}
            manifest['Dref'] = read_json(lock_path)['Dref']
            manifest['Dref_source'] = dict(path='docs/validation/mv/p7/pilot_lock_C.json', sha256=core.file_sha(lock_path))
        else:
            comparison_root = Path(comparison_root) if comparison_root else h.ROOT/'docs/validation/mv/p3b'
            manifest['comparison_inputs'] = {name: core.file_sha(comparison_root/name) for name in COMPARISON}
            lock = read_json(comparison_root/'pilot_lock_p3b3.json')
            manifest['Dref'] = lock['Dref']
    names = ['condition.json', 'materials.json', 'inputs.npz', 'source.zip', 'approval.json', 'preview_snapshot.json', 'preview.html']
    if cond['engine'] == 'colab_t4':
        names += ['reference.npz', 'scorer.npz']
    manifest['files'] = {name: core.file_sha(directory/name) for name in names}
    core.write_json(manifest_path, manifest)
    return manifest


def load(directory):
    directory = Path(directory)
    manifest = read_json(directory/'manifest.json')
    verify_files(directory, manifest['files'])
    cond, mat = read_json(directory/'condition.json'), read_json(directory/'materials.json')
    approval = read_json(directory/'approval.json')
    for key, value in (('condition_hash', digest(cond)), ('materials_hash', digest(mat))):
        check_equal(value, manifest[key], key)
        check_equal(value, approval[key], 'approval '+key)
    data = core.read_npz(directory/'inputs.npz')
    core.check_inputs(data, manifest['inputs'])
    check_equal(materials(data), mat, 'bundle material reconstruction')
    # Reconstruct from verified arrays even if a manifest file check is accidentally bypassed.
    check_equal(cond['edges_sha256'], core.array_sha(data['edges']), 'edges_sha256')
    check_equal(cond['cum_sha256'], core.array_sha(data['cum']), 'cum_sha256')
    return data, manifest


def batch_paths(directory, b):
    root = Path(directory)
    return root/f'batch_{b:03}.npz', root/f'batch_{b:03}.json', root/f'observation_{b:03}.json'


def save_batch(directory, b, result, record, cond, mat, session):
    path, meta_path, obs_path = batch_paths(directory, b)
    if any(p.exists() for p in (path, meta_path, obs_path)):
        raise core.Stop('refuse batch overwrite')
    h.validate(result, nspec=len(result['nspec']))
    core.require(h.health(result), 'batch health')
    core.write_json(obs_path, record)
    core.save_npz(path, result)
    n = cond['batch_histories']
    meta = dict(condition_hash=digest(cond), materials_hash=digest(mat), batch=b, seed=cond['seed'],
                start=b*n, end=(b+1)*n, outputs=core.describe(result),
                output_sha256=core.file_sha(path), observation_sha256=core.file_sha(obs_path), session_id=session)
    core.write_json(meta_path, meta)
    return meta


def read_batch(directory, b, cond, mat):
    path, meta_path, obs_path = batch_paths(directory, b)
    meta = read_json(meta_path)
    n = cond['batch_histories']
    expected = dict(condition_hash=digest(cond), materials_hash=digest(mat), batch=b, seed=cond['seed'], start=n*b, end=n*(b+1))
    check_equal({k: meta[k] for k in expected}, expected, 'batch provenance')
    if not meta.get('session_id'):
        raise core.Stop('batch session_id missing')
    verify_files(directory, {path.name: meta['output_sha256'], obs_path.name: meta['observation_sha256']})
    record = read_json(obs_path)
    verify_observation(record, cond, mat, n=n, start=n*b)
    result = core.read_npz(path)
    check_equal(core.describe(result), meta['outputs'], 'batch output describe')
    h.validate(result, nspec=len(result['nspec']))
    core.require(h.health(result), 'collected batch health')
    if int(result['nspec'].sum()) != n:
        raise core.Stop('batch n mismatch')
    if np.any(result['regions'][..., h.layout('p6')['bins']:] != 0):
        raise core.Stop('unused region capacity is nonzero')
    return result


def existing(directory, cond, mat):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    found = set()
    for path in root.glob('batch_*'):
        try:
            b = int(path.stem.split('_')[1])
        except (ValueError, IndexError):
            raise core.Stop('invalid batch filename')
        if not 0 <= b < 100:
            raise core.Stop('batch index out of range')
        found.add(b)
    for b in found:
        read_batch(root, b, cond, mat)
    return found


def collect(directory, cond, mat):
    found = existing(directory, cond, mat)
    if found != set(range(100)):
        raise core.Stop('batches missing: '+str(sorted(set(range(100))-found)))
    rows = [read_batch(directory, b, cond, mat) for b in range(100)]
    ranges = [read_json(batch_paths(directory, b)[2])['observed'] for b in range(100)]
    cursor = 0
    for meta in ranges:
        if meta['start'] != cursor or meta['n'] <= 0:
            raise core.Stop('batch ranges overlap/gap')
        cursor = meta['start']+meta['n']
    if cursor != cond['histories']:
        raise core.Stop('histories mismatch')
    return rows
