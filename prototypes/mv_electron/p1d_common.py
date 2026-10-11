"""P1d immutable inputs, fixed run matrix and atomic checkpoints (relative paths)."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/validation/mv/p1d'
SEED = 20261009
FIXED_SEED = 20261010
REGRESSION_SEED = 20261007
BOOTSTRAP_SEED = 20261008
BACKENDS = ('dcslib', 'mf')
KEYS = ('batches', 'ledger', 'counts', 'cross_sum', 'cross_sum2', 'cross_n', 'maxdepth')
TABLES = {'dcslib': 'docs/validation/mv/p1b/dcslib/data/gs.npz',
          'mf': 'docs/validation/mv/p1c/mf/data/gs.npz'}
OLD = {'dcslib': ROOT / 'docs/validation/mv/p1b/dcslib',
       'mf': ROOT / 'docs/validation/mv/p1c/mf'}
CONFIGS = {
    'base_2': dict(T=2., N=1000000, K=.025, fe=.05, split=0., shift=0., mode=0),
    'split_aligned': dict(T=2., N=1000000, K=.025, fe=.05, split=.1, shift=0., mode=0),
    'split_shifted': dict(T=2., N=1000000, K=.025, fe=.05, split=.1, shift=.037, mode=0),
    'ms_half': dict(T=2., N=1000000, K=.0125, fe=.05, split=0., shift=0., mode=0),
    'energy_half': dict(T=2., N=1000000, K=.025, fe=.025, split=0., shift=0., mode=0),
    'reference_2': dict(T=2., N=300000, K=.025, fe=.05, split=0., shift=0., mode=1),
    'base_10': dict(T=10., N=1000000, K=.025, fe=.05, split=0., shift=0., mode=0),
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    path = Path(path)
    if not path.is_absolute():
        path = OUT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)

def save_npz(path, **values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez(f, **values)
    tmp.replace(path)

def table(backend):
    g = np.load(ROOT / TABLES[backend])
    assert str(g['backend']) == backend
    assert .025 in g['K'] and .0125 in g['K']
    return g, tuple(g[x] for x in ('T', 'K', 'u', 'y')), dict(backend_rates=g['rates'], backend_single=g['single'])

def status(c):
    if not c['statistics_ok']:
        return 'statistics-insufficient'
    return 'pass' if c['pass'] else 'fail'

def aggregate(states):
    states = list(states)
    if 'fail' in states: return 'fail'
    if 'statistics-insufficient' in states: return 'statistics-insufficient'
    if not states or 'not-evaluated' in states: return 'not-evaluated'
    return 'pass'

LABELS = {'pass': '合格', 'fail': '不合格', 'statistics-insufficient': '統計不足', 'not-evaluated': '未評価'}

def verify():
    if (OUT / 'amendment1_hashes.json').exists():
        from .run_p1d_amendment1 import verify_freeze2
        return verify_freeze2()
    frozen = json.loads((OUT / 'initial_hashes.json').read_text())
    # New files in historical result trees are also prohibited.
    added = []
    for name in ('p1', 'p1b', 'p1c'):
        for p in (ROOT / 'docs/validation/mv' / name).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.relative_to(ROOT).as_posix() not in frozen:
                added.append(p.relative_to(ROOT).as_posix())
    code = json.loads((OUT / 'code_hashes.json').read_text())
    changed = [p for p, h in {**frozen, **code}.items() if not (ROOT / p).is_file() or sha(ROOT / p) != h]
    r = dict(status='pass' if not changed and not added else 'fail', initial_files=len(frozen),
             frozen_additional_files=len(code), changed=changed, added_historical_files=added)
    if r['status'] != 'pass': raise RuntimeError('R0 freeze mismatch: stop; see untouched.json')
    return r

def regression_saved():
    checks = {}
    for backend in BACKENDS:
        for k in (.05, .025):
            prefix = OUT / 'regression' / f'{backend}_{k}'
            with np.load(str(prefix) + '_before.npz') as a, np.load(str(prefix) + '_after.npz') as b:
                checks[f'{backend}_{k}'] = set(a.files) == set(b.files) and all(np.array_equal(a[x], b[x]) for x in a.files)
    if not all(checks.values()): raise RuntimeError('R0 regression mismatch: stop')
    return dict(status='pass', checks=checks, N=10000, seed=REGRESSION_SEED)

def freeze():
    target = OUT / 'code_hashes.json'
    if target.exists():
        verify(); regression_saved(); return
    paths = list((ROOT / 'prototypes/mv_electron').glob('*p1d*.py'))
    paths += list((ROOT / 'prototypes/mv_electron/tests').glob('*p1d*.py'))
    paths += [OUT / 'PREREGISTRATION.md', OUT / 'preregistration.sha256', OUT / 'initial_hashes.json', OUT / 'regression_before.json']
    paths += list((OUT / 'regression').glob('*_before.npz'))
    write(target, {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)})
    verify()
    from .transport import run
    for backend in BACKENDS:
        _, args, kw = table(backend)
        for k in (.05, .025):
            r = run(2., 10000, REGRESSION_SEED, .05, k, np.array([0., 1.2]), .01, *args, **kw)
            save_npz(OUT / 'regression' / f'{backend}_{k}_after.npz', **dict(zip(KEYS, r)))
    write('regression.json', regression_saved()); verify()
