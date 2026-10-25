"""Preserve the exact outer-batch tables accepted by the original pool.py.

Run from any directory with the repository on PYTHONPATH. Raw inputs are read only.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys

import numpy as np
import scipy

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[3]
CACHE = Path.home() / '.cache/chatcarlo'
COLUMNS = dict(units='MeV/history/(g/cm2)', count=220,
               order='depth bins in edges_cm; material layer in bin_layer',
               grid_units=dict(edges_cm='cm', mass_g_cm2='g/cm2', bin_layer='layer index'))


def load_pool():
    """Run the unchanged readers once, in plain mode; restore caller state on failure too."""
    argv, cwd = sys.argv, Path.cwd()
    try:
        sys.argv = [str(DIRECTORY / 'pool.py')]
        os.chdir(ROOT)
        return runpy.run_path(str(DIRECTORY / 'pool.py'), run_name='__main__')
    finally:
        sys.argv = argv
        os.chdir(cwd)


def source_file(files, path, expected=None):
    path = path.resolve()
    base = 'cache' if path.is_relative_to(CACHE) else 'repo'
    relative = path.relative_to(CACHE if base == 'cache' else ROOT).as_posix()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None:
        assert sha == expected, path
    files.setdefault(base, {})[relative] = sha


def tables(namespace):
    """Supplement provenance only; all q arrays come from the original readers."""
    arrays = dict(namespace['egs'][0]['grid'])
    records, sources = [], {}
    for group in ('egs', 'gpu'):
        for i, run in enumerate(namespace[group]):
            ident = f'{group}_{i}'
            files = {}
            if group == 'egs':
                npz, report, _, seed = namespace['EGS'][i]
                source_file(files, ROOT / npz)
                source_file(files, ROOT / report)
            elif i == 0:
                seeds = set()
                for b in range(100):
                    p = namespace['P8F'] / f'batch_{b:03}'
                    rec = json.loads(p.with_suffix('.json').read_text())
                    seeds.add(rec['seed'])
                    source_file(files, p.with_suffix('.npz'), rec['sha256'])
                    source_file(files, p.with_suffix('.json'))
                assert len(seeds) == 1
                seed = seeds.pop()
            else:
                directory = CACHE / 'p8g3/colab/results'
                complete = json.loads((directory / 'complete.json').read_text())
                seed = complete['seed']
                source_file(files, directory / 'complete.json')
                for j in range(400):
                    source_file(files, directory / f'call_{j:03}.npz', complete['calls'][str(j)]['sha256'])
            q = run['q']
            assert q.shape == (100, 220) and q.dtype == np.dtype('float64')
            arrays.update({f'{ident}_q': q, f'{ident}_mean': q.mean(0),
                           f'{ident}_sem': q.std(0, ddof=1) / np.sqrt(100)})
            records.append(dict(id=ident, group=group, order=i, name=run['name'],
                                N=run['N'], seed=seed, shape=list(q.shape), dtype=str(q.dtype)))
            sources[ident] = files
    return arrays, dict(runs=records, orig=dict(egs=['egs_0'], gpu=['gpu_0']),
                        columns=COLUMNS, source_files=sources)


def main():
    arrays, metadata = tables(load_pool())
    metadata.update(stage='statsstack', pool_sha256=hashlib.sha256((DIRECTORY / 'pool.py').read_bytes()).hexdigest(),
                    commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
                                          capture_output=True, text=True).stdout.strip(),
                    versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                    source_roots=dict(repo='repository root', cache='~/.cache/chatcarlo'))
    with (DIRECTORY / 'batch_tables.npz').open('xb') as output:
        np.savez(output, **arrays)
    with (DIRECTORY / 'batch_tables.json').open('x') as output:
        json.dump(metadata, output, ensure_ascii=False, indent=2)
        output.write('\n')


if __name__ == '__main__':
    main()
