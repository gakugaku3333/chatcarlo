"""Preserve the exact 4 MV quantities accepted by the original pool.py.

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
COLUMNS = dict(units='Gy/history', count=228, NR=36,
               order='PDD 120 columns, then folded OCR 3 x 36 columns at depths 5, 10, 20 cm',
               PDD_bin_cm=0.25, OCR_bin_cm=0.25,
               OCR_directions='mean of +x, -x, +y, -y, outward from the axis')


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
    """Supplement provenance only; do not duplicate quantities or raw readers."""
    arrays, records, sources = {}, [], {}
    for group in ('egs', 'gpu'):
        for i, run in enumerate(namespace[group]):
            ident = f'{group}_{i}'
            files = {}
            if group == 'egs':
                path, _, seed = namespace['EGS'][i]
                for name in ('result.npz', 'config.json', 'summary.json'):
                    source_file(files, ROOT / path / name, run['sha'] if name == 'result.npz' else None)
            else:
                path = ROOT / namespace['GPU'][i]
                for name in ('condition.json', 'materials.json'):
                    source_file(files, path / 'bundle' / name)
                seeds = set()
                for b in range(100):
                    directory = path / 'results'
                    meta_path = directory / f'batch_{b:03}.json'
                    meta = json.loads(meta_path.read_text())
                    seeds.add(meta['seed'])
                    source_file(files, meta_path)
                    source_file(files, directory / f'batch_{b:03}.npz', meta['output_sha256'])
                    source_file(files, directory / f'observation_{b:03}.json', meta['observation_sha256'])
                assert seeds == {run['seed']}
                seed = seeds.pop()
            q = run['q']
            assert q.shape == (100, 228) and q.dtype == np.dtype('float64')
            arrays.update({f'{ident}_q': q, f'{ident}_mean': q.mean(0),
                           f'{ident}_sem': q.std(0, ddof=1) / np.sqrt(100)})
            records.append(dict(id=ident, group=group, order=i, name=run['name'],
                                N=run['N'], seed=seed, shape=list(q.shape), dtype=str(q.dtype)))
            sources[ident] = files
    return arrays, dict(runs=records, orig=dict(egs=['egs_0'], gpu=['gpu_0']),
                        columns=COLUMNS, source_files=sources)


def main():
    arrays, metadata = tables(load_pool())
    metadata.update(stage='stats4mv', pool_sha256=hashlib.sha256((DIRECTORY / 'pool.py').read_bytes()).hexdigest(),
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
