"""Fixed P1d runs, no EGS5 execution. Resume only unfinished runs at original seed."""
import argparse
import json
import sys
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from prototypes.mv_electron.p1d_common import *

def transport_run(backend, name):
    from prototypes.mv_electron.transport import run
    verify()
    cfg = CONFIGS[name]
    path = OUT / backend / 'runs' / (name + '.npz')
    meta = path.with_suffix('.json')
    if path.exists():
        if not meta.exists(): raise RuntimeError('saved run lacks completion receipt: ' + str(path.relative_to(ROOT)))
        receipt = json.loads(meta.read_text())
        assert receipt['sha256'] == sha(path) and receipt['config'] == cfg and receipt['seed'] == SEED
        return
    g, args, kw = table(backend)
    T, n, fe, k = (cfg[x] for x in ('T', 'N', 'fe', 'K'))
    thick, dz = (1.2, .01) if T == 2 else (6., .05)
    faces = np.array([0., thick]) if not cfg['split'] else np.unique(np.r_[0., np.arange(cfg['shift'] or cfg['split'], thick, cfg['split']), thick])
    write(f'{backend}/progress_{name}.json', dict(active_backend=backend, active_run=name, config=cfg, seed=SEED,
                               resume_command='.venv/bin/python prototypes/mv_electron/run_p1d.py'))
    t = time.perf_counter()
    run(T, 1, SEED, fe, k, faces, dz, *args, mode=cfg['mode'], **kw)
    jit = time.perf_counter() - t
    t = time.perf_counter()
    r = run(T, n, SEED, fe, k, faces, dz, *args, mode=cfg['mode'], **kw)
    seconds = time.perf_counter() - t
    values = dict(zip(KEYS, r)); values['batches'] = r[0] / (n / 100 * dz)
    # Commit receipt first using the temporary NPZ hash; finalize NPZ last.
    pending = path.with_suffix('.pending.npz')
    save_npz(pending, **values, T=T, N=n, fe=fe, K=k, faces=faces, dz=dz, seed=SEED,
             mode=cfg['mode'], backend=backend, wall_seconds=seconds, jit_seconds=jit)
    write(meta, dict(config=cfg, seed=SEED, backend=backend, sha256=sha(pending),
                     table_sha256=sha(ROOT / TABLES[backend]), seconds=seconds, jit_seconds=jit))
    pending.replace(path)
    verify()
    print(backend, name, 'saved', round(seconds, 2), 'seconds', flush=True)

def fixed_runs(backend):
    from prototypes.mv_electron.reference import fixed
    _, args, kw = table(backend)
    for length in (.05, .2):
        for condensed in (False, True):
            verify()
            path = OUT / backend / 'fixed' / f'{length}_{condensed}.npz'
            seed = FIXED_SEED if condensed else SEED
            if path.exists():
                with np.load(path) as a:
                    assert int(a['N']) == 1000000 and int(a['seed']) == seed and float(a['K']) == .025
                receipt = json.loads(path.with_suffix('.json').read_text())
                assert receipt['sha256'] == sha(path)
                continue
            fixed(2., length, 1, seed, condensed, .025, *args, **kw)
            t = time.perf_counter(); a = fixed(2., length, 1000000, seed, condensed, .025, *args, **kw)
            seconds = time.perf_counter() - t
            pending = path.with_suffix('.pending.npz')
            save_npz(pending, values=a, N=1000000, seed=seed, K=.025, path_cm=length, condensed=condensed, seconds=seconds)
            write(path.with_suffix('.json'), dict(sha256=sha(pending), seed=seed, N=1000000, seconds=seconds))
            pending.replace(path)
            print(backend, 'K9i', length, condensed, 'saved', round(seconds, 2), flush=True)
    verify()

def benchmark(backend):
    """P1b timing method, adopted K only, 3 x 20000; no statistics pooled.
    All timing repetitions use SEED (no new seed selection). JIT excluded.
    Table build is out of scope; retain existing table build time as provenance.
    """
    from prototypes.mv_electron.transport import run, FIELDS
    target = OUT / backend / 'benchmark.json'
    saved = json.loads(target.read_text()) if target.exists() else dict(rows=[])
    g, args, kw = table(backend)
    for T in (2., 10.):
        for scoring in (True, False):
            rows = [r for r in saved['rows'] if r['T'] == T and r['scoring'] == scoring]
            row = rows[0] if rows else dict(T=T, scoring=scoring, N=20000, K=.025, fe=.05, seed=SEED, wall_seconds=[], counts_per_primary=[])
            if not rows: saved['rows'].append(row)
            faces = np.array([0., 1.2 if T == 2 else 6.]); dz = .01 if T == 2 else .05
            if len(row['wall_seconds']) == 3: continue
            run(T, 1, SEED, .05, .025, faces, dz, *args, scoring=scoring, **kw)
            while len(row['wall_seconds']) < 3:
                verify(); t = time.perf_counter()
                r = run(T, 20000, SEED, .05, .025, faces, dz, *args, scoring=scoring, **kw)
                row['wall_seconds'].append(time.perf_counter() - t)
                row['counts_per_primary'].append((r[2] / 20000).tolist())
                row['us_per_primary'] = float(np.median(row['wall_seconds']) / 20000 * 1e6)
                saved.update(method='single process; median 3 x 20000, same fixed seed; JIT excluded; no pooled statistics',
                             table_bytes=sum(g[x].nbytes for x in g.files), stack_bytes=4096 * FIELDS * 8,
                             historical_table_build_seconds=float(g['build_seconds']))
                write(target, saved)
            print(backend, 'timing', T, scoring, row['us_per_primary'], flush=True)

def main():
    p = argparse.ArgumentParser(); p.add_argument('--freeze', action='store_true')
    p.add_argument('--worker', choices=BACKENDS)
    p.add_argument('--run', choices=('fixed', *CONFIGS))
    a = p.parse_args()
    if a.freeze: freeze(); print('P1d code frozen; 4/4 regressions identical', flush=True); return
    verify(); regression_saved()
    from prototypes.mv_electron.run_checks_p1d import saved as replay, k5b
    if a.worker:
        if not a.run: p.error('--worker requires --run')
        if a.run == 'fixed': fixed_runs(a.worker)
        else: transport_run(a.worker, a.run)
        return
    if a.run: p.error('--run requires --worker')
    for backend in BACKENDS:
        k5b(backend, execute=True)
    def worker(backend, name):
        command = ['.venv/bin/python', 'prototypes/mv_electron/run_p1d.py', '--worker', backend, '--run', name]
        r = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        output = (r.stdout + r.stderr).replace(str(ROOT), '.').replace(str(Path.home()), '~')
        log = OUT / backend / (name + '.log'); log.write_text(output)
        if r.returncode: raise RuntimeError('worker failed: ' + str(log.relative_to(ROOT)))
        return backend, name, output
    # Independent processes preserve each function's fixed-seed random stream.
    # Only the parent performs saved-data reports; timing runs are serialized later.
    for jobs in ([ (b, 'fixed') for b in BACKENDS ], [(b, name) for name in CONFIGS for b in BACKENDS]):
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(worker, *job) for job in jobs]
            for future in as_completed(futures):
                backend, name, output = future.result()
                print(output.strip(), flush=True)
                write('progress.json', dict(last_completed_backend=backend, last_completed_run=name,
                      resume_command='.venv/bin/python prototypes/mv_electron/run_p1d.py'))
                replay()
    # No concurrent transport during the timing measurements.
    for backend in BACKENDS: benchmark(backend)
    replay(); write('untouched.json', verify())
    write('progress.json', dict(status='complete', resume_command='.venv/bin/python prototypes/mv_electron/run_p1d.py'))

if __name__ == '__main__': main()
