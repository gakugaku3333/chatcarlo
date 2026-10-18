"""P7 entry point: field size and energy variants of P6's 11 quantities.

Plan: docs/ai/plans/2026-10-18-mv-p7-field-energy-variants.md
Commands:
  local      gate 0 (deterministic)
  gate1      gate 1 (saved P3b-3 replay, Numba equality, planted faults)
  pilot      CPU Philox 4M + EGS5 4M per condition, then the pilot lock (Dref, N_main)
  egs        EGS5 production per condition (1 job per condition, at most 2 at a time)
  export / analyse are separate later steps.
Default work directory: ~/.cache/chatcarlo/p7/work (outside the repository).
EGS5 runs reuse run_p3b2.run_egs unchanged; only its module paths and the recorded
spectrum SHA are scoped to each call (the frozen Fortran is not edited).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from contextlib import contextmanager
import json
from pathlib import Path
import numpy as np
from scipy.stats import chisquare

from . import run_p3b2 as egs, analysis_p7 as stats
from .gpu import p7_core as h

ROOT = h.ROOT
DEFAULT_WORK = Path('~/.cache/chatcarlo/p7/work').expanduser()
OUT = ROOT/'docs/validation/mv/p7'
EGS_CACHE = h.CACHE
PILOT_N = 4000000


@contextmanager
def egs_scope(name):
    """Point the unchanged P3b-2 runner at P7's directories and the condition's spectrum SHA."""
    c = h.condition(name)
    old = egs.CACHE, egs.OUT, egs.CHECKS, egs.r.SPECTRUM_SHA
    egs.CACHE, egs.OUT, egs.CHECKS = EGS_CACHE, OUT, OUT/'checks_p7_none.json'
    egs.r.SPECTRUM_SHA = c['spectrum_sha256']
    try:
        yield
    finally:
        egs.CACHE, egs.OUT, egs.CHECKS, egs.r.SPECTRUM_SHA = old


def egs_label(stage, name):
    return f'{stage}_egs_{name}'


def egs_record(name, stage, N):
    """F7e: keep spectrum identity, spectrum top, the config E and the PEGS limits apart."""
    c = h.condition(name)
    return dict(condition=name, stage=stage, N=N, seed=c['seeds'][stage+'_egs'], spectrum=c['spectrum'],
                spectrum_sha256=c['spectrum_sha256'], spectrum_top_MeV=c['spectrum_top'],
                config_E_note='config E=10 is the P3b-2 runner constant; with spectrum.dat present p3b_egs.f sets emaxe to the spectrum top',
                pegs_limits=dict(UE=20.511, UP=21.0), field=[c['field_half'], c['ssd']], regions=h.layout('p7')['regions'])


def source_check(v, name, N):
    """Plan section 7: |x|,|y| <= half width, direction consistent with SSD, x and y uniform in 10 cells."""
    c = h.condition(name); src = v['sources']; fh = c['field_half']
    direction = np.column_stack((src[:, :2], np.full(len(src), c['ssd'])))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    error = float(np.max(abs(direction-src[:, 2:])/np.maximum(abs(direction), 1e-300)))
    p = [float(chisquare(np.histogram(src[:, k], bins=10, range=(-fh, fh))[0]).pvalue) for k in (0, 1)]
    ok = (bool(np.all(abs(src[:, :2]) <= fh)) and error <= 1e-12 and min(p) >= .001
          and int(v['source_hist'].sum()) == N)
    return dict(p=p, direction_relative_error=error, records=len(src), status='pass' if ok else 'fail')


def run_egs(name, stage, N):
    c = h.condition(name); edges, _, cum = h.spectrum(c['spectrum']); label = egs_label(stage, name)
    seed = c['seeds'][stage+'_egs']; dest = OUT/'egs5'/label
    record = egs_record(name, stage, N)
    with egs_scope(name):
        if not (dest/'result.npz').exists():
            work = EGS_CACHE/'egs5'/label
            resume = (work/'checkpoint.bin').is_file()
            egs.run_egs(label, N, seed, edges, cum, field=[c['field_half'], c['ssd']],
                        regions=h.layout('p7')['regions'], resume=resume)
    return read_egs(name, stage, N, record)


def egs_config(name, stage, N):
    """The complete config run_p3b2.run_egs writes for this condition (same construction, P7 SHA)."""
    c = h.condition(name); edges, _, cum = h.spectrum(c['spectrum'])
    return egs.r.plain(dict(N=N, seed=c['seeds'][stage+'_egs'], E=10., IUNRST=0, noflight=0, thickness=30., batches=100,
        input=egs.r.pegs_input(0), user_code='p3b_egs', field=[float(c['field_half']), float(c['ssd'])],
        regions=list(map(float, h.layout('p7')['regions'])), egs_input_hashes=egs.egs_hashes(),
        source_hashes=egs.source_hashes(), **egs.r.spec_config(edges, cum, c['spectrum_sha256'])))


def read_egs(name, stage, N, record=None):
    from .gpu import p5b_core as core
    c = h.condition(name); label = egs_label(stage, name); dest = OUT/'egs5'/label
    files = {f: core.file_sha(dest/f) for f in ('result.npz', 'summary.json', 'config.json')}
    rec = dest/'p7_record.json'
    if rec.exists() and 'files' in json.loads(rec.read_text()) and json.loads(rec.read_text())['files'] != files:
        raise h.Stop('EGS5 result/summary/config changed after completion: '+label)
    v = dict(np.load(dest/'result.npz', allow_pickle=False)); summary = json.loads((dest/'summary.json').read_text())
    config = json.loads((dest/'config.json').read_text())
    if config != json.loads(json.dumps(egs_config(name, stage, N))):
        raise h.Stop('EGS5 config differs from condition '+name)
    if (summary.get('status') != '合格' or not isinstance(summary.get('tests'), dict) or not summary['tests']
            or not all(x is True for x in summary['tests'].values())):
        raise h.Stop('EGS5 summary checks failed')
    if int(v['N']) != N or int(v['seed']) != c['seeds'][stage+'_egs']:
        raise h.Stop('EGS5 result N/seed differ')
    if np.asarray(v['region_ledger']).shape != (15,) or np.any(v['region_ledger']):
        raise h.Stop('EGS5 region ledger violation or shape')
    if not all(np.all(np.isfinite(x)) for x in v.values() if np.asarray(x).dtype.kind in 'fiu'):
        raise h.Stop('EGS5 result has non-finite values')
    check = source_check(v, name, N)
    if check['status'] != 'pass':
        raise h.Stop('EGS5 source check failed: '+json.dumps(check))
    if not rec.exists() or 'files' not in json.loads(rec.read_text()):
        # Fixed at first read after completion (records written before this fix get their hashes added once,
        # before any judgement value exists); every later read compares against these hashes.
        rec.write_text(json.dumps(dict(record or egs_record(name, stage, N), source_check=check, files=files), ensure_ascii=False, indent=1))
    data = {k: v[k] for k in ('axis_coarse', 'axis_fine', 'slab_coarse', 'profile', 'profile_fine')}
    data.update(N=np.array(N), nbatch=np.array(100))
    stats.dose(data, stats.spec(c['field_half']))
    return data


def _egs_job(job):
    name, stage, N = job
    run_egs(name, stage, N)
    return name


def parallel_egs(jobs):
    """One EGS5 job per condition, at most two at a time. Separate spawn processes, so the scoped
    module paths / spectrum SHA of one condition can never leak into another."""
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context('spawn')) as pool:
        for name in pool.map(_egs_job, jobs):
            print('EGS5 done', name, flush=True)


def proto_pilot(work, name):
    from .gpu import p5b_core as core
    target = Path(work)/'pilot'/name
    report = json.loads((target/'pilot.json').read_text())
    if report.get('status') != 'pass' or any(core.file_sha(target/p) != s for p, s in report['files'].items()):
        raise h.Stop('CPU pilot incomplete or changed: '+name)
    r = core.read_npz(target/'result.npz'); r.pop('metadata')
    return r, report


def lock_content(work, name):
    from .gpu import p5b_core as core
    c = h.condition(name); s = stats.spec(c['field_half'])
    left, report = proto_pilot(work, name); right = read_egs(name, 'pilot', PILOT_N)
    reference = stats.dref(stats.dose(right, s)['axis_coarse'])
    comparisons = stats.compare(stats.dose(left, s), stats.dose(right, s), reference, s)
    stats.stop_on_missing(comparisons)
    required = stats.required(comparisons, PILOT_N)
    N = h.main_size(required)
    dest = OUT/'egs5'/egs_label('pilot', name)
    hashes = dict(proto=report['files']['result.npz'], egs=core.file_sha(dest/'result.npz'),
                  egs_summary=core.file_sha(dest/'summary.json'), egs_config=core.file_sha(dest/'config.json'))
    return dict(condition=name, Dref=reference, required=required, N_main=N, spec=s, statistics=comparisons,
                pilot_hashes=hashes, pilot_label='参考（判定に使わない。本番に足さない）')


def pilot(work, data):
    from .gpu import p7
    p7.require_gates(work, data)
    for name in 'ABC':
        p7.pilot_proto(work, data, name)
    parallel_egs([(n, 'pilot', PILOT_N) for n in 'ABC'])
    locks = {}
    for name in 'ABC':
        path = OUT/f'pilot_lock_{name}.json'
        content = lock_content(work, name)
        if path.exists():
            if json.loads(path.read_text()) != json.loads(json.dumps(content)):
                raise h.Stop('pilot lock changed: '+name)
        else:
            OUT.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(content, ensure_ascii=False, indent=1))
        locks[name] = dict(Dref=content['Dref'], N_main=content['N_main'])
    return dict(status='pass', checks={n: dict(status='pass', **v) for n, v in locks.items()})


def require_lock(name, work=DEFAULT_WORK):
    """Re-derive the lock from the hashed pilot data (CPU pilot + EGS5 pilot) and compare in full."""
    path = OUT/f'pilot_lock_{name}.json'
    if not path.exists():
        raise h.Stop('pilot lock missing: '+name)
    lock = json.loads(path.read_text())
    if lock != json.loads(json.dumps(lock_content(work, name))):
        raise h.Stop('pilot lock does not re-derive from the pilot data: '+name)
    return lock


def production_egs(work, data):
    locks = {n: require_lock(n) for n in 'ABC'}
    # Longest first so the two parallel lanes finish closer together.
    order = sorted('ABC', key=lambda n: -locks[n]['N_main'])
    parallel_egs([(n, 'main', locks[n]['N_main']) for n in order])
    return dict(status='pass', checks={n: dict(status='pass', N=locks[n]['N_main']) for n in 'ABC'})


def export(work, data, destination=None):
    """Drive bundle: per-condition inputs, sources, CPU pilot references and the pilot locks."""
    import shutil, zipfile
    from .gpu import p7, p5b_core as core
    work = Path(work); idents = p7.require_gates(work, data)
    dest = Path(destination) if destination else work/'export'; dest.mkdir(parents=True, exist_ok=True)
    files = []; conditions = {}
    for name in 'ABC':
        lock = require_lock(name); _, report = proto_pilot(work, name)
        if report['identity'] != idents[name]:
            raise h.Stop('pilot identity differs: '+name)
        d = h.condition_inputs(data, name); core.save_npz(dest/f'p7_inputs_{name}.npz', d); files.append(dest/f'p7_inputs_{name}.npz')
        conditions[name] = dict(identity=idents[name], pilot=report, Dref=lock['Dref'], N_main=lock['N_main'],
                                lock_sha256=core.file_sha(OUT/f'pilot_lock_{name}.json'))
    with zipfile.ZipFile(dest/'p7_sources.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
        # Sources only; never spectrum values or transport tables (those travel in p7_inputs_*.npz on the user's Drive).
        for p in sorted(h.HERE.iterdir()):
            if p.suffix in ('.py', '.h', '.cpp', '.cu', '.ipynb'):
                z.write(p, Path('prototypes/mv_electron/gpu')/p.name)
        for rel in ('prototypes/mv_electron/analysis_p3b3.py', 'prototypes/mv_electron/analysis_p7.py', 'prototypes/__init__.py',
                    'prototypes/mv_electron/__init__.py'):
            if (ROOT/rel).exists():
                z.write(ROOT/rel, rel)
    files.append(dest/'p7_sources.zip')
    with zipfile.ZipFile(dest/'p7_reference.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name in 'ABC':
            target = Path(work)/'pilot'/name
            for p in sorted(target.rglob('*')):
                if p.is_file() and p.suffix in ('.npz', '.json'):
                    z.write(p, Path(name)/p.relative_to(target))
    files.append(dest/'p7_reference.zip')
    shutil.copyfile(h.HERE/'p7_colab.ipynb', dest/'p7_colab.ipynb')  # not hashed: Colab saves outputs into it
    manifest = dict(status='pass', files={p.name: dict(sha256=core.file_sha(p), bytes=p.stat().st_size) for p in files},
                    conditions=conditions)
    core.write_json(dest/'p7_manifest.json', manifest)
    print(str(dest), flush=True)
    return dict(status='pass', checks={n: dict(status='pass', N_main=c['N_main']) for n, c in conditions.items()})


def check_bundle_import(bundle):
    """Plan section 7: the bundle alone must be importable (sources zip extracted into an empty directory)."""
    import subprocess, sys, tempfile, zipfile
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(Path(bundle)/'p7_sources.zip') as z:
            z.extractall(tmp)
        code = 'from prototypes.mv_electron.gpu import p7_colab, p7_core; from prototypes.mv_electron import analysis_p7; print("ok")'
        p = subprocess.run([sys.executable, '-I', '-c', 'import sys; sys.path.insert(0, %r); ' % tmp + code],
                           capture_output=True, text=True, cwd=tmp)
        return dict(status='pass' if p.stdout.strip() == 'ok' else 'fail', stderr=p.stderr[-2000:])


def analyse(work, data, results=None, bundle=None):
    """Judgement per condition (plan section 8): GPU production vs EGS5 production, 11 quantities,
    fixed Dref from the EGS5 pilot, ranges from analysis_p7.spec. Report-only items alongside."""
    from .gpu import p7, p5b_core as core
    from .gpu.p7_colab import require_gate
    work = Path(work); idents = p7.require_gates(work, data)
    bundle = Path(bundle) if bundle else work/'export'; results = Path(results) if results else work/'results'
    manifest = json.loads((bundle/'p7_manifest.json').read_text()); out = dict(conditions={})
    for name in 'ABC':
        c = h.condition(name); ident = idents[name]; lock = require_lock(name, work); m = manifest['conditions'][name]
        if m['identity'] != ident or m['Dref'] != lock['Dref'] or m['N_main'] != lock['N_main'] \
                or m['lock_sha256'] != core.file_sha(OUT/f'pilot_lock_{name}.json'):
            raise h.Stop('manifest / pilot lock mismatch: '+name)
        require_gate(json.loads((results/f'gate2_{name}.json').read_text()), ident, 'gate2')
        require_gate(json.loads((results/f'gate3_{name}.json').read_text()), ident, 'gate3')
        N = lock['N_main']
        rows = [h.read_checkpoint(results/f'production_{name}'/f'batch_{b:03}.npz', ident, 'production', name, b, N) for b in range(100)]
        gpu = core.read_npz(results/f'production_{name}'/'result.npz'); meta = json.loads(str(gpu.pop('metadata')))
        expected = h.stacked(rows, 'production', name, N)
        if meta != dict(identity=ident, contract=h.contract('production', name, N), outputs=core.describe(expected)) \
                or h.exact(gpu, expected)['status'] != 'pass':
            raise h.Stop('production stacked data or metadata mismatch: '+name)
        if any(np.any(r['region_ledger']) for r in rows):
            raise h.Stop('region ledger violation: '+name)
        egs = read_egs(name, 'main', N)
        before = {p: core.file_sha(p) for p in (results/f'production_{name}'/'result.npz', OUT/'egs5'/egs_label('main', name)/'result.npz')}
        s = stats.spec(c['field_half']); gd = stats.dose(gpu, s); ed = stats.dose(egs, s)
        comparison = stats.compare(gd, ed, lock['Dref'], s)
        stats.stop_on_missing(comparison)  # plan section 3: a missing 50% crossing stops, never a verdict
        statuses = [x['status'] for group in comparison.values() for x in group]
        if '判定できない' in statuses:
            raise h.Stop('undeterminable quantity: '+name)
        verdict = ('統計不足' if '統計不足' in statuses else
                   '合格' if all(x in ('1%以内', '0.1 cm以内') for x in statuses) else '不合格')
        out['conditions'][name] = dict(status=verdict, Dref=lock['Dref'], N_main=N, spec=s, comparison=comparison,
            report_only=dict(reports=stats.reports(gd, ed, lock['Dref'], s), slab_ratios=stats.slab_ratios(gd, ed),
                             symmetry={k: stats.symmetry(v['profile'], s) for k, v in (('GPU', gd), ('EGS5', ed))}),
            production_sha256=core.file_sha(results/f'production_{name}'/'result.npz'),
            egs_sha256=core.file_sha(OUT/'egs5'/egs_label('main', name)/'result.npz'))
        if {p: core.file_sha(p) for p in before} != before:
            raise h.Stop('comparison inputs changed during analysis: '+name)
    out['note'] = '条件ごとの判定。3条件まとめての合否は出さない。厳密な同等性の検定ではない。'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'analysis.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return dict(status='done', checks={n: dict(status=v['status']) for n, v in out['conditions'].items()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('local', 'gate1', 'pilot', 'egs', 'export', 'analyse'))
    p.add_argument('--work', type=Path, default=DEFAULT_WORK)
    a = p.parse_args()
    from .gpu import p7
    from .gpu.p5a import load_inputs
    data = load_inputs(full=True)
    if a.command in ('local', 'gate1'):
        r = getattr(p7, a.command)(a.work, data)
    elif a.command == 'pilot':
        r = pilot(a.work, data)
    elif a.command == 'egs':
        r = production_egs(a.work, data)
    elif a.command == 'analyse':
        r = analyse(a.work, data)
    else:
        r = export(a.work, data)
        check = check_bundle_import(a.work/'export')
        if check['status'] != 'pass':
            raise h.Stop('bundle-only import failed: '+check['stderr'])
    print(json.dumps(dict(status=r.get('status'), checks={k: v['status'] for k, v in r['checks'].items()},
                          work=str(a.work)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
