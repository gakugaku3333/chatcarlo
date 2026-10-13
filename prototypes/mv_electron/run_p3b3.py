"""P3b-3 pilot, production and analysis; fixed rules, no additional histories."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import numpy as np
from . import analysis_p3b3 as stats, batch_kernels_p3b as kernel
from . import run_p3a as io, run_p3b2 as egs
from .batch_runner_p3u import run_batches, batch_seeds
from .batch_kernels_p3u import spectrum
from .run_p3b1 import params, REGIONS

CACHE = Path(os.environ.get('CHATCARLO_P3B_CACHE', '~/.cache/chatcarlo/p3b')).expanduser()
OUT = io.ROOT/'docs/validation/mv/p3b'
SEEDS = dict(pilot_proto=20291760, pilot_egs=20291761,
             main_proto=20291762, main_egs=20291763)
PILOT_N = 4000000


def read_json(path):
    return json.loads(Path(path).read_text())


def fixed():
    p = params()
    return io.plain(dict(sources={name:io.sha(io.CODE/name) for name in
                                 ('analysis_p3b3.py', 'run_p3b3.py')},
                         fingerprint=kernel.fingerprint(p), egs_sources=egs.source_hashes(),
                         rules=stats.RULES, regions=REGIONS, params=p, seeds=SEEDS))


def numeric(data):
    for key, value in data.items():
        a = np.asarray(value)
        if np.issubdtype(a.dtype, np.number) and not np.all(np.isfinite(a)):
            raise RuntimeError('non-finite output: '+key)


def validate(data, code, N, seed, summary=None):
    stats.dose(data)
    numeric(data)
    if np.asarray(data['seed']).shape != () or data['seed'] != seed or data['N'] != N:
        raise RuntimeError('result conditions differ')
    ledger_shape = (100, 15) if code == 'proto' else (15,)
    if np.asarray(data['region_ledger']).shape != ledger_shape or np.any(data['region_ledger']):
        raise RuntimeError('region ledger violation or shape')
    if code == 'proto':
        if data['nbatch'] != 100 or not np.array_equal(data['batch_seeds'], batch_seeds(seed, 100)):
            raise RuntimeError('invalid batch metadata')
        required_shapes = dict(totals=(100, 6), counts=(100, 16), violations=(100, 3),
                               brems=(100, 10), coarse=(100, 150), fine=(100, 600))
        for key, shape in required_shapes.items():
            if np.asarray(data[key]).shape != shape:
                raise RuntimeError('invalid shape: '+key)
        for key in kernel.KEYS:
            if key not in data or np.asarray(data[key]).shape[0] != 100:
                raise RuntimeError('missing or invalid batches: '+key)
        for b in range(100):
            if not kernel.healthy({key:data[key][b] for key in kernel.KEYS}):
                raise RuntimeError('unhealthy prototype batch: '+str(b))
        if 'batch_totals' in data and not np.array_equal(data['batch_totals'], data['totals']):
            raise RuntimeError('batch totals mismatch')
        if np.asarray(data['wall_seconds']).shape != () or data['wall_seconds'] <= 0:
            raise RuntimeError('invalid wall time')
    elif code == 'egs':
        if (summary is None or summary.get('status') != '合格' or
            not isinstance(summary.get('tests'), dict) or not summary['tests'] or
            not all(v is True for v in summary['tests'].values())):
            raise RuntimeError('EGS5 summary checks failed')
        if np.asarray(data['totals']).shape != (8,) or np.asarray(data['batchin']).shape != (100,):
            raise RuntimeError('invalid EGS5 ledger shape')
        if not np.array_equal(data['totals'], summary['ledger']):
            raise RuntimeError('EGS5 summary ledger mismatch')
    else:
        raise ValueError('unknown code')
    return data


def proto_config(N, seed):
    return io.plain(dict(params=params(), fingerprint=kernel.fingerprint(params()),
                         N=N, seed=seed, nbatch=100))


def proto_path(name):
    return OUT/'runs'/name/'result.npz'


def read_proto(name, N, seed):
    path = proto_path(name)
    if read_json(path.parent/'config.json') != proto_config(N, seed):
        raise RuntimeError('saved prototype config mismatch')
    data = io.load(path)
    if 'batch_totals' not in data:
        raise RuntimeError('saved prototype batch_totals missing')
    return validate(data, 'proto', N, seed)


def run_proto(name, N, seed):
    target = proto_path(name); config = proto_config(N, seed)
    if target.exists() or (target.parent/'config.json').exists():
        if not target.exists() or not (target.parent/'config.json').exists():
            raise RuntimeError('incomplete saved prototype result')
        return read_proto(name, N, seed)
    source = run_batches(CACHE/name, 'p3b_field', params(), N, seed, 2)
    data = io.load(source)
    validate(data, 'proto', N, seed)
    saved = {key:data[key] for key in kernel.KEYS}
    saved.update({key:data[key] for key in ('N', 'nbatch', 'seed', 'batch_seeds', 'wall_seconds')})
    saved['batch_totals'] = data['totals']
    io.save(target, **saved)
    io.write(target.parent/'config.json', config)
    return read_proto(name, N, seed)


@contextmanager
def egs_paths():
    # The unchanged P3b-2 runner uses module paths; scope overrides to this call.
    old = egs.CACHE, egs.OUT, egs.CHECKS
    egs.CACHE, egs.OUT, egs.CHECKS = CACHE, OUT, OUT/'checks_p3b2.json'
    try:
        yield
    finally:
        egs.CACHE, egs.OUT, egs.CHECKS = old


def egs_config(N, seed, edges, cum):
    return io.plain(dict(N=N, seed=seed, E=10., IUNRST=0, noflight=0, thickness=30.,
        batches=100, input=egs.r.pegs_input(0), user_code='p3b_egs', field=[5., 100.],
        regions=list(map(float, REGIONS)), egs_input_hashes=egs.egs_hashes(),
        source_hashes=egs.source_hashes(), **egs.r.spec_config(edges, cum, egs.r.SPECTRUM_SHA)))


def egs_state(name, config):
    work = CACHE/'egs5'/name; dest = OUT/'egs5'/name
    result, summary = dest/'result.npz', dest/'summary.json'
    if result.exists() != summary.exists():
        raise RuntimeError('partial EGS5 result')
    if result.exists():
        if read_json(dest/'config.json') != config:
            raise RuntimeError('completed EGS5 config mismatch')
        if work.exists() and read_json(work/'config.json') != config:
            raise RuntimeError('EGS5 work config mismatch')
        return 'complete'
    if not work.exists() and not dest.exists():
        return 'new'
    if not (work/'checkpoint.bin').is_file():
        raise RuntimeError('interrupted EGS5 job without checkpoint')
    egs.verify_config(work, dest, config)
    egs.verify_checkpoint(work, config)
    return 'resume'


def read_egs(name, N, seed):
    dest = OUT/'egs5'/name
    return validate(io.load(dest/'result.npz'), 'egs', N, seed, read_json(dest/'summary.json'))


def run_reference(name, N, seed, allow_launch=True):
    edges, cum = spectrum(params())
    config = egs_config(N, seed, edges, cum)
    with egs_paths():
        egs.refuse_failed()
        state = egs_state(name, config)
        if state != 'complete':
            if not allow_launch:
                raise RuntimeError('EGS5 result not complete')
            kw = dict(field=[5., 100.], regions=REGIONS)
            if state == 'resume':
                kw['resume'] = True
            egs.run_egs(name, N, seed, edges, cum, **kw)
            if egs_state(name, config) != 'complete':
                raise RuntimeError('EGS5 did not complete')
    return read_egs(name, N, seed)


def pilot_hashes():
    paths = dict(proto=proto_path('pilot_proto'),
                 proto_config=proto_path('pilot_proto').parent/'config.json',
                 egs=OUT/'egs5/pilot_egs/result.npz',
                 egs_summary=OUT/'egs5/pilot_egs/summary.json',
                 egs_config=OUT/'egs5/pilot_egs/config.json')
    return {key:io.sha(path) for key, path in paths.items()}


def lock_content(left, right):
    reference = stats.dref(stats.dose(right)['axis_coarse'])
    comparisons = stats.compare(stats.dose(left), stats.dose(right), reference)
    if any(v['status'] == '判定できない' for v in comparisons['J3']):
        raise RuntimeError('pilot crossing unavailable; stop')
    size = stats.main_size(comparisons)
    return io.plain(dict(fixed=fixed(), pilot_hashes=pilot_hashes(), Dref=reference,
                         statistics=comparisons, pilot_label='参考（判定に使わない。本番に足さない）', **size))


def require_lock():
    path = OUT/'pilot_lock_p3b3.json'
    if not path.is_file():
        raise RuntimeError('pilot lock missing')
    lock = read_json(path)
    if lock.get('status') != '合格':
        raise RuntimeError('pilot lock not passed; 相談')
    if lock.get('fixed') != fixed() or lock.get('pilot_hashes') != pilot_hashes():
        raise RuntimeError('pilot lock hashes or conditions changed')
    left = read_proto('pilot_proto', PILOT_N, SEEDS['pilot_proto'])
    right = run_reference('pilot_egs', PILOT_N, SEEDS['pilot_egs'], allow_launch=False)
    # Recompute statistics, Dref and N_main, including the 1.5 history margin.
    if lock != lock_content(left, right):
        raise RuntimeError('pilot lock statistics or N_main changed')
    return lock


def pilot():
    if (OUT/'pilot_lock_p3b3.json').exists():
        return require_lock()
    before = fixed()
    left = run_proto('pilot_proto', PILOT_N, SEEDS['pilot_proto'])
    right = run_reference('pilot_egs', PILOT_N, SEEDS['pilot_egs'])
    if fixed() != before:
        raise RuntimeError('pilot inputs changed')
    lock = lock_content(left, right)
    io.write(OUT/'pilot_lock_p3b3.json', lock)
    return lock


def production():
    lock = require_lock(); N = lock['N_main']
    run_proto('main_proto', N, SEEDS['main_proto'])
    require_lock()
    run_reference('main_egs', N, SEEDS['main_egs'])
    require_lock()
    return dict(N_main=N, status='完了')


def legacy_report(data, code, summary=None):
    if code == 'proto':
        totals = data['batch_totals'].sum(axis=0)
        incoming, deposited = totals[:2]
        seconds = float(data['wall_seconds'])
    else:
        totals = data['totals']; incoming, deposited = totals[:2]
        seconds = float(summary['wall_seconds'])
    if not np.isfinite(incoming) or incoming <= 0:
        raise RuntimeError('invalid total incoming energy')
    return dict(deposited_over_incoming=float(deposited/incoming), totals=totals,
                legacy_coarse=data['coarse'], legacy_fine=data['fine'],
                legacy_unit='MeV per batch' if code == 'proto' else 'Gy cm2 per history',
                microseconds_per_history=seconds/float(data['N'])*1e6,
                timing_kind='wall', CPU_seconds=None if summary is None else summary['CPU_seconds'],
                checks={key:data[key] for key in ('counts', 'violations', 'maxima', 'brems', 'region_ledger')}
                if code == 'proto' else summary)


def figures(left, right, reference, comparisons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    depth = (np.arange(120)+.5)*.25
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
    for code, data in (('prototype', left), ('EGS5', right)):
        ax[0].plot(depth, data['axis_coarse'].mean(0), label=code)
    ax[0].set_ylabel('Gy / incident photon'); ax[0].legend()
    ax[1].plot(depth, (left['axis_coarse'].mean(0)-right['axis_coarse'].mean(0))*100/reference)
    for value in (-1, 1):
        ax[1].axhline(value, color='gray', linestyle='--')
    for row in comparisons['J1']:
        ax[1].errorbar(np.mean(row['interval_cm']), row['delta'], yerr=1.96*row['SEM'], fmt='o')
    for a in ax:
        a.axvspan(0, 5, color='gray', alpha=.15, label='not judged (0–5 cm)')
    ax[1].legend(); ax[1].set(xlabel='depth (cm)', ylabel='difference (% Dref)')
    fig.tight_layout(); fig.savefig(OUT/'pdd_p3b3.png'); plt.close(fig)
    fig, ax = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    radius = (np.arange(36)+.5)*.25
    for j, depth in enumerate(stats.DEPTHS):
        for code, data in (('prototype', left), ('EGS5', right)):
            curve = stats.fold(data['profile']).mean(0)[j]
            ax[0, j].plot(radius, curve, label=code)
            position = stats.crossing(curve)
            if position is not None:
                ax[0, j].axvline(position, linestyle='--')
        ax[0, j].set_title(f'{depth} cm'); ax[0, j].legend()
        ax[1, j].plot(radius, (stats.fold(left['profile']).mean(0)[j]-stats.fold(right['profile']).mean(0)[j])*100/reference)
        ax[1, j].set(xlabel='|x| (cm)', ylabel='difference (% Dref)')
    ax[0, 0].set_ylabel('Gy / incident photon')
    fig.tight_layout(); fig.savefig(OUT/'profiles_p3b3.png'); plt.close(fig)
    fig, ax = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for i, (code, data) in enumerate((('prototype', left), ('EGS5', right))):
        curves = stats.directions(data['profile']).mean(0)
        for j, depth in enumerate(stats.DEPTHS):
            for k, name in enumerate(('+x', '-x', '+y', '-y')):
                ax[i, j].plot(radius, curves[k, j], label=name)
            ax[i, j].set_title(f'{code}, {depth} cm'); ax[i, j].legend()
            ax[i, j].set(xlabel='radius (cm)', ylabel='Gy / incident photon')
    fig.tight_layout(); fig.savefig(OUT/'symmetry_p3b3.png'); plt.close(fig)


def analyse():
    lock = require_lock(); N = lock['N_main']
    # No comparisons or launching until both production results are present.
    paths = [proto_path('main_proto'), OUT/'egs5/main_egs/result.npz',
             OUT/'egs5/main_egs/summary.json']
    if not all(path.is_file() for path in paths):
        raise RuntimeError('both production results required')
    hashes = {str(p.relative_to(OUT)):io.sha(p) for p in
              paths+[OUT/'pilot_lock_p3b3.json']}
    left = read_proto('main_proto', N, SEEDS['main_proto'])
    right = run_reference('main_egs', N, SEEDS['main_egs'], allow_launch=False)
    ld, rd = stats.dose(left), stats.dose(right)
    symmetry = dict(proto=stats.symmetry(ld['profile']), egs=stats.symmetry(rd['profile']))
    warnings = []
    for code, values in symmetry.items():
        for row in values['contrasts']:
            if row['status'] == '偏りを検出':
                warnings.append(f"対称性に偏りを検出（{code}, {row['depth_cm']} cm, {row['contrast']}, z={row['z']}）。原因は未調査")
    comparisons = stats.compare(ld, rd, lock['Dref'])
    result = dict(input_hashes=hashes, Dref=lock['Dref'], N_main=N,
                  conclusion_prefix=warnings, symmetry=symmetry, comparisons=comparisons,
                  report_only=stats.reports(ld, rd, lock['Dref']), caveats=stats.CAVEATS,
                  ledgers=dict(proto=legacy_report(left, 'proto'),
                               egs=legacy_report(right, 'egs', read_json(paths[2]))))
    require_lock()
    if hashes != {str(p.relative_to(OUT)):io.sha(p) for p in paths+[OUT/'pilot_lock_p3b3.json']}:
        raise RuntimeError('analysis inputs changed')
    io.write(OUT/'analysis_p3b3.json', result)
    figures(ld, rd, lock['Dref'], comparisons)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('pilot', 'production', 'analyse'))
    command = parser.parse_args().command
    result = globals()[command]()
    print(json.dumps(io.plain(result), ensure_ascii=False, allow_nan=False))
    if result.get('status') == '相談':
        raise SystemExit('N_main exceeds 160 million; consult user before production')


if __name__ == '__main__':
    main()
