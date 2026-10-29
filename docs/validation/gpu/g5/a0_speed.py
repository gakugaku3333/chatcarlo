"""A0 VM measurement driver. Imports frozen engines; never copies transport."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def load_configs(path):
    cfg = json.loads(Path(path).read_text())
    if cfg['format'] != 'g5-a0-v1':
        raise ValueError('unknown config format')
    tasks = [f"{c['id']}:{n}" for c in cfg['configurations'] for n in c['sizes']]
    if len(cfg['configurations']) != 6 or len(tasks) != len(set(tasks)):
        raise ValueError('six distinct configurations/tasks required')
    if cfg['warmup_order'] != tasks or cfg['round_orders'] != [tasks[i:] + tasks[:i] for i in (0, 2, 4)]:
        raise ValueError('predeclared rotated orders required')
    for c in cfg['configurations']:
        if c['sizes'] != list(dict.fromkeys([c['production_n'], 100000])) or c['start'] != 0:
            raise ValueError('production/100k sizes and start required')
    return cfg, sha(path)


class EventDelegate:
    """Transparent native-call delegation, including kwargs and return value."""
    def __init__(self, native, cp):
        self.native, self.cp, self.seconds = native, cp, []

    def __call__(self, *args, **kwargs):
        first, last = self.cp.cuda.Event(), self.cp.cuda.Event()
        first.record()
        result = self.native(*args, **kwargs)
        last.record()
        last.synchronize()
        self.seconds.append(float(self.cp.cuda.get_elapsed_time(first, last)) / 1000)
        return result


def resident_blocks(attrs, threads, props):
    """sm_75: 256 registers/warp, 256 bytes/shared allocation, 16 blocks/SM."""
    if threads > int(attrs['max_threads_per_block']):
        raise ValueError('kernel thread limit')
    registers = math.ceil(int(attrs['num_regs']) * 32 / 256) * 256 * math.ceil(threads / 32)
    shared = math.ceil(int(attrs['shared_size_bytes']) / 256) * 256
    limits = dict(blocks=16, threads=int(props['maxThreadsPerMultiProcessor']) // threads)
    if registers:
        limits['registers'] = int(props['regsPerMultiprocessor']) // registers
    if shared:
        limits['shared'] = int(props['sharedMemPerMultiprocessor']) // shared
    return dict(blocks_per_sm=min(limits.values()), registers_per_block=registers,
                shared_bytes_per_block=shared, limits=limits)


def resources(cp, kernel, c):
    attrs = {k: int(v) for k, v in kernel.attributes.items()}
    threads = c['launch']['threads']
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    return dict(attributes=attrs, launch=c['launch'], manual=resident_blocks(attrs, threads, props),
                occupancy_api_blocks_per_sm=int(cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(
                    kernel.kernel.ptr, threads, 0)))


def construct(cp, c, bundles):
    path = Path(bundles) / c['bundle']
    if c['kind'] == 'diagnostic':
        from chatcarlo.gpu.bundle import load
        from chatcarlo.gpu.core import Core
        a, source, spectrum, settings, manifest = load(path)
        if c['grid']:
            from chatcarlo.gpu.g2_core import Core, grid_from
            obj = Core(a, 'cuda', grid=grid_from(c['grid']))
            kernel = obj.native
            obj.native = EventDelegate(kernel, cp)
            timer = obj.native
        else:
            obj = Core(a, 'cuda')
            kernel = obj.fn
            obj.fn = EventDelegate(kernel, cp)
            timer = obj.fn
        if settings['scene_sha256'] != c['scene_model_sha256']:
            raise ValueError('bundle scene mismatch')
        if settings['fluorescence'] != c['flags']['fluorescence']:
            raise ValueError('bundle fluorescence mismatch')
        expected_detector = c['detector']
        if expected_detector is None:
            if a['detector'].size:
                raise ValueError('unexpected detector')
        else:
            d = expected_detector
            expected = np.r_[d['center_cm'], d['normal'], d['u_axis'], np.cross(d['normal'], d['u_axis']), d['size_cm'], d['shape']]
            if not np.array_equal(a['detector'], expected):
                raise ValueError('detector mismatch')
        return dict(obj=obj, kernel=kernel, timer=timer, source=source, spectrum=spectrum)
    if c['kind'] == 'p7':
        from prototypes.mv_electron.gpu import p7_colab as p7
        data, _, manifest = p7.load_bundle(path)
        contract = p7.h.contract('production', c['condition'], manifest['conditions'][c['condition']]['N_main'])
        cond = p7.h.condition(c['condition'])
        if (contract['n'], contract['seed'], cond['field_half'], cond['ssd']) != (c['production_n'], c['seed'], c['field_half_cm'], c['ssd_cm']):
            raise ValueError('P7 production configuration mismatch')
        obj = p7.GPU(cp, data[c['condition']], c['condition'], blocks=c['launch']['blocks'])
        return dict(obj=obj, kernel=obj.kernels['p7_kernel'])
    from prototypes.mv_electron.gpu import p8g2_colab as p8
    _, materials, arrays = p8.load(path)
    scoring = c['scoring']
    for name, expected in [('faces', scoring['faces_cm']), ('ids', scoring['material_ids']),
                           ('rho', scoring['layer_density_g_cm3']), ('sizes', scoring['sizes'])]:
        if not np.array_equal(arrays[name], expected):
            raise ValueError('P8g-2 scoring configuration mismatch: ' + name)
    launch = c['launch']
    obj = p8.GPU(cp, materials, arrays, capacity=launch['capacity'], check=launch['check'],
                 fault=launch['fault'], blocks=launch['blocks'], threads=launch['threads'])
    return dict(obj=obj, kernel=obj.kernel)


def diagnostic_health(measured, replay, energy):
    # Both Core.run calls already validate shape, finiteness, counters and deposits.
    # The absorbed ending retains the last incident energy in d[:,0].
    # Only escaped/detected photons carry energy out of transport.
    remaining = np.where(replay['i'][:, 3] == 0, 0., replay['d'][:, 0])
    deposited = replay['d'][:, 1]
    error = np.abs(deposited + remaining - energy)
    maximum = float(error.max(initial=0))
    if np.any(error > 1e-10 * np.maximum(energy, 1)):
        raise ValueError('diagnostic per-history energy balance')
    if not np.allclose(measured['deposits'], replay['deposits'], rtol=1e-10, atol=1e-8) or not np.array_equal(measured['endings'], replay['endings']):
        raise ValueError('diagnostic replay aggregate mismatch')
    return dict(status='pass', method='untimed same-input record replay', max_energy_error_keV=maximum,
                histories=len(energy), incident_keV=float(energy.sum()),
                deposited_keV=float(deposited.sum()), remaining_keV=float(remaining.sum()))


def measure(c, state, n):
    obj = state['obj']
    flags = c['flags']
    if c['kind'] == 'diagnostic':
        from chatcarlo.gpu.source_host import sample_source_photons
        inputs = sample_source_photons(state['source'], n, np.random.default_rng(c['source_seed']), state['spectrum'])
        state['timer'].seconds.clear()
        begin = time.perf_counter()
        result, _ = obj.run(*inputs, seed=c['seed'], start=c['start'], **flags)
        call_s = time.perf_counter() - begin
        if len(state['timer'].seconds) != 1:
            raise ValueError('expected one native launch')
        kernel_s = state['timer'].seconds[0]
        replay_flags = dict(flags, record=True, observe=False)
        replay = obj.run(*inputs, seed=c['seed'], start=c['start'], **replay_flags)
        health = diagnostic_health(result, replay, inputs[2])
    elif c['kind'] == 'p7':
        from prototypes.mv_electron.gpu import p7_colab as p7
        begin = time.perf_counter()
        result, timing = obj.measure(n, c['seed'], start=c['start'], **flags)
        call_s = time.perf_counter() - begin
        kernel_s = timing['kernel_ms'] / 1000
        health = p7.h.health(result)
        p7.core.require(health, 'A0 P7 health')
        health = dict(status='pass', report=health)
    else:
        from prototypes.mv_electron.gpu.p8g2_core import check_visits
        from prototypes.mv_electron.gpu.p8f_colab import run_health
        begin = time.perf_counter()
        result, _, visits, _ = obj.run(n, c['seed'], start=c['start'], **flags)
        call_s = time.perf_counter() - begin
        kernel_s = float(result['seconds'])
        check_visits(visits)
        if not run_health(result):
            raise ValueError('A0 P8g-2 health')
        health = dict(status='pass', visits='all exactly one', run_health=True)
    return dict(kernel_s=kernel_s, call_s=call_s, us_per_history=kernel_s * 1e6 / n, health=health)


def summarize(rows, cfg, digest, session):
    if any(r['config_sha256'] != digest or r['session_id'] != session or r['health']['status'] != 'pass' for r in rows):
        raise ValueError('mixed configuration/session or failed health')
    expected = {(f"{c['id']}:{n}", phase, repeat) for c in cfg['configurations'] for n in c['sizes']
                for phase, repeats in [('warmup', [-1]), ('valid', range(3))] for repeat in repeats}
    actual = [(r['task'], r['phase'], r['repeat']) for r in rows]
    if any(not math.isfinite(r['call_s']) or r['call_s'] <= 0 for r in rows):
        raise ValueError('invalid call timing')
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('incomplete/duplicate measurements')
    spreads, medians = {}, []
    for c in cfg['configurations']:
        for n in c['sizes']:
            values = [r['kernel_s'] for r in rows if r['task'] == f"{c['id']}:{n}" and r['phase'] == 'valid']
            if any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError('invalid timing')
            median = statistics.median(values)
            medians.append(dict(configuration=c['id'], n=n, median_us=median * 1e6 / n))
            if n == c['production_n']:
                spreads[c['id']] = (max(values) - min(values)) / median
    worst = max(spreads, key=spreads.get)
    s = spreads[worst]
    ratio = 1.05 if s <= .025 else 1 + 2 * s if s <= .05 else None
    return dict(status='baseline-defined' if ratio else 'baseline-undefined; repeat all six in a new session; stop if second attempt',
                s=s, worst_configuration=worst, spreads=spreads, tolerance_ratio=ratio, medians=medians)


def read_results(path, configs):
    cfg, digest = load_configs(configs)
    value = json.loads(Path(path).read_text())
    summary = summarize(value['measurements'], cfg, digest, value['session_id'])
    if value['config_sha256'] != digest or value['summary'] != summary or value['status'] != 'complete':
        raise ValueError('result summary/config/status mismatch')
    return value


def fingerprints(bundles):
    """Transfer identity includes drivers. Configuration identity excludes them."""
    from chatcarlo.gpu.bundle import verify
    from prototypes.mv_electron.gpu.p7_colab import load_bundle
    from prototypes.mv_electron.gpu.p8g2_colab import load
    bundles = Path(bundles)
    manifests = {name: verify(bundles / name) for name in ('g2b', 'g4')}
    manifests['p7'] = load_bundle(bundles / 'p7')[2]
    manifests['p8g2'] = load(bundles / 'p8g2')[0]
    transfer, identity = {}, {}
    for name in manifests:
        root = bundles / name
        transfer[name] = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
        # Numeric input files plus the compiled source closure, never Python drivers.
        identity[name] = {p: digest for p, digest in transfer[name].items()
                          if p.endswith(('.h', '.cpp', '.cu')) or
                          (Path(p).name in ('tables.npz', 'geometry.npz', 'g2_tables.npz', 'spectrum.npz', 'source.json', 'settings.json', 'p8f_inputs.npz', 'water_bundle.npz', 'lung_bundle.npz', 'bone_bundle.npz', 'lead_bundle.npz') or p.startswith('p7_inputs_'))}
    from chatcarlo.gpu import core as diagnostic
    from prototypes.mv_electron.gpu import p7_core, p8g2_core
    # The expanded CUDA source captures transitive includes and no launch tools.
    identity['compiled_sources'] = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in
                                    [('diagnostic', diagnostic.cuda_source()), ('p7', p7_core.source()), ('p8g2', p8g2_core.cuda_source())]}
    return dict(transfer=transfer, configuration=identity)


def profile(args, cfg):
    import cupy as cp
    c = next(c for c in cfg['configurations'] if c['id'] == args.profile)
    # Exactly one target launch; constructor initialization kernels have other names.
    state = construct(cp, c, args.bundles)
    measure(c, state, c['production_n'])


def run(args):
    import cupy as cp
    cfg, digest = load_configs(args.configs)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'measurements.json').exists():
        raise ValueError('fresh output directory required; sessions must not be combined')
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
    if 'T4' not in name or (props['major'], props['minor']) != (7, 5):
        raise ValueError('T4 sm_75 required')
    session = str(uuid.uuid4())
    record = dict(status='running', session_id=session, config_sha256=digest, gpu=name,
                  fingerprints=fingerprints(args.bundles),
                  control_files={p.name: sha(p) for p in (Path(__file__), HERE / 'a0_bootstrap.py', Path(args.configs))},
                  construction=[], resources=[], measurements=[], ncu=[])
    def save():
        write_json(output / 'measurements.json', record)
    states = {}
    try:
        # All six objects compile before any warmup.
        for c in cfg['configurations']:
            begin = time.perf_counter()
            states[c['id']] = construct(cp, c, args.bundles)
            cp.cuda.Device().synchronize()
            record['construction'].append(dict(configuration=c['id'], seconds=time.perf_counter() - begin,
                                               config_sha256=digest, session_id=session))
            record['resources'].append(dict(configuration=c['id'], config_sha256=digest, session_id=session,
                                             **resources(cp, states[c['id']]['kernel'], c)))
            save()
        by_id = {c['id']: c for c in cfg['configurations']}
        for repeat, order in [(-1, cfg['warmup_order']), *enumerate(cfg['round_orders'])]:
            for task in order:
                cid, size = task.split(':')
                row = dict(task=task, configuration=cid, n=int(size), phase='warmup' if repeat == -1 else 'valid',
                           repeat=repeat, config_sha256=digest, session_id=session,
                           seed=by_id[cid]['seed'], start=by_id[cid]['start'])
                try:
                    row.update(measure(by_id[cid], states[cid], int(size)))
                except Exception as exc:
                    row['health'] = dict(status='fail', error=str(exc))
                    record['measurements'].append(row)
                    raise
                record['measurements'].append(row)
                save()
        record['summary'] = summarize(record['measurements'], cfg, digest, session)
        # Profiles run only after ordinary timing, in separate processes.
        for c in cfg['configurations']:
            kernel_name = dict(g2b='g2_histories_kernel', g4='g1_histories_kernel', p8g2='p8g2_kernel').get(c['id'], 'p7_kernel')
            stem = output / (c['id'] + '_full')
            command = ['ncu', '--set', 'full', '--target-processes', 'all', '--kernel-name', kernel_name,
                       '--launch-count', '1', '--force-overwrite', '--export', str(stem), '--csv',
                       '--log-file', str(stem.with_suffix('.csv')), sys.executable, str(Path(__file__).resolve()),
                       '--configs', str(Path(args.configs).resolve()), '--bundles', str(Path(args.bundles).resolve()),
                       '--output', str(output.resolve()), '--profile', c['id']]
            receipt = dict(configuration=c['id'], launch=c['launch'], kernel=kernel_name, command=command,
                           config_sha256=digest, session_id=session, status='未取得', returncode=None, raw={})
            try:
                result = subprocess.run(command, timeout=1800, capture_output=True, text=True)
                receipt.update(returncode=result.returncode, stderr=result.stderr[-4000:])
                for p in (stem.with_suffix('.csv'), stem.with_suffix('.ncu-rep')):
                    if p.exists():
                        receipt['raw'][p.name] = sha(p)
                if result.returncode == 0 and len(receipt['raw']) == 2:
                    receipt['status'] = '取得'
            except (OSError, subprocess.TimeoutExpired) as exc:
                receipt['error'] = str(exc)
            record['ncu'].append(receipt)
            save()
        record['status'] = 'complete'
        save()
        (output / 'G5A0_DONE').write_text(session + '\n')
    except Exception as exc:
        record.update(status='failed; no baseline', error=str(exc))
        save()
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--configs', type=Path, default=HERE / 'a0_configs.json')
    p.add_argument('--bundles', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--profile')
    args = p.parse_args()
    cfg, _ = load_configs(args.configs)
    (profile(args, cfg) if args.profile else run(args))


if __name__ == '__main__':
    main()
