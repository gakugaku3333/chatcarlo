"""Local P5b bundle/local/compare CLI. P5a is imported only inside local functions."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from . import p5b_core as core

WORK = Path('~/.cache/chatcarlo/p5/p5b').expanduser()
SAVED = Path('~/.cache/chatcarlo/p5/p5a/philox').expanduser()


class CPU:
    def __init__(self, work, data, fault=0):
        from . import p5a
        core.fingerprints()
        core.check_inputs(data)
        if fault not in (0, 1, 2, 3):
            raise ValueError('unknown transport fault')
        work = Path(work)
        work.mkdir(parents=True, exist_ok=True)
        compiler = shutil.which('clang++')
        if compiler is None:
            raise RuntimeError('clang++ not available')
        self.flags = ['-O2', '-ffp-contract=off', '-std=c++11', '-shared', '-fPIC',
                      '-DP5_RNG_PHILOX', f'-DP5_TRANSPORT_FAULT={fault}', '-DP5_STACK_CAP=64']
        self.compiler = subprocess.check_output([compiler, '--version'], text=True).splitlines()[0]
        self.identity = core.sha(json.dumps(dict(flags=self.flags, compiler=self.compiler,
            sources={k: core.file_sha(core.HERE/k) for k in
                     (*core.FROZEN, 'p5b_wrap.h', 'p5b_cpu.cpp')}), sort_keys=True).encode())
        library = work/('p5b_'+self.identity[:16]+('.dylib' if sys.platform == 'darwin' else '.so'))
        build = subprocess.run([compiler, *self.flags, str(core.HERE/'p5b_cpu.cpp'), '-o', str(library)],
                               capture_output=True, text=True)
        (work/('p5b_build_'+self.identity[:16]+'.log')).write_text(build.stdout+build.stderr)
        build.check_returncode()
        self.data, self.tables = data, p5a.CPU.make_tables(data)
        self.lib = ct.CDLL(str(library))
        d = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        i = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
        i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
        u = np.ctypeslib.ndpointer(dtype=np.uint64, flags='C_CONTIGUOUS')
        u32 = np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS')
        t = ct.POINTER(p5a.Tables)
        signatures = dict(p5b_per_history=[t, ct.c_uint64, ct.c_uint64, ct.c_int, ct.c_int,
                                           d, d, d, d, d, i, i, i, i, i32, i, d, i, i],
                          p5b_fields=[t, ct.c_int, d, i32],
                          p5b_functions=[t, ct.c_int, ct.c_uint64, d, ct.c_int, d, u],
                          p5b_philox=[u, ct.c_int, u32, d],
                          p5b_rows=[ct.c_uint64, ct.c_int, i32, u])
        for name, signature in signatures.items():
            f = getattr(self.lib, name)
            f.argtypes, f.restype = signature, None

    def per_history(self, n, seed=core.SEED, start=0, maxsteps=core.MAXSTEPS):
        core.check_range(n, start, seed, maxsteps)
        r = core.empty_output(core.history_shapes(n))
        r['maxdepth'] = np.zeros(n, np.int32)
        self.lib.p5b_per_history(ct.byref(self.tables), seed, start, n, maxsteps,
                                *(r[k] for k in core.HISTORY_ORDER))
        r['maxdepth'] = r['maxdepth'].astype(np.int64)
        core.validate_output(r, core.history_shapes(n))
        return r

    def fields(self):
        fields = np.empty(len(core.field_reference(self.data)))
        dims = np.empty(11, np.int32)
        self.lib.p5b_fields(ct.byref(self.tables), len(fields), fields, dims)
        return dict(fields=fields, dimensions=dims)

    def function(self, name, x, seed):
        from . import p5a
        core.check_range(len(x), seed=seed)
        x = p5a.validate(name, x, self.data)
        out = np.empty((len(x), core.FUNCTIONS[name]))
        calls = np.empty(len(x), np.uint64)
        self.lib.p5b_functions(ct.byref(self.tables), list(core.FUNCTIONS).index(name), seed,
                              x, len(x), out, calls)
        if np.any(calls >= 2**63):
            raise core.Stop('function call counter exceeds int64')
        return out, calls.astype(np.int64)

    def philox(self, triples):
        if triples.dtype != np.uint64 or triples.ndim != 2 or triples.shape[1] != 3:
            raise ValueError('invalid Philox triples')
        triples = np.ascontiguousarray(triples)
        words, values = np.empty((len(triples), 4), np.uint32), np.empty(len(triples))
        self.lib.p5b_philox(triples, len(triples), words, values)
        return dict(philox_words=words, philox_values=values)

    def rows(self, n, start):
        core.check_range(n, start)
        rows, histories = np.empty(n, np.int32), np.empty(n, np.uint64)
        self.lib.p5b_rows(start, n, rows, histories)
        return rows, histories


def load_local_inputs(saved):
    from . import p5a
    core.fingerprints()
    config = json.loads((Path(saved)/'config.json').read_text())
    flags = ['-O2', '-ffp-contract=off', '-std=c++11', '-shared', '-fPIC',
             '-DP5_RNG_PHILOX', '-DP5_TRANSPORT_FAULT=0', '-DP5_STACK_CAP=64']
    if (config['sources'] != core.FROZEN or config['flags'] != flags or config['rng'] != 'philox'
        or config['seed'] != core.SEED or config['batches'] != 100 or config['per_batch'] != 100000):
        raise core.Stop('saved P5a configuration mismatch')
    data = p5a.load_inputs(full=True)
    core.check_inputs(data, config['arrays'])
    if len(data['cum']) != 20:
        raise core.Stop('fixed spectrum dimension mismatch')
    return data, config


def saved_batch(saved, b, config):
    path = Path(saved)/f'batch_{b:03}.npz'
    r = core.read_npz(path)
    expected_sha = core.sha(json.dumps(config, sort_keys=True).encode())
    if (str(r['config_sha256']) != expected_sha or int(r['N']) != 100000
        or int(r['seed']) != core.SEED or int(r['start']) != 100000*b
        or int(r['end']) != 100000*(b+1) or str(r['rng']) != 'philox'):
        raise core.Stop('saved CPU batch metadata mismatch')
    r = {k: r[k] if k != 'nan_count' else np.array(0, np.int64) for k in core.RUN_ORDER}
    core.validate_output(r, core.run_shapes(20))
    core.require(core.health(r), 'saved CPU health')
    return r


def function_inputs(data):
    from . import p5a
    critical = [.01, .02, .02/(1-2e-7), 1., *data['thresholds']]
    electron_nodes = np.r_[p5a.physics.ET, data['ts'], data['rates'][:, 0]]
    result = {}
    for name in ('collision', 'hard', 'coefficients', 'table_rates'):
        if name == 'coefficients':
            nodes, lo, hi = data['pes'], data['pes'][0], data['pes'][-1]
        elif name == 'table_rates':
            nodes = data['rates'][:, 0]
            lo, hi = nodes[0], nodes[-1]
        else:
            nodes, lo, hi = electron_nodes, np.nextafter(.01, 0), 20.
        result[name] = p5a.energy_points(nodes, lo, hi, critical)
    result['sample_W'] = p5a.logarithmic_points(np.nextafter(.02, np.inf), 10., 1000)
    result['gs_mu_cached'] = np.column_stack((p5a.logarithmic_points(data['ts'][0], data['ts'][-1], 1000),
                                             np.resize(data['ks'], 1000)))
    return result


def reference_checks(cpu):
    from . import p5a0
    r = cpu.fields()
    r['philox_triples'] = p5a0.rng_inputs()
    r.update(cpu.philox(r['philox_triples']))
    # Each random probe uses the same ten seeds as A1; each input starts afresh.
    for name, x in function_inputs(cpu.data).items():
        r['input_'+name] = x
        seeds = range(20291940, 20291950) if name in ('sample_W', 'gs_mu_cached') else [20291940]
        values, calls = zip(*(cpu.function(name, x, seed) for seed in seeds))
        r['value_'+name] = np.concatenate(values)
        r['calls_'+name] = np.concatenate(calls)
    for n in (20000, 100000):
        rows, histories = cpu.rows(n, 0)
        r['rows_'+str(n)], r['histories_'+str(n)] = rows, histories
    return r


def drive_directory():
    root = Path('~/Library/CloudStorage').expanduser()
    candidates = [p/'マイドライブ/viveMonte/p5/p5b' for p in root.glob('GoogleDrive-*@gmail.com')]
    if len(candidates) != 1:
        raise core.Stop('specify --drive with the gmail Drive synchronized p5b directory')
    return candidates[0]


def bundle(work, saved, drive=None):
    from . import p5a
    data, config = load_local_inputs(saved)
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    cpu = CPU(work/'build', data)
    reference = reference_checks(cpu)
    reference.update({'history_'+k: v for k, v in cpu.per_history(20000).items()})
    expected = [f'batch_{b:03}.npz' for b in range(100)]
    if sorted(p.name for p in Path(saved).glob('batch*.npz')) != expected:
        raise core.Stop('saved CPU batches missing/duplicate/out of range')
    if any('pending' in p.name for p in Path(saved).iterdir()):
        raise core.Stop('unfinished saved CPU file')
    batches = [saved_batch(saved, b, config) for b in range(100)]
    from prototypes.mv_electron.run_coupled_p2 import MEV_TO_GY_G
    reference['conversion'] = np.array(MEV_TO_GY_G)
    reference['Dref'] = np.array(p5a.dref())
    reference['c4_blocks'] = np.stack([core.deep_blocks(r['batches'].sum(axis=0), 100000, MEV_TO_GY_G)
                                      for r in batches])
    for k in ('nspec', 'counts', 'step_batches'):
        reference['c4_'+k] = np.stack([r[k] for r in batches])
    reference['saved_config_sha256'] = np.array(core.file_sha(Path(saved)/'config.json'))
    core.save_npz(work/'p5b_inputs.npz', data)
    core.save_npz(work/'p5b_reference.npz', reference)
    files = {name: dict(bytes=(work/name).stat().st_size, sha256=core.file_sha(work/name))
             for name in ('p5b_inputs.npz', 'p5b_reference.npz')}
    manifest = dict(sources=core.fingerprints(), source_sha256=core.sha(core.source().encode()),
                    arrays=core.describe(data), reference_arrays=core.describe(reference), files=files,
                    cpu=dict(compiler=cpu.compiler, flags=cpu.flags, identity=cpu.identity))
    core.write_json(work/'p5b_manifest.json', manifest)
    drive = Path(drive) if drive is not None else drive_directory()
    drive.mkdir(parents=True, exist_ok=True)
    for name in (*files, 'p5b_manifest.json'):
        if (work/name).resolve() != (drive/name).resolve():
            shutil.copyfile(work/name, drive/name)
        if (work/name).stat().st_size != (drive/name).stat().st_size or core.file_sha(work/name) != core.file_sha(drive/name):
            raise core.Stop('Drive copy verification failed: '+name)
    print(json.dumps(dict(files=files, source_sha256=manifest['source_sha256'], drive=str(drive)), indent=2))
    return manifest


def local(work, saved):
    from . import p5a
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    result = dict(checks={}, overall='incomplete', source_sha256=core.sha(core.source().encode()))
    try:
        data, config = load_local_inputs(saved)
        core.record(result, 'L0', core.verdict(True, sources=core.fingerprints(), arrays=core.describe(data)))
        cpu = CPU(work/'build', data)
        production = p5a.CPU(work/'build', data, rng='philox')
        if production.compiler != config['compiler']:
            raise core.Stop('saved CPU compiler mismatch')
        histories = cpu.per_history(20000)
        core.save_npz(work/'local_histories.npz', histories)
        checks = [core.l1(histories, production.run(20000, core.SEED), 20)]
        core.require(checks[0], 'L1 20000')
        for b in (0, 2):
            r = cpu.per_history(100000, start=100000*b)
            check = core.l1(r, saved_batch(saved, b, config), 20)
            checks.append(check)
            core.require(check, 'L1 saved batch '+str(b))
        core.record(result, 'L1', core.verdict(True, comparisons=checks))
        rows = [core.rows_check(*cpu.rows(n, 100000), 100000, n) for n in (20000, 100000)]
        core.record(result, 'L2', core.verdict(all(v['status'] == 'pass' for v in rows), comparisons=rows))
        faults = {}
        for f in (3, 2, 1):
            other = CPU(work/'build', data, fault=f).per_history(20000)
            if f == 3:
                checks = dict(calls=core.c2_calls(histories, other))
                ok = checks['calls']['status'] == 'fail'
            elif f == 2:
                checks = dict(bins=core.c2_bins(histories, other, l3=True))
                ok = checks['bins']['status'] == 'fail'
            else:
                checks = dict(health=core.health(other), calls=core.c2_calls(histories, other),
                              bins=core.c2_bins(histories, other, l3=True))
                ok = all(v['status'] == 'pass' for v in checks.values())
            faults['F'+str(f)] = core.verdict(ok, checks=checks)
            core.require(faults['F'+str(f)], 'L3 F'+str(f))
        merge = core.c3(histories, core.sum_histories(histories, 20, merge_fault=True), 20)
        faults['M'] = core.verdict(merge['status'] == 'fail', comparison=merge)
        core.record(result, 'L3', core.verdict(all(v['status'] == 'pass' for v in faults.values()), faults=faults))
        result['overall'] = 'pass'
    except core.Stop as exc:
        result['overall'], result['stop'] = 'fail', str(exc)
        raise
    finally:
        core.write_json(work/'local.json', result)
    return result


def compare(work, results):
    data, reference, manifest = core.load_bundle(work)
    results = Path(results)
    compile_result = json.loads((results/'p5b_compile.json').read_text())
    core.require(core.compile_check(compile_result, manifest), 'L4/C0 compile record')
    identity = core.batch_identity(manifest, compile_result['environment'])
    if json.loads((results/'p5c_identity.json').read_text()) != identity:
        raise core.Stop('C0 result identity mismatch')
    def artifact(name):
        r = core.read_npz(results/name)
        if json.loads(str(r.pop('metadata'))) != dict(identity=identity, outputs=core.describe(r)):
            raise core.Stop('result fingerprint/shape/dtype/hash mismatch: '+name)
        return r
    result = dict(checks={})
    core.record(result, 'C0', core.verdict(True))
    core.record(result, 'C1', core.c1(reference, artifact('c1.npz')))
    cpu = {k: reference['history_'+k] for k in core.HISTORY_ORDER}
    gpu = artifact('c2.npz')
    core.record(result, 'C2', core.c2(cpu, gpu, float(reference['Dref']), float(reference['conversion'])))
    ordinary, faulty = core.c3(gpu, artifact('c3.npz'), 20), core.c3(gpu, artifact('c3_fault.npz'), 20)
    split = core.split_check(gpu, artifact('c3_split.npz'))
    core.record(result, 'C3', core.verdict(ordinary['status'] == 'pass' and faulty['status'] == 'fail'
                and split['status'] == 'pass', ordinary=ordinary, merge_fault=faulty, split=split))
    present = core.resume(results/'batches', identity)
    if len(present) != 100:
        raise core.Stop('C4: incomplete saved GPU batches')
    batches = [core.read_checkpoint(present[b], identity, b) for b in range(100)]
    core.record(result, 'C4', core.c4(reference, batches, len(data['cum'])))
    result['checks']['C5'] = core.c5(batches[0], artifact('c5.npz'))
    result['overall'] = 'pass'
    core.write_json(Path(work)/'compare.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('bundle', 'local', 'compare'):
        p = sub.add_parser(name)
        p.add_argument('--work', type=Path, default=WORK)
        if name != 'compare':
            p.add_argument('--saved', type=Path, default=SAVED)
        if name == 'bundle':
            p.add_argument('--drive', type=Path)
        if name == 'compare':
            p.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'bundle':
            bundle(args.work, args.saved, args.drive)
        elif args.command == 'local':
            local(args.work, args.saved)
        else:
            compare(args.work, args.results)
    except core.Stop as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
