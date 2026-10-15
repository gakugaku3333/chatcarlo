"""Colab-only compile/run entry. No P5a, Numba, physics, or private data imports."""
import argparse
import io
import json
from pathlib import Path
import sys

from . import p5b_core as core


def environment(cp):
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props['name']
    if isinstance(name, bytes):
        name = name.decode()
    return dict(cupy=cp.__version__, cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                cuda_driver=cp.cuda.runtime.driverGetVersion(), gpu=name,
                compute_capability=[props['major'], props['minor']])


def compile_modules(cp, text, report):
    kernels = {}
    report['kernels'] = {}
    for fault in (0, 1):
        options = core.OPTIONS + (('-DP5B_MERGE_FAULT=1',) if fault else ())
        for name in (('p5b_run_kernel',) if fault else core.KERNELS):
            key = name+('_fault' if fault else '')
            log = io.StringIO()
            kernel = cp.RawKernel(text, name, options=options, backend='nvrtc')
            try:
                kernel.compile(log_stream=log)
            except Exception as exc:
                report['kernels'][key] = dict(status='fail', options=list(options),
                                              log=log.getvalue(), error=str(exc))
                raise
            report['kernels'][key] = dict(status='pass', options=list(options), log=log.getvalue())
            kernels[key] = kernel
    return kernels


def compile_only(bundle, results):
    import cupy as cp
    _, _, manifest = core.load_bundle(bundle)
    results = Path(results)
    text = core.source()
    report = dict(status='incomplete', source_sha256=core.sha(text.encode()),
                  options=list(core.OPTIONS), environment=environment(cp),
                  bundle_sha256={k: v['sha256'] for k, v in manifest['files'].items()})
    try:
        compile_modules(cp, text, report)
        report['status'] = 'pass'
    except Exception as exc:
        report['status'], report['error'] = 'fail', str(exc)
        raise
    finally:
        core.write_json(results/'p5b_compile.json', report)
        print(json.dumps(report, indent=2), flush=True)
    return report


class GPU:
    def __init__(self, cp, data, kernels):
        self.cp, self.data, self.kernels = cp, data, kernels
        core.check_inputs(data)
        self.tables = {k: cp.asarray(data[k]) for k in core.POINTERS}
        np = core.np
        dims = core.dimensions(data)
        self.args = tuple(self.tables[k] for k in core.POINTERS[:11])
        self.args += tuple(np.int32(v) for v in dims[:8])
        self.args += tuple(self.tables[k] for k in core.POINTERS[11:])
        self.args += tuple(np.float64(data[k].item()) for k in core.SCALARS)
        self.args += tuple(np.int32(v) for v in dims[8:])

    def launch(self, name, n, args):
        self.kernels[name](((n+127)//128,), (128,), args)

    def download(self, values):
        np, cp = core.np, self.cp
        result = {}
        for k, v in values.items():
            a = cp.asnumpy(v)
            if a.dtype == np.uint64:
                if np.any(a >= np.uint64(2**63)):
                    raise core.Stop('GPU counter exceeds int64: '+k)
                a = a.astype(np.int64)
            elif k == 'maxdepth':
                a = a.astype(np.int64)
            result[k] = a
        return result

    def output(self, shapes):
        np, cp = core.np, self.cp
        return {k: cp.zeros(shape, dtype=np.int32 if k == 'maxdepth' else
                          (np.uint64 if k in core.INTEGER else np.float64)) for k, shape in shapes.items()}

    def per_history(self, n, start=0):
        core.check_range(n, start)
        np = core.np
        r = self.output(core.history_shapes(n))
        args = self.args+(np.uint64(core.SEED), np.uint64(start), np.int32(n), np.int32(core.MAXSTEPS))
        self.launch('p5b_per_history_kernel', n, args+tuple(r[k] for k in core.HISTORY_ORDER))
        out = self.download(r)
        core.validate_output(out, core.history_shapes(n))
        return out

    def production(self, n, start=0, fault=False):
        core.check_range(n, start)
        np = core.np
        r = self.output(core.run_shapes(len(self.data['cum'])))
        args = self.args+(np.uint64(core.SEED), np.uint64(start), np.int32(n), np.int32(core.MAXSTEPS))
        self.launch('p5b_run_kernel'+('_fault' if fault else ''), n, args+tuple(r[k] for k in core.RUN_ORDER))
        out = self.download(r)
        core.validate_output(out, core.run_shapes(len(self.data['cum'])))
        return out

    def checks(self, reference):
        cp, np = self.cp, core.np
        out = cp.empty(reference['fields'].shape, dtype=np.float64)
        dims = cp.empty(11, dtype=np.int32)
        self.launch('p5b_fields_kernel', len(out), self.args+(np.int32(len(out)), out, dims))
        r = dict(fields=cp.asnumpy(out), dimensions=cp.asnumpy(dims))
        triples = cp.asarray(reference['philox_triples'])
        words, values = cp.empty((len(triples), 4), dtype=np.uint32), cp.empty(len(triples))
        self.launch('p5b_philox_kernel', len(triples), (triples, np.int32(len(triples)), words, values))
        r.update(philox_words=cp.asnumpy(words), philox_values=cp.asnumpy(values))
        for n in (20000, 100000):
            rows, histories = cp.empty(n, dtype=np.int32), cp.empty(n, dtype=np.uint64)
            self.launch('p5b_rows_kernel', n, (np.uint64(0), np.int32(n), rows, histories))
            r['rows_'+str(n)], r['histories_'+str(n)] = cp.asnumpy(rows), cp.asnumpy(histories)
        for which, (name, width) in enumerate(core.FUNCTIONS.items()):
            x = cp.asarray(reference['input_'+name])
            seeds = range(20291940, 20291950) if which >= 4 else [20291940]
            outputs, counters = [], []
            for seed in seeds:
                v, c = cp.empty((len(x), width)), cp.empty(len(x), dtype=np.uint64)
                self.launch('p5b_functions_kernel', len(x), self.args+
                            (np.int32(which), np.uint64(seed), x, np.int32(len(x)), v, c))
                outputs.append(cp.asnumpy(v))
                counters.append(self.download(dict(calls=c))['calls'])
            r['value_'+name], r['calls_'+name] = np.concatenate(outputs), np.concatenate(counters)
        return r


def run(bundle, results):
    import cupy as cp
    np = core.np
    data, reference, manifest = core.load_bundle(bundle)
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    report = dict(checks={}, overall='incomplete')
    try:
        if any('pending' in p.name for p in results.iterdir()):
            raise core.Stop('unfinished result file present')
        recorded = json.loads((results/'p5b_compile.json').read_text())
        env = environment(cp)
        if 'T4' not in env['gpu']:
            raise core.Stop('P5c requires the approved Colab T4 GPU')
        core.require(core.compile_check(recorded, manifest, env), 'L4/C0 compile record')
        identity = core.batch_identity(manifest, env)
        identity_path = results/'p5c_identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise core.Stop('C0: existing result identity mismatch')
        core.write_json(identity_path, identity)
        core.record(report, 'C0', core.verdict(True, environment=env, identity=identity))
        compile_report = {}
        kernels = compile_modules(cp, core.source(), compile_report)
        gpu = GPU(cp, data, kernels)
        def save(name, values):
            core.save_npz(results/name, values, dict(identity=identity, outputs=core.describe(values)))
        values = gpu.checks(reference)
        save('c1.npz', values)
        core.record(report, 'C1', core.c1(reference, values))
        histories = gpu.per_history(20000)
        save('c2.npz', histories)
        cpu = {k: reference['history_'+k] for k in core.HISTORY_ORDER}
        core.record(report, 'C2', core.c2(cpu, histories, float(reference['Dref']), float(reference['conversion'])))
        prod = gpu.production(20000)
        save('c3.npz', prod)
        ordinary = core.c3(histories, prod, 20)
        core.require(ordinary, 'C3 ordinary')
        faulty = gpu.production(20000, fault=True)
        save('c3_fault.npz', faulty)
        merge = core.c3(histories, faulty, 20)
        if merge['status'] != 'fail':
            raise core.Stop('C3: injected merge fault was not rejected')
        split = [gpu.per_history(5000, start=start) for start in range(0, 20000, 5000)]
        split = {k: np.concatenate([r[k] for r in split]) for k in core.HISTORY_ORDER}
        save('c3_split.npz', split)
        split_verdict = core.split_check(histories, split)
        core.record(report, 'C3', core.verdict(split_verdict['status'] == 'pass', ordinary=ordinary,
                                             merge_fault=merge, split=split_verdict))
        present = core.resume(results/'batches', identity)
        for b in range(100):
            if b in present:
                continue
            batch = gpu.production(100000, start=100000*b)
            core.require(core.health(batch), 'C4 batch health '+str(b))
            if not core.bits(batch['nspec'], reference['c4_nspec'][b]):
                raise core.Stop('C4: spectrum mismatch in batch '+str(b))
            path = results/'batches'/f'batch_{b:03}.npz'
            core.save_npz(path, batch, core.checkpoint_meta(identity, b, batch))
            present[b] = path
            print('C4 batch', b, 'saved', flush=True)
        batches = [core.read_checkpoint(present[b], identity, b) for b in range(100)]
        core.record(report, 'C4', core.c4(reference, batches, 20))
        repeat = gpu.production(100000)
        save('c5.npz', repeat)
        report['checks']['C5'] = core.c5(batches[0], repeat)
        report['overall'] = 'pass'
    except Exception as exc:
        report['overall'], report['stop'] = 'fail', str(exc)
        raise
    finally:
        core.write_json(results/'p5c.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('compile', 'run'))
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    try:
        (compile_only if args.command == 'compile' else run)(args.bundle, args.results)
    except core.Stop as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
