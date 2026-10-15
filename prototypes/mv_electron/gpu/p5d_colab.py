"""P5d Colab measurements; CuPy imported only when starting a session."""
import io
import os
from pathlib import Path
import subprocess
import time
from . import p5b_core as core
from . import p5d_core as d
from .p5b_colab import GPU, environment


def command(args):
    r = subprocess.run(args, capture_output=True, text=True)
    return dict(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)


def telemetry():
    return command(['nvidia-smi', '--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit', '--format=csv'])


def timed_gpu(gpu, n, history=False, block=128):
    core.check_range(n)
    cp = gpu.cp
    cp.cuda.Device().synchronize()
    begin = time.perf_counter()
    values = gpu.output(core.history_shapes(n) if history else core.run_shapes(len(gpu.data['cum'])))
    start, end = cp.cuda.Event(), cp.cuda.Event()
    args = gpu.args+(core.np.uint64(core.SEED), core.np.uint64(0), core.np.int32(n), core.np.int32(core.MAXSTEPS))
    name = 'p5b_per_history_kernel' if history else 'p5b_run_kernel'
    order = core.HISTORY_ORDER if history else core.RUN_ORDER
    start.record()
    gpu.kernels[name](((n+block-1)//block,), (block,), args+tuple(values[k] for k in order))
    end.record()
    out = gpu.download(values)
    end.synchronize()
    elapsed = time.perf_counter()-begin
    times = dict(kernel_ms=float(cp.cuda.get_elapsed_time(start, end)), call_s=elapsed,
                 output_bytes=sum(v.nbytes for v in values.values()))
    core.validate_output(out, core.history_shapes(n) if history else core.run_shapes(len(gpu.data['cum'])))
    core.require(core.health(out), 'GPU health')
    if (len(out['nspec_index']) if history else int(out['nspec'].sum())) != n:
        raise core.Stop('GPU history count mismatch')
    return out, times


def setup(bundle, results):
    import cupy as cp
    data, reference, manifest = d.load_bundle(bundle)
    report = dict(overall='incomplete', checks={}, measurements=[], costs={})
    # Count calls within this runtime, without touching or clearing either cache.
    report['runtime_repeat'] = getattr(setup, 'calls', 0) > 0
    setup.calls = getattr(setup, 'calls', 0)+1
    begin = time.perf_counter()
    cp.cuda.runtime.getDeviceCount()
    small = cp.zeros(1)
    cp.cuda.Device().synchronize()
    report['costs']['context_s'] = time.perf_counter()-begin
    report['environment'] = environment(cp)
    report['t4'] = 'T4' in report['environment']['gpu']
    report['cpu_environment'] = dict(lscpu=command(['lscpu']), nproc=command(['nproc']))
    props = cp.cuda.runtime.getDeviceProperties(0)
    report['device_attributes'] = {k: int(props[k]) for k in
        ('multiProcessorCount', 'maxThreadsPerMultiProcessor', 'regsPerMultiprocessor')}
    cache = Path(os.environ.get('CUPY_CACHE_DIR', '~/.cupy/kernel_cache')).expanduser()
    report['cache_directory'] = str(cache)
    report['cache_files_before'] = sorted(str(p.relative_to(cache)) for p in cache.rglob('*') if p.is_file())
    kernels = {}
    report['compilation'] = dict(options=list(core.OPTIONS), source_sha256=core.sha(core.source().encode()), kernels={})
    for name in ('p5b_run_kernel', 'p5b_per_history_kernel'):
        log = io.StringIO()
        kernel = cp.RawKernel(core.source(), name, options=core.OPTIONS, backend='nvrtc')
        begin = time.perf_counter()
        kernel.compile(log_stream=log)
        cp.cuda.Device().synchronize()
        report['compilation']['kernels'][name] = dict(
            cached_compile_request_s=time.perf_counter()-begin, log=log.getvalue(),
            attributes={k: int(kernel.attributes[k]) for k in
                ('num_regs', 'local_size_bytes', 'max_threads_per_block', 'shared_size_bytes')})
        kernels[name] = kernel
    report['cache_files_after'] = sorted(str(p.relative_to(cache)) for p in cache.rglob('*') if p.is_file())
    begin = time.perf_counter()
    gpu = GPU(cp, data, kernels)
    cp.cuda.Device().synchronize()
    report['costs']['tables_transfer_s'] = time.perf_counter()-begin
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    cpu = d.CPU(results/'build', data, compiler='g++', compat=True)
    report['cpu_compiler'], report['cpu_flags'] = cpu.compiler, cpu.flags
    return gpu, cpu, reference, report


def smoke(bundle, results):
    gpu, cpu, reference, report = setup(bundle, results)
    try:
        _, times = d.timed_cpu(cpu, 200)
        report['measurements'].append(dict(**d.metadata('smoke-C', 200), **times))
        _, times = timed_gpu(gpu, 1000)
        report['measurements'].append(dict(**d.metadata('smoke-D', 1000, block=128), **times))
        report['overall'] = 'healthy; no correctness gates'
    finally:
        core.write_json(Path(results)/'smoke.json', report)
    return report


def run(bundle, results):
    gpu, cpu, reference, report = setup(bundle, results)
    results = Path(results)
    baseline = None
    staged_history = {}
    def measure(label, n, repeat, history=False, block=128, gate=None, warmup=False):
        meta = dict(**d.metadata(label, n, block=block), repeat=repeat,
                    history_output=history, warmup=warmup, telemetry=telemetry())
        try:
            out, times = timed_gpu(gpu, n, history, block)
        except core.Stop:
            raise
        except Exception as exc:
            failure = d.no_output(label, n, exc, cuda=isinstance(exc, gpu.cp.cuda.runtime.CUDARuntimeError))
            report['measurements'].append(dict(**meta, **failure))
            core.write_json(results/'colab.json', report)
            if failure['restart_required']:
                raise core.Stop(failure['instruction']) from exc
            return None
        if gate is not None:
            check = gate(out)
            report['checks'][f'{label}_{n}_{repeat}'] = check
            if check['status'] != 'pass':
                if label == 'D3':
                    report['measurements'].append(dict(**meta, status='diagnostic_rejected'))
                    return None
                raise core.Stop(label+' gate failed')
        core.save_npz(results/f'{label}_{n}_{repeat}_{block}.npz', out, meta)
        if history:
            staged_history['measurement'] = dict(**meta, **times, status='pass')
        else:
            report['measurements'].append(dict(**meta, **times, status='pass'))
        core.write_json(results/'colab.json', report)
        return out
    try:
        for i in range(3):
            out, times = d.timed_cpu(cpu, 20000)
            core.require(d.c_gate(out, reference, len(cpu.data['cum'])), 'C integers/deep blocks; investigate any path mismatch')
            meta = d.metadata('C', 20000)
            core.save_npz(results/f'C_{i}.npz', out, meta)
            report['measurements'].append(dict(**meta, repeat=i, **times))
        measure('warmup', 1000, 0, warmup=True)
        # D-g(i) also supplies all three D2 repeats.
        for i in range(3):
            histories = measure('D2-history', 20000, i, history=True)
            measure('D2-production', 20000, i, gate=lambda r: core.c3(histories, r, len(cpu.data['cum'])))
            report['measurements'].append(staged_history['measurement'])
        measure('D-g', 100000, 0, gate=lambda r: d.d_gate(r, reference))
        for series, n in enumerate((10000, 100000, 1000000, 100000)):
            for i in range(3):
                out = measure('D', n, series*3+i)
                if n == 100000 and out is not None and baseline is None:
                    baseline = out
        for block in (64, 256):
            for i in range(3):
                measure('D3', 100000, i, block=block, gate=lambda r: d.compare_gate(
                    r, baseline, 100000, float(reference['conversion'])))
        report['overall'] = 'pass'
    except Exception as exc:
        report.update(overall='stopped', stop=str(exc))
        raise
    finally:
        core.write_json(results/'colab.json', report)
    return report
