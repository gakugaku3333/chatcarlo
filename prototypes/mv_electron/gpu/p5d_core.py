"""P5d NumPy-only contracts and CPU ABI; no transport imports."""
import ctypes as ct
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
import numpy as np
from . import p5b_core as core
HERE = core.HERE
DP = ct.POINTER(ct.c_double)
OUTPUTS = ('batches', 'fine_batches', 'totals', 'maxima', 'violations', 'counts',
           'maxdepth', 'nspec', 'step_batches', 'brems')
FLAGS = ['-O2', '-ffp-contract=off', '-std=c++11', '-shared', '-fPIC',
         '-DP5_RNG_PHILOX', '-DP5_TRANSPORT_FAULT=0', '-DP5_STACK_CAP=64']
GPU_OPTIONS = ('-DP5_GPU', '-DP5_RNG_PHILOX', '--fmad=false',
               '-DP5_TRANSPORT_FAULT=0', '-DP5_STACK_CAP=64')
SOURCE_SHA = '2ce5fe575ee6317b1be5cd3c0f319ed1cd9bc07b19fb104e1addcb11782832d0'
BUNDLE_SHA = {'p5b_inputs.npz': '7b02611de118c11add0471744e64fc1b77cfb0aeb4fe285e4f29b9e7699d920a',
              'p5b_reference.npz': '02f01237752230937def6c9ffdecefbbab398d9d600a5ec890b7ad16a08b289f'}


def load_bundle(directory):
    if core.OPTIONS != GPU_OPTIONS:
        raise core.Stop('GPU compilation options mismatch')
    data, reference, manifest = core.load_bundle(directory)
    if manifest['source_sha256'] != SOURCE_SHA or any(
        manifest['files'][k]['sha256'] != v for k, v in BUNDLE_SHA.items()):
        raise core.Stop('P5d pinned bundle/source mismatch')
    if manifest['cpu']['flags'] != FLAGS:
        raise core.Stop('P5d compilation flags mismatch')
    return data, reference, manifest


class Tables(ct.Structure):
    _fields_ = [(k, DP) for k in ('constants', 'loget', 'ed', 'ts', 'ks', 'us',
                                'ys', 'rates', 'pes', 'water', 'thresholds')]
    _fields_ += [(k, ct.c_int) for k in ('ne', 'nt', 'nk', 'nu', 'nrates',
                                      'ratecols', 'np', 'watercols')]
    _fields_ += [(k, DP) for k in ('er', 'Z', 'NJ', 'dl', 'brems_e', 'brems_v', 'edges', 'cum')]
    _fields_ += [(k, ct.c_double) for k in ('bpar', 'delcm', 'delpos')]
    _fields_ += [(k, ct.c_int) for k in ('nz', 'nbrems', 'nspec')]


class CPU:
    def __init__(self, work, data, compiler='clang++', flags=None, compat=False):
        core.fingerprints()
        core.check_inputs(data)
        if flags is not None and flags != FLAGS:
            raise core.Stop('CPU compilation flags mismatch')
        if core.sha(core.source().encode()) != SOURCE_SHA:
            raise core.Stop('source fingerprint mismatch')
        exe = shutil.which(compiler)
        if exe is None:
            raise core.Stop('compiler unavailable: '+compiler)
        work = Path(work)
        work.mkdir(parents=True, exist_ok=True)
        self.flags = list(FLAGS)
        sources = dict(core.FROZEN)
        if compat:
            header = (HERE/'p5d_gxx_compat.h').resolve()
            self.flags += ['-include', str(header)]
            sources[header.name] = core.sha(header.read_bytes())
        self.compiler = subprocess.check_output([exe, '--version'], text=True).splitlines()[0]
        self.identity = core.sha(json.dumps(dict(compiler=self.compiler, flags=self.flags,
                                                sources=sources), sort_keys=True).encode())
        library = work/('p5d_'+self.identity[:16]+('.dylib' if sys.platform == 'darwin' else '.so'))
        build = subprocess.run([exe, *self.flags, str(HERE/'p5_cpu.cpp'), '-o', str(library)],
                               capture_output=True, text=True)
        (work/('build_'+self.identity[:16]+'.log')).write_text(build.stdout+build.stderr)
        build.check_returncode()
        self.data, self.tables = data, self.make_tables(data)
        self.lib = ct.CDLL(str(library))
        d = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        i = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
        self.lib.p5a_run.argtypes = [ct.POINTER(Tables), ct.c_uint64, ct.c_uint64,
            ct.c_int, ct.c_int, ct.c_int, d, d, d, d, i, i, i, i, d, i, d, ct.c_int, i]
        self.lib.p5a_run.restype = None

    @staticmethod
    def make_tables(data):
        for name, _ in Tables._fields_[:11]:
            a = data[name]
            if a.dtype != np.float64 or not a.flags.c_contiguous or not np.isfinite(a).all():
                raise ValueError('invalid table: '+name)
        if (data['constants'].shape != (6,) or data['ed'].shape != data['loget'].shape
            or data['ys'].shape != (len(data['ts']), len(data['ks']), len(data['us']))
            or data['rates'].ndim != 2 or data['rates'].shape[1] not in (2, 3)
            or data['water'].shape != (len(data['pes']), 7)
            or data['thresholds'].shape != (2,)):
            raise ValueError('table shape mismatch')
        for name in ('loget', 'ts', 'ks', 'us', 'pes'):
            if data[name].ndim != 1 or len(data[name]) < 2 or np.any(np.diff(data[name]) <= 0):
                raise ValueError('invalid grid: '+name)
        if data['rates'].shape[1] == 3 and np.any(np.diff(data['rates'][:, 0]) <= 0):
            raise ValueError('invalid independent rate grid')
        if np.any(data['rates'][:, -2:] <= 0):
            raise ValueError('nonpositive rates')
        values = [data[name].ctypes.data_as(DP) for name, _ in Tables._fields_[:11]]
        sizes = [len(data['loget']), len(data['ts']), len(data['ks']), len(data['us']),
                 len(data['rates']), data['rates'].shape[1], len(data['pes']), 7]
        other = [data[name].ctypes.data_as(DP) if name in data else DP() for name in
                 ('er', 'Z', 'NJ', 'dl', 'brems_e', 'brems_v', 'edges', 'cum')]
        scalars = [float(data[k].item()) for k in ('bpar', 'delcm', 'delpos')]
        dimensions = [len(data['Z']), len(data.get('brems_e', [])), len(data.get('cum', []))]
        if 'cum' in data:
            if (data['brems_v'].shape != (len(data['brems_e']), 5)
                or data['edges'].shape != (len(data['cum'])+1,)
                or not np.isfinite(data['brems_v']).all()
                or np.any(np.diff(data['edges']) <= 0) or np.any(np.diff(data['cum']) <= 0)
                or data['cum'][-1] != 1. or not 0 < len(data['cum']) <= 64):
                raise ValueError('invalid full input tables')
        return Tables(*values, *sizes, *other, *scalars, *dimensions)

    def run(self, n, seed, start=0, history_rng=False, diagnostic=False, maxsteps=10000000):
        if 'cum' not in self.data:
            raise ValueError('transport inputs required')
        if (not isinstance(n, int) or n <= 0 or n > 20000000 or start < 0
            or seed < 0 or seed >= 2**64 or start+n > 2**64 or maxsteps <= 0):
            raise ValueError('invalid run range')
        r = dict(batches=np.zeros((100, 150)), fine_batches=np.zeros((100, 600)),
                 totals=np.zeros(6), maxima=np.zeros(3), violations=np.zeros(3, np.int64),
                 counts=np.zeros(16, np.int64), nspec=np.zeros((len(self.data['cum']), 10), np.int64),
                 step_batches=np.zeros(100, np.int64), brems=np.zeros(10))
        maxdepth = np.zeros(1, np.int64)
        capacity = min(n*maxsteps, 2000000) if diagnostic else 0
        trace, meta = np.empty((capacity, 7)), np.zeros(2, np.int64)
        begin = time.perf_counter()
        self.lib.p5a_run(ct.byref(self.tables), seed, start, n, int(history_rng), maxsteps,
                        r['batches'], r['fine_batches'], r['totals'], r['maxima'], r['violations'],
                        r['counts'], r['nspec'], r['step_batches'], r['brems'], maxdepth,
                        trace, capacity, meta)
        seconds = time.perf_counter()-begin
        r['maxdepth'] = np.array(maxdepth[0], dtype=np.int64)
        r['seconds'] = np.array(seconds)
        if diagnostic:
            r['diagnostic'] = trace[:min(capacity, int(meta[0]))]
            r['diagnostic_meta'] = meta
        return r


def cpu_health(r, n, nspec=20):
    values = {k: v for k, v in r.items() if k != 'seconds'}
    shapes = {k: v for k, v in core.run_shapes(nspec).items() if k != 'nan_count'}
    try:
        core.validate_output(values, shapes)
    except (core.Stop, AttributeError, TypeError):
        return core.verdict(False, reason='fields/shape/dtype')
    ok = (all(np.isfinite(v).all() for v in values.values())
          and not np.any(values['violations']) and not np.any(values['counts'][5:7])
          and int(values['nspec'].sum()) == n)
    return core.verdict(bool(ok))


def blocks(r, n, conversion):
    return core.deep_blocks(r['batches'].sum(axis=0), n, conversion)


def compare_gate(actual, expected, n, conversion, expected_blocks=None):
    integers = {k: core.bits(actual[k], expected[k]) for k in ('nspec', 'counts', 'step_batches')}
    ref = blocks(expected, n, conversion) if expected_blocks is None else expected_blocks
    difference = core.relative(blocks(actual, n, conversion), ref)
    return core.verdict(all(integers.values()) and bool(np.all(np.isfinite(difference))
                        and np.all(difference <= 1e-8)), integers=integers,
                        deep_relative=difference.tolist())


def c_gate(actual, reference, nspec):
    histories = {k: reference['history_'+k] for k in core.HISTORY_ORDER}
    expected = core.sum_histories(histories, nspec)
    return compare_gate(actual, expected, len(histories['steps']), float(reference['conversion']))


def d_gate(actual, reference):
    expected = {k: reference['c4_'+k][0] for k in ('nspec', 'counts', 'step_batches')}
    return compare_gate(actual, expected, 100000, float(reference['conversion']), reference['c4_blocks'][0])


def metadata(label, n, seed=core.SEED, start=0, history_rng=False, block=None):
    return dict(label=label, n=n, seed=seed, start=start, history_rng=history_rng,
                maxsteps=core.MAXSTEPS, block=block, time=datetime.now(timezone.utc).isoformat())


def timed_cpu(cpu, n, seed=core.SEED, **kwargs):
    begin = time.perf_counter()
    out = cpu.run(n, seed, **kwargs)
    elapsed = time.perf_counter()-begin
    core.require(cpu_health(out, n, len(cpu.data['cum'])), 'CPU health')
    return out, dict(inner_s=float(out['seconds']), call_s=elapsed)


def no_output(label, n, exc, cuda=False):
    if isinstance(exc, core.Stop) or not (n == 1000000 or label == 'D3'):
        raise core.Stop('execution stopped: '+str(exc)) from exc
    return dict(status='no_output', error=str(exc), restart_required=bool(cuda),
                instruction='Restart notebook and rerun remaining D including warmup' if cuda else '')
