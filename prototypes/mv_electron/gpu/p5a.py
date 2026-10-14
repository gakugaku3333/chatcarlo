"""P5a fixed-setting CPU translation, with a mandatory bitwise gate.

The history kernel is added only after A1 succeeds. No file deletion, downloads,
or GPU entry points. Work products are retained under the caller's work path.
"""
import argparse
import ctypes as ct
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from numba import njit

from prototypes.mv_electron import physics
from prototypes.mv_electron.backend_sampling import table_rates
from prototypes.mv_electron.photon_data_p2 import coefficients
from prototypes.mv_electron.sampling import gs_mu_cached
from prototypes.mv_electron.transport import local, score_segment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CHECKS = ('A1', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7',
          'C1', 'C2', 'C3', 'C4')
DP = ct.POINTER(ct.c_double)
OUTPUTS = ('batches', 'fine_batches', 'totals', 'maxima', 'violations', 'counts',
           'maxdepth', 'nspec', 'step_batches', 'brems')
P3R = Path('~/.cache/chatcarlo/p3r/prototype/p3r_proto').expanduser()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def array_hash(a):
    return sha(np.ascontiguousarray(a).tobytes())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='p5a_pending_',
                                     suffix='.json', delete=False) as f:
        f.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        pending = Path(f.name)
    pending.replace(path)


class Tables(ct.Structure):
    _fields_ = [(k, DP) for k in ('constants', 'loget', 'ed', 'ts', 'ks', 'us',
                                'ys', 'rates', 'pes', 'water', 'thresholds')]
    _fields_ += [(k, ct.c_int) for k in ('ne', 'nt', 'nk', 'nu', 'nrates',
                                      'ratecols', 'np', 'watercols')]
    _fields_ += [(k, DP) for k in ('er', 'Z', 'NJ', 'dl', 'brems_e', 'brems_v', 'edges', 'cum')]
    _fields_ += [(k, ct.c_double) for k in ('bpar', 'delcm', 'delpos')]
    _fields_ += [(k, ct.c_int) for k in ('nz', 'nbrems', 'nspec')]


def load_inputs(full=False):
    """Read the fixed inputs through the existing frozen input verifier."""
    from prototypes.mv_electron.run_p3s import inputs
    args = inputs('elsepa')
    data = dict(zip(('ts', 'ks', 'us', 'ys', 'rates', 'single', 'pes', 'water',
                     'thresholds', 'dl', 'bpar', 'delcm', 'delpos', 'nist_E',
                     'nist_muen'), args))
    data.update(constants=np.array([physics.M, physics.C, physics.I, physics.RE,
                                    physics.NE, physics.ALPHA]), loget=physics.LOGET,
                ed=physics.ED, er=physics.ER, Z=physics.Z, NJ=physics.NJ)
    if full:
        from prototypes.mv_electron.bremsstrahlung_p3r import tables
        from prototypes.mv_electron.run_p3a import load_spectrum
        data['brems_e'], data['brems_v'] = tables()
        data['edges'], _, data['cum'] = load_spectrum()
    return {k: np.ascontiguousarray(v, dtype=np.float64) for k, v in data.items()}


class CPU:
    def __init__(self, work, data, rng='mt', fault=0, cap=64):
        work = Path(work)
        work.mkdir(parents=True, exist_ok=True)
        exe = shutil.which('clang++')
        if exe is None:
            raise RuntimeError('clang++ not available')
        if rng not in ('mt', 'philox'):
            raise ValueError('unknown RNG')
        self.flags = ['-O2', '-ffp-contract=off', '-std=c++11', '-shared', '-fPIC',
                      '-DP5_RNG_'+rng.upper(), f'-DP5_TRANSPORT_FAULT={fault}', f'-DP5_STACK_CAP={cap}']
        self.rng, self.fault, self.cap = rng, fault, cap
        self.compiler = subprocess.check_output([exe, '--version'], text=True).splitlines()[0]
        self.sources = {name: sha((HERE/name).read_bytes()) for name in
                        ('p5_cpu.cpp', 'p5_kernel.h', 'p5_rng.h', 'p5a0_common.h', 'p5a.py')}
        self.identity = sha(json.dumps(dict(sources=self.sources, flags=self.flags,
                                           compiler=self.compiler), sort_keys=True).encode())
        output = work/('p5a_'+self.identity[:16]+('.dylib' if sys.platform == 'darwin' else '.so'))
        build = subprocess.run([exe, *self.flags, str(HERE/'p5_cpu.cpp'), '-o', str(output)],
                               capture_output=True, text=True)
        (work/('build_'+self.identity[:16]+'.log')).write_text(build.stdout+build.stderr)
        build.check_returncode()
        self.data = data
        self.tables = self.make_tables(data)
        self.lib = ct.CDLL(str(output))
        d = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        u = np.ctypeslib.ndpointer(dtype=np.uint64, flags='C_CONTIGUOUS')
        i = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
        signatures = {
            'p5a_deterministic': [ct.c_int, d, ct.c_int, ct.POINTER(Tables), d],
            'p5a_random': [ct.c_int, ct.c_uint64, d, ct.c_int, ct.POINTER(Tables), d, d, u],
            'p5a_score': [ct.c_int, d, ct.c_int, ct.c_int, ct.c_double, d],
            'p5a_rng': [ct.c_uint64, ct.c_uint64, ct.c_int, d],
            'p5a_run': [ct.POINTER(Tables), ct.c_uint64, ct.c_uint64, ct.c_int, ct.c_int, ct.c_int,
                        d, d, d, d, i, i, i, i, d, i, d, ct.c_int, i],
        }
        for name, signature in signatures.items():
            func = getattr(self.lib, name)
            func.argtypes, func.restype = signature, None

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

    def deterministic(self, name, x):
        x = validate(name, x, self.data)
        which, width = {'collision': (0, 1), 'hard': (1, 2),
                        'coefficients': (2, 4), 'table_rates': (3, 2)}[name]
        out = np.empty((len(x), width))
        self.lib.p5a_deterministic(which, x, len(x), ct.byref(self.tables), out)
        return out

    def random(self, name, seed, x):
        x = validate(name, x, self.data)
        out, next_value, calls = np.empty(len(x)), np.empty(1), np.empty(1, np.uint64)
        self.lib.p5a_random(int(name == 'gs_mu_cached'), seed, x, len(x),
                           ct.byref(self.tables), out, next_value, calls)
        return out, next_value, int(calls[0])

    def score(self, name, x, nb, dz):
        x = validate(name, x, self.data)
        if nb not in (150, 600) or dz != (0.2 if nb == 150 else 0.05):
            raise ValueError('fixed depth scorer required')
        out = np.zeros((len(x), nb))
        self.lib.p5a_score(int(name == 'local'), x, len(x), nb, dz, out)
        return out


def validate(name, x, data):
    x = np.ascontiguousarray(x, dtype=np.float64)
    width = {'gs_mu_cached': 2, 'score_segment': 3, 'local': 2}.get(name)
    if (x.ndim != (2 if width else 1) or (width and x.shape[1] != width)
        or not np.isfinite(x).all()):
        raise ValueError('invalid inputs: '+name)
    if name in ('collision', 'hard', 'sample_W'):
        if np.any(x <= (0.02 if name == 'sample_W' else 0)) or np.any(x > 20):
            raise ValueError('electron energy outside gate domain')
    elif name == 'coefficients':
        if np.any(x < data['pes'][0]) or np.any(x > data['pes'][-1]):
            raise ValueError('photon energy outside table range')
    elif name == 'table_rates':
        grid = data['rates'][:, 0] if data['rates'].shape[1] == 3 else data['ts']
        if np.any(x < grid[0]) or np.any(x > grid[-1]):
            raise ValueError('rate energy outside table range')
    elif name == 'gs_mu_cached':
        if (np.any(x[:, 0] < data['ts'][0]) or np.any(x[:, 0] > data['ts'][-1])
            or np.any(x[:, 1] < data['ks'][0]) or np.any(x[:, 1] > data['ks'][-1])):
            raise ValueError('GS input outside table range')
    elif name in ('score_segment', 'local'):
        if np.any(x[:, -1] < 0) or np.any(x[:, :-1] < 0) or np.any(x[:, :-1] > 30):
            raise ValueError('scorer input outside slab')
    return x


@njit
def deterministic_reference(which, x, ts, rates, pes, water, thresholds):
    width = 1 if which == 0 else (4 if which == 2 else 2)
    out = np.empty((len(x), width))
    for j in range(len(x)):
        if which == 0:
            out[j, 0] = physics.collision(x[j])
        elif which == 1:
            out[j] = physics.hard(x[j])
        elif which == 2:
            out[j] = coefficients(x[j], pes, water, thresholds)
        else:
            out[j] = table_rates(x[j], ts, rates)
    return out


@njit
def sample_w_reference(seed, x):
    np.random.seed(seed)
    out = np.empty(len(x))
    for j in range(len(x)):
        out[j] = physics.sample_W(x[j])
    return out, np.random.random()


@njit
def gs_reference(seed, x, ts, ks, us, ys, rates):
    np.random.seed(seed)
    out = np.empty(len(x))
    for j in range(len(x)):
        total, G1 = table_rates(x[j, 0], ts, rates)
        out[j] = gs_mu_cached(x[j, 0], x[j, 1], ts, ks, us, ys, total, G1)
    return out, np.random.random()


@njit
def score_reference(which, x, nb, dz):
    out = np.zeros((len(x), nb))
    for j in range(len(x)):
        if which == 0:
            score_segment(out[j], x[j, 0], x[j, 1], x[j, 2], dz)
        else:
            local(out[j], x[j, 0], x[j, 1], dz)
    return out


def reference(name, x, data, seed=None, nb=None, dz=None):
    x = validate(name, x, data)
    if name in ('collision', 'hard', 'coefficients', 'table_rates'):
        which = ('collision', 'hard', 'coefficients', 'table_rates').index(name)
        return deterministic_reference(which, x, data['ts'], data['rates'],
                                       data['pes'], data['water'], data['thresholds'])
    if name == 'sample_W':
        out, next_value = sample_w_reference(seed, x)
        return out, np.array([next_value])
    if name == 'gs_mu_cached':
        out, next_value = gs_reference(seed, x, data['ts'], data['ks'], data['us'],
                                       data['ys'], data['rates'])
        return out, np.array([next_value])
    return score_reference(int(name == 'local'), x, nb, dz)


def mismatch(ref, actual):
    if ref.shape != actual.shape or ref.dtype != actual.dtype:
        raise ValueError('comparison shape/dtype mismatch')
    a = np.ascontiguousarray(ref).view(np.uint8).reshape(-1, ref.dtype.itemsize)
    b = np.ascontiguousarray(actual).view(np.uint8).reshape(-1, actual.dtype.itemsize)
    flat = np.flatnonzero(np.any(a != b, axis=1))
    result = dict(values=int(ref.size), mismatches=int(len(flat)),
                  reference_sha256=array_hash(ref), cpu_sha256=array_hash(actual),
                  nonfinite=int(np.count_nonzero(~np.isfinite(ref))+np.count_nonzero(~np.isfinite(actual))))
    if len(flat):
        index = tuple(int(i) for i in np.unravel_index(int(flat[0]), ref.shape))
        r, c = float(ref[index]), float(actual[index])
        result['first'] = dict(index=list(index), reference=r, cpu=c,
                               reference_hex=r.hex(), cpu_hex=c.hex(),
                               abs_difference=abs(r-c))
    return result


class GateStop(Exception):
    pass


def logarithmic_points(lo, hi, n=100000):
    """Four-operation point construction; one fixed ratio, then multiplication.

    The ratio is calculated once, outside the point loop. Intermediate points
    use multiplication only, independent of the translated functions.
    """
    ratio = (hi/lo)**(1/(n-1))
    x = np.empty(n)
    x[0] = lo
    for i in range(1, n):
        x[i] = x[i-1]*ratio
    x[-1] = hi
    return np.minimum(hi, np.maximum(lo, x))


def energy_points(nodes, lo, hi, thresholds):
    nodes = np.asarray(nodes)
    thresholds = np.asarray(thresholds)
    a = np.concatenate((nodes, np.nextafter(nodes, 0), np.nextafter(nodes, np.inf),
                        thresholds, np.nextafter(thresholds, 0),
                        np.nextafter(thresholds, np.inf), logarithmic_points(lo, hi)))
    return a[(a >= lo) & (a <= hi)]


def gate(work):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    data = load_inputs()
    cpu = CPU(work, data)
    result = dict(checks={k: dict(status='not_run') for k in CHECKS},
                  environment=dict(system=platform.system(), machine=platform.machine(),
                                   python=platform.python_version(), numpy=np.__version__,
                                   compiler=cpu.compiler, flags=cpu.flags),
                  sources=cpu.sources,
                  input_arrays={k: dict(shape=list(v.shape), sha256=array_hash(v)) for k, v in data.items()},
                  overall='incomplete', p5b_ready=False)
    rows = []
    result['checks']['A1'] = dict(status='not_run', functions=rows)

    def compare(name, inputs, ref, actual, **details):
        row = dict(function=name, **details, **mismatch(ref, actual))
        rows.append(row)
        if row['mismatches'] or row['nonfinite']:
            row['status'] = 'fail'
            if 'first' in row:
                row['first']['input'] = np.asarray(inputs[row['first']['index'][0]]).tolist()
            result['checks']['A1']['status'] = 'fail'
            result['stop'] = dict(check='A1', function=name,
                                  classification='未特定（Claudeによる最初の差の調査待ち）',
                                  first=row.get('first'))
            print(json.dumps(result['stop'], ensure_ascii=False), flush=True)
            raise GateStop('A1 mismatch: '+name)
        row['status'] = 'pass'
        print('A1', name, 'pass', row['values'], 'values, mismatch 0', flush=True)

    try:
        critical = [.01, .02, .02/(1-2e-7), 1., *data['thresholds']]
        electron_nodes = np.r_[physics.ET, data['ts'], data['rates'][:, 0]]
        for name in ('collision', 'hard', 'coefficients', 'table_rates'):
            if name == 'coefficients':
                nodes, lo, hi = data['pes'], data['pes'][0], data['pes'][-1]
            elif name == 'table_rates':
                nodes = data['rates'][:, 0]
                lo, hi = nodes[0], nodes[-1]
            else:
                nodes, lo, hi = electron_nodes, np.nextafter(.01, 0), 20.
            x = energy_points(nodes, lo, hi, critical)
            compare(name, x, reference(name, x, data), cpu.deterministic(name, x), cases=len(x))

        for nb, dz in ((150, .2), (600, .05)):
            boundaries = np.arange(nb+1)*dz
            z = np.r_[boundaries, np.nextafter(boundaries, 0), np.nextafter(boundaries, np.inf),
                      logarithmic_points(1e-12, 30.)]
            z = z[(z >= 0) & (z <= 30)]
            # Forward/reverse, stationary and zero-loss cases, all with nonzero
            # losses that exercise division in the segment apportioning.
            inputs = np.column_stack((z, 30-z, .01+(z/30)*.9))
            extra = np.array([[0., 0., 1.], [30., 30., 1.], [dz, dz+5e-16, 1.],
                              [1., 29., 0.], [29., 1., 0.]])
            inputs = np.vstack((inputs, extra))
            for name, points in (('score_segment', inputs),
                                 ('local', np.column_stack((z, .01+(z/30)*.9)))):
                for start in range(0, len(points), 1000):
                    x = points[start:start+1000]
                    # One aggregate row per function/grid; record the first
                    # differing global input index without retaining arrays.
                    ref, actual = reference(name, x, data, nb=nb, dz=dz), cpu.score(name, x, nb, dz)
                    d = mismatch(ref, actual)
                    if d['mismatches'] or d['nonfinite']:
                        compare(name, x, ref, actual, nb=nb, dz=dz, input_offset=start)
                rows.append(dict(function=name, nb=nb, dz=dz, status='pass',
                                 cases=len(points), values=len(points)*nb, mismatches=0, nonfinite=0))
                print('A1', name, nb, 'pass', len(points), 'cases, mismatch 0', flush=True)

        for name in ('gs_mu_cached', 'sample_W'):
            if name == 'sample_W':
                x = logarithmic_points(np.nextafter(.02, np.inf), 10., 1000)
            else:
                e = logarithmic_points(data['ts'][0], data['ts'][-1], 1000)
                kval = np.resize(data['ks'], 1000)
                x = np.column_stack((e, kval))
            for seed in range(20291940, 20291950):
                ref, next_ref = reference(name, x, data, seed=seed)
                actual, next_actual, calls = cpu.random(name, seed, x)
                compare(name, x, ref, actual, seed=seed, cases=len(x), rng_calls=calls)
                compare(name+'_next_uniform', np.zeros((1,)), next_ref, next_actual, seed=seed)

        bad = {'collision': [np.array([0.]), np.array([-1.]), np.array([np.nan])],
               'hard': [np.array([0.]), np.array([-1.]), np.array([np.inf])],
               'coefficients': [np.array([np.nextafter(data['pes'][0], 0)]),
                                np.array([np.nextafter(data['pes'][-1], np.inf)])],
               'table_rates': [np.array([np.nextafter(data['rates'][0, 0], 0)]),
                               np.array([np.nextafter(data['rates'][-1, 0], np.inf)])],
               'sample_W': [np.array([.02]), np.array([0.])],
               'gs_mu_cached': [np.array([[0., .025]]), np.array([[1., 0.]])],
               'score_segment': [np.array([[-1., 0., 1.]]), np.array([[0., 1., -1.]])],
               'local': [np.array([[31., 1.]]), np.array([[0., -1.]])]}
        for name, cases in bad.items():
            rejected = 0
            for x in cases:
                for side in ('reference', 'cpu'):
                    try:
                        if side == 'reference':
                            reference(name, x, data, seed=20291940, nb=150, dz=.2)
                        elif name in ('score_segment', 'local'):
                            cpu.score(name, x, 150, .2)
                        elif name in ('sample_W', 'gs_mu_cached'):
                            cpu.random(name, 20291940, x)
                        else:
                            cpu.deterministic(name, x)
                    except ValueError:
                        rejected += 1
            row = dict(function=name+'_domain_rejection', cases=len(cases),
                       rejected=rejected, expected=2*len(cases),
                       status='pass' if rejected == 2*len(cases) else 'fail')
            rows.append(row)
            if row['status'] != 'pass':
                raise GateStop('A1 domain rejection: '+name)
        result['checks']['A1']['status'] = 'pass'
    except GateStop:
        result['checks']['A1']['status'] = 'fail'
        result['overall'] = 'fail'
    finally:
        write_json(work/'gate.json', result)
    return result


def p4_run(data, n, seed, history_rng=False, maxstack=4096, maxsteps=10000000):
    from prototypes.mv_electron.photon_transport_p4 import run_coupled
    from prototypes.mv_electron.run_p3s import rule_kwargs
    args = tuple(data[k] for k in ('ts', 'ks', 'us', 'ys', 'rates', 'single', 'pes', 'water',
                                  'thresholds', 'dl', 'bpar', 'delcm', 'delpos', 'nist_E', 'nist_muen'))
    # Material scalars have their original scalar types for Numba's signature.
    args = args[:10]+tuple(float(a.item()) for a in args[10:13])+args[13:]
    begin = time.perf_counter()
    v = run_coupled(10., n, seed, .05, .025, 30., *args,
                    annihilate_in_flight=True, spectrum_edges=data['edges'], spectrum_cum=data['cum'],
                    brems_mode=2, score_kerma=False, score_pair=False, history_rng=history_rng,
                    maxstack=maxstack, maxsteps=maxsteps, **rule_kwargs('own'))
    indices = (0, 1, 5, 6, 7, 8, 9, 10, 12, 13)
    r = {name: np.asarray(v[i]) for name, i in zip(OUTPUTS, indices)}
    r['maxdepth'] = np.asarray(r['maxdepth'], dtype=np.int64)
    r['seconds'] = np.array(time.perf_counter()-begin)
    return r


def compare_outputs(ref, actual):
    details = {k: mismatch(np.asarray(ref[k]), np.asarray(actual[k])) for k in OUTPUTS}
    ok = all(v['mismatches'] == 0 and v['nonfinite'] == 0 for v in details.values())
    first = next((dict(output=k, **details[k]['first']) for k in OUTPUTS if 'first' in details[k]), None)
    return dict(status='pass' if ok else 'fail', outputs=details, first=first)


def health(r):
    finite = all(np.isfinite(r[k]).all() for k in OUTPUTS)
    v = dict(finite=bool(finite), violations=np.asarray(r['violations']).tolist(),
             step_limit=int(r['counts'][5]), stack_or_invalid_step=int(r['counts'][6]),
             brems_envelope_excess=float(r['brems'][2]), brems_energy_excess=float(r['brems'][3]),
             unresolved=float(r['totals'][3]), maxdepth=int(r['maxdepth']))
    ok = (finite and not np.any(r['violations']) and not np.any(r['counts'][5:7])
          and not np.any(r['brems'][2:4]) and r['totals'][3] == 0)
    return dict(status='pass' if ok else 'fail', **v)


def validate_result(r, nspec):
    shapes = dict(batches=(100, 150), fine_batches=(100, 600), totals=(6,), maxima=(3,),
                  violations=(3,), counts=(16,), maxdepth=(), nspec=(nspec, 10),
                  step_batches=(100,), brems=(10,))
    integer = ('violations', 'counts', 'maxdepth', 'nspec', 'step_batches')
    for k, shape in shapes.items():
        if k not in r or r[k].shape != shape or r[k].dtype != (np.int64 if k in integer else np.float64):
            raise GateStop('batch output shape/dtype mismatch: '+k)


def save_output(path, r, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent, prefix='p5a_pending_',
                                     suffix='.npz', delete=False) as f:
        np.savez(f, **r, **metadata)
        pending = Path(f.name)
    pending.replace(path)


@contextmanager
def metadata_lock(dest):
    # Retained lock file. No unlink or cleanup, and transport is outside this lock.
    with (Path(dest)/'metadata.lock').open('a') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def first_history(cpu, data, n, seed, history_rng=True, maxstack=4096, maxsteps=10000000):
    """Find the first differing history without changing the preregistered seed."""
    if history_rng:
        for h in range(n):
            s = (seed+h) % 2**32
            ref = p4_run(data, 1, s, True, maxstack, maxsteps)
            actual = cpu.run(1, s, history_rng=True, maxsteps=maxsteps)
            comparison = compare_outputs(ref, actual)
            if comparison['status'] == 'fail':
                actual = cpu.run(1, s, history_rng=True, maxsteps=maxsteps, diagnostic=True)
                return h, comparison['first'], actual
    else:
        # Prefixes retain the single continuous MT state. The last prefix marks
        # the first history that can cause an output mismatch.
        for h in range(n):
            ref = p4_run(data, h+1, seed, False, maxstack, maxsteps)
            actual = cpu.run(h+1, seed, maxsteps=maxsteps)
            comparison = compare_outputs(ref, actual)
            if comparison['status'] == 'fail':
                actual = cpu.run(h+1, seed, maxsteps=maxsteps, diagnostic=True)
                return h, comparison['first'], actual
    return None, None, None


def require_gate(work):
    path = Path(work)/'gate.json'
    if not path.exists():
        gate(work)
    if json.loads(path.read_text())['checks']['A1']['status'] != 'pass':
        raise GateStop('A1 pass required; run gate first')
    return json.loads(path.read_text())


def run_config(cpu, data):
    return dict(sources=cpu.sources, flags=cpu.flags, compiler=cpu.compiler,
                arrays={k: dict(sha256=array_hash(a), shape=list(a.shape), dtype=str(a.dtype))
                        for k, a in data.items()}, rng=cpu.rng)


def batch_run(cpu, data, dest, indices, total_batches=100, per_batch=100000, seed=20291960):
    """Independent history ranges; one process per caller, at most two parts.

    Missing later batches are work to do. Missing earlier completed batches,
    extra files, duplicates, or inconsistent metadata are refused on resume.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    config = dict(**run_config(cpu, data), seed=seed, batches=total_batches, per_batch=per_batch)
    config_path = dest/'config.json'
    with metadata_lock(dest):
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise GateStop('batch configuration mismatch')
        if not config_path.exists():
            write_json(config_path, config)
    config_sha = sha(json.dumps(config, sort_keys=True).encode())
    present = {}
    for path in dest.glob('batch*.npz'):
        try:
            b = int(path.stem.split('_')[1])
        except (ValueError, IndexError) as exc:
            raise GateStop('invalid batch name') from exc
        if path.name != f'batch_{b:03}.npz' or b in present or not 0 <= b < total_batches:
            raise GateStop('duplicate or unexpected batch')
        with np.load(path, allow_pickle=False) as z:
            if (str(z['config_sha256']) != config_sha or int(z['N']) != per_batch
                or int(z['seed']) != seed or int(z['start']) != b*per_batch
                or int(z['end']) != (b+1)*per_batch or str(z['rng']) != 'philox'):
                raise GateStop('batch metadata mismatch')
            validate_result(z, len(data['cum']))
            if health(z)['status'] != 'pass':
                raise GateStop('unhealthy saved batch')
        present[b] = path
    progress_path = dest/'progress.json'
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get('config_sha256') != config_sha or any(b not in present for b in progress['completed']):
            raise GateStop('missing previously completed batch')
    for b in indices:
        if b not in present:
            r = cpu.run(per_batch, seed, start=b*per_batch)
            if health(r)['status'] != 'pass':
                save_output(dest/f'failed_{b:03}.npz', r)
                raise GateStop('C2: unhealthy batch '+str(b))
            save_output(dest/f'batch_{b:03}.npz', r, config_sha256=np.array(config_sha),
                        N=per_batch, seed=np.uint64(seed), start=np.uint64(b*per_batch),
                        end=np.uint64((b+1)*per_batch), rng=np.array('philox'))
            present[b] = dest/f'batch_{b:03}.npz'
        with metadata_lock(dest):
            earlier = json.loads(progress_path.read_text())['completed'] if progress_path.exists() else []
            if any(not (dest/f'batch_{i:03}.npz').exists() for i in earlier):
                raise GateStop('missing previously completed batch')
            write_json(progress_path, dict(config_sha256=config_sha, completed=sorted(set(earlier) | set(present))))
        print('Philox batch', b, 'complete', flush=True)
    return sorted(present)


def check_b7(work, data):
    cpu = CPU(work, data, rng='philox')
    # Retain examples from earlier builds; never reuse them with a new source.
    whole = Path(work)/('b7_whole_'+cpu.identity[:16])
    split = Path(work)/('b7_split_'+cpu.identity[:16])
    batch_run(cpu, data, whole, range(4), 4, 500)
    batch_run(cpu, data, split, [0], 4, 500)
    batch_run(cpu, data, split, [1], 4, 500)
    batch_run(cpu, data, split, [2, 3], 4, 500)
    details = []
    for b in range(4):
        with np.load(whole/f'batch_{b:03}.npz') as a, np.load(split/f'batch_{b:03}.npz') as c:
            details.append(compare_outputs(a, c))
    return dict(status='pass' if all(d['status'] == 'pass' for d in details) else 'fail',
                histories=2000, batches=4, comparisons=details)


def verify_saved_inputs(data):
    """Read-only B3/C precondition against P3r's original input hashes."""
    from prototypes.mv_electron import run_p3a, run_p3s
    config = json.loads((P3R/'config.json').read_text())
    hashes = run_p3a.table_hashes('elsepa', run_p3s.inputs('elsepa'))
    spec = run_p3a.spec_config(data['edges'], data['cum'], run_p3a.SPECTRUM_SHA)
    for key, value in spec.items():
        if config.get(key) != value:
            raise GateStop('saved spectrum hash/configuration mismatch: '+key)
    if config['table_hashes'] != hashes:
        raise GateStop('saved table hashes mismatch')
    fixed = dict(N=10000000, seed=20291621, E=10., backend='elsepa', fe=.05, k=.025,
                 thickness=30., annihilate_in_flight=True, batches=100, brems_mode=2,
                 rule='own', score_kerma=False, score_pair=False)
    if any(config.get(k) != v for k, v in fixed.items()):
        raise GateStop('saved fixed configuration mismatch')
    return dict(status='pass', table_hashes=hashes, spectrum_sha256=run_p3a.SPECTRUM_SHA)


def saved_form(r):
    return {dict(batches='coarse', fine_batches='fine', step_batches='steps').get(k, k):
            np.sum(r[k], axis=0) if k in ('batches', 'fine_batches') else r[k] for k in OUTPUTS}


def chain0(work, skip_saved):
    work = Path(work)
    result = require_gate(work)
    data = load_inputs(full=True)
    cpu = CPU(work, data)
    # Verify the A1 artifact still describes the representative source and data.
    if any(cpu.sources[k] != v for k, v in result['sources'].items()):
        raise GateStop('gate source changed; rerun A1 before chain0')
    for k, info in result['input_arrays'].items():
        if array_hash(data[k]) != info['sha256'] or list(data[k].shape) != info['shape']:
            raise GateStop('input array changed since A1: '+k)
    result['transport_config'] = run_config(cpu, data)
    try:
        saved_b1 = None
        for name, seed, hr in (('B1', 20291950, True), ('B2', 20291951, False)):
            actual = cpu.run(20000, seed, history_rng=hr)
            ref = p4_run(data, 20000, seed, history_rng=hr)
            save_output(work/(name+'_cpu.npz'), actual)
            save_output(work/(name+'_p4.npz'), ref)
            row = compare_outputs(ref, actual)
            row['health'] = health(actual)
            row['histories'] = 20000
            row['seed'] = seed
            result['checks'][name] = row
            print(name, row['status'], 'health', row['health']['status'], flush=True)
            if row['status'] != 'pass' or row['health']['status'] != 'pass':
                h, first, diagnostic = first_history(cpu, data, 20000, seed, hr)
                if diagnostic is not None:
                    save_output(work/(name+'_first_history.npz'), diagnostic, history=np.array(h))
                result['stop'] = dict(check=name, history=h, first=first or row['first'],
                                      classification='未特定（最初のステップと原因はClaudeが調査）')
                raise GateStop('chain0 mismatch: '+name)
            if name == 'B1':
                saved_b1 = actual
        diag = cpu.run(20000, 20291950, history_rng=True, diagnostic=True)
        result['checks']['B5'] = compare_outputs(saved_b1, diag)
        result['checks']['B5']['diagnostic_records'] = len(diag['diagnostic'])
        result['checks']['B5']['diagnostic_overflow'] = int(diag['diagnostic_meta'][1])
        if result['checks']['B5']['status'] != 'pass':
            raise GateStop('B5 mismatch')
        failures = []
        for cap, maxsteps in ((2, 10000000), (64, 2)):
            variant = CPU(work, data, cap=cap)
            actual = variant.run(200, 20291950, history_rng=True, maxsteps=maxsteps)
            ref = p4_run(data, 200, 20291950, True, cap, maxsteps)
            row = compare_outputs(ref, actual)
            row.update(capacity=cap, maxsteps=maxsteps, counts=actual['counts'][5:7].tolist(),
                       unresolved=float(actual['totals'][3]))
            failures.append(row)
            if row['status'] != 'pass':
                h, first, diagnostic = first_history(variant, data, 200, 20291950, True, cap, maxsteps)
                if diagnostic is not None:
                    save_output(work/'B6_first_history.npz', diagnostic, history=np.array(h))
                result['checks']['B6'] = dict(status='fail', cases=failures)
                result['stop'] = dict(check='B6', history=h, first=first or row['first'])
                raise GateStop('B6 mismatch')
        result['checks']['B6'] = dict(status='pass', cases=failures)
        faults = []
        for fault in (1, 2, 3):
            variant = CPU(work, data, fault=fault)
            actual = variant.run(20000, 20291950, history_rng=True)
            row = compare_outputs(saved_b1, actual)
            h, first, diagnostic = first_history(variant, data, 20000, 20291950)
            if diagnostic is not None:
                save_output(work/f'B4_fault{fault}_first.npz', diagnostic, history=np.array(h))
            faults.append(dict(fault=fault, status='pass' if row['status'] == 'fail' and h is not None else 'fail',
                               first_history=h, first=first, comparison=row))
            if faults[-1]['status'] != 'pass':
                result['checks']['B4'] = dict(status='fail', cases=faults)
                raise GateStop('B4 failed to detect fault '+str(fault))
        result['checks']['B4'] = dict(status='pass', cases=faults)
        result['checks']['B7'] = check_b7(work, data)
        if result['checks']['B7']['status'] != 'pass':
            raise GateStop('B7 mismatch')
        if not skip_saved:
            precondition = verify_saved_inputs(data)
            cases = []
            for b, seed in ((0, 4067114529), (2, 3112960579)):
                with np.load(P3R/f'batch_{b:03}.npz') as z:
                    if int(z['N']) != 100000 or int(z['seed']) != seed:
                        raise GateStop('B3 saved N/seed mismatch')
                    actual = cpu.run(100000, seed)
                    transformed = saved_form(actual)
                    differences = {k: mismatch(np.asarray(z[k]), np.asarray(v)) for k, v in transformed.items()}
                    row = dict(batch=b, seed=seed, outputs=differences, health=health(actual),
                               status='pass' if all(d['mismatches'] == 0 for d in differences.values()) else 'fail')
                    cases.append(row)
                    if row['status'] != 'pass' or row['health']['status'] != 'pass':
                        result['checks']['B3'] = dict(status='fail', precondition=precondition, cases=cases)
                        first = next((dict(output=k, **v['first']) for k, v in differences.items()
                                      if 'first' in v), None)
                        # Saved batches have no per-history reference. First
                        # confirm a fresh P4 call matches the saved aggregate;
                        # only then use P4 to localize the C-side difference.
                        reference_batch = p4_run(data, 100000, seed)
                        saved_reference = saved_form(reference_batch)
                        reference_matches_saved = all(z[k].tobytes() == v.tobytes()
                                                      for k, v in saved_reference.items())
                        h, diagnostic = None, None
                        if reference_matches_saved:
                            h, local_first, diagnostic = first_history(cpu, data, 100000, seed, False)
                            first = local_first or first
                        if diagnostic is not None:
                            save_output(work/f'B3_batch{b}_first.npz', diagnostic, history=np.array(h))
                        result['stop'] = dict(check='B3', batch=b, history=h, first=first,
                                              fresh_P4_matches_saved=reference_matches_saved,
                                              classification='未特定（Claudeによる調査待ち）')
                        raise GateStop('B3 saved batch mismatch')
            result['checks']['B3'] = dict(status='pass', precondition=precondition, cases=cases)
        result['overall'] = 'incomplete' if skip_saved else 'chain0_pass'
        result['p5b_ready'] = False
    except GateStop as exc:
        result['overall'] = 'fail'
        result.setdefault('stop', dict(reason=str(exc)))
    finally:
        write_json(work/'chain0.json', result)
    return result


def dref():
    lock = json.loads((ROOT/'docs/validation/mv/p3a/pilot_lock.json').read_text())
    value = float(lock['Dref'])
    if value != 9.798535080485913e-12:
        raise GateStop('fixed Dref mismatch')
    return value


def mt_batches(data):
    verify_saved_inputs(data)
    paths = sorted(P3R.glob('batch*.npz'))
    expected = [P3R/f'batch_{b:03}.npz' for b in range(100)]
    if paths != expected:
        raise GateStop('saved MT batches missing or duplicated')
    seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(20291621).spawn(100)]
    rows = []
    for b, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as z:
            if int(z['N']) != 100000 or int(z['seed']) != seeds[b]:
                raise GateStop('saved MT batch metadata mismatch')
            r = {k: z[k] for k in z.files}
        selected = {dict(batches='coarse', fine_batches='fine', step_batches='steps').get(k, k): k
                    for k in OUTPUTS}
        # Health uses output names, while saved bins have already been summed.
        if health({value: r[key] for key, value in selected.items()})['status'] != 'pass':
            raise GateStop('saved MT batch is unhealthy')
        rows.append(r)
    return rows


def dose_samples(rows, new=False):
    from prototypes.mv_electron.run_coupled_p2 import MEV_TO_GY_G
    coarse = np.stack([np.sum(r['batches'], axis=0) if new else r['coarse'] for r in rows])
    return coarse*MEV_TO_GY_G/(100000*.2)


def blocks(doses):
    # Form the mean within each batch BEFORE estimating the sample variance.
    return np.column_stack([np.mean(doses[:, a:a+25], axis=1) for a in (25, 50, 75, 100, 125)])


def sem_check(work):
    work = Path(work)
    data = load_inputs(full=True)
    cpu = CPU(work, data, rng='philox')
    require_chain0(work, cpu, data)
    rows = mt_batches(data)
    values = blocks(dose_samples(rows))
    sem = np.std(values, axis=0, ddof=1)/np.sqrt(100)
    pct = sem/dref()*100
    result = dict(status='pass' if np.all(pct <= .23) else 'fail', MT_SEM_pct=pct.tolist(),
                  projected_equal_variance_difference_SEM_pct=(np.sqrt(2)*pct).tolist(),
                  stop_limit_pct=.23, Dref=dref(), batches=100, histories=10000000,
                  arrays={k: array_hash(a) for k, a in data.items()})
    write_json(work/'sem_check.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'arrays'}, ensure_ascii=False), flush=True)
    return result


def require_chain0(work, cpu, data):
    path = Path(work)/'chain0.json'
    if not path.exists():
        raise GateStop('chain0 including B3 required')
    result = json.loads(path.read_text())
    if any(result['checks'][k]['status'] != 'pass' for k in
           ('A1', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7')):
        raise GateStop('chain0 including B3 is incomplete or failed')
    expected = run_config(cpu, data)
    saved = result['transport_config']
    # Philox changes only the selected RNG macro; everything else is frozen.
    if (saved['sources'] != expected['sources'] or saved['arrays'] != expected['arrays']
        or saved['compiler'] != expected['compiler']
        or [f for f in saved['flags'] if not f.startswith('-DP5_RNG_')] !=
           [f for f in expected['flags'] if not f.startswith('-DP5_RNG_')]):
        raise GateStop('source/build/input changed since chain0')
    return result


def chain1(work, part):
    work = Path(work)
    data = load_inputs(full=True)
    cpu = CPU(work, data, rng='philox')
    require_chain0(work, cpu, data)
    path = work/'sem_check.json'
    if not path.exists():
        raise GateStop('sem-check must precede chain1')
    pre = json.loads(path.read_text())
    if pre['status'] != 'pass' or pre['arrays'] != {k: array_hash(a) for k, a in data.items()}:
        raise GateStop('SEM check failed or input changed')
    # Each independently invoked part is sequential. No process pool is spawned.
    indices = range(50*part, 50*(part+1))
    batch_run(cpu, data, work/'philox', indices)
    return dict(status='pass', part=part, completed_batches=50, histories=5000000,
                overall='incomplete', p5b_ready=False)


def difference_summary(left, right, ref):
    lm, rm = np.mean(left, axis=0), np.mean(right, axis=0)
    variance = np.var(left, axis=0, ddof=1)/len(left)+np.var(right, axis=0, ddof=1)/len(right)
    delta, sem = (lm-rm)/ref*100, np.sqrt(variance)/ref*100
    return dict(delta_pct=np.asarray(delta).tolist(), SEM_pct=np.asarray(sem).tolist())


def analyse(work, summary):
    work = Path(work)
    data = load_inputs(full=True)
    cpu = CPU(work, data, rng='philox')
    result = require_chain0(work, cpu, data)
    # Empty indices validate every existing batch/configuration without transport.
    completed = batch_run(cpu, data, work/'philox', [])
    if completed != list(range(100)):
        raise GateStop('C1 batches missing; no partial analysis')
    philox = []
    for b in range(100):
        with np.load(work/'philox'/f'batch_{b:03}.npz') as z:
            philox.append({k: z[k] for k in z.files})
    mt = mt_batches(data)
    pdose, mdose = dose_samples(philox, True), dose_samples(mt)
    comp = difference_summary(blocks(pdose), blocks(mdose), dref())
    passed = np.abs(comp['delta_pct']) <= 1.
    precise = np.asarray(comp['SEM_pct']) <= .33
    result['checks']['C1'] = dict(status='pass' if np.all(passed & precise) else 'fail',
                                 Dref=dref(), difference=comp, tolerance_pct=1., precision_pct=.33,
                                 block_pass=(passed & precise).tolist(), histories_per_side=10000000,
                                 seed=20291960)
    healthy = [health(r) for r in philox]
    result['checks']['C2'] = dict(status='pass' if all(r['status'] == 'pass' for r in healthy) else 'fail',
                                 batch_checks=healthy)
    interactions = {}
    for name, key, index in (('Compton', 'counts', 8), ('photoelectric', 'counts', 9),
                             ('pair', 'counts', (10, 11)), ('bremsstrahlung', 'brems', 0),
                             ('Moller', 'counts', 2), ('annihilation_in_flight', 'counts', 14)):
        def samples(rows):
            return np.array([np.sum(r[key][list(index)]) if isinstance(index, tuple) else r[key][index]
                             for r in rows])/100000
        a, b = samples(philox), samples(mt)
        interactions[name] = dict(delta_per_history=float(a.mean()-b.mean()),
                                  SEM=float(np.sqrt(a.var(ddof=1)/100+b.var(ddof=1)/100)))
    result['checks']['C3'] = dict(status='report', interactions=interactions, deep_blocks=comp,
                                 buildup_0_2_cm=difference_summary(pdose[:, :10].mean(axis=1),
                                                                   mdose[:, :10].mean(axis=1), dref()))
    if result['checks']['C1']['status'] != 'pass' or result['checks']['C2']['status'] != 'pass':
        result['overall'], result['p5b_ready'] = 'fail', False
    else:
        # Untimed warmup, then fixed 50k histories in a single process per code.
        p4_run(data, 100, 20291960)
        cpu.run(100, 20291960)
        new = cpu.run(50000, 20291960)
        old = p4_run(data, 50000, 20291960)
        result['checks']['C4'] = dict(status='report', histories=50000, seed=20291960,
                                     processes=1, machine=platform.machine(),
                                     C_Philox_us_per_history=float(new['seconds'])/50000*1e6,
                                     P4_MT_us_per_history=float(old['seconds'])/50000*1e6)
        result['overall'], result['p5b_ready'] = 'pass', True
    write_json(work/'analysis.json', result)
    small = dict(overall=result['overall'], p5b_ready=result['p5b_ready'],
                 checks={k: dict(status=v['status']) for k, v in result['checks'].items()})
    for k in ('C1', 'C3', 'C4'):
        small['checks'][k] = result['checks'][k]
    small['checks']['C2'].update(finite=all(r['finite'] for r in healthy),
                                violations=np.sum([r['violations'] for r in healthy], axis=0).tolist(),
                                step_limit=sum(r['step_limit'] for r in healthy),
                                stack_or_invalid_step=sum(r['stack_or_invalid_step'] for r in healthy),
                                unresolved=sum(r['unresolved'] for r in healthy))
    write_json(summary, small)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('gate')
    p.add_argument('--work', type=Path, required=True)
    p = sub.add_parser('chain0')
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--skip-saved', action='store_true')
    for name in ('sem-check', 'chain1', 'analyse'):
        p = sub.add_parser(name)
        p.add_argument('--work', type=Path, required=True)
        if name == 'chain1':
            p.add_argument('--part', type=int, choices=(0, 1), required=True)
        if name == 'analyse':
            p.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'gate':
            result = gate(args.work)
        elif args.command == 'chain0':
            result = chain0(args.work, args.skip_saved)
        elif args.command == 'sem-check':
            result = sem_check(args.work)
        elif args.command == 'chain1':
            result = chain1(args.work, args.part)
        else:
            result = analyse(args.work, args.summary)
    except GateStop as exc:
        print('STOP:', str(exc), flush=True)
        return 1
    return 1 if result.get('overall', result.get('status')) == 'fail' else 0


if __name__ == '__main__':
    sys.exit(main())
