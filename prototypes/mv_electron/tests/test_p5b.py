"""Fast wrapper and fixed-gate tests; all build/artifact files stay in /tmp."""
import ast
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from prototypes.mv_electron.gpu import p5b, p5b_core as core


@pytest.fixture(scope='module')
def work():
    # Retain artifacts, including failed builds. No cleanup/deletion routine.
    return Path(tempfile.mkdtemp(prefix='chatcarlo_p5b_test_'))


@pytest.fixture(scope='module')
def data():
    from prototypes.mv_electron.gpu import p5a
    return p5a.load_inputs(full=True)


@pytest.fixture(scope='module')
def cpu(work, data):
    return p5b.CPU(work, data)


def artificial(n=200):
    r = core.empty_output(core.history_shapes(n))
    r['totals'][:, 0] = 1.
    r['calls'][:] = 3
    r['steps'][:] = 2
    r['diag_used'][:] = 2
    r['diagnostic'][:, 0, 5] = 2
    r['diagnostic'][:, 1, 5] = 3
    r['batches'][:, 149] = .25
    r['fine_batches'][:, 599] = .25
    r['totals'][:, 1] = .25
    return r


def test_frozen_fingerprints_and_single_source_expansion():
    assert core.fingerprints() == core.FROZEN
    text = core.source()
    assert text.count('// p5a0_common.h\n') == 1
    assert text.count('struct P5Tables {') == 1
    assert '#include "p5_kernel.h"' not in text
    assert '#include "p5_rng.h"' not in text
    assert '#include "p5a0_common.h"' not in text
    assert text.index('#define INFINITY') < text.index('void p5_boundary')


def test_colab_has_no_private_imports():
    for name in ('p5b_core.py', 'p5b_colab.py'):
        tree = ast.parse((core.HERE/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not any(k in (node.module or '') for k in ('p5a', 'physics', 'numba'))
            if isinstance(node, ast.Import):
                assert not any(k in a.name for a in node.names for k in ('p5a', 'physics', 'numba'))
    notebook = json.loads((core.HERE/'p5b_colab.ipynb').read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
            assert cell['outputs'] == []


def test_gpu_argument_order_and_signed_save(cpu, data):
    from prototypes.mv_electron.gpu.p5b_colab import GPU
    class Arrays:
        asarray = staticmethod(np.asarray)
        asnumpy = staticmethod(np.asarray)
    gpu = GPU(Arrays(), data, {})
    expected = [data[k] for k in core.POINTERS[:11]]
    expected += [np.int32(v) for v in core.dimensions(data)[:8]]
    expected += [data[k] for k in core.POINTERS[11:]]
    expected += [np.float64(data[k].item()) for k in core.SCALARS]
    expected += [np.int32(v) for v in core.dimensions(data)[8:]]
    assert len(gpu.args) == 33
    for actual, wanted in zip(gpu.args, expected):
        assert actual.dtype == wanted.dtype
        assert np.array_equal(actual, wanted)
    assert gpu.download({'counts': np.array([2**63-1], np.uint64)})['counts'].dtype == np.int64
    with pytest.raises(core.Stop, match='exceeds int64'):
        gpu.download({'counts': np.array([2**63], np.uint64)})


def test_compile_record_requires_all_kernels_and_bundle():
    manifest = dict(source_sha256='a'*64, files={'p5b_inputs.npz': {'sha256': 'b'*64}})
    report = dict(status='pass', source_sha256='a'*64, options=list(core.OPTIONS),
                  bundle_sha256={'p5b_inputs.npz': 'b'*64}, environment={'gpu': 'T4'},
                  kernels={k: dict(status='pass') for k in (*core.KERNELS, 'p5b_run_kernel_fault')})
    assert core.compile_check(report, manifest, {'gpu': 'T4'})['status'] == 'pass'
    assert core.compile_check(report, manifest, {'gpu': 'other'})['status'] == 'fail'
    report['kernels']['p5b_rows_kernel']['status'] = 'fail'
    assert core.compile_check(report, manifest)['status'] == 'fail'


def test_all_table_fields(cpu, data):
    actual = cpu.fields()
    assert core.bits(actual['fields'], core.field_reference(data))
    assert core.bits(actual['dimensions'], np.array(core.dimensions(data), np.int32))


@pytest.mark.parametrize('name', list(core.FUNCTIONS))
def test_probe_values_and_input_local_calls(cpu, data, name):
    from prototypes.mv_electron.gpu import p5a
    # Independent P5a calls supply exactly the same per-input Philox state.
    original = p5a.CPU(Path(tempfile.mkdtemp(prefix='chatcarlo_p5b_probe_')), data, rng='philox')
    x = np.array([.021, .5, 1., 2., 10.])
    if name == 'gs_mu_cached':
        x = np.column_stack((x, np.resize(data['ks'], len(x))))
    actual, calls = cpu.function(name, x, core.SEED)
    if name in ('sample_W', 'gs_mu_cached'):
        expected, _, expected_calls = original.random(name, core.SEED, x[:1])
        assert core.bits(actual[:1].ravel(), expected)
        assert calls[0] == expected_calls
        for j in range(len(x)):
            value, counter = cpu.function(name, x[j:j+1], core.SEED)
            if j == 0:
                assert core.bits(value, actual[j:j+1])
                assert counter[0] == calls[j]
            assert calls[j] >= 1
        # Reversing inputs changes their history keys, not a shared RNG stream.
        assert np.array_equal(cpu.function(name, x, core.SEED)[1], calls)
    else:
        assert core.bits(actual, original.deterministic(name, x))
        assert not np.any(calls)


def test_philox_words_and_conversion(cpu):
    from prototypes.mv_electron.gpu import p5a0
    triples, expected = p5a0.known_answers()
    actual = cpu.philox(triples)
    assert core.bits(actual['philox_words'], expected)
    assert core.bits(actual['philox_values'], p5a0.philox_python(triples)[1])


@pytest.mark.parametrize('n', [20000, 100000])
def test_fixed_row_mapping(cpu, n):
    assert core.rows_check(*cpu.rows(n, 2**32+7), 2**32+7, n)['status'] == 'pass'


@pytest.mark.parametrize('n,start', [(0, 0), (20000001, 0), (1, -1), (2, 2**64-1)])
def test_range_rejection(n, start):
    with pytest.raises(ValueError):
        core.check_range(n, start)


def test_cpu_history_sum_and_split(cpu, work, data):
    from prototypes.mv_electron.gpu import p5a
    histories = cpu.per_history(300)
    production = p5a.CPU(work, data, rng='philox').run(300, core.SEED)
    assert core.l1(histories, production, len(data['cum']))['status'] == 'pass'
    assert core.health(histories)['status'] == 'pass'
    assert np.all(histories['calls'] > 0)
    assert np.all(histories['diag_used'] == np.minimum(histories['steps'], 4))
    pieces = [cpu.per_history(100, start=b*100) for b in range(3)]
    split = {k: np.concatenate([r[k] for r in pieces]) for k in core.HISTORY_ORDER}
    assert core.split_check(histories, split)['status'] == 'pass'
    assert core.c3(histories, core.sum_histories(histories, 20), 20)['status'] == 'pass'


def test_real_faults_call_direct_gates(cpu, work, data):
    reference = cpu.per_history(300)
    f3 = p5b.CPU(work, data, 3).per_history(300)
    assert core.c2_calls(reference, f3)['status'] == 'fail'
    f2 = p5b.CPU(work, data, 2).per_history(300)
    assert core.c2_bins(reference, f2, l3=True)['status'] == 'fail'
    f1 = p5b.CPU(work, data, 1).per_history(300)
    assert core.health(f1)['status'] == 'pass'
    assert core.c2_calls(reference, f1)['status'] == 'pass'
    assert core.c2_bins(reference, f1, l3=True)['status'] == 'pass'
    assert core.c3(reference, core.sum_histories(reference, 20, merge_fault=True), 20)['status'] == 'fail'


def c1_arrays():
    r = dict(fields=np.array([0., 1.]), dimensions=np.arange(11, dtype=np.int32),
             philox_words=np.zeros((1, 4), np.uint32), philox_values=np.zeros(1))
    for n in (20000, 100000):
        r['rows_'+str(n)] = np.array([0], np.int32)
        r['histories_'+str(n)] = np.array([0], np.uint64)
    for name in core.FUNCTIONS:
        n = 1000 if name == 'gs_mu_cached' else 1
        r['value_'+name] = np.zeros((n, core.FUNCTIONS[name]))
        r['calls_'+name] = np.ones(n, np.int64)
    return r


def test_c1_zero_tolerance_and_gs_boundary():
    a = c1_arrays()
    b = {k: v.copy() for k, v in a.items()}
    assert core.c1(a, b)['status'] == 'pass'
    b['calls_gs_mu_cached'][0] += 1
    assert core.c1(a, b)['status'] == 'pass'
    b['calls_gs_mu_cached'][1] += 1
    assert core.c1(a, b)['status'] == 'fail'
    b = {k: v.copy() for k, v in a.items()}
    b['value_gs_mu_cached'][0, 0] = 1e-8
    assert core.c1(a, b)['status'] == 'pass'
    b['value_gs_mu_cached'][0, 0] = np.nextafter(1e-8, np.inf)
    assert core.c1(a, b)['status'] == 'fail'
    b = {k: v.copy() for k, v in a.items()}
    b['value_sample_W'][0, 0] = -0.
    assert core.c1(a, b)['status'] == 'fail'
    assert np.array_equal(core.relative([0., 0., 1.], [0., 1., 2.]), [0., 1., .5])


def test_c2_call_and_bin_boundaries():
    a = artificial(100)
    b = {k: v.copy() for k, v in a.items()}
    b['diagnostic'][0, 1, 5] += 1
    assert core.c2_calls(a, b)['status'] == 'pass'
    b['diagnostic'][1, 1, 5] += 1
    assert core.c2_calls(a, b)['status'] == 'fail'
    a['diag_used'][0], b['diag_used'][0] = 1, 1
    assert core.c2_calls(a, b)['status'] == 'pass'
    b = {k: v.copy() for k, v in a.items()}
    b['batches'][0, 0] = 1e-9
    assert core.c2_bins(a, b, l3=True)['status'] == 'pass'
    b['batches'][0:2, 0] = np.nextafter(1e-9, np.inf)
    assert core.c2_bins(a, b, l3=True)['status'] == 'fail'
    assert core.c2_bins(a, b)['status'] == 'report'
    b['nspec_index'][0] = 1
    assert core.c2_source(a, b)['status'] == 'fail'


@pytest.mark.parametrize('key,index,value', [('violations', (0, 0), 1), ('counts', (0, 5), 1),
    ('counts', (0, 6), 1), ('brems', (0, 2), 1.), ('brems', (0, 3), 1.),
    ('totals', (0, 3), 1.), ('nan_count', (0,), 1), ('batches', (0, 0), np.inf)])
def test_health_each_condition(key, index, value):
    r = artificial()
    r[key][index] = value
    assert core.health(r)['status'] == 'fail'


def test_l1_brems_roundoff_bound():
    a = artificial()
    a['brems'][:, 0] = 1.
    a['brems'][:, 1] = 1.
    b = core.sum_histories(a, 20)
    limit = core.l1(a, b, 20)['brems_1_bound']
    b['brems'][1] += limit/2
    assert core.l1(a, b, 20)['status'] == 'pass'
    b['brems'][1] += limit*2
    assert core.l1(a, b, 20)['status'] == 'fail'


def test_c3_signed_absolute_sum_and_fault():
    a = artificial()
    a['brems'][::2, 4] = 1.
    a['brems'][1::2, 4] = -1.
    b = core.sum_histories(a, 20)
    bound = len(a['steps'])*2**-52*len(a['steps'])
    b['brems'][4] = bound
    assert core.c3(a, b, 20)['status'] == 'pass'
    b['brems'][4] = np.nextafter(bound, np.inf)
    assert core.c3(a, b, 20)['status'] == 'fail'
    assert core.c3(a, core.sum_histories(a, 20, merge_fault=True), 20)['status'] == 'fail'


def test_paired_sem_covariance_and_zero_sem():
    a = np.arange(100.)[:, None]+np.arange(5.)[None, :]
    d = core.paired((a+.001)-a, 1.)
    assert np.max(d['SEM_pct']) < 1e-12
    z = core.paired(np.ones((100, 5)), 100.)
    assert z['z'] == ['—']*5
    assert z['SEM_pct'] == [0.]*5
    coarse = np.repeat(np.arange(100.)[:, None], 150, axis=1)
    blocks = core.deep_blocks(coarse, 1, .2)
    np.testing.assert_allclose(blocks, np.repeat(np.arange(100.)[:, None], 5, axis=1), rtol=2e-15)


def test_c4_fixed_thresholds_and_spectrum():
    batch = core.empty_output(core.run_shapes(20))
    reference = dict(c4_blocks=np.zeros((100, 5)), c4_nspec=np.zeros((100, 20, 10), np.int64),
                     c4_counts=np.zeros((100, 16), np.int64), c4_step_batches=np.zeros((100, 100), np.int64),
                     Dref=np.array(100.), conversion=np.array(20000.))
    batch['batches'][0] = .5
    assert core.c4(reference, [batch]*100, 20)['status'] == 'pass'
    batch['batches'][0] = .5001
    assert core.c4(reference, [batch]*100, 20)['status'] == 'fail'
    batch['batches'][:] = 0
    batch['nspec'][0, 0] = 1
    assert core.c4(reference, [batch]*100, 20)['status'] == 'fail'
    batch['nspec'][0, 0] = 0
    reference['c4_blocks'][::2] = 1.98
    reference['c4_blocks'][1::2] = -1.98
    assert core.c4(reference, [batch]*100, 20)['status'] == 'pass'
    reference['c4_blocks'] *= 2/1.98
    assert core.c4(reference, [batch]*100, 20)['status'] == 'fail'


def identity():
    return dict(source_sha256='a'*64, options=list(core.OPTIONS), bundle_sha256={'input': 'b'*64},
                seed=core.SEED, per_batch=100000, batches=100, nspec=20, environment=None)


def checkpoint(directory, b=0, ident=None):
    ident = identity() if ident is None else ident
    r = core.empty_output(core.run_shapes(20))
    path = directory/f'batch_{b:03}.npz'
    core.save_npz(path, r, core.checkpoint_meta(ident, b, r))
    return path


def test_checkpoint_resume_and_metadata(work):
    directory = work/'resume_valid'
    ident = identity()
    assert core.resume(directory, ident) == {}
    checkpoint(directory)
    assert list(core.resume(directory, ident)) == [0]
    changed = dict(ident, seed=ident['seed']+1)
    with pytest.raises(core.Stop, match='configuration'):
        core.resume(directory, changed)
    path = checkpoint(directory)
    r = core.read_npz(path)
    meta = json.loads(str(r['metadata']))
    meta['start'] = 1
    r['metadata'] = np.array(json.dumps(meta))
    core.save_npz(path, r)
    with pytest.raises(core.Stop, match='fingerprint/range'):
        core.resume(directory, ident)


@pytest.mark.parametrize('bad', ['pending', 'duplicate', 'range', 'gap', 'shape', 'dtype', 'hash'])
def test_resume_rejects_bad_files(work, bad):
    directory = work/('resume_'+bad)
    ident = identity()
    core.resume(directory, ident)
    path = checkpoint(directory, b=1 if bad == 'gap' else 0)
    if bad == 'pending':
        (directory/'p5b_pending_abc.npz').write_bytes(b'partial')
    elif bad == 'duplicate':
        (directory/'batch_000_copy.npz').write_bytes(path.read_bytes())
    elif bad == 'range':
        (directory/'batch_100.npz').write_bytes(path.read_bytes())
    elif bad in ('shape', 'dtype', 'hash'):
        r = core.read_npz(path)
        if bad == 'shape':
            r['batches'] = r['batches'][:, :-1]
        elif bad == 'dtype':
            r['counts'] = r['counts'].astype(np.int32)
        else:
            r['batches'][0, 0] = 1.
        core.save_npz(path, r)
    with pytest.raises(core.Stop):
        core.resume(directory, ident)
