"""Canonical approval identities and observations at the native launch boundary."""
import ctypes as ct
import json
import numpy as np
from ..gpu import p5b_core as core, p7_core as h

OPTIONS = core.OPTIONS + ('-DP6_FAULT=0',)
# P7 GPU.measure ABI, positions are fixed independently of the constructed tuple.
GPU_BASE = (tuple(core.POINTERS[:11]) +
            ('ne', 'nt', 'nk', 'nu', 'nrates', 'ratecols', 'np', 'watercols') +
            tuple(core.POINTERS[11:]) + tuple(core.SCALARS) + ('nz', 'nbrems', 'nspec'))
GPU_PREFIX = GPU_BASE + ('logs', 'diagnostics', 'geometry', 'seed', 'start', 'n',
                         'maxsteps', 'field', 'field_half_cm', 'ssd_cm', 'regions', 'next', 'history')
CPU_CONFIGURE = ('tables', 'logs', 'diagnostics', 'initialize')
CPU_RUN = ('seed', 'start', 'n', 'maxsteps', 'geometry', 'field', 'field_half_cm', 'ssd_cm', 'regions')
CARRIED = ('spectrum', 'spectrum_file_sha256', 'engine')
DIM_NAMES = ('ne', 'nt', 'nk', 'nu', 'nrates', 'ratecols', 'np', 'watercols', 'nz', 'nbrems', 'nspec')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return core.sha(canonical(value))


def condition(scene, data):
    name = scene['source']['spectrum']
    spec = h.SPECTRA[name]
    return dict(field_half_cm=5., ssd_cm=100., spectrum=name, spectrum_file_sha256=spec['sha256'],
                edges_sha256=core.array_sha(data['edges']), cum_sha256=core.array_sha(data['cum']),
                geometry_sha256=core.sha(bytes(h.geometry('p6')[0])), thickness_cm=30, scoring='pdd_ocr',
                histories=scene['run']['histories'], seed=scene['run']['seed'], batches=100,
                batch_histories=scene['run']['histories']//100, maxsteps=h.MAXSTEPS, field=1, regions=1,
                engine=scene['run']['engine'])


def materials(data):
    """Only kernel inputs enter this hash; extra table arrays are bundle-verified."""
    h.k.frozen()
    return dict(sources={n: core.file_sha(h.HERE/n) for n in (*h.j.KERNEL_SOURCES, *h.k.SOURCES, *h.SOURCES)},
                inputs=core.describe({k: data[k] for k in core.POINTERS}),
                scalars={k: float(data[k].item()) for k in core.SCALARS},
                dimensions=dict(zip(DIM_NAMES, core.dimensions(data))),
                source_sha256=core.sha(h.source().encode()), options=list(OPTIONS))


def load_data(name='mohan10'):
    from ..gpu.p5a import load_inputs
    data = load_inputs(full=True)
    edges, _, cum = h.spectrum(name)
    data.update(edges=np.ascontiguousarray(edges), cum=np.ascontiguousarray(cum))
    core.check_inputs(data)
    return data


def identities(scene, data=None):
    if data is None:
        name = scene['source']['spectrum']
        data = load_data() if name == 'mohan10' else load_data(name)
    return condition(scene, data), materials(data), data


def check_equal(actual, expected, label=''):
    if actual != expected:
        if isinstance(actual, dict) and isinstance(expected, dict):
            changed = [k for k in actual.keys() | expected.keys() if actual.get(k) != expected.get(k)]
            raise core.Stop(label + ': ' + ', '.join(sorted(changed)))
        raise core.Stop(label + ': mismatch')


def scalar(value, dtype, name):
    if type(value) is not dtype:
        raise core.Stop(name + ': ABI type mismatch')
    return dict(type=np.dtype(dtype).name, value=value.item())


def array(value, name, to_host):
    host = np.ascontiguousarray(to_host(value))
    return dict(type='array', **core.describe({name: host})[name])


def observe_gpu(args, expected_condition, expected_materials, to_host, source, options):
    """Called with the very tuple passed to RawKernel, before the launch."""
    if len(args) != 80:
        raise core.Stop('GPU argument count/position mismatch')
    rows = []
    for i, name in enumerate(GPU_PREFIX):
        if name in core.POINTERS or name in ('logs', 'diagnostics', 'geometry', 'next'):
            row = array(args[i], name, to_host)
            wanted = ('uint8' if name == 'geometry' else 'uint64' if name == 'diagnostics'
                      else 'int32' if name == 'next' else 'float64')
            if row['dtype'] != wanted:
                raise core.Stop(name + ': ABI dtype mismatch')
        else:
            dtype = np.uint64 if name in ('seed', 'start') else np.float64 if name in (*core.SCALARS, 'field_half_cm', 'ssd_cm') else np.int32
            row = scalar(args[i], dtype, name)
        rows.append(dict(position=i, name=name, **row))
    # Outputs are part of the same actual tuple; record every position as well.
    output_names = (*core.HISTORY_ORDER, *core.RUN_ORDER, 'regions', 'region_ledger', 'sources', 'regions', 'region_ledger')
    for i, name in enumerate(output_names, len(GPU_PREFIX)):
        rows.append(dict(position=i, name='output_'+name, **array(args[i], name, to_host)))
    for i in range(76, 80):
        if i == 79 and hasattr(args[i], 'shape') and args[i].ndim:
            row = array(args[i], 'additions', to_host)
        else:
            row = scalar(args[i], np.int32 if i == 77 else np.uint64, 'tail_'+str(i))
        rows.append(dict(position=i, name=('records', 'capacity', 'source_records', 'additions')[i-76], **row))
    mat = dict(expected_materials)
    mat['inputs'] = {k: {x: rows[GPU_PREFIX.index(k)][x] for x in ('shape', 'dtype', 'sha256')} for k in core.POINTERS}
    mat['dimensions'] = {k: rows[GPU_PREFIX.index(k)]['value'] for k in DIM_NAMES}
    mat['scalars'] = {k: rows[GPU_PREFIX.index(k)]['value'] for k in core.SCALARS}
    mat.update(source_sha256=core.sha(source.encode()), options=list(options))
    observed = {k: rows[GPU_PREFIX.index(k)]['value'] for k in ('seed', 'start', 'n', 'maxsteps', 'field', 'field_half_cm', 'ssd_cm', 'regions')}
    observed.update({k+'_sha256': rows[GPU_PREFIX.index(k)]['sha256'] for k in ('geometry', 'edges', 'cum')})
    return observation(rows, observed, mat, expected_condition, expected_materials)


def observation(rows, observed, mat, cond, expected_materials):
    rebuilt = dict(cond)
    for k, v in observed.items():
        if k not in ('start', 'n'):
            rebuilt[k] = v
    # Total histories are verified again from all 100 ranges upon collection.
    if observed['n'] != cond['batch_histories'] and observed['n'] != 2048:
        raise core.Stop('n: unexpected batch range')
    record = dict(arguments=rows, observed=observed, condition=rebuilt, materials=mat,
                  carried_not_independently_observed=list(CARRIED),
                  geometry_observes=['thickness_cm', 'scoring'])
    # Record faults too, so the VM negative control can be collected and checked locally.
    return record


def reconstruct(record):
    rows = record['arguments']
    gpu = 'configure' not in record
    prefix = GPU_PREFIX if gpu else CPU_RUN
    if [r['position'] for r in rows] != list(range(len(rows))):
        raise core.Stop('observation ABI positions mismatch')
    if [r['name'] for r in rows[:len(prefix)]] != list(prefix):
        raise core.Stop('observation ABI names mismatch')
    index = {r['name']: r for r in rows[:len(prefix)]}
    observed = {k: index[k]['value'] for k in ('seed', 'start', 'n', 'maxsteps', 'field', 'field_half_cm', 'ssd_cm', 'regions')}
    observed['geometry_sha256'] = index['geometry']['sha256']
    actual_materials = dict(record['materials'])
    if gpu:
        actual_materials['inputs'] = {k: {x: index[k][x] for x in ('shape', 'dtype', 'sha256')} for k in core.POINTERS}
        actual_materials['dimensions'] = {k: index[k]['value'] for k in DIM_NAMES}
        actual_materials['scalars'] = {k: index[k]['value'] for k in core.SCALARS}
    else:
        configuration = record['configure']
        tables = configuration['arguments'][0]
        for k in ('inputs', 'dimensions', 'scalars'):
            check_equal(configuration[k], tables[k], 'configure '+k)
            actual_materials[k] = tables[k]
    for key in ('edges', 'cum'):
        observed[key+'_sha256'] = actual_materials['inputs'][key]['sha256']
    check_equal(actual_materials, record['materials'], 'argument materials reconstruction')
    check_equal(observed, record['observed'], 'argument condition reconstruction')
    return observed, actual_materials


def verify_observation(record, cond, mat, n=None, start=None):
    observed, actual_materials = reconstruct(record)
    rebuilt = dict(cond)
    for k, v in observed.items():
        if k not in ('n', 'start'):
            rebuilt[k] = v
    check_equal(record['condition'], rebuilt, 'observation reconstruction')
    check_equal(rebuilt, cond, 'condition')
    check_equal(actual_materials, mat, 'materials')
    if n is not None and observed['n'] != n:
        raise core.Stop('n: mismatch')
    if start is not None and observed['start'] != start:
        raise core.Stop('start: mismatch')
    return digest(rebuilt), digest(record['materials'])


def observe_cpu_configure(args, data):
    """Read the P5Tables struct and its actual pointer contents, not self.data."""
    from ..gpu.p5d_core import Tables
    if len(args) != len(CPU_CONFIGURE):
        raise core.Stop('configure ABI count')
    tables = ct.cast(args[0], ct.POINTER(Tables)).contents
    dims = {k: int(getattr(tables, k)) for k in DIM_NAMES}
    # Check dimensions before dereferencing pointers to avoid reading beyond allocations.
    check_equal(dims, dict(zip(DIM_NAMES, core.dimensions(data))), 'configure dimensions')
    inputs = {}
    for name in core.POINTERS:
        pointer = getattr(tables, name)
        # Reject unknown addresses before dereferencing (also detects swapped table slots).
        if ct.cast(pointer, ct.c_void_p).value != data[name].ctypes.data:
            raise core.Stop('configure pointer: '+name)
        values = np.ctypeslib.as_array(pointer, shape=(data[name].size,)).reshape(data[name].shape)
        inputs[name] = core.describe({name: values})[name]
    scalars = {k: float(getattr(tables, k)) for k in core.SCALARS}
    diagnostics = np.ctypeslib.as_array(args[2], shape=(40,)).reshape(20, 2)
    if type(args[3]) is not int or args[3] not in (0, 1):
        raise core.Stop('configure initialize: ABI type/value mismatch')
    logs = array(args[1], 'logs', np.asarray)
    return dict(arguments=[dict(position=0, name='tables', type='P5Tables', inputs=inputs, dimensions=dims, scalars=scalars),
                           dict(position=1, name='logs', **logs),
                           dict(position=2, name='diagnostics', **array(diagnostics, 'diagnostics', np.asarray)),
                           dict(position=3, name='initialize', type='int32', value=args[3])],
                inputs=inputs, dimensions=dims, scalars=scalars, logs=logs, initialize=args[3])


def observe_cpu_run(args, configure, cond, mat, history=False):
    if len(args) != (30 if history else 23):
        raise core.Stop('CPU run argument count/position mismatch')
    rows = []
    observed = {}
    for i, name in enumerate(CPU_RUN):
        if name == 'geometry':
            geometry = ct.cast(args[i], ct.POINTER(h.p6.Geometry)).contents
            row = dict(type='geometry', sha256=core.sha(bytes(geometry)))
            observed['geometry_sha256'] = row['sha256']
        else:
            dtype = float if name in ('field_half_cm', 'ssd_cm') else int
            if type(args[i]) is not dtype:
                raise core.Stop(name+': CPU ABI type mismatch')
            row = dict(type='float64' if dtype is float else 'uint64' if name in ('seed', 'start') else 'int32', value=args[i])
            observed[name] = args[i]
        rows.append(dict(position=i, name=name, **row))
    output_names = (*core.HISTORY_ORDER, 'regions', 'region_ledger', 'sources', 'trace', 'capacity', 'meta') if history else (*core.RUN_ORDER, 'regions', 'region_ledger', 'additions')
    offset = 10 if history else 9
    if history:
        rows.append(dict(position=9, name='fresh', type='int32', value=args[9]))
    for i, name in enumerate(output_names, offset):
        value = args[i]
        if isinstance(value, np.ndarray):
            row = array(value, name, np.asarray)
        elif name == 'maxdepth':
            row = array(np.ctypeslib.as_array(value, shape=(args[2] if history else 1,)), name, np.asarray)
        elif name == 'capacity':
            row = dict(type='int32', value=int(value))
        elif name == 'additions':
            if value is None:
                row = dict(type='null')
            else:
                pointer = ct.cast(value, ct.POINTER(ct.c_uint64))
                row = array(np.ctypeslib.as_array(pointer, shape=(100*h.BINS,)), name, np.asarray)
        else:
            raise core.Stop('CPU ABI output type mismatch: '+name)
        rows.append(dict(position=i, name=name, **row))
    rebuilt_mat = dict(mat, **{k: configure[k] for k in ('inputs', 'dimensions', 'scalars')})
    for key in ('edges', 'cum'):
        observed[key+'_sha256'] = configure['inputs'][key]['sha256']
    record = observation(rows, observed, rebuilt_mat, cond, mat)
    record['configure'] = configure
    return record
