"""Shared P5b/P5c data contracts and frozen gates. NumPy is the only dependency.

No transport imports, GPU imports, deletion, or performance measurements.
"""
import hashlib
import json
from pathlib import Path
import re
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 20291960
MAXSTEPS = 10000000
OPTIONS = ('-DP5_GPU', '-DP5_RNG_PHILOX', '--fmad=false',
           '-DP5_TRANSPORT_FAULT=0', '-DP5_STACK_CAP=64')
KERNELS = ('p5b_run_kernel', 'p5b_per_history_kernel', 'p5b_fields_kernel',
           'p5b_functions_kernel', 'p5b_philox_kernel', 'p5b_rows_kernel')
FROZEN = {
    'p5_cpu.cpp': '11fd05ed68100db0b3bc02231c659f86b72b7403ae17fc88085d91791e9a7f81',
    'p5_kernel.h': '39d82e04f5c969c92c1ee35d026241119364134fecf1db801ff5d0a4a44bd549',
    'p5_rng.h': '6c3257588ee4be2d0881a05f5d0869b65b942f68fd596a9f0ea981bcaaa9760f',
    'p5a0_common.h': '8c209cfe016241bbbb5d586830dfe5a1ddff648f8f75d3b431dd2ae6d61e70c2',
    'p5a.py': '2609e2a6ce666b3d56214e11929dcd642df261765b6bf55cdc1364d059743934',
}
POINTERS = ('constants', 'loget', 'ed', 'ts', 'ks', 'us', 'ys', 'rates',
            'pes', 'water', 'thresholds', 'er', 'Z', 'NJ', 'dl',
            'brems_e', 'brems_v', 'edges', 'cum')
SCALARS = ('bpar', 'delcm', 'delpos')
INPUTS = set(POINTERS + SCALARS + ('single', 'nist_E', 'nist_muen'))
FUNCTIONS = dict(collision=1, hard=2, coefficients=4, table_rates=2,
                 sample_W=1, gs_mu_cached=1)
HISTORY_ORDER = ('batches', 'fine_batches', 'totals', 'maxima', 'brems',
                 'violations', 'counts', 'nspec_index', 'steps', 'maxdepth',
                 'calls', 'diagnostic', 'diag_used', 'nan_count')
RUN_ORDER = ('batches', 'fine_batches', 'totals', 'maxima', 'brems',
             'violations', 'counts', 'nspec', 'step_batches', 'maxdepth', 'nan_count')
INTEGER = {'violations', 'counts', 'nspec_index', 'steps', 'maxdepth', 'calls',
           'diag_used', 'nan_count', 'nspec', 'step_batches'}


class Stop(RuntimeError):
    """A failed gate or inconsistent artifact: retain files and stop."""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def array_sha(a):
    return sha(np.ascontiguousarray(a).tobytes())


def describe(arrays):
    return {k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=array_sha(v))
            for k, v in arrays.items()}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix='p5b_pending_',
                                     suffix='.json', delete=False) as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        pending = Path(f.name)
    pending.replace(path)


def save_npz(path, arrays, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = dict(arrays)
    if metadata is not None:
        values['metadata'] = np.array(json.dumps(metadata, sort_keys=True, allow_nan=False))
    with tempfile.NamedTemporaryFile('wb', dir=path.parent, prefix='p5b_pending_',
                                     suffix='.npz', delete=False) as f:
        np.savez(f, **values)
        pending = Path(f.name)
    pending.replace(path)


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        if len(z.files) != len(set(z.files)):
            raise Stop('duplicate npz entries')
        return {k: z[k] for k in z.files}


def fingerprints():
    actual = {k: file_sha(HERE/k) for k in FROZEN}
    if actual != FROZEN:
        raise Stop('L0: frozen P5a source fingerprint mismatch')
    return actual


def source():
    fingerprints()
    names = ('p5a0_common.h', 'p5_rng.h', 'p5_kernel.h', 'p5b_wrap.h')
    chunks = []
    for name in names:
        s = (HERE/name).read_text()
        s = re.sub(r'^\s*#include "(?:' + '|'.join(re.escape(n) for n in names) +
                   r')"\s*$', '', s, flags=re.MULTILINE)
        chunks.append('// '+name+'\n'+s+'\n')
    template = (HERE/'p5b_gpu.cu').read_text()
    if template.count('//@@COMMON@@') != 1:
        raise Stop('source insertion marker mismatch')
    return template.replace('//@@COMMON@@', ''.join(chunks))


def check_inputs(data, expected=None):
    if set(data) != INPUTS:
        raise Stop('expected exactly 25 input arrays')
    if any(a.dtype != np.float64 or not a.flags.c_contiguous or not np.isfinite(a).all()
           for a in data.values()):
        raise Stop('input dtype/contiguity/nonfinite mismatch')
    n = dimensions(data)
    ne, nt, nk, nu, nr, rc, np_, wc, nz, nb, ns = n
    shapes = dict(constants=(6,), ed=(ne,), er=(ne,), ys=(nt, nk, nu),
                  rates=(nr, rc), water=(np_, wc), thresholds=(2,), NJ=(nz,),
                  dl=(6, 6), brems_v=(nb, 5), edges=(ns+1,),
                  single=(nr, nu+1), nist_muen=data['nist_E'].shape)
    if rc not in (2, 3) or wc != 7 or not 0 < ns*10 <= 640:
        raise Stop('input dimensions out of domain')
    for k, shape in shapes.items():
        if data[k].shape != shape:
            raise Stop('input shape mismatch: '+k)
    for k in ('loget', 'ts', 'ks', 'us', 'pes', 'brems_e', 'edges', 'cum', 'nist_E'):
        if data[k].ndim != 1 or len(data[k]) < (1 if k == 'cum' else 2) or np.any(np.diff(data[k]) <= 0):
            raise Stop('invalid input grid: '+k)
    for k in SCALARS:
        if data[k].size != 1:
            raise Stop('invalid scalar: '+k)
    if data['cum'][0] <= 0 or data['cum'][-1] != 1 or np.any(data['rates'][:, -2:] <= 0):
        raise Stop('invalid cumulative spectrum or rates')
    if rc == 3 and np.any(np.diff(data['rates'][:, 0]) <= 0):
        raise Stop('invalid independent rate grid')
    if expected is not None and describe(data) != expected:
        raise Stop('L0: input hashes/shapes/dtypes differ from saved config')
    return describe(data)


def dimensions(data):
    return (len(data['loget']), len(data['ts']), len(data['ks']), len(data['us']),
            len(data['rates']), data['rates'].shape[1], len(data['pes']),
            data['water'].shape[1], len(data['Z']), len(data['brems_e']), len(data['cum']))


def field_reference(data):
    return np.concatenate([data[k].ravel() for k in POINTERS+SCALARS])


def check_range(n, start=0, seed=SEED, maxsteps=MAXSTEPS):
    if (type(n) is not int or not 0 < n <= 20000000 or type(start) is not int
        or start < 0 or start+n > 2**64 or type(seed) is not int or not 0 <= seed < 2**64
        or type(maxsteps) is not int or not 0 < maxsteps <= 2147483647):
        raise ValueError('invalid history range/seed/step limit')


def history_shapes(n):
    return dict(batches=(n, 150), fine_batches=(n, 600), totals=(n, 6), maxima=(n, 3),
                brems=(n, 10), violations=(n, 3), counts=(n, 16), nspec_index=(n,),
                steps=(n,), maxdepth=(n,), calls=(n,), diagnostic=(n, 4, 7),
                diag_used=(n,), nan_count=(n,))


def run_shapes(nspec):
    return dict(batches=(100, 150), fine_batches=(100, 600), totals=(6,), maxima=(3,),
                brems=(10,), violations=(3,), counts=(16,), nspec=(nspec, 10),
                step_batches=(100,), maxdepth=(), nan_count=())


def empty_output(shapes):
    return {k: np.zeros(s, np.int64 if k in INTEGER else np.float64) for k, s in shapes.items()}


def validate_output(r, shapes):
    if set(r) != set(shapes):
        raise Stop('output fields mismatch')
    for k, shape in shapes.items():
        if r[k].shape != shape or r[k].dtype != (np.int64 if k in INTEGER else np.float64):
            raise Stop('output shape/dtype mismatch: '+k)
        if k in INTEGER and np.any(r[k] < 0):
            raise Stop('negative output counter: '+k)
    if 'diag_used' in r and np.any(r['diag_used'] > 4):
        raise Stop('invalid diagnostic capacity')


def bit_elements(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise Stop('comparison shape/dtype mismatch')
    return np.all(np.ascontiguousarray(a).view(np.uint8).reshape(a.size, a.dtype.itemsize) ==
                  np.ascontiguousarray(b).view(np.uint8).reshape(b.size, b.dtype.itemsize), axis=1).reshape(a.shape)


def bits(a, b):
    return bool(np.all(bit_elements(a, b)))


def relative(a, b):
    a, b = np.asarray(a), np.asarray(b)
    den = np.maximum(np.abs(a), np.abs(b))
    return np.divide(np.abs(a-b), den, out=np.zeros_like(den, dtype=float), where=den != 0)


def verdict(ok, **details):
    return dict(status='pass' if ok else 'fail', **details)


def require(result, name):
    if result['status'] != 'pass':
        raise Stop(name+' failed')
    return result


def health(r):
    nonfinite = sum(int(np.count_nonzero(~np.isfinite(v))) for v in r.values())
    bad = dict(violations=int(np.count_nonzero(r['violations'])),
               step_limit=int(np.count_nonzero(r['counts'][..., 5])),
               stack_or_step=int(np.count_nonzero(r['counts'][..., 6])),
               brems_excess=int(np.count_nonzero(r['brems'][..., 2:4])),
               unresolved=int(np.count_nonzero(r['totals'][..., 3])),
               nan_count=int(np.sum(r['nan_count'])), nonfinite=nonfinite)
    return verdict(not any(bad.values()), **bad)


def sum_histories(r, nspec, merge_fault=False, absolute=False):
    n = len(r['steps'])
    check_range(n)
    validate_output(r, history_shapes(n))
    if np.any(r['nspec_index'] >= nspec*10):
        raise Stop('spectrum index out of range')
    out = empty_output(run_shapes(nspec))
    rows = np.arange(n)*100//n
    # Deliberately use history order; vector reductions may reorder additions.
    for i, row in enumerate(rows):
        for k in ('batches', 'fine_batches'):
            v = np.abs(r[k][i]) if absolute else r[k][i]
            if merge_fault and k == 'batches':
                v = v.copy()
                nz = np.flatnonzero(v)
                if len(nz):
                    v[nz[-1]] = 0
            out[k][row] += v
        for k in ('totals', 'brems'):
            out[k] += np.abs(r[k][i]) if absolute else r[k][i]
        for k in ('counts', 'violations'):
            out[k] += r[k][i]
        out['maxima'] = np.maximum(out['maxima'], r['maxima'][i])
        out['maxdepth'][...] = max(int(out['maxdepth']), int(r['maxdepth'][i]))
        out['step_batches'][row] += r['steps'][i]
        out['nspec'].flat[int(r['nspec_index'][i])] += 1
        out['nan_count'][...] += r['nan_count'][i]
    return out


def l1(r, production, nspec):
    total = sum_histories(r, nspec)
    exact = {k: bool(np.array_equal(total[k], production[k])) for k in RUN_ORDER
             if k not in ('brems', 'nan_count')}
    mask = np.arange(10) != 1
    exact['brems_except_1'] = bool(np.array_equal(total['brems'][mask], production['brems'][mask]))
    n = float(production['brems'][0])
    m = max(0., n-1)
    bound = m*2**-52*float(production['brems'][1])/(1-m*2**-53)
    diff = abs(float(total['brems'][1]-production['brems'][1]))
    return verdict(all(exact.values()) and diff <= bound and bool(np.all(r['calls'] > 0)),
                   exact=exact, brems_1_difference=diff, brems_1_bound=bound,
                   calls_positive=bool(np.all(r['calls'] > 0)))


def rows_check(rows, histories, start, n):
    return verdict(bits(rows, (np.arange(n)*100//n).astype(np.int32)) and
                   bits(histories, np.arange(n, dtype=np.uint64)+np.uint64(start)))


def c1(reference, actual):
    exact_keys = ('fields', 'dimensions', 'philox_words', 'philox_values',
                  'rows_20000', 'histories_20000', 'rows_100000', 'histories_100000')
    a = verdict(all(bits(reference[k], actual[k]) for k in exact_keys))
    deterministic = {}
    for name in ('collision', 'hard', 'coefficients', 'table_rates'):
        r, g = reference['value_'+name], actual['value_'+name]
        d = relative(r, g)
        deterministic[name] = verdict(bool(np.isfinite(d).all() and np.all(d <= 1e-8)),
            bit_fraction=float(np.mean(bit_elements(r, g))), max_relative=float(np.max(d)))
    b = verdict(all(v['status'] == 'pass' for v in deterministic.values()), functions=deterministic)
    c = verdict(bool(np.isfinite(actual['value_sample_W']).all()) and
                bool(np.isfinite(reference['value_sample_W']).all()) and
                bits(reference['value_sample_W'], actual['value_sample_W']) and
                bits(reference['calls_sample_W'], actual['calls_sample_W']))
    match = bit_elements(reference['calls_gs_mu_cached'], actual['calls_gs_mu_cached'])
    delta = np.abs(reference['value_gs_mu_cached']-actual['value_gs_mu_cached']).ravel()
    fraction = float(np.mean(match))
    d = verdict(fraction >= .999 and bool(np.isfinite(delta[match]).all() and np.all(delta[match] <= 1e-8)),
                call_fraction=fraction, differing_calls=int(np.count_nonzero(~match)),
                max_abs_matched=float(delta[match].max(initial=0)))
    return verdict(all(v['status'] == 'pass' for v in (a, b, c, d)), a=a, b=b, c=c, d=d)


def c2_source(a, b):
    return verdict(bits(a['totals'][:, 0], b['totals'][:, 0]) and bits(a['nspec_index'], b['nspec_index']))


def c2_calls(a, b):
    na, nb = np.minimum(a['diag_used'], 2), np.minimum(b['diag_used'], 2)
    same = na == nb
    for j in range(2):
        same &= (na <= j) | (a['diagnostic'][:, j, 5] == b['diagnostic'][:, j, 5])
    fraction = float(np.mean(same))
    return verdict(fraction >= .99, fraction=fraction, differing=int(np.count_nonzero(~same)))


def paired(delta, dref):
    delta = np.asarray(delta)
    if len(delta) < 2 or not np.isfinite(delta).all() or not np.isfinite(dref) or dref <= 0:
        raise Stop('invalid paired samples/Dref')
    mean, sem = delta.mean(axis=0), delta.std(axis=0, ddof=1)/np.sqrt(len(delta))
    z = ['—' if s == 0 else float(m/s) for m, s in zip(np.atleast_1d(mean), np.atleast_1d(sem))]
    return dict(mean=mean.tolist(), SEM=sem.tolist(),
                mean_pct=(mean/dref*100).tolist(), SEM_pct=(sem/dref*100).tolist(), z=z)


def deep_blocks(coarse, histories, conversion):
    doses = np.asarray(coarse)*conversion/(histories*.2)
    return np.stack([doses[..., a:a+25].mean(axis=-1) for a in (25, 50, 75, 100, 125)], axis=-1)


def c2_bins(a, b, start=0, dref=None, conversion=None, l3=False):
    same_steps = a['steps'] == b['steps']
    same_path = same_steps & (a['calls'] == b['calls'])
    bins_match = np.all(np.abs(a['batches']-b['batches']) <= a['totals'][:, :1]*1e-9, axis=1)
    fraction = float(np.mean(bins_match[same_path])) if np.any(same_path) else 0.
    skeletal = same_steps & bins_match
    differing = np.flatnonzero(~skeletal)
    result = dict(status=('pass' if fraction >= .99 else 'fail') if l3 else 'report',
                  same_path=int(np.count_nonzero(same_path)), bin_fraction=fraction,
                  matched_fraction=float(np.mean(skeletal)),
                  first_diverging_history=int(start+differing[0]) if len(differing) else None)
    if dref is not None:
        result['paired_blocks'] = paired(deep_blocks(b['batches']-a['batches'], 1, conversion), dref)
    return result


def c2(a, b, dref, conversion):
    for r in (a, b):
        validate_output(r, history_shapes(20000))
    checks = dict(a_cpu=health(a), a_gpu=health(b), b=c2_source(a, b), c=c2_calls(a, b),
                  d=c2_bins(a, b, dref=dref, conversion=conversion))
    return verdict(all(checks[k]['status'] == 'pass' for k in ('a_cpu', 'a_gpu', 'b', 'c')), **checks)


def c3(histories, production, nspec):
    validate_output(production, run_shapes(nspec))
    total, absolute = sum_histories(histories, nspec), sum_histories(histories, nspec, absolute=True)
    n = len(histories['steps'])
    max_k = int(np.bincount(np.arange(n)*100//n, minlength=100).max())
    details = {}
    for k in RUN_ORDER:
        if k in ('batches', 'fine_batches', 'totals', 'brems'):
            K = n if k in ('totals', 'brems') else max_k
            bound = K*2**-52*absolute[k]
            delta = np.abs(total[k]-production[k])
            ok = np.isfinite(delta) & (delta <= bound)
        else:
            ok = bit_elements(total[k], production[k])
        details[k] = int(np.count_nonzero(~ok))
    return verdict(not any(details.values()), exceeding=details)


def split_check(whole, split):
    return verdict(all(bits(whole[k], split[k]) for k in HISTORY_ORDER))


def c4(reference, batches, nspec):
    if len(batches) != 100:
        raise Stop('C4 requires exactly 100 batches')
    blocks, healthy, spectra = [], True, True
    for b, r in enumerate(batches):
        validate_output(r, run_shapes(nspec))
        healthy &= health(r)['status'] == 'pass'
        spectra &= bits(r['nspec'], reference['c4_nspec'][b])
        blocks.append(deep_blocks(r['batches'].sum(axis=0), 100000, float(reference['conversion'])))
    summary = paired(np.stack(blocks)-reference['c4_blocks'], float(reference['Dref']))
    passes = (np.abs(summary['mean_pct']) <= .5) & (np.array(summary['SEM_pct']) <= .2)
    reports = {}
    for k in ('counts', 'step_batches'):
        p = paired(np.stack([r[k] for r in batches])-reference['c4_'+k], 1.)
        reports[k] = dict(mean=p['mean'], SEM=p['SEM'], z=p['z'], unit='per batch')
    return verdict(healthy and spectra and bool(np.all(passes)), blocks=summary,
                   block_pass=passes.tolist(), healthy=healthy, spectrum_exact=spectra,
                   count_and_step_differences=reports)


def c5(a, b):
    return dict(status='report', max_abs={k: float(np.max(np.abs(a[k]-b[k])))
                                        for k in RUN_ORDER})


def load_bundle(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'p5b_manifest.json').read_text())
    fingerprints()
    if manifest['sources'] != FROZEN or manifest['source_sha256'] != sha(source().encode()):
        raise Stop('C0: source fingerprint mismatch')
    for name in ('p5b_inputs.npz', 'p5b_reference.npz'):
        info = manifest['files'][name]
        if (directory/name).stat().st_size != info['bytes'] or file_sha(directory/name) != info['sha256']:
            raise Stop('C0: bundle byte size/sha256 mismatch: '+name)
    data, reference = read_npz(directory/'p5b_inputs.npz'), read_npz(directory/'p5b_reference.npz')
    check_inputs(data, manifest['arrays'])
    if describe(reference) != manifest['reference_arrays']:
        raise Stop('reference shape/dtype/hash mismatch')
    return data, reference, manifest


def compile_check(report, manifest, env=None):
    names = (*KERNELS, 'p5b_run_kernel_fault')
    ok = (report.get('status') == 'pass' and report.get('source_sha256') == manifest['source_sha256']
          and report.get('options') == list(OPTIONS)
          and report.get('bundle_sha256') == {k: v['sha256'] for k, v in manifest['files'].items()}
          and all(report.get('kernels', {}).get(k, {}).get('status') == 'pass' for k in names)
          and (env is None or report.get('environment') == env))
    return verdict(ok)


def batch_identity(manifest, environment=None):
    return dict(source_sha256=manifest['source_sha256'], options=list(OPTIONS),
                bundle_sha256={k: v['sha256'] for k, v in manifest['files'].items()},
                seed=SEED, per_batch=100000, batches=100, nspec=20,
                environment=environment)


def checkpoint_meta(identity, b, arrays):
    if not 0 <= b < 100:
        raise Stop('batch outside fixed range')
    return dict(identity=identity, batch=b, start=100000*b, end=100000*(b+1),
                outputs=describe(arrays))


def read_checkpoint(path, identity, b):
    arrays = read_npz(path)
    try:
        meta = json.loads(str(arrays.pop('metadata')))
    except (KeyError, ValueError) as exc:
        raise Stop('missing/invalid checkpoint metadata') from exc
    validate_output(arrays, run_shapes(identity['nspec']))
    if meta != checkpoint_meta(identity, b, arrays):
        raise Stop('checkpoint fingerprint/range/shape/dtype/hash mismatch')
    require(health(arrays), 'checkpoint health')
    return arrays


def resume(directory, identity):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any('pending' in p.name or p.suffix in ('.tmp', '.part') for p in directory.iterdir()):
        raise Stop('unfinished checkpoint present; retain it and stop')
    config = directory/'config.json'
    if config.exists():
        if json.loads(config.read_text()) != identity:
            raise Stop('resume configuration mismatch')
    elif any(directory.iterdir()):
        raise Stop('checkpoint directory has no configuration')
    else:
        write_json(config, identity)
    found = {}
    for p in directory.iterdir():
        if p.name == 'config.json':
            continue
        match = re.fullmatch(r'batch_(\d{3})\.npz', p.name)
        if match is None or not 0 <= int(match[1]) < 100 or int(match[1]) in found:
            raise Stop('duplicate/unexpected/out-of-range checkpoint')
        b = int(match[1])
        read_checkpoint(p, identity, b)
        found[b] = p
    if sorted(found) != list(range(len(found))):
        raise Stop('missing earlier completed checkpoint')
    return found


def record(results, key, value):
    results['checks'][key] = value
    print(key, value['status'], flush=True)
    require(value, key)
