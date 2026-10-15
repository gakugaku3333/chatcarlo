"""P5e contracts, selection and artifact checks (NumPy only)."""
import json
import numpy as np
from . import p5b_core as core, p5d_core as d

VARIANTS = {f'V{i}': dict(allocation=a, stack=s, registers=r)
            for i, (a, s, r) in enumerate((a, s, r) for a in ('S', 'D')
                for s in (64, 16) for r in (None, 168, 128))}


def options(v, fault=False):
    x = VARIANTS[v]
    return core.OPTIONS[:-1] + (f'-DP5_STACK_CAP={x["stack"]}',) + (
        () if x['registers'] is None else (f'--maxrregcount={x["registers"]}',)) + (
        ('-DP5B_MERGE_FAULT=1',) if fault else ())


def source(v):
    text = core.source()
    return text if v == 'V0' else text+'\n'+(core.HERE/'p5e_gpu.cu').read_text()


def names(v):
    return ('p5b_run_kernel', 'p5b_per_history_kernel') if v == 'V0' else (
        'p5e_run_kernel', 'p5e_per_history_kernel')


def order_check(order, n):
    if order.dtype != np.int32 or order.shape != (n,) or not np.array_equal(np.sort(order), np.arange(n)):
        raise core.Stop('order is not an int32 permutation')
    return order


def ideal_order(steps):
    return order_check(np.lexsort((np.arange(len(steps)), -steps)).astype(np.int32), len(steps))


def e1(actual, expected):
    try:
        n = len(expected['steps'])
        for r in (actual, expected):
            core.validate_output(r, core.history_shapes(n))
            if not all(np.isfinite(x).all() for x in r.values()):
                return core.verdict(False)
        return core.verdict(all(core.bits(actual[k], expected[k]) for k in core.HISTORY_ORDER))
    except (core.Stop, KeyError, TypeError, AttributeError):
        return core.verdict(False)


def e2(actual, reference):
    core.validate_output(actual, core.run_shapes(20))
    return core.verdict(core.health(actual)['status'] == 'pass' and
                        d.d_gate(actual, reference)['status'] == 'pass')


def visits_check(visits, global_h, rows, n, start):
    return core.verdict(core.bits(visits, np.ones(n, np.uint64)) and
        core.bits(global_h, np.arange(n, dtype=np.uint64)+np.uint64(start)) and
        core.bits(rows, (np.arange(n)*100//n).astype(np.int32)))


def simulate(n, threads, start=0):
    visits = np.zeros(n, np.uint64)
    h, rows = np.empty(n, np.uint64), np.empty(n, np.int32)
    counter = 0
    while counter < n:
        for _ in range(threads):
            k = counter
            counter += 1
            if k < n:
                visits[k] += np.uint64(1); h[k] = start+k; rows[k] = k*100//n
    return visits, h, rows


def select(times):
    if 'V0' not in times or any(len(t) != 3 or not np.isfinite(t).all() or min(t) <= 0 for t in times.values()):
        raise core.Stop('incomplete timing rounds')
    eligible = [v for v in times if v != 'V0' and max(times[v]) < min(times['V0'])]
    if not eligible:
        return 'V0'
    fastest = min(eligible, key=lambda v: np.median(times[v]))
    overlap = [v for v in eligible if max(times[v]) >= min(times[fastest]) and
               min(times[v]) <= max(times[fastest])]
    def rank(v):
        x = VARIANTS[v]
        changes = (x['allocation'] != 'S', x['stack'] != 64, x['registers'] is not None)
        return (sum(changes), *changes, np.median(times[v]))
    return min(overlap, key=rank)


def identity(v, manifest, environment, launches):
    return dict(variant=v, configuration=VARIANTS[v], source_sha256=core.sha(source(v).encode()),
        options=list(options(v)), entries=list(names(v)), block=128, launches=launches,
        bundle_sha256={k: x['sha256'] for k, x in manifest['files'].items()},
        environment=environment, seed=core.SEED, per_batch=100000, batches=100, nspec=20)


def match_identity(path, expected):
    if path.exists() and json.loads(path.read_text()) != expected:
        raise core.Stop('configuration fingerprint mismatch')
    core.write_json(path, expected)
