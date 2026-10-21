"""Verified bundle -> gates -> resumable production. Also the local CPU driver."""
import importlib.metadata
import json
import os
from pathlib import Path
import time
import uuid
from ..gpu import p5b_core as core, p7_core as h
from . import bundle
from .condition import verify_observation, check_equal


def reject_cupy12():
    names = [d.metadata.get('Name', '').lower().replace('_', '-') for d in importlib.metadata.distributions()]
    if 'cupy-cuda12x' in names:
        raise core.Stop('cupy-cuda12x: conflict; not launched')


def gates(gpu, root, cond, mat, manifest):
    import numpy as np
    root = Path(root)
    cpu = core.read_npz(root/'reference.npz')
    actual = gpu.measure(2048, cond['seed'], history=True)
    verify_observation(gpu.last_observation, cond, mat, n=2048, start=0)
    # Same pass/fail as P7 c2, with reporting explicitly partitioned using p6.
    a = {k: cpu[k] for k in core.HISTORY_ORDER}; b = {k: actual[k] for k in core.HISTORY_ORDER}
    checks = dict(cpu_health=h.health(cpu), gpu_health=h.health(actual), source=core.c2_source(a, b), calls=core.c2_calls(a, b),
                  field_source=core.verdict(bool(np.all(np.abs(cpu['sources']-actual['sources']) <= 1e-13))))
    layout = h.layout('p6')
    regions = {}
    for j, (offset, size) in enumerate(zip(layout['offsets'], layout['sizes'])):
        delta = (actual['regions'][:, offset:offset+size]-cpu['regions'][:, offset:offset+size])*1.602176634e-10/layout['volumes'][j]
        regions[str(j)] = core.paired(delta, manifest['Dref'])
    two_a = core.verdict(all(v['status'] == 'pass' for v in checks.values()), checks=checks, regions_report_only=regions,
                             legacy_bins=core.c2_bins(a, b, dref=manifest['Dref'], conversion=1.602176634e-10))
    core.require(two_a, '2a')
    total = gpu.measure(2048, cond['seed'], count_additions=True)
    verify_observation(gpu.last_observation, cond, mat, n=2048, start=0)
    two_c = h.c3(actual, total, len(gpu.data['cum']), region_additions=gpu.region_additions)
    core.require(two_c, '2c')
    fixture = core.read_npz(root/'scorer.npz'); records = fixture.pop('records')
    scored = gpu.scorer(records)
    two_d = h.exact(scored, fixture)
    core.require(two_d, '2d')
    if any(np.any(x['regions'][..., layout['bins']:] != 0) for x in (actual, total, scored)):
        raise core.Stop('2d unused capacity nonzero')
    return {'2a': two_a, '2c': two_c, '2d': two_d}


def run(bundle_directory, results, session=None):
    begin = time.perf_counter()
    data, manifest = bundle.load(bundle_directory)
    root = Path(bundle_directory)
    cond, mat = bundle.read_json(root/'condition.json'), bundle.read_json(root/'materials.json')
    session = session or str(uuid.uuid4())
    results = Path(results); results.mkdir(parents=True, exist_ok=True)
    found = bundle.existing(results, cond, mat)
    timings = dict(bundle_verification=time.perf_counter()-begin)
    before = time.perf_counter()
    report = dict(status='incomplete', session_id=session, condition_hash=manifest['condition_hash'], materials_hash=manifest['materials_hash'])
    try:
        if cond['engine'] == 'cpu_local':
            from .gpu_adapter import CPU
            engine = CPU(results/'build', data, cond, mat)
            timings['compile'] = time.perf_counter()-before
            report['gates'] = dict(cpu='not GPU; local boundary test')
        else:
            reject_cupy12()
            import cupy as cp
            from ..gpu.p5b_colab import environment
            env = environment(cp)
            if 'T4' not in env['gpu']:
                raise core.Stop('T4 required')
            report['environment'] = env
            from .gpu_adapter import GPU
            engine = GPU(cp, data, cond, mat)
            timings['compile'] = time.perf_counter()-before
            before = time.perf_counter()
            report['gates'] = gates(engine, root, cond, mat, manifest)
            timings['gates'] = time.perf_counter()-before
        core.write_json(results/f'session_{session}.json', report)
        before = time.perf_counter()
        for b in range(100):
            if b in found:
                continue
            r = engine.measure(cond['batch_histories'], cond['seed'], start=b*cond['batch_histories'])
            # Fault control is retained as artifacts; normal runs stop before saving a discrepant batch.
            verify_observation(engine.last_observation, cond, mat, n=cond['batch_histories'], start=b*cond['batch_histories'])
            bundle.save_batch(results, b, r, engine.last_observation, cond, mat, session)
        timings['production'] = time.perf_counter()-before
        bundle.collect(results, cond, mat)
        if cond['engine'] == 'colab_t4':
            before = time.perf_counter()
            # Required negative control after production; collect this observation for local rejection.
            previous = os.environ.get('MV_ENTRY_FAULT')
            try:
                os.environ['MV_ENTRY_FAULT'] = 'field_half'
                engine.measure(2048, cond['seed'])
            finally:
                if previous is None:
                    os.environ.pop('MV_ENTRY_FAULT', None)
                else:
                    os.environ['MV_ENTRY_FAULT'] = previous
            core.write_json(results/'fault_observation.json', engine.last_observation)
            try:
                verify_observation(engine.last_observation, cond, mat, n=2048, start=0)
            except core.Stop as exc:
                report['fault'] = dict(status='detected', reason=str(exc), observation_sha256=core.file_sha(results/'fault_observation.json'))
            else:
                raise core.Stop('field_half fault was not detected')
            timings['fault'] = time.perf_counter()-before
        report['status'] = 'complete'
    except Exception as exc:
        report.update(status='failed', reason=str(exc))
        raise
    finally:
        report['timings'] = timings
        core.write_json(results/f'session_{session}.json', report)
        core.write_json(results/'driver_status.json', report)
    return report


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', required=True); p.add_argument('--results', required=True); p.add_argument('--session', required=True)
    args = p.parse_args()
    run(args.bundle, args.results, args.session)
