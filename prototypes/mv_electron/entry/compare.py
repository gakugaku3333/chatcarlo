"""P3b-3 comparison, after bundle, provenance, VM gates and comparison-input verification."""
from pathlib import Path
import numpy as np
from . import bundle
from .condition import digest, verify_observation, check_equal
from ..gpu import p5b_core as core, p7_core as h


def stacked(rows, cond):
    if len(rows) != 100:
        raise core.Stop('100 outer batches required')
    reduced = [h.reduced(row, 'p6') for row in rows]
    return dict(**{k: np.stack([r[k] for r in reduced]) for k in reduced[0]},
                N=np.array(cond['histories']), nbatch=np.array(100))


def verify_sessions(results, cond, mat):
    sessions = {bundle.read_json(bundle.batch_paths(results, b)[1])['session_id'] for b in range(100)}
    for session in sessions:
        report = bundle.read_json(bundle.safe_path(results, f'session_{session}.json'))
        check_equal(report['condition_hash'], digest(cond), 'VM condition_hash')
        check_equal(report['materials_hash'], digest(mat), 'VM materials_hash')
        if set(report['gates']) != {'2a', '2c', '2d'} or any(g['status'] != 'pass' for g in report['gates'].values()):
            raise core.Stop('VM gates failed/incomplete')
    status = bundle.read_json(Path(results)/'driver_status.json')
    if status['status'] != 'complete' or status.get('fault', {}).get('status') != 'detected':
        raise core.Stop('VM production/fault incomplete')
    bundle.verify_files(results, {'fault_observation.json': status['fault']['observation_sha256']})
    try:
        verify_observation(bundle.read_json(Path(results)/'fault_observation.json'), cond, mat, n=2048, start=0)
    except core.Stop as exc:
        if 'field_half_cm' not in str(exc):
            raise
    else:
        raise core.Stop('VM fault control not detected locally')


def verify_comparison(root, manifest, data, expected_config=None):
    from .. import run_p3b3 as p3
    root = Path(root)
    if set(manifest['comparison_inputs']) != set(bundle.COMPARISON):
        raise core.Stop('comparison_inputs incomplete')
    bundle.verify_files(root, manifest['comparison_inputs'])
    config = bundle.read_json(root/'egs5/main_egs/config.json')
    expected = expected_config if expected_config is not None else p3.egs_config(46000000, 20291763, data['edges'], data['cum'])
    check_equal(config, expected, 'EGS5 config.json')
    lock = bundle.read_json(root/'pilot_lock_p3b3.json')
    if lock['N_main'] != 46000000 or lock['Dref'] != manifest['Dref']:
        raise core.Stop('pilot_lock mismatch')
    egs = core.read_npz(root/'egs5/main_egs/result.npz')
    return p3.validate(egs, 'egs', 46000000, 20291763, bundle.read_json(root/'egs5/main_egs/summary.json'))


def statistics(gpu_mev, egs_mev, reference):
    from .. import analysis_p3b3 as stats, analysis_p7
    # Both sides are region energies in MeV: conversion must precede every comparison/report.
    gpu, egs = stats.dose(gpu_mev), stats.dose(egs_mev)
    comparison = stats.compare(gpu, egs, reference)
    reports = stats.reports(gpu, egs, reference)
    deep = comparison['J1']+comparison['J2']
    positions = comparison['J3']
    def summary(rows):
        delta = [abs(r['delta']) for r in rows if r['delta'] is not None]
        sem = [r['SEM'] for r in rows if r['SEM'] is not None]
        return dict(max_abs=max(delta) if len(delta) == len(rows) else None,
                    SEM_range=[min(sem), max(sem)] if len(sem) == len(rows) else None)
    deep_summary = summary(deep)
    deep_summary['RMS'] = float(np.sqrt(np.mean([r['delta']**2 for r in deep])))
    ratios = analysis_p7.slab_ratios(gpu, egs)
    symmetry = stats.symmetry(gpu['profile'])
    statuses = [r['status'] for group in comparison.values() for r in group]
    verdict = ('統計不足' if '統計不足' in statuses else '合格' if all(s in ('1%以内', '0.1 cm以内') for s in statuses) else '不合格または判定不能')
    return dict(status=verdict, comparison=comparison, reports=reports, slab_ratios=ratios, symmetry=symmetry,
                abstract=dict(deep_Dref_percent=deep_summary, position_cm=summary(positions),
                              buildup_slab=ratios[0], buildup_axis=reports['intervals'][0],
                              penumbra_GPU_minus_EGS5_cm=[None if r['proto']['width20_80'] is None or r['egs']['width20_80'] is None
                                                       else r['proto']['width20_80']-r['egs']['width20_80'] for r in reports['profiles']]),
                user_attention_required=symmetry['status'] == '偏りを検出',
                scope='この1条件・今回の環境の1回。ブロック・中央平均・50%位置まで。同等性ではない。',
                penumbra_note='抄録たたき台の「試作（CPU）− EGS5」と異なり、今回の半影は「GPU − EGS5」。')


def analyse(directory, results, comparison_root=None):
    data, manifest = bundle.load(directory)
    cond, mat = bundle.read_json(Path(directory)/'condition.json'), bundle.read_json(Path(directory)/'materials.json')
    rows = bundle.collect(results, cond, mat)
    verify_sessions(results, cond, mat)
    root = Path(comparison_root) if comparison_root else h.ROOT/'docs/validation/mv/p3b'
    egs = verify_comparison(root, manifest, data)
    result = statistics(stacked(rows, cond), egs, manifest['Dref'])
    bundle.verify_files(root, manifest['comparison_inputs'])
    return result
