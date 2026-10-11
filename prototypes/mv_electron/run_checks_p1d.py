"""P1d frozen decisions and saved-only replay. Never executes EGS5 or transport."""
import argparse
import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from prototypes.mv_electron.p1d_common import *
from prototypes.mv_electron.analysis import compare, dref, ranges
from prototypes.mv_electron.check_egs5 import parse

def k5b(backend, execute=False):
    target = OUT / backend / 'k5b.json'
    if not target.exists() and execute:
        command = ['.venv/bin/python', '-m', 'pytest', f'prototypes/mv_electron/tests/test_p1d.py::test_k5b_actual_table[{backend}]', '-q']
        r = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        output = (r.stdout + r.stderr).replace(str(ROOT), '.').replace(str(Path.home()), '~')
        (OUT / backend).mkdir(exist_ok=True)
        (OUT / backend / 'k5b.log').write_text(output)
        write(target, dict(status='pass' if r.returncode == 0 and 'skipped' not in output else 'fail',
                           command=' '.join(command), exit_code=r.returncode, log=f'{backend}/k5b.log'))
    return json.loads(target.read_text()) if target.exists() else dict(status='not-evaluated')

def r1(backend):
    rows = []
    old = json.loads((OLD[backend] / 'fixed_checks.json').read_text())['K9i']['rows']
    for length in (.05, .2):
        paths = [OUT / backend / 'fixed' / f'{length}_{c}.npz' for c in (False, True)]
        if not all(p.exists() for p in paths): continue
        loaded = []
        for path in paths:
            receipt = json.loads(path.with_suffix('.json').read_text())
            assert receipt['sha256'] == sha(path)
            with np.load(path) as a:
                expected_seed = FIXED_SEED if bool(a['condensed']) else SEED
                assert int(a['N']) == 1000000 and int(a['seed']) == expected_seed and float(a['K']) == .025
                loaded.append(a['values'])
        ref, pro = loaded; n = len(ref)
        difference = (pro.mean(0) - ref.mean(0)) / ref.mean(0)
        sem = np.sqrt(pro.var(0, ddof=1) / n + ref.var(0, ddof=1) / n) / ref.mean(0)
        c = dict(statistics_ok=bool(np.max(sem[:2]) <= .005), **{'pass': bool(np.max(abs(difference[:2])) <= .02)})
        previous = next(r for r in old if r['path_cm'] == length and r['K'] == .025)
        od = np.asarray(previous['relative']); os = np.asarray(previous['difference_sem_relative'])
        rows.append(dict(path_cm=length, K=.025, N=n, reference_seed=SEED, prototype_seed=FIXED_SEED,
                         reference=ref.mean(0).tolist(), prototype=pro.mean(0).tolist(), relative=difference.tolist(),
                         difference_sem_relative=sem.tolist(), status=status(c), statistics_ok=c['statistics_ok'],
                         historical_relative=od.tolist(), historical_SEM=os.tolist(),
                         new_minus_historical_over_combined_SEM=((difference - od) / np.sqrt(sem**2 + os**2)).tolist()))
    return dict(status=aggregate([r['status'] for r in rows] + ([] if len(rows) == 2 else ['not-evaluated'])), rows=rows)

def validate_run(backend, name):
    path = OUT / backend / 'runs' / (name + '.npz')
    receipt = json.loads(path.with_suffix('.json').read_text())
    cfg = CONFIGS[name]
    assert receipt['config'] == cfg and receipt['seed'] == SEED and receipt['sha256'] == sha(path)
    with np.load(path) as a:
        for key in ('N', 'T', 'K', 'fe', 'mode'): assert a[key] == cfg[key]
        assert int(a['seed']) == SEED and str(a['backend']) == backend

def r2(backend):
    directory = OUT / backend
    names = list(CONFIGS)[:-1]
    existing = [name for name in names if (directory / 'runs' / (name + '.npz')).exists()]
    for name in existing: validate_run(backend, name)
    if len(existing) != len(names):
        return dict(status='not-evaluated', completed_runs=existing, pending_runs=[n for n in names if n not in existing], K5b=k5b(backend))
    from prototypes.mv_electron import check_transport
    previous = check_transport.OUT
    try:
        check_transport.OUT = directory
        with contextlib.redirect_stdout(io.StringIO()): check_transport.main(directory)
    finally:
        check_transport.OUT = previous
    result = json.loads((directory / 'transport_checks.json').read_text())
    result['K5b'] = k5b(backend)
    for key in ('K7', 'K8'):
        for c in result[key]['rows'].values(): c['status'] = status(c)
        result[key]['status'] = aggregate(c['status'] for c in result[key]['rows'].values())
    result['status'] = aggregate(result[k]['status'] for k in ('K5', 'K5b', 'K7', 'K8', 'K9ii'))
    write(directory / 'transport_checks.json', result)
    return result

def comparison(backend, T):
    ref = parse(f'base_{T}'); path = OUT / backend / 'runs' / f'base_{T}.npz'
    validate_run(backend, f'base_{T}')
    with np.load(path) as p:
        mask = ref['q'] >= .2 * dref(ref['q'])
        c = compare(ref['batches'], p['batches'], ref['z'], tol=.03, range_tol=.03, mask=mask, bootstrap=2000 if T == 2 else 1000)
        c['prototype_back_energy_fraction'] = float(p['ledger'][:, 1].sum() / (T * int(p['N'])))
        c['prototype_back_fraction_SEM'] = float((p['ledger'][:, 1] / T).std(ddof=1) / np.sqrt(int(p['N'])))
    c['EGS5_back_energy_fraction'] = ref['ledger'][2] / ref['ledger'][0]
    c['status'] = status(c) if T == 2 else 'not-evaluated'
    if T == 10: c['reason'] = '参考値を記録。R4には合否の規則がないため未評価（計算済み）。'
    c['reference_run'] = f'docs/validation/mv/p1/egs5/base_{T}'
    c['fixed_mask_indices'] = np.flatnonzero(mask).tolist()
    return c

def reduction(old, new, ref, mask):
    """Point and bootstrap use exactly the same old run and fixed reference mask."""
    rq = ref.mean(0); D = dref(rq)
    dp = float(max(abs(old.mean(0)[mask] - rq[mask])) / D)
    dn = float(max(abs(new.mean(0)[mask] - rq[mask])) / D)
    rng = np.random.default_rng(BOOTSTRAP_SEED); samples = []
    for _ in range(1000):
        e = ref[rng.integers(len(ref), size=len(ref))].mean(0)
        p = old[rng.integers(len(old), size=len(old))].mean(0)
        n = new[rng.integers(len(new), size=len(new))].mean(0)
        Db = dref(e)
        do = max(abs(p[mask] - e[mask])) / Db; dnew = max(abs(n[mask] - e[mask])) / Db
        samples.append(float(1 - dnew / do))
    return dict(point=float(1 - dn / dp), CI95=np.percentile(samples, [2.5, 97.5]).tolist(),
                d_new=dn, d_old=dp, samples=samples, bootstrap_N=1000, seed=BOOTSTRAP_SEED,
                fixed_mask_indices=np.flatnonzero(mask).tolist(), method='batch bootstrap; shared EGS5 draw for numerator and denominator; percentile 95%')

def record_curve(path, ref):
    with np.load(path) as p:
        q = p['batches'].mean(0); D = dref(ref['batches'].mean(0)); mask = ref['q'] >= .2 * dref(ref['q'])
        rr = ranges(ref['batches'].mean(0), ref['z'], D); rp = ranges(q, ref['z'], D)
        return dict(max_dose_difference_over_Dref=float(max(abs(q[mask] - ref['batches'].mean(0)[mask])) / D),
                    R50_cm=float(rp[0]), Rp_cm=float(rp[1]), range_difference_cm=(rp - rr).tolist(),
                    range_difference_over_reference_R50=((rp - rr) / rr[0]).tolist(),
                    back_energy_fraction=float(p['ledger'][:, 1].sum() / (float(p['T']) * int(p['N']))),
                    status='not-evaluated', reason='補助記録のみ。旧判定とR3の停止規則を変更しない。')

def auxiliary():
    result = {}; ref = parse('base_2'); D = dref(ref['batches'].mean(0)); mask = ref['q'] >= .2 * dref(ref['q'])
    p1 = np.load(ROOT / 'docs/validation/mv/p1/runs/base_2.npz')['batches']
    for backend in BACKENDS:
        newpath = OUT / backend / 'runs/base_2.npz'
        if not newpath.exists(): continue
        validate_run(backend, 'base_2')
        with np.load(newpath) as p: new = p['batches']
        oldpath = OLD[backend] / 'runs/base_2.npz'
        with np.load(oldpath) as p: old = p['batches']
        with np.load(OLD[backend] / 'runs/ms_half.npz') as p: previous = p['batches']
        difference = new.mean(0) - previous.mean(0)
        sem = np.sqrt(new.var(0, ddof=1) / len(new) + previous.var(0, ddof=1) / len(previous))
        ratio = np.divide(difference, sem, out=np.zeros_like(difference), where=sem > 0)
        result[backend] = dict(old_K005=record_curve(oldpath, ref), new_K0025=record_curve(newpath, ref),
                               r_P1=reduction(p1, new, ref['batches'], mask), r_K=reduction(old, new, ref['batches'], mask),
                               saved_ms_half_consistency=dict(depth_cm=ref['z'].tolist(), difference_over_Dref=(difference / D).tolist(),
                                   combined_SEM_over_Dref=(sem / D).tolist(), difference_over_combined_SEM=ratio.tolist(),
                                   max_abs_difference_over_Dref=float(max(abs(difference)) / D),
                                   max_abs_ratio=float(max(abs(ratio))), bins_beyond_2SEM=int(sum(abs(ratio) > 2)),
                                   note='保存済みms_halfは整合の確認にのみ使用。EGS5とは比較しない。'))
    if all(b in result for b in BACKENDS):
        a = np.load(OUT / 'mf/runs/base_2.npz')['batches']; b = np.load(OUT / 'dcslib/runs/base_2.npz')['batches']
        diff = a.mean(0) - b.mean(0); sem = np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b))
        ii = np.flatnonzero(mask); i = int(ii[np.argmax(abs(diff[mask]))])
        R50 = ranges(ref['batches'].mean(0), ref['z'], D)[0]; tail = ref['z'] > R50
        result['mf_minus_dcslib'] = dict(max_difference_over_Dref=float(abs(diff[i]) / D), signed_difference_over_Dref=float(diff[i] / D),
            depth_cm=float(ref['z'][i]), reference_R50_cm=float(R50), depth_cm_array=ref['z'].tolist(), difference_over_Dref=(diff / D).tolist(),
            combined_SEM_over_Dref=(sem / D).tolist(), beyond_R50_all_bins_max_over_Dref=float(max(abs(diff[tail])) / D),
            beyond_R50_eligible_max_over_Dref=float(max(abs(diff[tail & mask])) / D),
            beyond_R50_indices=np.flatnonzero(tail).tolist())
    return result

def saved():
    untouched = verify(); regression = regression_saved()
    write('untouched.json', untouched)
    summary = dict(R0=dict(status='pass', untouched=untouched, regression=regression), backends={})
    reference_status = json.loads((ROOT / 'docs/validation/mv/p1/egs5_checks.json').read_text())['K4']['status']
    for backend in BACKENDS:
        one = dict(R1=r1(backend), R2=r2(backend))
        one['R3'] = dict(status='not-evaluated', reason='R1とR2が両方合格した場合だけ判定する。')
        if all(one[key]['status'] == 'pass' for key in ('R1', 'R2')) and reference_status == 'pass':
            one['R3'] = comparison(backend, 2)
        one['R4'] = comparison(backend, 10) if (OUT / backend / 'runs/base_10.npz').exists() else dict(status='not-evaluated', reason='参考run未完了')
        timing = OUT / backend / 'benchmark.json'
        one['timing'] = json.loads(timing.read_text()) if timing.exists() else dict(rows=[])
        summary['backends'][backend] = one
    summary['auxiliary'] = auxiliary()
    write('summary.json', summary)
    from prototypes.mv_electron.report_p1d import report
    report(summary)
    verify()
    return summary

def main():
    p = argparse.ArgumentParser(); p.add_argument('--from-saved', action='store_true'); p.add_argument('--verify-untouched', action='store_true')
    a = p.parse_args()
    if a.verify_untouched:
        r = verify(); r['regression'] = regression_saved(); write('untouched.json', r); print(json.dumps(r, indent=2)); return
    if not a.from_saved: p.error('choose --from-saved or --verify-untouched')
    r = saved()
    print(json.dumps({b: {k: LABELS[r['backends'][b][k]['status']] for k in ('R1', 'R2', 'R3', 'R4')} for b in BACKENDS}, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
