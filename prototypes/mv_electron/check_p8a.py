"""P8a strict data checks; incomplete checks cannot result in a pass.

Expensive direct ELSEPA checks are read-only. GS checks run the P1f numerical
criteria for each new mixture. Results and model-choice decisions are separate.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from .materials_p8a import (ROOT, OUT, CACHE, MATERIALS, MixtureElsepa, constants,
                            energy_grid, finite, kn_total, log_interp, photo_interp,
                            pair_interp, load_manifest, load_reference, sha, write,
                            verify_protected)


def aggregate(states):
    states = list(states)
    if not states or any(s not in ('pass', 'fail') for s in states):
        return 'not-evaluated'
    return 'pass' if all(s == 'pass' for s in states) else 'fail'


def aggregate_recorded_omission(states):
    from .generate_elsepa_p8a import OMISSION_STATUS
    states = list(states)
    result = aggregate('pass' if s == OMISSION_STATUS else s for s in states)
    return OMISSION_STATUS if result == 'pass' and OMISSION_STATUS in states else result


def fixed_omission(original, replacement, z, charge):
    from .generate_elsepa_p8a import RECORDED_OMISSIONS
    return dict(Z=z, charge=charge, original_energy_MeV=original,
                replacement_energy_MeV=replacement) in RECORDED_OMISSIONS


def canonical_recorded_omissions(items):
    """Stable order and unique physical points, including Ca shared by materials."""
    from .generate_elsepa_p8a import RECORDED_OMISSIONS
    return [dict(scope) for scope in RECORDED_OMISSIONS if scope in items]


def relative(actual, expected):
    actual, expected = finite(actual), finite(expected)
    if actual.shape != expected.shape or actual.size == 0:
        raise ValueError('comparison shape / empty data')
    if np.any((expected == 0) & (actual != 0)):
        return float('inf')
    return float(np.max(np.abs(np.divide(actual-expected, expected,
                     out=np.zeros_like(actual), where=expected != 0))))


def pair_comparison(energies, actual, expected, total, threshold):
    """Appendix 5: near threshold use absolute error / XCOM total attenuation."""
    energies, actual, expected, total = map(finite, (energies, actual, expected, total))
    if (energies.ndim != 1 or not energies.size
            or any(a.shape != energies.shape for a in (actual, expected, total))
            or np.any(total <= 0) or np.any(actual < 0) or np.any(expected < 0)):
        raise ValueError('pair comparison shape / coefficient')
    near = energies < 1.01*threshold
    absolute = float(np.max(abs(actual[near]-expected[near])/total[near])) if np.any(near) else 0.
    far_relative = relative(actual[~near], expected[~near]) if np.any(~near) else 0.
    return dict(max_relative=relative(actual, expected),
                near_threshold_count=int(np.sum(near)), near_threshold_limit_MeV=1.01*threshold,
                max_absolute_over_total=absolute, absolute_over_total_limit=1e-5,
                far_max_relative=far_relative, relative_limit=.001,
                status='pass' if absolute <= 1e-5 and far_relative <= .001 else 'fail')


def interpolation_checks(stopping, photon, ne):
    es, mid = energy_grid(stopping['T']), energy_grid(stopping['check_T'])
    if np.any(np.isin(mid, es)) or mid[0] <= es[0] or mid[-1] >= es[-1]:
        raise ValueError('stopping check points must be held out')
    if not np.all(np.isin(np.sqrt(es[:-1]*es[1:]), mid)):
        raise ValueError('missing stopping midpoints')
    report = {}
    for k in ('collision', 'radiative'):
        error = relative(log_interp(mid, es, stopping[k]), stopping['check_'+k])
        report[k] = dict(max_relative=error, status='pass' if error <= .001 else 'fail')
    e, check = energy_grid(photon['E']), energy_grid(photon['check_E'])
    if np.any(np.isin(check, e)) or not np.all(np.isin(np.sqrt(e[:-1]*e[1:]), check)):
        raise ValueError('photon held-out midpoints missing or reused')
    # Appendix 8: the midpoint between the two sides of an edge (relative
    # 2e-6 apart) sits on the edge itself; it tests no interpolation.
    pe = np.asarray(photon['photo_E'])
    on_edge = (pe[:-1] == pe[1:])
    if len(on_edge) != len(check):
        raise ValueError('photon check points do not pair with node intervals')
    keep = ~on_edge
    photo = photo_interp(check[keep], photon['photo_E'], photon['photo'])
    err = relative(photo, np.asarray(photon['check_photo'])[keep])
    report['photoelectric'] = dict(max_relative=err, excluded_edge_midpoints=int(on_edge.sum()),
                                   status='pass' if err <= .001 else 'fail')
    for i, process in enumerate(('pair_nuclear', 'pair_electron')):
        t = float(photon['thresholds'][i])
        comparison = pair_comparison(check, pair_interp(check, e, photon[process], t),
                                     photon['check_'+process], photon['check_total'], t)
        probe = pair_interp(np.array([np.nextafter(t, 0), t, np.nextafter(t, np.inf)]), e, photon[process], t)
        threshold = bool(probe[0] == probe[1] == 0 and probe[2] > 0)
        comparison['threshold'] = threshold
        if not threshold:
            comparison['status'] = 'fail'
        report[process] = comparison
    # Independent angular quadrature of free-electron Klein-Nishina.
    from numpy.polynomial.legendre import leggauss
    from .physics import M, RE
    mu, w = leggauss(512)
    q = 1/(1+e[:, None]/M*(1-mu))
    direct = np.pi*RE**2*ne*np.sum(w*q*q*(q+1/q-(1-mu*mu)), axis=1)
    err = relative(kn_total(e, ne), direct)
    # Quadrature at very high E and cancellation at low E can be less accurate
    # than the analytic formula; use an independent high-precision evaluation.
    import decimal
    with decimal.localcontext() as ctx:
        ctx.prec = 60
        D = decimal.Decimal
        independent = []
        for value in e:
            x = D(float(value))/D(M)
            l = (1+2*x).ln()
            r = (1+x)/x**2*(2*(1+x)/(1+2*x)-l/x)+l/(2*x)-(1+3*x)/(1+2*x)**2
            independent.append(float(2*D(str(np.pi))*D(RE)**2*D(ne)*r))
    err = relative(kn_total(e, ne), independent)
    report['compton'] = dict(max_relative=err, status='pass' if err <= 1e-12 else 'fail')
    # Exact stored side values and the upper-side-at-edge convention.
    sides = []
    for edge in photon['edges']:
        ids = np.flatnonzero(photon['photo_E'] == edge)
        if len(ids) != 2:
            raise ValueError('absorption edge sides missing')
        lower, upper = photon['photo'][ids]
        above = photo_interp(edge, photon['photo_E'], photon['photo'])
        below = photo_interp(np.nextafter(edge, 0), photon['photo_E'], photon['photo'])
        sides.append(bool(np.isclose(above, upper, rtol=1e-14) and np.isclose(below, lower, rtol=1e-12)))
    report['edges'] = dict(count=len(sides), exact_edge='upper', status='pass' if all(sides) else 'fail')
    return report


def gs_checks(provider, g, n_moments=1000000, n_cdf=10000000):
    """P1f T4 thresholds, with no P1f checkpoint paths or globals modified."""
    from .gs_backend import distribution, single_cdf
    from .dcs import moments
    from .check_gs_p1f import samples, single_samples, cdf_moments
    from scipy.stats import chisquare
    from numpy.polynomial.legendre import leggauss
    rows, refinement, cdfs = [], [], []
    for t in (.01, .02, .1, 1., 2., 10.):
        G, total, _ = moments(provider, t)
        for k in (.001, .01, .05, .2):
            mu = samples(t, k, n_moments, *(g[x] for x in ('T', 'K', 'u', 'y', 'rates')))
            p2 = (3*mu*mu-1)/2
            actual = np.array([mu.mean(), p2.mean()])
            expected = np.exp(-k*G[1:3]/G[1])
            sem = np.array([mu.std(ddof=1), p2.std(ddof=1)])/np.sqrt(n_moments)
            tol = np.where(expected >= .05, .01*expected, .001)
            ok = np.all(abs(actual-expected) <= tol) and np.all(sem <= tol/3)
            rows.append(dict(T=t, K=k, moments=actual.tolist(), expected=expected.tolist(), sem=sem.tolist(), status='pass' if ok else 'fail'))
            L = 1024
            while True:
                try:
                    a = distribution(provider, t, k, L, 16385, 16)
                    break
                except ValueError:
                    L *= 2
                    if L > 32768:
                        raise
            b = distribution(provider, t, k, 2*L, 32769, 32)
            change = float(max(abs(cdf_moments(*a[:3])-cdf_moments(*b[:3]))))
            def inverse(u, row):
                y = np.interp(u, row[1], row[0])
                return cdf_moments(y, u, row[2])
            u = g['u']
            uf = np.unique(np.r_[u, (u[:-1]+u[1:])/2])
            inv = float(max(abs(inverse(u, a)-inverse(uf, b))))
            refinement.append(dict(T=t, K=k, moment_change=change, inverse_change=inv,
                                   status='pass' if max(change, inv) <= 1e-4 else 'fail'))
        def check_cdf(vals, kind):
            edges = np.geomspace(1e-6, 2+1e-6, 21)-1e-6
            edges[0], edges[-1] = 0., 2.
            native = provider.edges(t)
            x, w = leggauss(32)
            exp = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                cuts = np.unique(np.r_[lo, native[(native > lo) & (native < hi)], hi])
                y = (cuts[:-1, None]+cuts[1:, None])/2+(cuts[1:, None]-cuts[:-1, None])*x/2
                exp.append(float(np.sum(provider.density(t, y)*w*(cuts[1:, None]-cuts[:-1, None])/2)/total*n_cdf))
            boundaries, acc = [len(exp)], 0.
            for i in range(len(exp)-1, -1, -1):
                acc += exp[i]
                if acc >= 100:
                    boundaries.append(i)
                    acc = 0.
            if boundaries[-1] != 0:
                boundaries[-1] = 0
            indices = np.array(sorted(boundaries))
            expected = np.array([sum(exp[lo:hi]) for lo, hi in zip(indices[:-1], indices[1:])])
            obs = np.histogram(vals, edges[indices])[0]
            expected *= obs.sum()/expected.sum()
            p = float(chisquare(obs, expected).pvalue)
            cdfs.append(dict(T=t, kind=kind, p=p, min_expected=float(expected.min()),
                             status='pass' if p > .001 and expected.min() >= 100 else 'fail'))
        check_cdf(single_samples(t, n_cdf, g['T'], g['u'], g['single']), 'runtime-single')
        if t == .02:
            y, c, *_ = distribution(provider, t, 1e-5/total*G[1], L=2048, n=32769)
            vals = np.interp(np.random.default_rng(20261013).random(n_cdf), c, y)
            check_cdf(vals, 'short-GS')
    norm = float(np.max(abs(finite(g['normalization'])-1)))
    monotone = bool(np.all(np.diff(g['y'], axis=2) >= 0))
    return dict(status=aggregate([r['status'] for r in rows+refinement+cdfs]+['pass' if norm <= 1e-6 and monotone else 'fail']),
                normalization_error=norm, monotone=monotone, moments=rows,
                refinement=refinement, cdf=cdfs, n_moments=n_moments, n_cdf=n_cdf)


def direct_coverage(nodes, direct, coverage=None):
    """Require every intended check to be either completed or explicitly omitted."""
    from .generate_elsepa_p8a import direct_energies
    z, charge = int(nodes['Z']), int(nodes['charge'])
    expected = direct_energies(energy_grid(nodes['energy_MeV']), z)
    actual = finite(direct['energy_MeV'])
    if (int(direct['Z']) != z or int(direct['charge']) != charge
            or actual.ndim != 1 or np.any(np.diff(actual) <= 0)):
        raise ValueError('ELSEPA direct identity/energy grid')
    if coverage is None:
        if not np.array_equal(actual, expected):
            raise ValueError('ELSEPA independent checks missing without omission receipt')
        return []
    omitted = coverage['omitted']
    missing = np.array([r['energy_MeV'] for r in omitted])
    if (coverage['Z'] != z or coverage['charge'] != charge
            or coverage['expected_energy_MeV'] != expected.tolist()
            or coverage['completed_count'] != len(actual) or coverage['not_evaluated_count'] != len(omitted)
            or coverage['status'] != ('not-evaluated' if omitted else 'pass')
            or len(set(missing)) != len(missing) or np.any(np.isin(missing, actual))
            or not np.array_equal(np.sort(np.r_[actual, missing]), expected)):
        raise ValueError('ELSEPA direct coverage mismatch')
    for r in omitted:
        if (r['Z'] != z or r['charge'] != charge or r['status'] != 'not-evaluated'
                or not r.get('reason') or r['timeout_seconds'] != 600):
            raise ValueError('ELSEPA invalid timeout omission')
    return omitted


def inspect_atomic(a, z, charge, energy, kind):
    """Extend only P8a's warning policy; P1f's frozen checks stay unchanged."""
    from .check_elsepa_p1f import inspect
    result = inspect(a, z, charge, energy, kind)
    accepted, values, rejected = [], [], []
    # The parser retains three-line blocks: the TST heading may be followed
    # by the printed total. Do not whitelist unrelated continuation warnings.
    number = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?'
    pattern = (r'\s*WARNING:\s*TST\s*=\s*('+number+r')\.\s*Check convergence\.'
               r'(?:\s+Total elastic cross section\s*=\s*'+number+r'\s+cm\*\*2)?\s*')
    series = np.asarray(result['table_over_series_relative'])
    doubling = np.asarray(result['doubling_relative'])
    consistent = (series.shape == (3,) and np.all(np.isfinite(series))
                  and abs(series[0]) <= .005 and np.all(abs(series[1:]) <= .001))
    stable = bool(np.all(np.isfinite(doubling)) and np.max(abs(doubling)) <= 1e-6)
    for warning in result['rejected_warnings']:
        match = re.fullmatch(pattern, warning, flags=re.I)
        tst = float(match[1].replace('D', 'E').replace('d', 'e')) if match else None
        if tst is not None and 0 <= tst <= 1e-4 and consistent and stable:
            accepted.append(warning)
            values.append(tst)
        else:
            rejected.append(warning)
    result.update(accepted_tst_warnings=accepted, accepted_tst_max=max(values, default=None),
                  rejected_warnings=rejected)
    if not rejected:
        result['failure_quantities'] = [r for r in result['failure_quantities'] if r != 'rejected numerical warning']
    result['status'] = 'fail' if result['failure_quantities'] else 'pass'
    return result


def tst_summary(rows):
    """Counts are points (not duplicate parser blocks), grouped for S8 RESULTS."""
    groups = {}
    for row in rows:
        if 'accepted_tst_warnings' not in row:
            continue
        group = groups.setdefault((row['Z'], row['charge']),
                                  dict(Z=row['Z'], charge=row['charge'], accepted_point_count=0, max_TST=None))
        if row['accepted_tst_warnings']:
            group['accepted_point_count'] += 1
            group['max_TST'] = max(group['max_TST'] or 0., row['accepted_tst_max'])
    return list(groups.values())


def direct_point_checks(nodes, data, index, kind='direct', curve=None):
    """Exactly the same angular, warning, interpolation and CDF gates for both points."""
    from .check_elsepa import Atomic, integrated
    from .gs_backend import single_cdf
    t = float(data['energy_MeV'][index])
    z, charge = int(data['Z']), int(data['charge'])
    atom = {k: data[k][index] for k in ('dcs', 'error', 'series_xs', 'table_xs')}
    atom.update(mu=data['mu'], warnings=np.array(json.loads(str(data['warnings'][index]))),
                notices=np.array(json.loads(str(data['notices'][index]))), convergence='[]')
    angular = inspect_atomic(atom, z, charge, t, kind)
    if curve is None:
        curve = CubicSpline(np.log(nodes['energy_MeV']), np.log(nodes['dcs']), axis=0, bc_type='natural')
    approx = Atomic(nodes['mu'], np.exp(curve(np.log(t))))
    direct = Atomic(data['mu'], data['dcs'][index])
    err = relative(integrated(approx, t), integrated(direct, t))
    yi, ci = single_cdf(approx, t)
    yd, cd = single_cdf(direct, t)
    grid = np.unique(np.r_[yi, yd])
    cdf = float(np.max(abs(np.interp(grid, yi, ci)-np.interp(grid, yd, cd))))
    interpolation = dict(kind='midpoint' if kind == 'direct' else 'replacement-interpolation',
                         Z=z, charge=charge, T=t, max_relative=err, cdf_difference=cdf,
                         status='pass' if err <= .002 and cdf <= .002 else 'fail')
    return dict(status=aggregate([angular['status'], interpolation['status']]),
                angular=angular, interpolation=interpolation)


def replacement_pairs(nodes, omitted, group):
    """Validate one replacement per saved timeout, including its original identity."""
    from .generate_elsepa_p8a import direct_energies, replacement_energy
    expected = direct_energies(nodes['energy_MeV'], int(nodes['Z']))
    original = {r['energy_MeV']: r for r in omitted}
    pairs = {}
    for pair in group['pairs']:
        t = pair['original']['energy_MeV']
        if (t not in original or pair['original'] != original[t] or t in pairs
                or pair['timeout_seconds'] != 600 or pair['status'] not in ('pass', 'fail', 'not-evaluated')
                or pair['energy_MeV'] != replacement_energy(t, nodes['energy_MeV'], expected)):
            raise ValueError('ELSEPA invalid/duplicate replacement pair')
        pairs[t] = pair
    return pairs


def atomic_checks(tables, replacements=None):
    """P1f T2 angular and independent energy-interpolation criteria."""
    from .generate_elsepa_p8a import OMISSION_STATUS
    rows, omissions, active, recorded = [], [], [], []
    for name, ref in tables.items():
        a = load_reference(ref, 'element')
        es = energy_grid(a['energy_MeV'])
        for i, t in enumerate(es):
            atom = {k: a[k][i] for k in ('dcs', 'error', 'series_xs', 'table_xs')}
            atom.update(mu=a['mu'], warnings=np.array(json.loads(str(a['warnings'][i]))),
                        notices=np.array(json.loads(str(a['notices'][i]))), convergence='[]')
            rows.append(inspect_atomic(atom, int(a['Z']), int(a['charge']), t, 'node'))
            active.append(rows[-1]['status'])
        if 'direct' not in ref:
            # H/O only: require P1f's sealed completed T2 rather than pretend
            # independent direct cases were executed by this P8a invocation.
            prior = json.loads((ROOT/'docs/validation/mv/p1f/t2.json').read_text())
            if prior['status'] != 'pass' or prior['table_hashes'][name] != ref['sha256']:
                raise ValueError('P1f elemental check missing or hash mismatch')
            continue
        b = load_reference(ref['direct'], 'element')
        coverage = load_reference(ref['direct_status'], 'element') if 'direct_status' in ref else None
        missing = direct_coverage(a, b, coverage)
        rows.extend(dict(r, kind='direct-timeout') for r in missing)
        omission = dict(Z=int(a['Z']), charge=int(a['charge']),
                        not_evaluated_count=len(missing), omitted=missing, replacement_pairs=[])
        omissions.append(omission)
        curve = CubicSpline(np.log(es), np.log(a['dcs']), axis=0, bc_type='natural')
        for i, t in enumerate(b['energy_MeV']):
            result = direct_point_checks(a, b, i, curve=curve)
            rows.extend((result['angular'], result['interpolation']))
            active.append(result['status'])
        group = (replacements or {}).get(name)
        pairs = {}
        if group is not None:
            if (group['Z'] != int(a['Z']) or group['charge'] != int(a['charge'])
                    or group['nodes_sha256'] != ref['sha256'] or group['direct_sha256'] != ref['direct']['sha256']
                    or group['coverage_sha256'] != ref['direct_status']['sha256']):
                raise ValueError('ELSEPA replacement base table identity/hash mismatch')
            pairs = replacement_pairs(a, missing, group)
        resolved, permitted = 0, 0
        for original in missing:
            pair = pairs.get(original['energy_MeV'])
            if pair is None:
                active.append('not-evaluated')
                continue
            is_permitted = False
            if pair['status'] == 'not-evaluated':
                raw = load_reference(pair['timeout_record'], 'element')
                if (raw['Z'] != int(a['Z']) or raw['charge'] != int(a['charge'])
                        or raw['energy_MeV'] != pair['energy_MeV'] or raw['timeout_seconds'] != 600
                        or raw['status'] != 'not-evaluated' or not pair.get('reason')):
                    raise ValueError('ELSEPA replacement timeout identity mismatch')
                result = dict(status='not-evaluated', reason=pair['reason'])
                rows.append(dict(result, kind='replacement-timeout', T=pair['energy_MeV'], Z=int(a['Z']), charge=int(a['charge'])))
                scope = dict(Z=int(a['Z']), charge=int(a['charge']), original_energy_MeV=original['energy_MeV'],
                             replacement_energy_MeV=pair['energy_MeV'])
                if (group.get('recorded_omission_policy') == scope
                        and fixed_omission(original['energy_MeV'], pair['energy_MeV'], int(a['Z']), int(a['charge']))):
                    from .generate_elsepa_p8a import _read_timeout, input_text
                    expected = dict(Z=int(a['Z']), charge=int(a['charge']), energy_MeV=pair['energy_MeV'], IHEF=0,
                                    input=input_text(int(a['Z']), int(a['charge']), pair['energy_MeV'], ihef=0),
                                    executable_sha256=raw['identity']['executable_sha256'])
                    if _read_timeout(Path(pair['timeout_record']['path']).parent, expected) != raw:
                        raise ValueError('ELSEPA recorded omission timeout evidence mismatch')
                    permitted += 1
                    is_permitted = True
                    recorded.append(scope)
            else:
                data = load_reference(pair['data'], 'element')
                if (int(data['Z']) != int(a['Z']) or int(data['charge']) != int(a['charge'])
                        or int(data['IHEF']) != 0 or not np.array_equal(data['energy_MeV'], [pair['energy_MeV']])):
                    raise ValueError('ELSEPA replacement data identity mismatch')
                result = direct_point_checks(a, data, 0, kind='replacement', curve=curve)
                result['status'] = aggregate([pair['status'], result['status']])
                rows.extend((result['angular'], result['interpolation']))
            active.append(OMISSION_STATUS if is_permitted else result['status'])
            resolved += result['status'] == 'pass'
            omission['replacement_pairs'].append(dict(pair, checks=result))
        omission.update(replacement_count=len(pairs), unresolved_count=len(missing)-resolved,
                        recorded_omission_count=permitted)
    return dict(status=aggregate_recorded_omission(active), rows=rows, direct_omissions=omissions,
                recorded_omissions=canonical_recorded_omissions(recorded), tst_warnings=tst_summary(rows))


def shell_checks(shells, photon, definition=None):
    import xraylib
    from .pegs_p8a import shell_probabilities
    binding = finite(shells['binding_MeV'])
    if definition is not None:
        c = constants(definition)
        if not np.array_equal(shells['Z'], c['Z']) or not np.array_equal(shells['NJ'], c['NJ']):
            raise ValueError('shell material composition mismatch')
    rows = []
    for i, z in enumerate(shells['Z']):
        for j, shell in enumerate((xraylib.K_SHELL, xraylib.L1_SHELL, xraylib.L2_SHELL, xraylib.L3_SHELL, xraylib.M1_SHELL)):
            try:
                ref = xraylib.EdgeEnergy(int(z), shell)*1e-3
            except ValueError:
                if binding[i, j] > 0:
                    rows.append(dict(Z=int(z), shell=j, status='report-only', reason='xraylib has no corresponding shell'))
                continue
            diff = float(binding[i, j]/ref-1)
            # EGS effective M averages several unresolved outer shells.
            reason = 'EGS5 effective M binding represents unresolved M/N/outer shells; M1 is only a comparator' if j == 4 else ''
            rows.append(dict(Z=int(z), shell=j, relative=diff, reason=reason,
                             status='pass' if abs(diff) <= .01 else ('report-only' if reason else 'fail')))
    sums = []
    for e in photon['E']:
        prob, local = shell_probabilities(e, shells)
        sums.append(abs(float(prob.sum())+local-1) <= 1e-12 and np.all(prob >= 0))
    return dict(status='pass' if all(sums) and not any(r['status'] == 'fail' for r in rows) else 'fail', binding=rows,
                probability_nodes=len(sums), local_absorption_included=True)


def verify_pair(ref, material, definition):
    from .pegs_p8a import pair_parameters
    a = load_reference(ref, material)
    if str(a['material']) != material or a['dl'].shape != (6, 6) or a['bpar'].shape != (2,) or a['delpos'].shape != (2,):
        raise ValueError('pair material / shape')
    source = Path(ref['source'])
    if sha(source) != ref['source_sha256'] or str(a['source_sha256']) != ref['source_sha256']:
        raise ValueError('pair source hash mismatch')
    b = pair_parameters(source.read_text(), 'P8A_'+material.upper(), definition, expected_start=ref['medium_line'])
    if int(a['dl_line']) != ref['dl_line'] or int(b['dl_line']) != ref['dl_line']:
        raise ValueError('pair reading position mismatch')
    for k, v in b.items():
        if np.asarray(v).shape != a[k].shape or np.asarray(v).tobytes() != a[k].tobytes():
            raise ValueError('pair constants differ from source: '+k)
    return dict(status='pass')


def mixture_checks(definition, tables):
    """Sorted summation, macroscopic density scaling, invariant angular law."""
    import copy
    from .dcs import moments
    a = MixtureElsepa(definition, tables)
    d = copy.deepcopy(definition)
    d['weights'] = dict(reversed(list(d['weights'].items())))
    d['atomic_weights'] = dict(reversed(list(d['atomic_weights'].items())))
    b = MixtureElsepa(d, tables)
    d['density_g_cm3'] *= 2
    c = MixtureElsepa(d, tables)
    order_ok, density_ok, angular_ok = True, True, True
    for t in (.01, .02, .1, 1., 2., 10.):
        y = a.edges(t)
        av, bv, cv = a.density(t, y), b.density(t, y), c.density(t, y)
        order_ok &= av.tobytes() == bv.tobytes()
        density_ok &= np.allclose(cv, 2*av, rtol=2e-14, atol=0)
        ga, ta, _ = moments(a, t)
        gc, tc, _ = moments(c, t)
        angular_ok &= np.allclose(ga[1:]/ta, gc[1:]/tc, rtol=2e-14, atol=0)
    return dict(status='pass' if order_ok and density_ok and angular_ok else 'fail',
                order_bytes=bool(order_ok), density_rate=bool(density_ok), angular_law=bool(angular_ok),
                pure_element=len(definition['weights']) == 1)


def water_checks(definition, elastic, gs):
    from . import physics
    from .materials_p8a import legacy_water
    from .generate_elsepa_p8a import arrays_identical
    from .pegs_p8a import pair_parameters
    legacy = legacy_water()
    # Actual legacy read path must agree in dtype/shape/bytes with source arrays.
    with np.load(physics.DATA/'estar.npz') as f:
        old = {k: f[k].copy() for k in f.files}
    stop_ok = arrays_identical(legacy['stopping'], old)
    source = ROOT/'docs/validation/mv/p1/egs5/base_2/pgs5job.pegs5dat'
    pair = pair_parameters(source.read_text(), 'H2O')
    # Protected historical constants and the formerly used parser's exact slice.
    lines = source.read_text().splitlines()
    values = np.array([float(v) for line in lines[6:16] for v in line.split()])
    pair_ok = pair['dl'].tobytes() == values[:36].reshape(6, 6).tobytes() and pair['bpar'][0] == values[38] and pair['delcm'] == values[36] and pair['delpos'][0] == values[39]
    tables = {int(z): load_reference(elastic[f'z{z}_q-1.npz'], 'element') for z in (1, 8)}
    p = MixtureElsepa(definition, tables, legacy_water=True)
    from .p1f_common import provider
    old_p = provider()
    points = np.geomspace(.01, 10, 17)
    elastic_ok = all(p.density(t, np.linspace(0, 2, 41)).tobytes() == old_p.density(t, np.linspace(0, 2, 41)).tobytes() for t in points)
    gs_ok = arrays_identical(gs, legacy['gs'])
    from .photon_data_p2 import TABLE
    with np.load(TABLE, allow_pickle=False) as f:
        photon_ok = arrays_identical(legacy['photon'], {k: f[k] for k in f.files})
    c = constants(definition)
    return dict(status='pass' if stop_ok and pair_ok and elastic_ok and gs_ok and photon_ok and legacy['NE'] == physics.NE and legacy['I'] == physics.I and legacy['NJ'].tobytes() == physics.NJ.tobytes() else 'fail',
                stopping_bytes=stop_ok, pair_bytes=pair_ok, elastic_bytes=elastic_ok, gs_bytes=gs_ok, photon_bytes=photon_ok,
                composition_consistency=dict(status='report-only', NE_relative=c['NE']/physics.NE-1,
                    NJ_relative=(np.array(c['NJ'])/physics.NJ-1).tolist(), I_relative=c['I_MeV']/physics.I-1))


def load_replacement_report(out):
    """A sidecar can resolve omissions only for the exact saved S4 inputs."""
    from .materials_p8a import implementation_hashes
    path = Path(out)/'elsepa_replacements_v10.json'
    v11 = Path(out)/'elsepa_replacements_v11.json'
    v12 = Path(out)/'elsepa_replacements_v12.json'
    if v11.exists():
        path = v11
    if v12.exists():
        path = v12
    if not path.exists():
        return {}
    report = json.loads(path.read_text())
    identity = report['identity']
    from .generate_elsepa_p8a import RECORDED_OMISSION, RECORDED_OMISSIONS
    expected = 'P8a appendix 12' if path == v12 else 'P8a appendix 11' if path == v11 else 'P8a appendix 10'
    if (report['authorization'] != expected or identity['replacement_factor'] != 1+1e-3
            or identity['timeout_seconds'] != 600 or identity['implementations'] != implementation_hashes()
            or (path == v11 and identity.get('recorded_omission') != RECORDED_OMISSION)
            or (path == v12 and identity.get('recorded_omissions') != RECORDED_OMISSIONS)
            or report['inputs'].get(str(Path(out)/'manifest.json')) != sha(Path(out)/'manifest.json')):
        raise ValueError('ELSEPA replacement report conditions mismatch')
    for name, expected in report['inputs'].items():
        if sha(name) != expected:
            raise ValueError('ELSEPA replacement input hash mismatch')
    return report['groups']


def material_checks_allow_s7(checks):
    """Accept the scoped appendix 12 result, never a general incomplete S6.

    Validate the S6 snapshot as well as its label: every original omission must
    have a passing replacement or be one of the two fixed double timeouts. Other
    rows and acceptance sections must pass. No cache or simulation is needed.
    """
    from .generate_elsepa_p8a import OMISSION_STATUS
    if checks.get('status') == 'pass':
        return True  # Preserve the pre-existing fully passed entry gate.
    declared = checks.get('recorded_omissions', [])
    if checks.get('status') != OMISSION_STATUS or not declared or canonical_recorded_omissions(declared) != declared:
        return False
    if set(checks.get('materials', {})) != set(MATERIALS):
        return False
    def passed(value):
        if not isinstance(value, dict):
            return True
        if 'status' in value:
            return value['status'] == 'pass'
        return all(passed(v) for v in value.values())
    witnessed = []
    for material, sections in checks['materials'].items():
        required = {'interpolation', 'pair', 'shells', 'atomic', 'mixture', 'pegs_report', 'gs'}
        if material == 'water': required.add('compatibility')
        if not required <= set(sections) or not all(passed(v) for k, v in sections.items() if k != 'atomic'):
            return False
        atomic = sections['atomic']
        allowed_original, allowed_replacement, scopes = set(), set(), []
        for group in atomic.get('direct_omissions', []):
            z, q = group['Z'], group['charge']
            original = {r['energy_MeV']: r for r in group['omitted']}
            pairs = group['replacement_pairs']
            if (len(original) != len(group['omitted']) or len(original) != group['not_evaluated_count']
                    or len(pairs) != len(original) or group['replacement_count'] != len(pairs)):
                return False
            seen, missing = set(), 0
            for pair in pairs:
                t = pair['original']['energy_MeV']
                if (t not in original or pair['original'] != original[t] or t in seen
                        or original[t]['status'] != 'not-evaluated' or original[t]['Z'] != z or original[t]['charge'] != q
                        or pair['energy_MeV'] != t*(1+1e-3) or pair['timeout_seconds'] != 600):
                    return False
                seen.add(t)
                allowed_original.add((z, q, t))
                if pair['status'] == pair['checks']['status'] == 'pass':
                    continue
                if (pair['status'] != 'not-evaluated' or pair['checks']['status'] != 'not-evaluated'
                        or not fixed_omission(t, pair['energy_MeV'], z, q)):
                    return False
                allowed_replacement.add((z, q, pair['energy_MeV']))
                scopes.append(dict(Z=z, charge=q, original_energy_MeV=t, replacement_energy_MeV=pair['energy_MeV']))
                missing += 1
            if group['unresolved_count'] != missing or group['recorded_omission_count'] != missing:
                return False
        canonical = canonical_recorded_omissions(scopes)
        if (len(canonical) != len(scopes) or atomic.get('recorded_omissions', []) != canonical
                or atomic['status'] != (OMISSION_STATUS if scopes else 'pass')):
            return False
        originals, replacements = [], []
        for row in atomic['rows']:
            if row['status'] == 'pass':
                continue
            if row['status'] != 'not-evaluated':
                return False
            if row.get('kind') == 'direct-timeout':
                originals.append((row['Z'], row['charge'], row['energy_MeV']))
            elif row.get('kind') == 'replacement-timeout':
                replacements.append((row['Z'], row['charge'], row['T']))
            else:
                return False
        if (len(originals) != len(allowed_original) or set(originals) != allowed_original
                or len(replacements) != len(allowed_replacement) or set(replacements) != allowed_replacement):
            return False
        witnessed.extend(scopes)
    return canonical_recorded_omissions(witnessed) == declared


def check(out=OUT):
    verify_protected()
    m = load_manifest(out/'manifest.json')
    replacements = load_replacement_report(out)
    report = {}
    for material in MATERIALS:
        row = m['materials'][material]
        data = {k: load_reference(r, material) for k, r in row.items()}
        definition = data['definition']
        if material == 'lung' and (definition['density_g_cm3'] != .26 or json.loads((out/'lung_state.json').read_text())['status'] != 'pass'):
            raise ValueError('S2 lung state not checked')
        c = constants(definition)
        tables = {int(z): load_reference(data['elastic'][f'z{z}_q-1.npz'], 'element') for z in c['Z']}
        provider = MixtureElsepa(definition, tables, legacy_water=(material == 'water'))
        report[material] = dict(interpolation=interpolation_checks(data['stopping'], data['photon'], c['NE']),
                                pair=verify_pair(row['pair'], material, definition),
                                shells=shell_checks(data['shells'], data['photon'], definition),
                                atomic=atomic_checks(data['elastic'], replacements),
                                mixture=mixture_checks(definition, tables))
        pegs_report = json.loads((out/(material+'_pegs_stopping.json')).read_text())
        if (pegs_report['IUNRST'] != 4 or not pegs_report['unrestricted'] or pegs_report['AE'] != .521
                or pegs_report['AP'] != .01 or pegs_report['units'] != 'MeV cm2/g'
                or sha(Path(pegs_report['raw'])) != pegs_report['raw_sha256']):
            raise ValueError('PEGS unrestricted comparison conditions / raw hash mismatch')
        explanations_path = out/'pegs_explanations.json'
        explanations = json.loads(explanations_path.read_text()) if explanations_path.exists() else {}
        needs_explanation = []
        for process in ('collision', 'radiative'):
            for i, diff in enumerate(pegs_report[process+'_relative']):
                if abs(diff) > .01:
                    key = f'{material}/{process}/{i}'
                    if not explanations.get(key, '').strip():
                        needs_explanation.append(key)
        report[material]['pegs_report'] = dict(status='pass' if not needs_explanation else 'not-evaluated',
                comparison='report-only; no PEGS/ESTAR agreement verdict',
                missing_explanations=needs_explanation, details=pegs_report)
        if material == 'water':
            report[material]['compatibility'] = water_checks(definition, data['elastic'], data['gs'])
            # P1f GS numerical checks must already be sealed and successful.
            prior = json.loads((ROOT/'docs/validation/mv/p1f/t4.json').read_text())
            from .check_gs_p1f import from_saved
            from_saved(provider, data['gs'], prior)
            report[material]['gs'] = dict(status=prior['status'], method='P1f saved numerical checks replayed')
        else:
            report[material]['gs'] = gs_checks(provider, data['gs'])
        write(out/'checks_progress.json', report)
    def states(value):
        if isinstance(value, dict):
            if 'status' in value:
                yield value['status']
            else:
                for item in value.values():
                    yield from states(item)
    recorded = canonical_recorded_omissions([scope for r in report.values() for scope in r['atomic']['recorded_omissions']])
    result = dict(status=aggregate_recorded_omission(states(report)), materials=report,
                  recorded_omissions=recorded,
                  manifest_sha256=sha(out/'manifest.json'),
                  protected_files=verify_protected(), model_choice='separate S7 fluorescence decision')
    write(out/'checks.json', result)
    return result


def main():
    argparse.ArgumentParser(__doc__).parse_args()
    result = check()
    print(result['status'])
    from .generate_elsepa_p8a import OMISSION_STATUS
    if result['status'] not in ('pass', OMISSION_STATUS):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
