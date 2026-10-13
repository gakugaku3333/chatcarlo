"""Fixed P4a batch statistics, sizing and history comparison (MeV inputs)."""
import math
import numpy as np

ENERGIES = (.1, 1., 10.)
RANGES = (.0144, .44, 5.)
DIFF_LIMIT = .01
SEM_LIMIT = 1/300
PILOT_N = 200000
MAX_N = 20000000


def geometry(index, system, margin=2.):
    """A/B and their homogeneous controls have the same mass faces."""
    R = RANGES[index]
    if system in ('A', 'H_A'):
        masses = np.array([margin, .5, .5, .5, margin])*R
        rho = np.array([1., .5, 2., .25, 1.])
    elif system in ('B', 'H_B'):
        masses = np.array([margin, .5, 1/40, .5, margin])*R
        rho = np.array([1., 1., 1/1024, 1., 1.])
    else:
        raise ValueError('unknown system')
    if system.startswith('H_'):
        rho[:] = 1.
    mf = np.r_[0., np.cumsum(masses)]
    faces = np.r_[0., np.cumsum(masses/rho)]
    return dict(layer_faces=faces, layer_rho=rho, mass_faces=mf,
                mass_bin=R/40, thickness=float(faces[-1]))


def expected_energy(n, T, delta_m, total_mass):
    if n <= 0 or T <= 0 or delta_m <= 0 or total_mass <= 0:
        raise ValueError('positive expectation inputs required')
    return n*T*delta_m/total_mass


def classify(difference, sem, valid=True):
    if not valid or not np.all(np.isfinite([difference, sem])) or sem < 0:
        return '判定できない'
    if sem > SEM_LIMIT:
        return '統計不足'
    if abs(difference) > DIFF_LIMIT:
        return '不合格'
    return '合格'


def healthy(result):
    return (not np.any(result['violations']) and int(result['mass_ledger']) == 0
            and not np.any(result['counts'][5:7]))


def layer_statistics(batches, n, T, mass_faces, mass_bin, valid=True):
    a = np.asarray(batches, float)
    mf = np.asarray(mass_faces, float)
    if n <= 0 or n % 100 or a.shape != (100, round(mf[-1]/mass_bin)):
        raise ValueError('100 equal batches and matching mass bins required')
    edges = np.rint(mf/mass_bin).astype(int)
    rows = []
    for layer in range(len(mf)-1):
        delta = mf[layer+1]-mf[layer]
        exp = expected_energy(n, T, delta, mf[-1])
        energy = a[:, edges[layer]:edges[layer+1]].sum(axis=1)
        ratios = energy/(exp/100)-1
        diff = float(np.mean(ratios))
        sem = float(np.std(ratios, ddof=1)/10)
        rows.append(dict(layer=layer, energy=float(energy.sum()), expected=exp,
                         difference=diff, sem=sem, status=classify(diff, sem, valid)))
    return rows


def required_histories(sems):
    a = np.asarray(sems, float)
    if a.size == 0 or not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError('finite nonnegative SEMs required')
    need = PILOT_N*float(np.max(a/SEM_LIMIT))**2
    n = max(PILOT_N, math.ceil(1.5*need/PILOT_N)*PILOT_N)
    if n > MAX_N:
        raise ValueError('required histories exceed 20000000; user decision required')
    return n


def compare_histories(left, right, incoming):
    a, b = np.asarray(left, float), np.asarray(right, float)
    if a.shape != b.shape or a.ndim != 2 or len(a) == 0:
        raise ValueError('matching nonempty history matrices required')
    energy = np.broadcast_to(np.asarray(incoming, float), (len(a),))
    if np.any(energy <= 0) or not np.all(np.isfinite(energy)):
        raise ValueError('positive finite incoming energy required')
    maxima = np.max(np.abs(a-b), axis=1)
    matched = np.isfinite(maxima) & (maxima <= energy*1e-9)
    # Only matched histories participate in the aggregate comparison.
    total = float(np.sum(a[matched]))
    error = float(abs(np.sum(a[matched])-np.sum(b[matched])))
    aggregate = error <= abs(total)*1e-9
    count = int(np.count_nonzero(matched))
    indices = np.flatnonzero(~matched)[:20]
    return dict(matched=count, n=len(a), aggregate_difference=error,
                aggregate_total=total, aggregate_match=bool(aggregate),
                mismatches=[dict(history=int(i), maximum=float(maxima[i])) for i in indices],
                status='合格' if count >= math.ceil(.999*len(a)) and aggregate else '不合格')


def report_only_doses(batches, mass_faces, mass_bin):
    """Dose per layer mass and pairwise relative differences; no Fano expectation."""
    edges = np.rint(np.asarray(mass_faces)/mass_bin).astype(int)
    doses = [np.sum(batches[:, edges[j]:edges[j+1]], axis=1)/
             (mass_faces[j+1]-mass_faces[j]) for j in (1, 2, 3)]
    means = [float(np.mean(d)) for d in doses]
    sems = [float(np.std(d, ddof=1)/10) for d in doses]
    pairs = []
    for j, k in ((0, 1), (0, 2), (1, 2)):
        contrast = (doses[j]-doses[k])/means[k]
        pairs.append(dict(layers=[j+1, k+1], difference=float(np.mean(contrast)),
                          sem=float(np.std(contrast, ddof=1)/10)))
    return dict(dose_per_mass=means, sem=sems, pairs=pairs)
