"""P7 statistics: P3b-3's rules with condition-dependent central / outside ranges.

Everything that does not depend on the layout is imported from analysis_p3b3 unchanged.
Plan: docs/ai/plans/2026-10-18-mv-p7-field-energy-variants.md (section 3, D5).
"""
import math
import numpy as np
from . import analysis_p3b3 as base

MEV_TO_GY = base.MEV_TO_GY
BLOCKS = base.BLOCKS
DEPTHS = base.DEPTHS
BOOTSTRAPS = base.BOOTSTRAPS
BOOT_SEED = base.BOOT_SEED
mean_sem = base.mean_sem
classify = base.classify
difference = base.difference
smooth = base.smooth
dref = base.dref
VOLUMES = base.VOLUMES
CENTRAL_FRACTION = .8
OUTSIDE_START = 1.5
OUTSIDE_WIDTH = 2.
COARSE = .25


def half_width(field_half, depth, ssd=100.):
    return field_half*(ssd+depth)/ssd


def spec(field_half=None, extent=20., ssd=100., legacy=False):
    """Bin counts on the coarse (0.25 cm) radius grid. legacy=True reproduces P3b-3 (16 bins, 28:36, +-9 cm)."""
    if legacy:
        return dict(extent=9., radius_bins=36, central=[16, 16, 16], outside=[(28, 36)]*3)
    central, outside = [], []
    for depth in DEPTHS:
        h = half_width(field_half, depth, ssd)
        central.append(int(math.floor(CENTRAL_FRACTION*h/COARSE+1e-9)))
        lo = int(math.ceil(OUTSIDE_START*h/COARSE-1e-9)); outside.append((lo, lo+int(round(OUTSIDE_WIDTH/COARSE))))
    nb = int(round(extent/COARSE))
    if any(hi > nb for _, hi in outside) or min(central) < 1:
        raise ValueError('ranges outside the scored extent')
    return dict(extent=extent, radius_bins=nb, central=central, outside=outside)


def shapes(s):
    n = 2*s['radius_bins']
    return dict(axis_coarse=(100, 120), axis_fine=(100, 300), slab_coarse=(100, 120),
                profile=(100, 2, 3, n), profile_fine=(100, 2, 3, 2*n))


def dose(data, s):
    try:
        n = np.asarray(data['N'])
        if n.shape != () or not np.isfinite(n) or n <= 0 or n != int(n) or int(n) % 100:
            raise ValueError('invalid N')
        if 'nbatch' in data and np.asarray(data['nbatch']).item() != 100:
            raise ValueError('nbatch must be 100')
        out = {}
        for key, shape in shapes(s).items():
            x = np.asarray(data[key], dtype=float)
            if x.shape != shape or not np.all(np.isfinite(x)) or np.any(x < 0):
                raise ValueError('invalid region: '+key)
            out[key] = x * MEV_TO_GY / (int(n)/100 * VOLUMES[key])
        return out
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('invalid input') from exc


def directions(profile):
    """+x, -x, +y, -y in outward order (batch, direction, depth, radius); any even length."""
    x = np.asarray(profile, float)
    if x.ndim != 4 or x.shape[1:3] != (2, 3) or x.shape[-1] % 2 or not np.all(np.isfinite(x)):
        raise ValueError('invalid profile')
    mid = x.shape[-1]//2
    return np.stack((x[:, 0, :, mid:], x[:, 0, :, :mid][:, :, ::-1],
                     x[:, 1, :, mid:], x[:, 1, :, :mid][:, :, ::-1]), axis=1)


def fold(profile):
    return directions(profile).mean(axis=1)


def crossing(curve, width, central_bins, fraction=.5):
    """P3b-3's crossing with the central level taken over `central_bins` coarse bins
    (2x as many on the 0.125 cm grid: the same physical edge, not re-rounded)."""
    x = np.asarray(curve, float)
    if width not in (.25, .125) or x.ndim != 1 or not np.all(np.isfinite(x)):
        return None
    m = central_bins if width == .25 else 2*central_bins
    central = float(x[:m].mean())
    if not np.isfinite(central) or central <= 0:
        return None
    level = central*fraction
    hits = np.flatnonzero(x < level)
    if not len(hits) or hits[0] == 0:
        return None
    j = int(hits[0]); inner = (j-.5)*width
    if x[j-1] == level:
        return inner
    return float(inner+width*(x[j-1]-level)/(x[j-1]-x[j]))


def bootstrap_position(batches, width, central_bins, fraction=.5):
    x = np.asarray(batches, float)
    if x.ndim != 2 or x.shape[0] != 100 or not np.all(np.isfinite(x)):
        raise ValueError('invalid crossing batches')
    position = crossing(x.mean(axis=0), width, central_bins, fraction)
    rng = np.random.default_rng(BOOT_SEED)
    values = [crossing(x[rng.integers(0, 100, 100)].mean(axis=0), width, central_bins, fraction)
              for _ in range(BOOTSTRAPS)]
    missing = sum(v is None for v in values)
    sem = None
    if not missing and position is not None:
        sem = float(np.std(values, ddof=1))
        if np.ptp(values) <= 1e-12:
            sem = 0.
    return dict(position=position, SEM=sem, missing=missing)


def compare(left, right, reference, s):
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError('invalid Dref')
    a, b = fold(left['profile']), fold(right['profile'])
    out = dict(J1=[], J2=[], J3=[])
    for lo, hi in BLOCKS:
        item = difference(left['axis_coarse'][:, lo:hi].mean(1), right['axis_coarse'][:, lo:hi].mean(1), 100/reference)
        out['J1'].append(dict(item, interval_cm=[lo*.25, hi*.25]))
    for j, depth in enumerate(DEPTHS):
        m = s['central'][j]
        out['J2'].append(dict(difference(a[:, j, :m].mean(1), b[:, j, :m].mean(1), 100/reference),
                              depth_cm=depth, central_cm=m*COARSE))
        lp = bootstrap_position(a[:, j], .25, m); rp = bootstrap_position(b[:, j], .25, m)
        delta = None if lp['position'] is None or rp['position'] is None else lp['position']-rp['position']
        sem = None if lp['SEM'] is None or rp['SEM'] is None else float(np.hypot(lp['SEM'], rp['SEM']))
        out['J3'].append(dict(depth_cm=depth, proto=lp, egs=rp, delta=delta, SEM=sem, status=classify(delta, sem, .1)))
    return out


def stop_on_missing(statistics):
    """Plan section 3: any bootstrap sample without a crossing stops the stage."""
    for row in statistics['J3']:
        for side in ('proto', 'egs'):
            if row[side]['position'] is None or row[side]['missing']:
                raise ValueError('50% position not found in every bootstrap sample; stop')


def required(statistics, pilot_N=4000000):
    """Histories needed so the worst of the 11 quantities meets its precision condition."""
    dose_sem = [v['SEM'] for key in ('J1', 'J2') for v in statistics[key]]
    pos_sem = [v['SEM'] for v in statistics['J3']]
    if len(dose_sem) != 8 or len(pos_sem) != 3:
        raise ValueError('invalid pilot statistics')
    values = np.asarray(dose_sem+pos_sem, float)
    if not np.all(np.isfinite(values)) or np.any(values < 0) or not np.any(values > 0):
        raise ValueError('外挿できない')
    need = max(np.max((values[:8]/(1/3))**2), np.max((values[8:]/(.1/3))**2))
    return float(pilot_N*need)


def symmetry(profile, s):
    d = directions(profile)
    sides = np.stack([d[:, :, j, :s['central'][j]].mean(axis=-1) for j in range(3)], axis=2)
    central = sides.mean(axis=1).mean(axis=0)
    contrasts = np.stack((sides[:, 0]-sides[:, 1], sides[:, 2]-sides[:, 3],
                          (sides[:, 0]+sides[:, 1]-sides[:, 2]-sides[:, 3])/2), axis=-1)
    means, sems = mean_sem(contrasts)
    rows = []
    for j, depth in enumerate(DEPTHS):
        for k, name in enumerate(('+x - -x', '+y - -y', 'x - y')):
            m, sd = float(means[j, k]), float(sems[j, k])
            z = m/sd if sd else (None if m == 0 else math.copysign(math.inf, m))
            status = ('判定できない（SEM 0）' if z is None else '偏りを検出' if abs(z) >= 2.77 else '偏りは検出されず')
            rows.append(dict(depth_cm=depth, contrast=name, mean=m, SEM=sd,
                             z=z if z is None or np.isfinite(z) else ('+inf' if z > 0 else '-inf'),
                             status=status, detection_percent=float(100*3.61*sd/central[j]) if central[j] > 0 else None))
    state = ('偏りを検出' if any(x['status'] == '偏りを検出' for x in rows) else
             '判定できない（SEM 0）' if any(x['z'] is None for x in rows) else '偏りは検出されず')
    return dict(status=state, contrasts=rows, note=base.CAVEATS)


def reports(left, right, reference, s):
    result = dict(intervals=[], slab_blocks=[], profiles=[], peaks={})
    for lo, hi in ((0, 8), (8, 12), (12, 20)):
        result['intervals'].append(dict(difference(left['axis_coarse'][:, lo:hi].mean(1),
            right['axis_coarse'][:, lo:hi].mean(1), 100/reference), interval_cm=[lo*.25, hi*.25]))
    for lo, hi in BLOCKS:
        result['slab_blocks'].append(dict(difference(left['slab_coarse'][:, lo:hi].mean(1),
            right['slab_coarse'][:, lo:hi].mean(1), 100/reference), interval_cm=[lo*.25, hi*.25]))
    for code, data in (('proto', left), ('egs', right)):
        result['peaks'][code] = (int(np.argmax(smooth(data['axis_fine'].mean(0))))+.5)*.1
    for j, depth in enumerate(DEPTHS):
        lo, hi = s['outside'][j]; m = s['central'][j]
        row = dict(depth_cm=depth, outside_cm=[lo*COARSE, hi*COARSE],
                   outside=difference(fold(left['profile'])[:, j, lo:hi].mean(1), fold(right['profile'])[:, j, lo:hi].mean(1), 100/reference))
        for code, data in (('proto', left), ('egs', right)):
            curve = fold(data['profile']).mean(0)[j]
            p20, p80 = crossing(curve, .25, m, .2), crossing(curve, .25, m, .8)
            row[code] = dict(position20=p20, position80=p80, width20_80=None if p20 is None or p80 is None else p20-p80,
                             position50_fine=crossing(fold(data['profile_fine']).mean(0)[j], .125, m),
                             unfolded=directions(data['profile']).mean(0)[:, j].tolist())
        result['profiles'].append(row)
    for key in ('intervals', 'slab_blocks'):
        for row in result[key]:
            row.pop('status')
    for row in result['profiles']:
        row['outside'].pop('status')
    return result


SLAB_INTERVALS = ((0, 8), (8, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 120))


def slab_ratios(left, right):
    """Report only (P3b-3 method): whole-slab deposit ratio - 1 per depth interval, SEMs propagated."""
    rows = []
    for lo, hi in SLAB_INTERVALS:
        lm, ls = mean_sem(left['slab_coarse'][:, lo:hi].mean(1)); rm, rs = mean_sem(right['slab_coarse'][:, lo:hi].mean(1))
        ratio = float(lm/rm)
        rows.append(dict(interval_cm=[lo*COARSE, hi*COARSE], ratio_minus_1_percent=100*(ratio-1),
                         SEM_percent=float(100*ratio*np.hypot(ls/lm, rs/rm))))
    return rows
