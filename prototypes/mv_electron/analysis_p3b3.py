"""Fixed P3b-3 statistics on batches; no file I/O or transport."""
import math
import numpy as np

MEV_TO_GY = 1.602176634e-10
SHAPES = dict(axis_coarse=(100, 120), axis_fine=(100, 300),
              slab_coarse=(100, 120), profile=(100, 2, 3, 72),
              profile_fine=(100, 2, 3, 144))
VOLUMES = dict(axis_coarse=1., axis_fine=.4, slab_coarse=.25,
               profile=.5, profile_fine=.25)
BLOCKS = [(20+20*i, 40+20*i) for i in range(5)]
DEPTHS = (5, 10, 20)
BOOTSTRAPS = 2000
BOOT_SEED = 20291770
RULES = dict(plan='2026-10-13-mv-p3b3-pilot-production-analysis.md',
             dose_tolerance_percent=1., dose_sem_percent=1/3,
             position_tolerance_cm=.1, position_sem_cm=.1/3,
             blocks=BLOCKS, central_bins=16, bootstraps=BOOTSTRAPS,
             bootstrap_seed=BOOT_SEED, symmetry_z=2.77,
             pilot_N=4000000, margin=1.5, quantum=2000000, maximum_N=160000000)
CAVEATS = ('対称性の水準と検出できる大きさは正規近似による名目の対比ごとの値。'
           '自由度99のt分布では全体の水準は約6%。対比間の相関があり実際の全体の誤検出率ではない。'
           '検出できる大きさより小さい偏りがないとは主張しない。'
           '厳密な同等性の検定ではない。Drefの不確かさは伝播せず、多重比較を補正しない。'
           '試し計算からの外挿と1.5倍の余裕は本番の精度を保証しない。統計不足でも粒子を追加しない。')


def dose(data):
    """Both input formats contain region sums in MeV, never legacy dose arrays."""
    try:
        n = np.asarray(data['N'])
        if n.shape != () or not np.isfinite(n) or n <= 0 or n != int(n) or int(n) % 100:
            raise ValueError('invalid N')
        if 'nbatch' in data and np.asarray(data['nbatch']).item() != 100:
            raise ValueError('nbatch must be 100')
        out = {}
        for key, shape in SHAPES.items():
            x = np.asarray(data[key], dtype=float)
            if x.shape != shape or not np.all(np.isfinite(x)) or np.any(x < 0):
                raise ValueError('invalid region: '+key)
            out[key] = x * MEV_TO_GY / (int(n)/100 * VOLUMES[key])
        return out
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('invalid input') from exc


def directions(profile):
    """Return +x, -x, +y, -y in outward order (batch, direction, depth, radius)."""
    x = np.asarray(profile, float)
    if x.shape not in (SHAPES['profile'], SHAPES['profile_fine']) or not np.all(np.isfinite(x)):
        raise ValueError('invalid profile')
    mid = x.shape[-1]//2
    return np.stack((x[:, 0, :, mid:], x[:, 0, :, :mid][:, :, ::-1],
                     x[:, 1, :, mid:], x[:, 1, :, :mid][:, :, ::-1]), axis=1)


def fold(profile):
    return directions(profile).mean(axis=1)


def mean_sem(values):
    x = np.asarray(values, float)
    if x.shape[0] != 100 or not np.all(np.isfinite(x)):
        raise ValueError('invalid batch values')
    return x.mean(axis=0), x.std(axis=0, ddof=1)/10


def classify(delta, sem, tolerance=1.):
    if delta is None or sem is None or not np.all(np.isfinite([delta, sem])) or sem < 0:
        return '判定できない'
    if sem > tolerance/3:
        return '統計不足'
    unit = '1%' if tolerance == 1 else '0.1 cm'
    return unit+'以内' if abs(delta) <= tolerance else unit+'を超える'


def difference(left, right, scale=1., tolerance=1.):
    lm, ls = mean_sem(left); rm, rs = mean_sem(right)
    d = float((lm-rm)*scale); s = float(np.hypot(ls, rs)*scale)
    return dict(delta=d, SEM=s, proto=float(lm), egs=float(rm),
                upper_abs_95=abs(d)+1.96*s, status=classify(d, s, tolerance))


def smooth(curve):
    return np.convolve(curve, np.ones(5)/5, mode='same')


def dref(axis):
    x = np.asarray(axis, float)
    if x.shape != (100, 120) or not np.all(np.isfinite(x)):
        raise ValueError('invalid Dref curve')
    value = float(smooth(x.mean(axis=0)).max())
    if not np.isfinite(value) or value <= 0:
        raise ValueError('Dref must be positive and finite')
    return value


def crossing(curve, width=.25, fraction=.5):
    """First outward downward crossing of this curve's own central level."""
    x = np.asarray(curve, float)
    if width not in (.25, .125) or x.shape != (int(9/width),) or not np.all(np.isfinite(x)):
        return None
    central = float(x[:int(4/width)].mean())
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


def bootstrap_position(batches, width=.25, fraction=.5):
    x = np.asarray(batches, float)
    if x.shape != (100, int(9/width)) or not np.all(np.isfinite(x)):
        raise ValueError('invalid crossing batches')
    position = crossing(x.mean(axis=0), width, fraction)
    rng = np.random.default_rng(BOOT_SEED)
    values = [crossing(x[rng.integers(0, 100, 100)].mean(axis=0), width, fraction)
              for _ in range(BOOTSTRAPS)]
    missing = sum(v is None for v in values)
    sem = None
    if not missing and position is not None:
        # Roundoff in proportional curves is not physical sampling uncertainty.
        sem = float(np.std(values, ddof=1))
        if np.ptp(values) <= 1e-12:
            sem = 0.
    return dict(position=position, SEM=sem, missing=missing)


def compare(left, right, reference):
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError('invalid Dref')
    a, b = fold(left['profile']), fold(right['profile'])
    out = dict(J1=[], J2=[], J3=[])
    for lo, hi in BLOCKS:
        item = difference(left['axis_coarse'][:, lo:hi].mean(1),
                          right['axis_coarse'][:, lo:hi].mean(1), 100/reference)
        out['J1'].append(dict(item, interval_cm=[lo*.25, hi*.25]))
    for j, depth in enumerate(DEPTHS):
        out['J2'].append(dict(difference(a[:, j, :16].mean(1), b[:, j, :16].mean(1),
                                        100/reference), depth_cm=depth))
        lp = bootstrap_position(a[:, j]); rp = bootstrap_position(b[:, j])
        delta = None if lp['position'] is None or rp['position'] is None else lp['position']-rp['position']
        sem = None if lp['SEM'] is None or rp['SEM'] is None else float(np.hypot(lp['SEM'], rp['SEM']))
        out['J3'].append(dict(depth_cm=depth, proto=lp, egs=rp, delta=delta, SEM=sem,
                              status=classify(delta, sem, .1)))
    return out


def main_size(statistics):
    dose_sem = [v['SEM'] for key in ('J1', 'J2') for v in statistics[key]]
    pos_sem = [v['SEM'] for v in statistics['J3']]
    if len(dose_sem) != 8 or len(pos_sem) != 3:
        raise ValueError('invalid pilot statistics')
    try:
        values = np.asarray(dose_sem+pos_sem, float)
        if not np.all(np.isfinite(values)) or np.any(values < 0) or not np.any(values > 0):
            raise ValueError('外挿できない')
        need = max(np.max((values[:8]/(1/3))**2), np.max((values[8:]/(.1/3))**2))
        n = max(2000000, math.ceil(4000000*1.5*need/2000000)*2000000)
    except (TypeError, OverflowError) as exc:
        raise ValueError('外挿できない') from exc
    return dict(N_main=n, status='相談' if n > 160000000 else '合格')


def symmetry(profile):
    sides = directions(profile)[:, :, :, :16].mean(axis=-1)
    central = sides.mean(axis=1).mean(axis=0)
    contrasts = np.stack((sides[:, 0]-sides[:, 1], sides[:, 2]-sides[:, 3],
                          (sides[:, 0]+sides[:, 1]-sides[:, 2]-sides[:, 3])/2), axis=-1)
    means, sems = mean_sem(contrasts)
    rows = []
    for j, depth in enumerate(DEPTHS):
        for k, name in enumerate(('+x - -x', '+y - -y', 'x - y')):
            m, s = float(means[j, k]), float(sems[j, k])
            z = m/s if s else (None if m == 0 else math.copysign(math.inf, m))
            status = ('判定できない（SEM 0）' if z is None else
                      '偏りを検出' if abs(z) >= 2.77 else '偏りは検出されず')
            rows.append(dict(depth_cm=depth, contrast=name, mean=m, SEM=s,
                             z=z if z is None or np.isfinite(z) else ('+inf' if z > 0 else '-inf'),
                             status=status, detection_percent=float(100*3.61*s/central[j])
                             if central[j] > 0 else None))
    state = ('偏りを検出' if any(x['status']=='偏りを検出' for x in rows) else
             '判定できない（SEM 0）' if any(x['z'] is None for x in rows) else '偏りは検出されず')
    return dict(status=state, contrasts=rows, note=CAVEATS)


def reports(left, right, reference):
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
        row = dict(depth_cm=depth, outside=difference(fold(left['profile'])[:, j, 28:36].mean(1),
                    fold(right['profile'])[:, j, 28:36].mean(1), 100/reference))
        for code, data in (('proto', left), ('egs', right)):
            curve = fold(data['profile']).mean(0)[j]
            p20, p80 = crossing(curve, fraction=.2), crossing(curve, fraction=.8)
            row[code] = dict(position20=p20, position80=p80,
                            width20_80=None if p20 is None or p80 is None else p20-p80,
                            position50_fine=crossing(fold(data['profile_fine']).mean(0)[j], .125),
                            unfolded=directions(data['profile']).mean(0)[:, j].tolist())
        result['profiles'].append(row)
    # Report-only quantities carry no pass/fail classification.
    for key in ('intervals', 'slab_blocks'):
        for row in result[key]:
            row.pop('status')
    for row in result['profiles']:
        row['outside'].pop('status')
    return result
