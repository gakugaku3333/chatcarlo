"""NumPy-only candidate maps. Correctness comes from device-side certification."""
import numpy as np
NAMES = ('us', 'brems_e', 'rates', 'loget', 'pes', 'ts')
SITES = ('density/loget', 'radiation/loget', 'brems/brems_e', 'photoelectric/pes',
         'pair0/pes', 'pair1/pes', 'scattering/rates', 'GS/ts', 'GS/us')


def grids(data):
    return (data['us'], data['brems_e'],
            data['rates'][:, 0] if data['rates'].shape[1] == 3 else data['ts'],
            data['loget'], data['pes'], data['ts'])


def generate(data):
    ranges, meta, arrays = [], [], {}
    for name, raw in zip(NAMES, grids(data)):
        a = np.asarray(raw, dtype=np.float64)
        if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all() or np.any(np.diff(a) < 0):
            raise ValueError('grid must be finite, nondecreasing, length >= 2: '+name)
        if name == 'pes':
            if np.any(a <= 0): raise ValueError('nonpositive photon grid')
            a = np.log(a)
        lo, hi = float(a[0]), float(a[-1])
        if not hi > lo: raise ValueError('zero grid span: '+name)
        bins = 2*len(a)
        edges = np.linspace(lo, hi, bins+1)
        # Adjacent bins plus one index on each side. These are predictions only.
        L = np.maximum(0, np.searchsorted(a, edges[np.maximum(0, np.arange(bins)-1)], side='left')-1)
        H = np.minimum(len(a), np.searchsorted(a, edges[np.minimum(bins, np.arange(bins)+2)], side='right')+1)
        r = np.column_stack((L, H)).astype(np.int32).ravel()
        meta.extend((lo, hi, bins/(hi-lo), bins, sum(len(v) for v in ranges)))
        ranges.append(r)
        arrays['grid_'+name] = np.ascontiguousarray(raw)
        arrays['edges_'+name] = edges
    starts = np.array([np.searchsorted(data['pes'], th, side='right') for th in data['thresholds']], np.int32)
    if np.any(starts >= len(data['pes'])): raise ValueError('pair threshold beyond grid')
    arrays.update(ranges=np.concatenate(ranges), meta=np.array(meta), pair_start=starts,
                  thresholds=np.array(data['thresholds'], copy=True))
    return arrays


def vectors(raw, edges, logarithmic=False, randoms=1000000):
    """Endpoint/device-log cases are added by the native gate kernel too."""
    a = np.log(raw) if logarithmic else raw
    v = np.concatenate((a, edges))
    rng = np.random.default_rng(20261016)
    return np.ascontiguousarray(np.concatenate((v, np.nextafter(v, -np.inf),
        np.nextafter(v, np.inf), [np.nan, np.inf, -np.inf, 0., -0., -1.,
        np.finfo(float).max, -np.finfo(float).max, np.finfo(float).tiny],
        rng.uniform(a[0], a[-1], randoms))))
