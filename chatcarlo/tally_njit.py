"""Numba-compiled scalar DDA for the kernel dose-grid path."""
from __future__ import annotations

import numpy as np
from numba import njit

from . import tally

_PARALLEL_EPS = 1e-12
_SCRATCH_MARGIN = 16


@njit(cache=True)
def _enumerate_intersections(origin, direction, enter, leave, lo, shape, h,
                             eps_plane, ts):
    """Append clipped endpoints and eligible grid-plane intersections to ``ts``."""
    n_t = 0
    if n_t >= ts.size:
        raise ValueError("DDA intersection scratch array is too small")
    ts[n_t] = enter
    n_t += 1
    if n_t >= ts.size:
        raise ValueError("DDA intersection scratch array is too small")
    ts[n_t] = leave
    n_t += 1

    for k in range(3):
        d = direction[k]
        if abs(d) < _PARALLEL_EPS:
            continue
        p_enter = (origin[k] + enter * d - lo[k]) / h
        p_leave = (origin[k] + leave * d - lo[k]) / h
        p_small = min(p_enter, p_leave)
        p_big = max(p_enter, p_leave)
        m_lo = int(np.ceil(p_small + eps_plane))
        m_hi = int(np.floor(p_big - eps_plane))
        for m in range(m_lo, m_hi + 1):
            if n_t >= ts.size:
                raise ValueError("DDA intersection scratch array is too small")
            ts[n_t] = (lo[k] + m * h - origin[k]) / d
            n_t += 1
    return n_t


@njit(cache=True)
def _dda_njit(origin, direction, length_cm, weights_a, weights_b,
              lo, shape, h, target_a, target_b, eps_plane):
    """Reference-equivalent scalar intersection enumeration and midpoint scoring."""
    hi0 = lo[0] + shape[0] * h
    hi1 = lo[1] + shape[1] * h
    hi2 = lo[2] + shape[2] * h
    for s in range(origin.shape[0]):
        enter = 0.0
        leave = length_cm[s]
        valid = True
        for k in range(3):
            o = origin[s, k]
            d = direction[s, k]
            lk = lo[k]
            hk = hi0 if k == 0 else (hi1 if k == 1 else hi2)
            if abs(d) < _PARALLEL_EPS:
                if o < lk or o > hk:
                    valid = False
                    break
            else:
                ta = (lk - o) / d
                tb = (hk - o) / d
                if ta > tb:
                    tmp = ta
                    ta = tb
                    tb = tmp
                if ta > enter:
                    enter = ta
                if tb < leave:
                    leave = tb
        if not valid or leave <= enter:
            continue

        ts = np.empty(shape[0] + shape[1] + shape[2] + _SCRATCH_MARGIN,
                      dtype=np.float64)
        n_t = _enumerate_intersections(
            origin[s], direction[s], enter, leave, lo, shape, h, eps_plane, ts)

        for i in range(1, n_t):
            value = ts[i]
            j = i - 1
            while j >= 0 and ts[j] > value:
                ts[j + 1] = ts[j]
                j -= 1
            ts[j + 1] = value

        for i in range(n_t - 1):
            dl = ts[i + 1] - ts[i]
            if dl <= 0.0:
                continue
            mid = (ts[i] + ts[i + 1]) * 0.5
            ix = int(np.floor((origin[s, 0] + direction[s, 0] * mid - lo[0]) / h))
            iy = int(np.floor((origin[s, 1] + direction[s, 1] * mid - lo[1]) / h))
            iz = int(np.floor((origin[s, 2] + direction[s, 2] * mid - lo[2]) / h))
            if (ix < 0 or ix >= shape[0] or iy < 0 or iy >= shape[1]
                    or iz < 0 or iz >= shape[2]):
                continue
            target_a[ix, iy, iz] += weights_a[s] * dl
            target_b[ix, iy, iz] += weights_b[s] * dl


def accumulate_track_length_multi_njit(
        target_weight_pairs, grid, origin, direction, length_cm):
    """Accumulate two track-length tallies with the scalar njit DDA.

    This has the same call signature as
    :func:`chatcarlo.tally.accumulate_track_length_multi`. Inputs are expected
    to be float64, C-contiguous arrays and writable targets, matching the
    existing function's operating assumptions. Other dtypes, non-contiguous
    arrays, and read-only arrays are undefined.
    """
    if len(target_weight_pairs) != 2:
        raise ValueError("target_weight_pairs must contain exactly two pairs")
    n = length_cm.shape[0] if length_cm.ndim == 1 else -1
    if origin.shape != (n, 3):
        raise ValueError(f"origin must have shape ({n}, 3)")
    if direction.shape != (n, 3):
        raise ValueError(f"direction must have shape ({n}, 3)")
    if length_cm.shape != (n,):
        raise ValueError(f"length_cm must have shape ({n},)")
    for target, weight in target_weight_pairs:
        if target.shape != grid.shape:
            raise ValueError(f"target must have shape {grid.shape}")
        if weight.shape != (n,):
            raise ValueError(f"weight must have shape ({n},)")

    (target_a, weights_a), (target_b, weights_b) = target_weight_pairs
    _dda_njit(
        origin, direction, length_cm, weights_a, weights_b,
        grid.origin_cm, np.asarray(grid.shape, dtype=np.int64),
        grid.voxel_size_cm, target_a, target_b, tally._EPS_PLANE)
