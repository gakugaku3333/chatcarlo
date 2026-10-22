"""Exact production PCHIP coefficients and ICRP table, without resampling."""
import numpy as np


def pack(names):
    from ..materials import _XAAMDI_FILES, _muen_interpolator, density
    from ..dose_coefficients import _table
    curves = [_muen_interpolator(_XAAMDI_FILES[n.lower().strip()]) for n in names]
    stride = max(len(p.x) for p in curves)
    x = np.zeros((len(names), stride))
    c = np.zeros((len(names), 4, stride))
    for j, p in enumerate(curves):
        x[j, :len(p.x)] = p.x
        c[j, :, :p.c.shape[1]] = p.c
    e, h = _table()
    return dict(g2_table=np.r_[x.ravel(), c.ravel(), [density(n) for n in names],
                              np.log(e), np.log(h)],
                g2_sizes=np.array([stride, len(e), *[len(p.x) for p in curves]], dtype=np.int64))


def config(arrays, grid=None, boxes=(), water=-1):
    sizes = arrays['g2_sizes']
    head = [sizes[0], sizes[1], *sizes[2:], int(grid is not None)]
    head += [*(grid.origin_cm if grid else [0]*3), *(grid.shape if grid else [0]*3),
             grid.voxel_size_cm if grid else 1, len(boxes), water]
    return np.array(head + [v for _, lo, hi in boxes for v in [*lo, *hi]], float)
