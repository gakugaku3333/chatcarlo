"""Portable P9a input schema constants; NumPy-only Colab dependency."""
GPU_SEED = 920101
PILOT_SEED = 920102
CALL_N = 200000


def shapes_from_sizes(sizes, mode='gpu'):
    nl, nb, nc, nf, ns = map(int, sizes)
    if not (1 <= nl <= 16 and 0 < nb <= 220 and 0 < nc <= 108 and 0 < nf <= 430 and 0 <= ns <= 64):
        raise ValueError('input capacity')
    shapes = dict(dose=[nb], p8_batches=[100,nb], coarse=[100,nc],
                  fine=[100,nf], additions=[100,nb])
    if mode == 'gpu':
        shapes.update(_groups_d=[100,nb+nc+nf+37], _groups_i=[100,79+nb],
                      _absolute=[100,nb+nc+nf+37])
    return shapes
