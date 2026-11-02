import numpy as np

def c2_calls(a, b):
    na, nb = np.minimum(a['diag_used'], 2), np.minimum(b['diag_used'], 2)
    same = na == nb
    for j in range(2):
        same &= (na <= j) | (a['diagnostic'][:, j, 5] == b['diagnostic'][:, j, 5])
    fraction = float(np.mean(same))
    return dict(status='pass' if fraction>=.99 else 'fail',fraction=fraction,differing=int(np.count_nonzero(~same)))

