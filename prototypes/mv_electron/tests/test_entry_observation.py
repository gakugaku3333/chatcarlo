import copy
import numpy as np
import pytest
from prototypes.mv_electron.entry import condition as c
from prototypes.mv_electron.gpu import p5b_core as core, p7_core as h


def gpu_tuple(data):
    dims = dict(zip(c.DIM_NAMES, core.dimensions(data)))
    base = []
    for name in c.GPU_BASE:
        if name in core.POINTERS:
            base.append(data[name])
        elif name in core.SCALARS:
            base.append(np.float64(data[name].item()))
        else:
            base.append(np.int32(dims[name]))
    base += [np.zeros(3), np.zeros((20, 2), np.uint64), np.frombuffer(bytes(h.geometry('p6')[0]), np.uint8),
             np.uint64(1), np.uint64(0), np.int32(20), np.int32(h.MAXSTEPS), np.int32(1),
             np.float64(5), np.float64(100), np.int32(1), np.zeros(1, np.int32), np.int32(0)]
    hist = h.empty(h.history_shapes(1)); run = h.empty(h.run_shapes(len(data['cum'])))
    base += [hist[k] for k in core.HISTORY_ORDER]+[run[k] for k in core.RUN_ORDER]
    base += [hist[k] for k in ('regions', 'region_ledger', 'sources')]+[run[k] for k in ('regions', 'region_ledger')]
    base += [np.uint64(0), np.int32(0), np.uint64(0), np.uint64(0)]
    return base


@pytest.fixture(scope='module')
def context():
    from prototypes.mv_electron.entry import scene
    from pathlib import Path
    raw = scene.load(Path(__file__).resolve().parents[3]/'examples/therapy/water_10mv_10x10.yaml')
    raw['run'].update(histories=2000, seed=1, engine='cpu_local')
    cond, mat, data = c.identities(raw)
    return cond, mat, data


@pytest.mark.parametrize('slot', [25, 26, 35, 11, 27, 41, 37])
def test_entry_gpu_position_map_detects_swap(context, slot):
    cond, mat, data = context
    args = gpu_tuple(data)
    normal = c.observe_gpu(tuple(args), cond, mat, np.asarray, h.source(), c.OPTIONS)
    assert len(normal['arguments']) == 80
    c.verify_observation(normal, cond, mat, n=20, start=0)
    if slot in (25, 26):
        args[slot] = data['brems_e']
    elif slot == 35:
        args[slot] = args[slot].copy(); args[slot][0] ^= 1
    elif slot in (11, 37):
        args[slot] += type(args[slot])(1)
    else:
        args[slot] += type(args[slot])(.5)
    with pytest.raises(core.Stop):
        changed = c.observe_gpu(tuple(args), cond, mat, np.asarray, h.source(), c.OPTIONS)
        c.verify_observation(changed, cond, mat, n=20, start=0)


def test_entry_gpu_abi_type(context):
    cond, mat, data = context
    args = gpu_tuple(data); args[41] = np.int32(5)
    with pytest.raises(core.Stop, match='type mismatch'):
        c.observe_gpu(tuple(args), cond, mat, np.asarray, h.source(), c.OPTIONS)
