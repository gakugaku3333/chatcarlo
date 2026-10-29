"""G2 entry points; G1 observation positions and native ABI remain unchanged."""
import ctypes
import os
from types import SimpleNamespace
import numpy as np
from . import core
from .g2_tables import config


class Core(core.Core):
    def __init__(self, arrays, device='cpu', *, grid=None, boxes=(), water=-1,
                 count_additions=False, **kwargs):
        if len(boxes) > 64:
            raise ValueError('G2 supports at most 64 validation boxes')
        super().__init__(arrays, device, **kwargs)
        self.grid = grid
        self.boxes = boxes
        self.count_additions = count_additions
        self.cfg = config(arrays, grid, boxes, water)
        self.base_types = self.fn.argtypes if device == 'cpu' else None
        if device == 'cpu':
            self.native = self.lib.g2_histories
            self.native.restype = None
            self.native.argtypes = self.base_types + [ctypes.c_void_p]*8
        else:
            self.native = self.module.get_function('g2_histories_kernel')
            self.native.compile()
        def invoke(*args):
            extra = [self.a['g2_table'], self.cfg, *self.outputs]
            fault = os.environ.get('CHATCARLO_G4_FAULT')
            if fault == 'g2_cfg':
                extra[1] = self.cfg.copy()
                extra[1][self.nm+9] *= 2
            elif fault == 'g2_cfg_inplace':
                self.cfg[self.nm+8] -= 1
            elif fault == 'g2_table':
                extra[0] = extra[0].copy()
                extra[0][0] = np.nextafter(extra[0][0], np.inf)
            if self.device == 'cpu':
                converted = [v.ctypes.data if v is not None else None for v in extra]
                final = final_arguments(args, converted, fault)
                check_outputs(final, self.outputs, 'cpu')
                if self._observed is not None:
                    observed = core.Core._observe(self, final, self._observation_arrays)
                    observed.update({f'{19+i}:{name}': core.array_sha(next(v for v in extra[:2] if v.ctypes.data == final[19+i]))
                                     for i, name in enumerate(('g2_table', 'cfg'))})
                    self._observed.update(observed)
                self.native(*final)
            else:
                gpu = [self.cp.asarray(v) if v is not None else np.uint64(0) for v in extra]
                final = final_arguments(args[2], gpu, fault)
                check_outputs(final, gpu[2:], 'cuda')
                if self._observed is not None:
                    observed = core.Core._observe(self, final, self._observation_arrays)
                    observed.update({f'{25+i}:{name}': core.array_sha(self.cp.asnumpy(final[25+i]))
                                     for i, name in enumerate(('g2_table', 'cfg'))})
                    self._observed.update(observed)
                self.native(args[0], args[1], final)
                self.cp.cuda.Stream.null.synchronize()
                for i, target in enumerate(self.outputs):
                    if target is not None:
                        # Recover from the very device buffer checked at its ABI position.
                        target[:] = self.cp.asnumpy(final[27+i])
        invoke.argtypes = self.base_types
        self.fn = invoke

    def _observe(self, args, arrays):
        self._observation_arrays = arrays
        self._observed = {}
        return self._observed

    def run(self, *args, **kwargs):
        self._observed = None
        nv = int(np.prod(self.grid.shape)) if self.grid else 0
        nb = len(self.boxes)
        self.outputs = [np.zeros(nv), np.zeros(nv),
                        np.zeros(nv) if self.count_additions else None,
                        np.zeros(nb), np.zeros(nb),
                        np.zeros(nb) if self.count_additions else None]
        result = super().run(*args, **kwargs)
        r = result[0] if isinstance(result, tuple) else result
        for name, value in zip(('kerma', 'h10', 'grid_additions', 'box_sum',
                                'box_sum2', 'box_additions'), self.outputs):
            if value is not None:
                if not np.isfinite(value).all() or np.any(value < 0):
                    raise ValueError('invalid G2 output: '+name)
                r[name] = value.copy()
        return result


OUTPUT_NAMES = ('kerma', 'h10', 'grid_additions', 'box_sum', 'box_sum2', 'box_additions')


def final_arguments(base, extra, fault=None):
    """Pure ABI assembly, also testable for CUDA without CuPy."""
    final = list(base) + list(extra)
    if fault == 'g2_swap':
        offset = len(base) + 2
        final[offset], final[offset+1] = final[offset+1], final[offset]
    return tuple(final)


def check_outputs(args, outputs, device):
    offset = 21 if device == 'cpu' else 27
    def pointer(value):
        if value is None: return 0
        # Absent CUDA outputs are passed as np.uint64(0); NumPy scalars also have .data (a memoryview).
        if isinstance(value, (int, np.integer)): return int(value)
        if device == 'cpu':
            return value.ctypes.data if isinstance(value, np.ndarray) else int(value)
        return int(value.data.ptr) if hasattr(value, 'data') else int(value)
    for i, name in enumerate(OUTPUT_NAMES):
        if pointer(args[offset+i]) != pointer(outputs[i]):
            raise ValueError('output mapping mismatch: '+name)


def expected_observation(a, cfg, *args):
    observed = core.expected_observation(a, *args)
    offset = 19 if args[-1] == 'cpu' else 25
    observed.update({f'{offset}:g2_table': core.array_sha(a['g2_table']),
                     f'{offset+1}:cfg': core.array_sha(cfg)})
    return observed


def grid_from(spec):
    if (set(spec) != {'origin_cm', 'shape', 'voxel_size_cm', 'layout_version'}
            or spec['layout_version'] != 'g2-grid-v1'
            or len(spec['origin_cm']) != 3 or len(spec['shape']) != 3
            or not all(type(v) is int and v > 0 for v in spec['shape'])
            or not np.isfinite(spec['origin_cm']).all()
            or not np.isfinite(spec['voxel_size_cm']) or spec['voxel_size_cm'] <= 0):
        raise ValueError('invalid job.grid')
    return SimpleNamespace(origin_cm=np.array(spec['origin_cm'], float),
                           shape=tuple(spec['shape']), voxel_size_cm=float(spec['voxel_size_cm']))


def health(result, grid):
    for name in ('kerma', 'h10'):
        value = result.get(name)
        if (not isinstance(value, np.ndarray) or value.dtype != np.dtype('float64')
                or value.shape != (int(np.prod(grid.shape)),)
                or not np.isfinite(value).all() or np.any(value < 0)):
            raise ValueError('invalid G2 output: '+name)
