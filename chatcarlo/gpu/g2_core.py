"""G2 entry points; G1 observation positions and native ABI remain unchanged."""
import ctypes
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
            self.native = self.cp.RawKernel(core.cuda_source(), 'g2_histories_kernel',
                                           backend='nvrtc', options=core.CUDA_OPTIONS)
            self.native.compile()
        def invoke(*args):
            extra = [self.a['g2_table'], self.cfg, *self.outputs]
            if self.device == 'cpu':
                self.native(*args, *[v.ctypes.data if v is not None else None for v in extra])
            else:
                gpu = [self.cp.asarray(v) if v is not None else np.uint64(0) for v in extra]
                self.native(args[0], args[1], (*args[2], *gpu))
                self.cp.cuda.Stream.null.synchronize()
                for target, value in zip(self.outputs, gpu[2:]):
                    if target is not None:
                        target[:] = self.cp.asnumpy(value)
        invoke.argtypes = self.base_types
        self.fn = invoke

    def run(self, *args, **kwargs):
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
