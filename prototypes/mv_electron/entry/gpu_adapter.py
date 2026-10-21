"""P7 adapters with p6 geometry and observation at the actual native call boundary."""
import ctypes as ct
import os
import numpy as np
from ..gpu import p7, p7_core as h, p5b_core as core
from . import condition as identity


class NativeCall:
    def __init__(self, call, owner, history):
        self.call, self.owner, self.history = call, owner, history

    def __call__(self, *args):
        args = list(args)
        if os.environ.get('MV_ENTRY_FAULT') == 'field_half':
            args[6] = 5.5
        owner = self.owner
        owner.last_observation = identity.observe_cpu_run(tuple(args), owner.config_observation,
                                                         owner.condition, owner.materials, self.history)
        return self.call(*args)


class CPU(p7.CPU):
    def __init__(self, work, data, condition, materials):
        self.condition, self.materials = condition, materials
        super().__init__(work, data, 'philox', layout='p6')
        self.lib.p7_run = NativeCall(self.lib.p7_run, self, False)
        self.lib.p7_histories = NativeCall(self.lib.p7_histories, self, True)

    def configure(self, initialize=False):
        args = (ct.byref(self.tables), self.logs, self.log_counts.ctypes.data_as(p7.p5j.UP), int(initialize))
        self.config_observation = identity.observe_cpu_configure(args, self.data)
        self.lib.p5j_configure(*args)

    def measure(self, n, seed, start=0, history=False, **kwargs):
        method = self.histories if history else self.run
        return method(n, seed, float(self.condition['field_half_cm']), float(self.condition['ssd_cm']),
                      start=start, maxsteps=self.condition['maxsteps'], **kwargs)


class GPU:
    """Wrap P7's launch construction; replace only geometry and condition, observe RawKernel."""
    def __init__(self, cp, data, condition, materials):
        from ..gpu.p7_colab import GPU as P7GPU
        from ..gpu.p5b_colab import GPU as BaseGPU
        from ..gpu import p5j as maps
        self.cp, self.data = cp, data
        self.condition, self.materials = condition, materials
        # Same initialization as P7, without its A/B/C-only condition/geometry.
        gpu = P7GPU.__new__(P7GPU)
        gpu.options = identity.OPTIONS
        gpu.condition = dict(field_half=condition['field_half_cm'], ssd=condition['ssd_cm'])
        gpu.blocks, gpu.fault, gpu.launches = 40, 0, {}
        self.source = h.source()
        kernels = {}
        for name in ('p7_kernel', 'p7_score_kernel', 'p5j_initialize_kernel'):
            kernel = cp.RawKernel(self.source, name, options=gpu.options, backend='nvrtc')
            kernel.compile()
            kernels[name] = kernel
        BaseGPU.__init__(gpu, cp, data, kernels)
        layout = maps.layout(data)
        gpu.logs = cp.empty(int(layout['lengths'].sum())+3, np.float64)
        gpu.counts = cp.zeros((20, 2), np.uint64)
        gpu.args += (gpu.logs, gpu.counts)
        gpu.kernels['p5j_initialize_kernel']((1,), (1,), gpu.args)
        cp.cuda.Device().synchronize()
        gpu.generated = core.describe(dict(logs=cp.asnumpy(gpu.logs), **layout))
        gpu.geometry = cp.asarray(np.frombuffer(bytes(h.geometry('p6')[0]), np.uint8))
        gpu.next, gpu.nspec = cp.zeros(1, np.int32), len(data['cum'])
        launch = gpu.kernels['p7_kernel']

        def observed_launch(grid, block, args):
            if grid != (40,) or block != (256,):
                raise core.Stop('GPU launch dimensions changed')
            args = list(args)
            if os.environ.get('MV_ENTRY_FAULT') == 'field_half':
                args[41] = np.float64(5.5)
            args = tuple(args)
            self.last_observation = identity.observe_gpu(args, condition, materials, cp.asnumpy, self.source, gpu.options)
            launch(grid, block, args)
        gpu.kernels['p7_kernel'] = observed_launch
        self.gpu = gpu

    def measure(self, n, seed, start=0, history=False, **kwargs):
        result, self.timing = self.gpu.measure(n, seed, start=start, history=history,
                                               maxsteps=self.condition['maxsteps'], **kwargs)
        self.region_additions = self.gpu.region_additions
        return result

    def scorer(self, records):
        return self.gpu.scorer(records)
