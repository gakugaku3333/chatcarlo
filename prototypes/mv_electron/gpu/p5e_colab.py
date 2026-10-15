"""P5e exploration and adopted C0–C4; CuPy is loaded at session start only."""
import json
import time
import uuid
from pathlib import Path
import numpy as np
from . import p5b_core as core, p5d_core as d, p5e_core as e
from .p5b_colab import GPU, environment
from .p5d_colab import telemetry


class NumericalFailure(core.Stop):
    def __init__(self, out, times):
        super().__init__('GPU health failed')
        self.out, self.times = out, times


class VariantGPU(GPU):
    def __init__(self, cp, data, v, ordered=False):
        self.v, self.ordered = v, ordered
        text = e.source(v) if not ordered else core.source()+'\n'+(core.HERE/'p5e_gpu.cu').read_text()
        names = e.names(v) if not ordered else ('p5e_run_kernel', 'p5e_per_history_kernel')
        self.entries = names
        kernels, self.launches = {}, {}
        sm = cp.cuda.runtime.getDeviceProperties(0)['multiProcessorCount']
        for name in (*names, 'p5e_visits_kernel'):
            if name == 'p5e_visits_kernel' and v == 'V0' and not ordered:
                continue
            kernel = cp.RawKernel(text, name, options=e.options(v), backend='nvrtc')
            kernel.compile()
            try:
                blocks = cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(kernel.kernel.ptr, 128, 0)
            except Exception as exc:
                raise core.Stop('occupancy API unavailable; stop') from exc
            if blocks <= 0:
                raise core.Stop('occupancy API returned zero')
            self.launches[name] = dict(method='CuPy occupancy API', blocks_per_sm=int(blocks),
                T=int(sm*blocks*128), attributes={k: int(kernel.attributes[k]) for k in
                    ('num_regs', 'local_size_bytes', 'max_threads_per_block', 'shared_size_bytes')})
            kernels[name] = kernel
        fault = cp.RawKernel(text, names[0], options=e.options(v, True), backend='nvrtc')
        fault.compile()
        kernels[names[0]+'_fault'] = fault
        # Fault occupancy is independently queried too.
        blocks = cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(fault.kernel.ptr, 128, 0)
        if blocks <= 0:
            raise core.Stop('fault occupancy is zero')
        self.launches[names[0]+'_fault'] = dict(T=int(sm*blocks*128), blocks_per_sm=int(blocks),
            method='CuPy occupancy API', attributes={k: int(fault.attributes[k]) for k in ('num_regs','local_size_bytes')})
        super().__init__(cp, data, kernels)
        self.counter = cp.zeros(1, dtype=np.uint64)

    def measure(self, n, start=0, history=False, fault=False, order=None):
        core.check_range(n, start)
        cp = self.cp
        cp.cuda.Device().synchronize()
        begin = time.perf_counter()
        values = self.output(core.history_shapes(n) if history else core.run_shapes(len(self.data['cum'])))
        name = self.entries[int(history)]+('_fault' if fault else '')
        args = self.args+(np.uint64(core.SEED), np.uint64(start), np.int32(n), np.int32(core.MAXSTEPS))
        mode = 2 if order is not None else int(e.VARIANTS[self.v]['allocation'] == 'D')
        if name.startswith('p5e'):
            self.counter.fill(0)
            args += (self.counter, self.cp.zeros(1, np.int32) if order is None else order, np.int32(mode))
        T = self.launches[name]['T'] if mode == 1 else ((n+127)//128)*128
        first, last = cp.cuda.Event(), cp.cuda.Event()
        first.record()
        self.kernels[name]((T//128,), (128,), args+tuple(values[k] for k in
            (core.HISTORY_ORDER if history else core.RUN_ORDER)))
        last.record()
        out = self.download(values)
        last.synchronize()
        times = dict(kernel_ms=float(cp.cuda.get_elapsed_time(first,last)), call_s=time.perf_counter()-begin,
                     output_bytes=sum(v.nbytes for v in values.values()))
        core.validate_output(out, core.history_shapes(n) if history else core.run_shapes(len(self.data['cum'])))
        if core.health(out)['status'] != 'pass':
            raise NumericalFailure(out, times)
        return out, times

    def per_history(self, n, start=0):
        return self.measure(n,start,history=True)[0]

    def production(self, n, start=0, fault=False):
        return self.measure(n,start,fault=fault)[0]

    def visits(self, n, start):
        cp = self.cp
        values = (cp.zeros(n,np.uint64), cp.zeros(n,np.uint64), cp.zeros(n,np.int32))
        self.counter.fill(0)
        name = 'p5e_visits_kernel'
        self.kernels[name]((self.launches[name]['T']//128,), (128,),
            (np.uint64(start),np.int32(n),self.counter,cp.zeros(1,np.int32),np.int32(1),*values))
        return dict(zip(('visits','global_h','rows'), (cp.asnumpy(v) for v in values)))


def cuda_error(cp, exc):
    return isinstance(exc, (cp.cuda.runtime.CUDARuntimeError, cp.cuda.driver.CUDADriverError))


def explore(bundle, results):
    import cupy as cp
    data, ref, manifest = d.load_bundle(bundle)
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    session = uuid.uuid4().hex
    root = results/session
    root.mkdir()
    report = dict(session=session, complete=False, environment=environment(cp), variants={}, measurements=[])
    report['t4'] = 'T4' in report['environment']['gpu']
    accepted = []
    def save(v, label, out):
        core.save_npz(root/f'{v}_{label}.npz',out)
    try:
        for v in e.VARIANTS:
            record = report['variants'][v] = dict(status='incomplete')
            try:
                gpu = VariantGPU(cp,data,v)
            except Exception as exc:
                if cuda_error(cp,exc):
                    raise core.Stop('CUDA error: restart runtime and rerun from first cell') from exc
                if isinstance(exc, core.Stop):
                    raise
                # Only compiler and allocation failures are variant exclusions.
                if not isinstance(exc, (cp.cuda.compiler.CompileException, cp.cuda.memory.OutOfMemoryError, MemoryError)):
                    raise
                record.update(status='excluded', kind='compile', error=str(exc))
                if v == 'V0': raise core.Stop('V0 compilation failed') from exc
                continue
            record['identity'] = e.identity(v,manifest,report['environment'],gpu.launches)
            checks = record['checks'] = {}
            try:
                out = gpu.per_history(20000)
                save(v,'e1',out)
                baseline = core.read_npz(root/'V0_e1.npz')
                checks['E1'] = e.e1(out,baseline)
                del out, baseline
                out = gpu.production(100000)
                save(v,'e2',out)
                checks['E2'] = e.e2(out,ref)
                del out
                if e.VARIANTS[v]['allocation'] == 'D':
                    T = gpu.launches['p5e_visits_kernel']['T']
                    ok = True
                    for i,(n,start) in enumerate(((2,0),(T-1,0),(T,0),(T+1,0),(3*T+17,0),(T+1,7000),(T+1,7000))):
                        out = gpu.visits(n,start)
                        save(v,f'e3_{i}',out)
                        ok &= e.visits_check(**out,n=n,start=start)['status']=='pass'
                    out = gpu.per_history(2000,7000)
                    save(v,'e3_transport',out)
                    static = VariantGPU(cp,data,'V0').per_history(2000,7000)
                    save(v,'e3_static',static)
                    checks['E3'] = core.verdict(ok and e.e1(out,static)['status']=='pass')
                    del out,static
                passed = all(x['status']=='pass' for x in checks.values())
            except (cp.cuda.memory.OutOfMemoryError, MemoryError) as exc:
                passed=False
                record.update(kind='resource', error=str(exc))
            except NumericalFailure as exc:
                passed=False
                save(v,'health_failure',exc.out)
                record['health_failure']=str(exc)
            record['status']='pass' if passed else 'excluded'
            if not passed and v=='V0': raise core.Stop('V0 E2 failed')
            if passed: accepted.append(v)
            del gpu
            core.write_json(root/'explore.json',report)
        def measure(v,n,label,repeat):
            if v not in accepted:
                return
            try:
                gpu = VariantGPU(cp,data,v)
                out,times = gpu.measure(n)
            except (NumericalFailure, cp.cuda.compiler.CompileException,
                    cp.cuda.memory.OutOfMemoryError, MemoryError) as exc:
                if isinstance(exc,NumericalFailure):
                    save(v,f'{label}_{repeat}_unhealthy',exc.out)
                report['variants'][v].update(status='excluded',error=str(exc))
                accepted.remove(v)
                if v=='V0': raise core.Stop('V0 measurement failed') from exc
                core.write_json(root/'explore.json',report)
                return
            save(v,f'{label}_{repeat}',out)
            report['measurements'].append(dict(**d.metadata(label,n,block=128),variant=v,repeat=repeat,
                session=session,telemetry=telemetry(),**times))
            core.write_json(root/'explore.json',report)
        for v in list(accepted): measure(v,1000,'warmup',0)
        for r in range(3):
            for v in accepted[::(-1 if r%2 else 1)]: measure(v,100000,'round',r)
        times={v:[x['kernel_ms'] for x in report['measurements'] if x['label']=='round' and x['variant']==v] for v in accepted}
        report['selection_input']=times
        report['selected']=e.select(times)
        if report['selected']!='V0':
            for v in ('V0',report['selected']): measure(v,1000000,'million',0)
            if report['selected'] not in accepted:
                report['selection_input']={v:t for v,t in times.items() if v in accepted}
                report['selected']=e.select(report['selection_input'])
        gpu=VariantGPU(cp,data,'V0',ordered=True)
        begin=time.perf_counter()
        order=e.ideal_order(ref['history_steps'])
        device_order=cp.asarray(order)
        cp.cuda.Device().synchronize()
        report['ideal_order_setup_s']=time.perf_counter()-begin
        save('ideal','order',dict(order=order))
        out,_=gpu.measure(20000,history=True,order=device_order)
        save('ideal','history',out)
        report['ideal_check']=e.e1(out,core.read_npz(root/'V0_e1.npz'))
        del out
        report['ideal_times']=[]
        if report['ideal_check']['status']=='pass':
            for _ in range(3):
                _,t=gpu.measure(20000,order=device_order)
                report['ideal_times'].append(t)
        report['complete']=True
    except Exception as exc:
        report['stop']=str(exc)
        if cuda_error(cp,exc):
            raise core.Stop('CUDA error: restart runtime and rerun from first cell') from exc
        raise
    finally:
        core.write_json(root/'explore.json',report)
    return report


def adopt(bundle, results, variant):
    import cupy as cp
    data,ref,manifest=d.load_bundle(bundle)
    results=Path(results); results.mkdir(parents=True,exist_ok=True)
    gpu=VariantGPU(cp,data,variant)
    env=environment(cp)
    if 'T4' not in env['gpu']: raise core.Stop('adoption requires T4')
    identity=e.identity(variant,manifest,env,gpu.launches)
    e.match_identity(results/'identity.json',identity)
    report=dict(checks={},overall='incomplete',identity=identity)
    def record(k,v): core.record(report,k,v)
    def save(name,out): core.save_npz(results/(name+'.npz'),out,dict(identity=identity,outputs=core.describe(out)))
    try:
        record('C0',core.verdict(True))
        # C1 diagnostic functions compiled with the adopted options and source.
        for name in core.KERNELS[2:]:
            kernel=cp.RawKernel(e.source(variant),name,options=e.options(variant),backend='nvrtc')
            kernel.compile(); gpu.kernels[name]=kernel
        values=gpu.checks(ref); save('c1',values); record('C1',core.c1(ref,values))
        histories=gpu.per_history(20000); save('c2',histories)
        record('C2',core.c2({k:ref['history_'+k] for k in core.HISTORY_ORDER},histories,float(ref['Dref']),float(ref['conversion'])))
        prod=gpu.production(20000); save('c3',prod)
        normal=core.c3(histories,prod,20)
        faulty=gpu.production(20000,fault=True); save('c3_fault',faulty)
        fault=core.c3(histories,faulty,20)
        pieces=[gpu.per_history(5000,start) for start in range(0,20000,5000)]
        split={k:np.concatenate([r[k] for r in pieces]) for k in core.HISTORY_ORDER}
        save('c3_split',split)
        record('C3',core.verdict(normal['status']=='pass' and fault['status']=='fail' and core.split_check(histories,split)['status']=='pass',ordinary=normal,merge_fault=fault))
        found=core.resume(results/'batches',identity)
        for b in range(100):
            if b not in found:
                out=gpu.production(100000,100000*b)
                core.require(core.health(out),'C4 health')
                if not core.bits(out['nspec'],ref['c4_nspec'][b]): raise core.Stop('C4 spectrum')
                path=results/'batches'/f'batch_{b:03}.npz'
                core.save_npz(path,out,core.checkpoint_meta(identity,b,out)); found[b]=path
        record('C4',core.c4(ref,[core.read_checkpoint(found[b],identity,b) for b in range(100)],20))
        report['overall']='pass'
    except Exception as exc:
        report.update(overall='not adopted',stop=str(exc))
        if cuda_error(cp,exc): raise core.Stop('CUDA error: restart runtime from first cell') from exc
        raise
    finally: core.write_json(results/'adopt.json',report)
    return report
