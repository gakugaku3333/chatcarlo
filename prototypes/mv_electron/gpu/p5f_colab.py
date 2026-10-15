"""P5f-0 M0–M2; imports no CUDA or physics packages until execution."""
import argparse
import glob
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
import numpy as np
from . import p5b_core as core, p5d_core as d, p5f_derive as f
from .p5b_colab import GPU, environment


class DiagnosticGPU(GPU):
    def __init__(self, cp, data, condition):
        self.condition = condition
        text, self.identity = f.derive(condition)
        kernels, self.launches = {}, {}
        for name in ('p5f_run_kernel', 'p5f_per_history_kernel'):
            k = cp.RawKernel(text, name, options=f.OPTIONS, backend='nvrtc')
            k.compile()
            blocks = cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(k.kernel.ptr,128,0)
            if blocks <= 0:
                raise core.Stop('occupancy API returned zero')
            self.launches[name] = dict(blocks_per_sm=int(blocks), attributes={
                key:int(k.attributes[key]) for key in ('num_regs','local_size_bytes')})
            kernels[name] = k
        super().__init__(cp,data,kernels)

    def measure(self, n=100000, history=False):
        cp = self.cp
        cp.cuda.Device().synchronize()
        begin = time.perf_counter()
        shapes = core.history_shapes(n) if history else core.run_shapes(len(self.data['cum']))
        shapes = dict(shapes)
        if 'W' in self.condition:
            shapes['batches'] = (n if history else 100,2)
            shapes['fine_batches'] = (0,)
        shapes['reads'] = (n,) if history else (1,)
        if not history: shapes['calls'] = (1,)
        values = self.output({k:v for k,v in shapes.items() if k != 'reads'})
        values['reads'] = cp.zeros(shapes['reads'],np.uint64)
        order = (*core.HISTORY_ORDER,'reads') if history else (*core.RUN_ORDER,'calls','reads')
        args = self.args+(np.uint64(core.SEED),np.uint64(0),np.int32(n),
                         np.int32(core.MAXSTEPS),np.int32((24*len(self.data['ks'])+4)*len(self.data['us'])))
        first,last = cp.cuda.Event(),cp.cuda.Event()
        first.record()
        name = 'p5f_per_history_kernel' if history else 'p5f_run_kernel'
        self.launch(name,n,args+tuple(values[k] for k in order))
        last.record()
        out = self.download(values)
        last.synchronize()
        times = dict(kernel_ms=float(cp.cuda.get_elapsed_time(first,last)),call_s=time.perf_counter()-begin)
        return out,times


def m0(root):
    """Probe a tiny kernel in a separate process; retain full profiler diagnostics."""
    root = Path(root)
    candidates = [shutil.which('ncu'), *glob.glob('/usr/local/cuda*/bin/ncu'),
                  *glob.glob('/opt/nvidia/nsight-compute/*/ncu')]
    ncu = next((x for x in candidates if x and Path(x).is_file()),None)
    lookup = subprocess.run(['sh','-c','command -v ncu'],capture_output=True,text=True)
    report = dict(candidates=candidates,available=False,command_v=dict(
        returncode=lookup.returncode,stdout=lookup.stdout,stderr=lookup.stderr))
    if ncu:
        version = subprocess.run([ncu,'--version'],capture_output=True,text=True)
        report['version'] = version.stdout+version.stderr
        cmd = [ncu,'--set','basic',sys.executable,'-m',
               'prototypes.mv_electron.gpu.p5f_ncu_target','--probe']
        r = subprocess.run(cmd,capture_output=True,text=True)
        report.update(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,
                      available=r.returncode==0 and 'ERR_' not in r.stdout+r.stderr,ncu=ncu)
    core.write_json(root/'m0.json',report)
    return report


def m1(probe,bundle,root):
    if not probe['available']:
        return dict(status='unavailable',reason='M0 performance counters unavailable')
    command = [probe['ncu'],'--set','full','--replay-mode','kernel','--cache-control','all',
               '--kernel-name','p5b_run_kernel','--launch-skip','1','--launch-count','1',
               '--export',str(Path(root)/'v0_profile'),
               sys.executable,'-m','prototypes.mv_electron.gpu.p5f_ncu_target','--bundle',str(bundle)]
    r = subprocess.run(command,capture_output=True,text=True)
    return dict(status='pass' if r.returncode==0 else 'unavailable',command=command,
                returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,
                sections='full',replay='kernel',cache_control='all',passes='see full ncu stdout/report',
                timing_policy='profiler times excluded from M2')


def run(bundle, results, probe=None):
    import cupy as cp
    data,ref,manifest = d.load_bundle(bundle)
    root = Path(results)/uuid.uuid4().hex
    root.mkdir(parents=True)
    report = dict(complete=False,environment=environment(cp),manifest=manifest,
                  conditions={},measurements=[],interpretation={
                      'T':'表の読み出しの局所性を大きく良くしたときの感度。DRAMの費用そのものではない。',
                      'W':'線量の集計の経路全体を縮めた対照。',
                      'selection':'結果から最初の候補1つと根拠・未分離の原因・次の比較を別途記述する。'})
    try:
        report['M0'] = m0(root) if probe is None else probe
        report['M1'] = m1(report['M0'],bundle,root)
        if 'T4' not in report['environment']['gpu']:
            raise core.Stop('P5f-0 requires T4')
        v0 = GPU(cp,data,{name:cp.RawKernel(core.source(),name,options=core.OPTIONS,backend='nvrtc')
                          for name in core.KERNELS[:2]})
        baseline = v0.per_history(20000)
        core.save_npz(root/'v0_history.npz',baseline)
        histories = {}
        for c in f.CONDITIONS:
            gpu = DiagnosticGPU(cp,data,c)
            report['conditions'][c] = dict(identity=gpu.identity,launches=gpu.launches)
            h,_ = gpu.measure(20000,history=True)
            core.save_npz(root/(c+'_history.npz'),h)
            histories[c] = {k:h[k] for k in ('calls','steps','counts','nspec_index','maxdepth')}
            if c=='B' and not all(core.bits(h[k],baseline[k]) for k in core.HISTORY_ORDER):
                raise core.Stop('P1 B per-history mismatch')
            p,_ = gpu.measure()
            core.save_npz(root/(c+'_preflight.npz'),p)
            if c=='B' and (not f.health(p,100000) or core.health(p)['status']!='pass' or d.d_gate(p,ref)['status']!='pass'):
                raise core.Stop('P1 B production mismatch')
            report['conditions'][c]['preflight_health'] = f.health(p,100000)
            if c in ('W','TW'):
                other = histories['T' if c=='TW' else 'B']
                if not all(core.bits(histories[c][k],other[k]) for k in other):
                    raise core.Stop('P2 transport mismatch: '+c)
            del gpu,h,p
        # P1 and P2 have passed before timed rounds begin.
        report['P1_P2'] = 'pass'
        accepted = {c for c in f.CONDITIONS if report['conditions'][c]['preflight_health']}
        for c in f.CONDITIONS:
            gpu=DiagnosticGPU(cp,data,c)
            p,_=gpu.measure(1000)
            core.save_npz(root/(c+'_warmup.npz'),p)
            if not f.health(p,1000): accepted.discard(c)
        for r in range(3):
            for c in f.CONDITIONS[::(-1 if r%2 else 1)]:
                gpu=DiagnosticGPU(cp,data,c)
                p,times=gpu.measure()
                core.save_npz(root/f'{c}_round{r}.npz',p)
                healthy=f.health(p,100000)
                if not healthy: accepted.discard(c)
                report['measurements'].append(dict(condition=c,round=r,n=100000,start=0,
                    health=healthy,**times,steps=int(p['step_batches'].sum()),
                    gs_calls=int(p['counts'][1]),eight_point_reads=int(p['reads'].sum()),
                    rng_calls=int(p['calls'].sum())))
                core.write_json(root/'report.json',report)
        report['comparison']={c:dict(status='eligible' if c in accepted else 'その診断の条件は成立しなかった',
            times=[x for x in report['measurements'] if x['condition']==c] if c in accepted else []) for c in f.CONDITIONS}
        report['complete']=True
    except Exception as exc:
        report['stop']=str(exc)
        raise
    finally:
        core.write_json(root/'report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',required=True)
    parser.add_argument('--results',required=True)
    a=parser.parse_args()
    run(a.bundle,a.results)
