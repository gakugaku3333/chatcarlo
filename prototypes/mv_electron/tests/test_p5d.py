"""P5d gates, independent ABI, timing order and isolation."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
import pytest
from prototypes.mv_electron.gpu import p5d_core as d, p5b_core as core


@pytest.fixture(scope='module')
def cpus():
    from prototypes.mv_electron.gpu import p5a
    data = p5a.load_inputs(full=True)
    work = Path(tempfile.mkdtemp(prefix='p5d_test_'))
    return d.CPU(work, data), p5a.CPU(work, data, rng='philox')


@pytest.mark.parametrize('start,seed,maxsteps', [(0,core.SEED,core.MAXSTEPS),
    (7000,core.SEED,core.MAXSTEPS),(0,20291961,core.MAXSTEPS),(0,core.SEED,2)])
def test_abi(cpus,start,seed,maxsteps):
    from prototypes.mv_electron.gpu import p5a
    cpu, original = cpus
    r = cpu.run(2000,seed,start=start,maxsteps=maxsteps)
    assert p5a.compare_outputs(original.run(2000,seed,start=start,maxsteps=maxsteps),r)['status']=='pass'
    assert (d.cpu_health(r,2000)['status']=='pass') == (maxsteps!=2)


def test_abi_with_compat(cpus, tmp_path):
    from prototypes.mv_electron.gpu import p5a
    cpu, original = cpus
    compat = d.CPU(tmp_path, cpu.data, compiler='clang++', compat=True)
    header = (d.HERE/'p5d_gxx_compat.h').resolve()
    assert compat.flags == d.FLAGS + ['-include', str(header)]
    identity = dict(compiler=compat.compiler, flags=compat.flags,
                    sources={**core.FROZEN, header.name: core.sha(header.read_bytes())})
    assert compat.identity == core.sha(json.dumps(identity, sort_keys=True).encode())
    assert compat.identity != cpu.identity
    assert p5a.compare_outputs(original.run(2000, core.SEED),
                               compat.run(2000, core.SEED))['status'] == 'pass'


def valid():
    r = core.empty_output(core.run_shapes(20))
    r.pop('nan_count')
    r['nspec'][0,0]=200
    r['batches'][:]=1.
    return r


@pytest.mark.parametrize('bad',['seconds','missing','extra','shape','dtype','nonfinite','step','stack','count','violations'])
def test_health(bad):
    r=valid()
    if bad=='seconds': r['seconds']=np.array(np.nan)
    elif bad=='missing': r.pop('brems')
    elif bad=='extra': r['other']=np.zeros(1)
    elif bad=='shape': r['totals']=np.zeros(7)
    elif bad=='dtype': r['counts']=r['counts'].astype(np.int32)
    elif bad=='nonfinite': r['totals'][0]=np.inf
    elif bad in ('step','stack'): r['counts'][5 if bad=='step' else 6]=1
    elif bad=='count': r['nspec'][0,0]-=1
    elif bad=='violations': r['violations'][0]=1
    assert (d.cpu_health(r,200)['status']=='pass') == (bad=='seconds')


def test_real_health(cpus):
    assert d.cpu_health(cpus[0].run(200,core.SEED),200)['status']=='pass'


@pytest.mark.parametrize('delta,ok',[(1e-7,False),(1e-9,True)])
def test_gates(delta,ok):
    from test_p5b import artificial
    histories=artificial(200)
    histories['batches'][:]=1.
    r=core.sum_histories(histories,20)
    ref={'history_'+k:v for k,v in histories.items()}
    ref['conversion']=np.array(1.)
    r['batches']*=1+delta
    assert (d.c_gate(r,ref,20)['status']=='pass')==ok
    baseline=core.sum_histories(histories,20)
    c4={'c4_'+k:baseline[k][None] for k in ('nspec','counts','step_batches')}
    c4.update(conversion=np.array(1.),c4_blocks=d.blocks(baseline,100000,1.)[None])
    assert (d.d_gate(r,c4)['status']=='pass')==ok
    for k in ('nspec','counts','step_batches'):
        changed={key:v.copy() for key,v in baseline.items()}
        changed[k].flat[0]+=1
        assert d.c_gate(changed,ref,20)['status']=='fail'
        assert d.d_gate(changed,c4)['status']=='fail'


def test_pins_and_flags(monkeypatch,cpus):
    with pytest.raises(core.Stop,match='flags'):
        d.CPU('/tmp/p5d_bad_flags',cpus[0].data,flags=['-O3'])
    manifest=dict(source_sha256=d.SOURCE_SHA,files={k:dict(sha256=v) for k,v in d.BUNDLE_SHA.items()},cpu=dict(flags=d.FLAGS))
    monkeypatch.setattr(core,'load_bundle',lambda _: ({},{},manifest))
    d.load_bundle('unused')
    manifest['cpu']['flags']=['-O3']
    with pytest.raises(core.Stop, match='flags'): d.load_bundle('unused')
    manifest['cpu']['flags']=d.FLAGS
    monkeypatch.setattr(core, 'OPTIONS', ('--use_fast_math',))
    with pytest.raises(core.Stop, match='options'): d.load_bundle('unused')
    monkeypatch.setattr(core, 'OPTIONS', d.GPU_OPTIONS)
    manifest['source_sha256']='wrong'
    with pytest.raises(core.Stop): d.load_bundle('unused')
    manifest['source_sha256']=d.SOURCE_SHA
    manifest['files']['p5b_inputs.npz']['sha256']='wrong'
    with pytest.raises(core.Stop): d.load_bundle('unused')


def test_real_bundle_path_rejects(tmp_path):
    core.write_json(tmp_path/'p5b_manifest.json',dict(sources=core.FROZEN,source_sha256='wrong'))
    with pytest.raises(core.Stop,match='source'): d.load_bundle(tmp_path)


def test_timing_order():
    from prototypes.mv_electron.gpu.p5d_colab import timed_gpu
    events=[]
    class Event:
        def __init__(self): self.name='start' if not events.count('start') else 'end'
        def record(self):
            self.name='start' if 'start' not in events else 'end'
            events.append(self.name)
        def synchronize(self): events.append('end-sync')
    def sync(): events.append('sync')
    def allocate(shapes):
        events.append('allocate'); return core.empty_output(shapes)
    def launch(grid,block,args): events.append('launch')
    def download(values):
        events.append('download'); values['nspec'][0,0]=200; return values
    cp=SimpleNamespace(cuda=SimpleNamespace(Device=lambda:SimpleNamespace(synchronize=sync),Event=Event,get_elapsed_time=lambda a,b:12.))
    gpu=SimpleNamespace(cp=cp,data={'cum':np.zeros(20)},args=(),output=allocate,
                        kernels={'p5b_run_kernel':launch},download=download)
    _,times=timed_gpu(gpu,200)
    assert events[:6]==['sync','allocate','start','launch','end','download']
    assert times['kernel_ms']==12. and times['call_s']>=0


def test_stopping():
    assert d.no_output('D',1000000,MemoryError())['status']=='no_output'
    assert d.no_output('D3',100000,MemoryError())['status']=='no_output'
    assert d.no_output('D',1000000,RuntimeError(),cuda=True)['restart_required']
    for label,n,exc in [('D',100000,MemoryError()),('D',1000000,core.Stop('health'))]:
        with pytest.raises(core.Stop): d.no_output(label,n,exc)


def test_isolated_imports_and_notebook():
    code="from prototypes.mv_electron.gpu import p5d_core,p5d_colab; import sys; assert not any(k == 'numba' or k == 'cupy' or k.endswith('.p5a') for k in sys.modules)"
    subprocess.run([sys.executable,'-c',code],check=True)
    notebook=json.loads((core.HERE/'p5d_colab.ipynb').read_text())
    for c in notebook['cells']:
        if c['cell_type']=='code':
            compile(''.join(c['source']),'<notebook>','exec')
            assert c['outputs']==[]
