"""P5f-1 CPU regression, uncommon paths and fail-closed artifact contracts."""
import json
import re
import subprocess
import sys
import numpy as np
import pytest
from prototypes.mv_electron.gpu import p5b_core as core, p5g_core as g


def test_cpu_2000(tmp_path):
    from prototypes.mv_electron.gpu import p5g, p5a
    data=p5a.load_inputs(full=True)
    new=p5g.CPU(tmp_path/'new',data)
    old=p5g.CPU(tmp_path/'old',data,baseline=True)
    a=new.run(2000,core.SEED,start=7000,diagnostic=True)
    b=old.run(2000,core.SEED,start=7000,diagnostic=True)
    core.require(g.exact(a,b,(*p5g.d.OUTPUTS,'diagnostic','diagnostic_meta')),'2000 CPU histories')
    assert not np.any(a['counts'][5:7])


def test_unusual_cpu(tmp_path):
    from prototypes.mv_electron.gpu import p5g, p5a
    report=p5g.unusual(tmp_path,p5a.load_inputs(full=True))
    assert report['status']=='pass'
    assert report['missing']==[]
    for path in tmp_path.glob('*_trace_*.npz'):
        v=core.read_npz(path)
        assert g.trace_check(v['trace'],v['meta'])['status']=='pass'
        assert v['trace'].shape[1]==len(g.TRACE_COLUMNS)


@pytest.mark.parametrize('key',core.HISTORY_ORDER)
def test_one_bit(key):
    a=core.empty_output(core.history_shapes(2));b={k:v.copy() for k,v in a.items()}
    assert g.exact(a,b,core.HISTORY_ORDER)['status']=='pass'
    b[key].view(np.uint64).flat[0]^=np.uint64(1)
    assert g.exact(a,b,core.HISTORY_ORDER)['status']=='fail'


def test_trace_contract():
    v=np.zeros((4,10));v[:,:2]=[[-1,0],[0,3],[3,4],[5,4]]
    assert g.trace_check(v,np.array([4,0]),stored=True)['status']=='pass'
    assert g.trace_check(v,np.array([4,1]),stored=True)['status']=='fail'
    assert g.trace_check(v,np.array([4,0]))['status']=='fail'
    v[1,:2]=[1,3]
    assert g.trace_check(v,np.array([4,0]),stored=True)['status']=='fail'


def test_source_scheduler():
    text=g.source();gpu=(core.HERE/'p5g_gpu.cu').read_text()
    assert '#include "p5' not in text
    assert gpu.index('unsigned hm=__ballot_sync')<gpu.index('if(pm&lane)')
    assert gpu.index('p5g_warp(t,seed,start+uint64_t(i),i<n')<gpu.index('if(i>=n)return;')
    assert '__activemask' not in gpu
    assert 'p5g_history(t,&rng' in (core.HERE/'p5g_cpu.cpp').read_text()
    assert text.count('struct P5Tables {')==1
    core.fingerprints()


def test_map_weights_and_csv():
    from prototypes.mv_electron.gpu.p5g_colab import parse_source_csv
    csv='Function,Address,SASS,File,Line,Instructions Executed,Thread Instructions Executed,Predicated-On Thread Instructions Executed\nk,0x0,DADD R1 R2,x.h,10,2,32,16\nk,0x10,@P0 DMUL R1 R2,x.h,10,8,8,8\nk,0x20,IADD R1 R2,x.h,10,10,160,120\n'
    records=parse_source_csv(csv);summary=g.weighted_map(records)
    by_kind={r['kind']:r for r in summary}
    assert by_kind['fp64']['lanes']==4
    assert by_kind['all']['lanes']==10
    with pytest.raises(core.Stop):parse_source_csv('no counters')


def test_sass_extracts_cached_elf(tmp_path,monkeypatch):
    from prototypes.mv_electron.gpu import p5g_colab
    cache=tmp_path/'nvrtc_cache';cache.mkdir()
    elf=b'\x7fELF\x00synthetic device code\xff'
    cached=b'CuPy cache header\n'+elf
    path=cache/'kernel.cubin';path.write_bytes(cached)
    monkeypatch.setattr(p5g_colab.shutil,'which',lambda name:'/fake/cuobjdump')
    calls=[]
    def dump(args,**kwargs):
        calls.append(args)
        assert args==['/fake/cuobjdump','--dump-sass',str(tmp_path/'sass_0.cubin')]
        assert (tmp_path/'sass_0.cubin').read_bytes()==elf
        return subprocess.CompletedProcess(args,0,'Function : target\n /*0000*/ DADD R1, R2, R3;\n','')
    monkeypatch.setattr(p5g_colab.subprocess,'run',dump)
    assert p5g_colab.sass(tmp_path,'target')==['/*0000*/ DADD R1, R2, R3;']
    assert len(calls)==1
    assert path.read_bytes()==cached


def test_sass_rejects_cache_without_elf(tmp_path,monkeypatch):
    from prototypes.mv_electron.gpu import p5g_colab
    cache=tmp_path/'nvrtc_cache';cache.mkdir()
    (cache/'kernel.cubin').write_bytes(b'CuPy cache header without device code')
    monkeypatch.setattr(p5g_colab.shutil,'which',lambda name:'/fake/cuobjdump')
    def unexpected_dump(*args,**kwargs):
        pytest.fail('cuobjdump must not run without ELF')
    monkeypatch.setattr(p5g_colab.subprocess,'run',unexpected_dump)
    with pytest.raises(core.Stop,match='ELF unavailable'):
        p5g_colab.sass(tmp_path,'target')
    assert not (tmp_path/'sass_0.cubin').exists()


def test_gate1_rejects_missing_and_stale(tmp_path):
    path=tmp_path/'gate1.json'
    report=dict(status='pass',batches=100,sources={k:core.file_sha(core.HERE/k) for k in (*core.FROZEN,'p5b_wrap.h',*g.SOURCES)},
                outputs=list(core.RUN_ORDER[:-1]),checks={str(b):dict(status='pass',fields={k:True for k in core.RUN_ORDER[:-1]}) for b in range(100)})
    core.write_json(path,report)
    assert g.require_gate1(path)['status']=='pass'
    report['checks']['1']['status']='fail';core.write_json(path,report)
    with pytest.raises(core.Stop):g.require_gate1(path)
    report['checks']['1']['status']='pass';report['sources']['p5g_kernel.h']='0'*64
    core.write_json(path,report)
    with pytest.raises(core.Stop):g.require_gate1(path)


def test_isolation_notebook():
    subprocess.run([sys.executable,'-c',"from prototypes.mv_electron.gpu import p5g_core,p5g_colab; import sys; assert not any(k in ('numba','cupy') or k.endswith('.p5a') for k in sys.modules)"],check=True)
    book=json.loads((core.HERE/'p5g_colab.ipynb').read_text())
    for cell in book['cells']:
        if cell['cell_type']=='code':
            compile(''.join(cell['source']),'<notebook>','exec')
            assert cell['outputs']==[]
    first=''.join(book['cells'][1]['source'])
    assert first.index("'fetch'")<first.index("'checkout'")
    assert re.search(r'CODE_COMMIT = "([0-9a-f]{40})?"',first)
    assert 'viveMonte/p5/p5g' in first
    flags=['APPROVE_MAP','APPROVE_GATE2','APPROVE_GATE3','APPROVE_TIMING','APPROVE_COMPARISON_MAP']
    for flag,cell in zip(flags,book['cells'][2:]):assert flag+' = False' in ''.join(cell['source'])


def test_map_notebook_contract():
    path=core.HERE/'p5g_map_colab.ipynb'
    book=json.loads(path.read_text())
    assert 'APPROVE_' not in path.read_text()
    cells=[cell for cell in book['cells'] if cell['cell_type']=='code']
    assert len(cells)==2
    for cell in cells:
        compile(''.join(cell['source']),'<map-notebook>','exec')
        assert cell['outputs']==[]
        assert cell['execution_count'] is None
    first=''.join(cells[0]['source'])
    original=json.loads((core.HERE/'p5g_colab.ipynb').read_text())
    assert cells[0]==original['cells'][1]
    assert first.index("'fetch'")<first.index("'checkout'")<first.index('from prototypes.')
    assert re.search(r'CODE_COMMIT = "([0-9a-f]{40})?"',first)


@pytest.mark.parametrize('has_gate2,has_gate3',[(False,False),(True,False),(False,True),(True,True)])
def test_map_notebook_runs_available_maps(tmp_path,capsys,has_gate2,has_gate3):
    from types import SimpleNamespace
    book=json.loads((core.HERE/'p5g_map_colab.ipynb').read_text())
    reports=[tmp_path/'gate2/gate2.json',tmp_path/'gate3/gate3.json']
    for present,path in zip((has_gate2,has_gate3),reports):
        if present:
            path.parent.mkdir(parents=True)
            path.write_text('{}')
    calls=[]
    def divergence_map(*args,**kwargs):
        calls.append((args,kwargs))
    exec(compile(''.join(book['cells'][2]['source']),'<map-notebook>','exec'),
         dict(p5g_colab=SimpleNamespace(divergence_map=divergence_map),
              BUNDLE='bundle',RESULTS=tmp_path,GATE1='gate1'))
    expected=[(('bundle',tmp_path/'map_v0','gate1'),{})]
    if has_gate2 and has_gate3:
        expected.append((('bundle',tmp_path/'map_comparison','gate1'),
                         dict(compare=True,gate2_report=reports[0],gate3_report=reports[1])))
        assert capsys.readouterr().out==''
    else:
        assert capsys.readouterr().out.strip()=='比較の地図は関門2・3の後'
    assert calls==expected
