"""P5l CPU acceptance, no gate1 or GPU execution and no deletion."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import numpy as np
import pytest
from prototypes.mv_electron.gpu import p5b_core as core,p5l,p5l_core as h,p5l_emulate as emulate


@pytest.fixture(scope='module')
def data():
    from prototypes.mv_electron.gpu.p5a import load_inputs
    return load_inputs(full=True)


@pytest.fixture(scope='module',params=[False,True],ids=['ordinary','checked'])
def cpu(request,tmp_path_factory,data):
    return p5l.CPU(tmp_path_factory.mktemp('p5l-retained'),data,request.param)


def test_management(cpu):
    assert p5l.management(cpu)['status']=='pass'


def test_transport_all_fields_traces_failures_and_reuse(cpu):
    result=p5l.compare_small(cpu)
    assert result['status']=='pass' and result['n']==2000 and result['start']==7000
    assert set(result['fields'])==set(h.VARIANTS)
    assert result['max_trace_records']<8192
    for fields in result['fields'].values():assert all(fields['fields'].values())


@pytest.mark.parametrize('variant',h.VARIANTS)
@pytest.mark.parametrize('fault',[1,2,3])
def test_actual_synthetic_faults_rejected(cpu,variant,fault):
    assert cpu.schedule(variant,2*3*256*h.CONFIG[h.VARIANTS.index(variant)][2]+17,3,fault=fault)['status']=='fail'


def test_multiple_block_transport(cpu):
    # Includes refill across multiple blocks and a nonzero global start.
    expected=cpu.histories(1553,'P5j',12345)
    for v in h.VARIANTS:
        out=cpu.histories(1553,v,12345,blocks=3)
        assert h.exact(out,expected)['status']=='pass'


def test_emulation_invariants():
    sequences=[np.array([0,1,1,2,0,3],np.int8),np.array([0,3],np.int8),np.array([0,1,2,3],np.int8)]
    for m in (1,2,4):
        for K in (1,2,4,8):
            report=emulate.utilization(sequences,m,K,histories=32*m*3+1)
            assert report['conserved'] and 0<report['utilization']<=1
            assert report['transitions']==sum(len(sequences[i%3]) for i in range(32*m*3+1))


def test_frozen_files_and_identity(data):
    core.fingerprints()
    for name,digest in h.FROZEN.items():assert core.file_sha(core.HERE/name)==digest
    identity=h.identity(data)
    assert not any(n.endswith('.py') for n in identity['sources'])
    assert identity['nspec']==20
    assert json.loads(json.dumps(identity))==identity
    text=h.source()
    assert '#include "p5' not in text
    # The frozen P5j translation unit is an exact prefix, not a rewrite.
    from prototypes.mv_electron.gpu import p5j_core
    assert text.startswith(p5j_core.source())
    assert tuple(c[2] for c in h.CONFIG)==(1,1,1,2,4)
    assert tuple(c[3] for c in h.CONFIG)==(1,4,4,4,4)
    cpp=(core.HERE/'p5l_cpu.cpp').read_text()
    assert '#include "p5k_cpu.cpp"' in cpp and 'p5k_run(-1,' in cpp
    assert 'memcmp' not in cpp


def test_notebook_export_and_no_deletion(tmp_path):
    nb=json.loads((core.HERE/'p5l_colab.ipynb').read_text())
    for cell in nb['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'<cell>','exec')
    text=str(nb)
    assert 'p5l_colab.run_all(' in text and "ROOT.parent/'p5b'" in text
    assert 'RUN_' not in text
    with zipfile.ZipFile(p5l.export_sources(tmp_path)) as z:
        for n in (*h.SOURCES,'p5l.py','p5l_core.py','p5l_colab.py','p5l_colab.ipynb','p5l_emulate.py','p5j_kernel.h'):
            assert 'prototypes/mv_electron/gpu/'+n in z.namelist()
    for path in core.HERE.glob('p5l*.py'):
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                fn=node.func
                assert not isinstance(fn,ast.Attribute) or fn.attr not in ('unlink','remove','rmtree','rmdir')
    subprocess.run([sys.executable,'-c','import sys; from prototypes.mv_electron.gpu import p5l,p5l_colab; assert "cupy" not in sys.modules; assert "numba" not in sys.modules'],check=True)


def test_gate_records_fail_closed(data,tmp_path):
    from prototypes.mv_electron.gpu import p5l_colab as colab
    path=tmp_path/'gate1.json';core.write_json(path,dict(status='pass',identity=h.identity(data),batches=100,checks={}))
    with pytest.raises(core.Stop):h.require_gate1(path,data)
    with pytest.raises(core.Stop):colab.require_gate2(dict(status='pass',identity=h.identity(data)),data)
    # A truncated trace compared with itself must still fail.
    trace=np.zeros((1,1,10));meta=np.array([[9,8]],np.int64)
    assert h.traces(trace,meta,trace,meta)['status']=='fail'


def test_colab_order_raw_csv_and_aggregation():
    text=(core.HERE/'p5l_colab.py').read_text()
    body=text[text.index('def run_all('):text.index("if __name__=='__main__'")]
    assert body.index('require_gate1')<body.index('gate2(cp')<body.index('timing(cp')<body.index('gate3(gpu')<body.index('raw_map(')
    assert 'parse_source_csv' not in text and 'read_csv' not in text
    assert 'core.c3(expected,old.production(20000),20)' in text
    assert 'core.c3(expected,new.production(20000),20)' in text
    assert "('ordinary',False),('checked',True)" in text
    assert "for first in range(0,20000,32)" in text
    assert 'gate3 unconfirmed' in text


def test_warp_contract_source_and_launch():
    text=(core.HERE/'p5l_gpu.cu').read_text()
    assert '__syncthreads' not in text
    assert '(blockIdx.x*8+warp)*R' in text
    assert '32*a+lane' in text
    assert text.count('__syncwarp()')>=6
    assert 'p5l_classify(slots,m,lists,counts)' in text
    assert 'p5l_work(s,t,maxsteps,K,trace,capacity)' in text
    py=(core.HERE/'p5l_colab.py').read_text()
    assert 'blocks=40' in py and '(256,)' in py
    assert 'stage_timing' not in py
    for section in ('SpeedOfLight','MemoryWorkloadAnalysis_Tables','ComputeWorkloadAnalysis'):
        assert section in py
    assert '1000000' in py and 'refill_audit(new,old)' in py


def test_synthetic_k_mixed_snapshot_and_warp_isolation(cpu):
    for v in range(5):assert cpu.lib.p5l_contract(v)==0


def test_refill_audit_record(cpu):
    # Fast independent focused refill; full 2081-history traces are in compare_small.
    for v in h.VARIANTS:
        out,tr,meta=cpu.histories(1041,v,blocks=1,capacity=8192)
        ref,rt,rm=cpu.histories(1041,'P5j',blocks=1,capacity=8192)
        assert h.exact(out,ref)['status']=='pass'
        assert h.traces(tr,meta,rt,rm)['status']=='pass'
        assert not np.any(meta[:,1])


def test_assembled_gpu_cpp_syntax(tmp_path):
    # Host stubs check C++ declarations/macros only, not CUDA execution or synchronization.
    # In particular P5B_HISTORY_ARGS already owns an output named `counts`.
    import shutil
    prelude='''#include <cmath>
#include <cstring>
#define P5_GPU
#define P5_RNG_PHILOX
#define P5_TRANSPORT_FAULT 0
#define P5_STACK_CAP 64
#define __device__
#define __global__
#define __shared__
struct Dim {int x;}; Dim threadIdx,blockIdx,blockDim;
inline void __syncthreads(){} inline void __syncwarp(){}
inline unsigned __ballot_sync(unsigned,bool){return 0;}
inline int __popc(unsigned a){return __builtin_popcount(a);}
inline double __longlong_as_double(unsigned long long a){union{unsigned long long i;double d;}v;v.i=a;return v.d;}
inline long long __double_as_longlong(double a){union{long long i;double d;}v;v.d=a;return v.i;}
template<class T,class U> T atomicAdd(T*a,U b){T old=*a;*a+=b;return old;}
template<class T,class U> T atomicMax(T*a,U b){T old=*a;if(b>*a)*a=b;return old;}
'''
    source=tmp_path/'assembled_gpu.cpp';source.write_text(prelude+h.source())
    result=subprocess.run([shutil.which('clang++'),'-std=c++11','-fsyntax-only',str(source)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
