"""P5k CPU acceptance, no gate1 or GPU execution and no deletion."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import numpy as np
import pytest
from prototypes.mv_electron.gpu import p5b_core as core,p5k,p5k_core as h,p5k_emulate as emulate


@pytest.fixture(scope='module')
def data():
    from prototypes.mv_electron.gpu.p5a import load_inputs
    return load_inputs(full=True)


@pytest.fixture(scope='module',params=[False,True],ids=['ordinary','checked'])
def cpu(request,tmp_path_factory,data):
    return p5k.CPU(tmp_path_factory.mktemp('p5k-retained'),data,request.param)


def test_management(cpu):
    assert p5k.management(cpu)['status']=='pass'


def test_transport_all_fields_traces_failures_and_reuse(cpu):
    result=p5k.compare_small(cpu)
    assert result['status']=='pass' and result['n']==2000 and result['start']==7000
    assert set(result['fields'])==set(h.VARIANTS)
    assert result['max_trace_records']<8192
    for fields in result['fields'].values():assert all(fields['fields'].values())


@pytest.mark.parametrize('variant',h.VARIANTS)
@pytest.mark.parametrize('fault',[1,2,3])
def test_actual_synthetic_faults_rejected(cpu,variant,fault):
    assert cpu.schedule(variant,1553,3,fault=fault)['status']=='fail'


def test_multiple_block_transport(cpu):
    # Includes refill across multiple blocks and a nonzero global start.
    expected=cpu.histories(1553,'P5j',12345)
    for v in h.VARIANTS:
        out=cpu.histories(1553,v,12345,blocks=3)
        assert h.exact(out,expected)['status']=='pass'


def test_emulation_invariants():
    sequences=[np.array([0,1,1,2,0,3],np.int8),np.array([0,3],np.int8),np.array([0,1,2,3],np.int8)]
    for grouped in (False,True):
        for R in (32,256,512,1024):
            report=emulate.utilization(sequences,R,grouped=grouped,histories=R*3+1)
            assert report['conserved'] and 0<report['utilization']<=1
            assert report['transitions']==sum(len(sequences[i%3]) for i in range(R*3+1))
    report=emulate.fixed(sequences);assert report['conserved'] and 0<report['utilization']<=1


def test_frozen_files_and_identity(data):
    core.fingerprints()
    for name,digest in h.FROZEN.items():assert core.file_sha(core.HERE/name)==digest
    identity=h.identity(data)
    assert not any(n.endswith('.py') for n in identity['sources'])
    assert identity['nspec']==20
    text=h.source()
    assert '#include "p5' not in text
    # The frozen P5j translation unit is an exact prefix, not a rewrite.
    from prototypes.mv_electron.gpu import p5j_core
    assert text.startswith(p5j_core.source())
    driver=(core.HERE/'p5k_driver.h').read_text()
    assert 's->L.maxdepth=o.maxdepth' in driver
    assert 's->diag_overflow=d.overflow' in driver and 's->trace_overflow=tr.overflow' in driver
    cpp=(core.HERE/'p5k_cpu.cpp').read_text()
    assert 'p5j_history(t,&s.rng,0,maxsteps,&o,&d,trace?&tr:0)' in cpp
    assert 'memcmp' not in cpp


def test_notebook_export_and_no_deletion(tmp_path):
    nb=json.loads((core.HERE/'p5k_colab.ipynb').read_text())
    for cell in nb['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'<cell>','exec')
    text=str(nb)
    assert 'p5k_colab.run_all(' in text and "ROOT.parent/'p5b'" in text
    assert 'RUN_' not in text
    with zipfile.ZipFile(p5k.export_sources(tmp_path)) as z:
        for n in (*h.SOURCES,'p5k.py','p5k_core.py','p5k_colab.py','p5k_colab.ipynb','p5k_emulate.py','p5j_kernel.h'):
            assert 'prototypes/mv_electron/gpu/'+n in z.namelist()
    for path in core.HERE.glob('p5k*.py'):
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                fn=node.func
                assert not isinstance(fn,ast.Attribute) or fn.attr not in ('unlink','remove','rmtree','rmdir')
    subprocess.run([sys.executable,'-c','import sys; from prototypes.mv_electron.gpu import p5k,p5k_colab; assert "cupy" not in sys.modules; assert "numba" not in sys.modules'],check=True)


def test_gate_records_fail_closed(data,tmp_path):
    from prototypes.mv_electron.gpu import p5k_colab as colab
    path=tmp_path/'gate1.json';core.write_json(path,dict(status='pass',identity=h.identity(data),batches=100,checks={}))
    with pytest.raises(core.Stop):h.require_gate1(path,data)
    with pytest.raises(core.Stop):colab.require_gate2(dict(status='pass',identity=h.identity(data)),data)
    # A truncated trace compared with itself must still fail.
    trace=np.zeros((1,1,10));meta=np.array([[9,8]],np.int64)
    assert h.traces(trace,meta,trace,meta)['status']=='fail'


def test_colab_order_raw_csv_and_aggregation():
    text=(core.HERE/'p5k_colab.py').read_text()
    body=text[text.index('def run_all('):text.index("if __name__=='__main__'")]
    assert body.index('require_gate1')<body.index('gate2(cp')<body.index('timing(cp')<body.index('gate3(gpu')<body.index('raw_map(')
    assert 'parse_source_csv' not in text and 'read_csv' not in text
    assert 'core.c3(expected,old.production(20000),20)' in text
    assert 'core.c3(expected,new.production(20000),20)' in text
    assert "('ordinary',False),('checked',True)" in text
    assert "for first in range(0,20000,32)" in text
    assert 'gate3 unconfirmed' in text
