"""Acceptance checks for the isolated P5a0 trial; retain all work products."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np
import pytest

from prototypes.mv_electron.gpu import p5a0 as p


@pytest.fixture(scope='module')
def work():
    return p.workdir(p.ROOT/'.p5a0_work'/('pytest_'+uuid.uuid4().hex))


@pytest.fixture(scope='module')
def cpu(work):
    return p.CPU(work)


def test_three_official_known_answers(cpu):
    inputs, expected=p.known_answers()
    assert p.same(cpu.rng(inputs)[0],expected)
    assert p.same(p.philox_python(inputs)[0],expected)


def test_million_philox_cases_and_high_words(cpu):
    inputs=p.rng_inputs()
    assert len(inputs)==1_000_125
    assert np.any(inputs[:,0]>>32) and np.any(inputs[:,1]>>32) and np.any(inputs[:,2]>>32)
    assert all(p.same(a,b) for a,b in zip(p.philox_python(inputs),cpu.rng(inputs)))


def test_toy_reference_faults_overflow_and_partition(cpu,work):
    normal=cpu.toy(1000)
    assert all(p.same(a,b) for a,b in zip(p.toy_python(),normal))
    assert normal[0][:,2].max()>=8
    assert not normal[0][:,3].any()
    assert p.same(normal[1].sum(axis=0),cpu.toy(1000,shared=True)[1])
    for fault in (1,2,3):
        actual=p.CPU(work,fault=fault).toy(1000)
        assert p.fault_details(normal,actual,fault)['detected']
    overflow=p.CPU(work,cap=4).toy(1000)
    assert np.count_nonzero(overflow[0][:,3]==1)>0
    parts=[cpu.toy(400),cpu.toy(600,start=400)]
    for j in range(2):
        assert p.same(normal[j],np.concatenate([s[j] for s in parts]))


@pytest.fixture(scope='module')
def local_result(work):
    bundle=work/'inputs.npz'
    p.bundle(bundle)
    result=p.run('local',bundle,work,work/'local.json')
    return result,bundle


def test_full_local_acceptance(local_result):
    result,_=local_result
    assert result['verdict']=='pass',result['checks']
    assert all(c['status']=='pass' for c in result['checks'].values() if c['required'])
    assert result['checks']['T4']['above_20_rejected']
    assert result['checks']['T3']['values']['count']==2_000_000


def test_colab_without_physics_import(local_result,work):
    _,bundle=local_result
    dest=work/'dryrun.json'
    env=dict(os.environ,PYTHONPATH=str(p.ROOT))
    result=subprocess.run([sys.executable,'-m','prototypes.mv_electron.gpu.p5a0','colab',
        '--bundle',str(bundle),'--work',str(work),'--out',str(dest)],env=env,capture_output=True,text=True)
    r=json.loads(dest.read_text())
    assert not r['physics_imported']
    assert 'cpu_outputs' in r
    if importlib.util.find_spec('cupy') is None:
        assert 'GPUなし' in result.stdout
        assert all(c['status']=='not_run' for c in r['checks'].values())
        assert r['verdict']=='fail'
        assert result.returncode==1


def test_fail_closed_and_nonfinite():
    assert p.verdict({'a':dict(required=True,status='not_run')})=='fail'
    assert p.verdict({'a':dict(required=False,status='fail')})=='fail'
    r={'checks':{'a':dict(required=False,status='not_run')}}
    with pytest.raises(p.TrialStop):
        p.record(r,'a',values=p.diff(np.array([1.]),np.array([np.inf])))
    assert r['checks']['a']['status']=='fail'


def test_compare_refuses_mixed_bundle_and_code(local_result,work):
    result,_=local_result
    a=work/'compare_a.json';b=work/'compare_b.json'
    a.write_text(json.dumps(result))
    for key in ('bundle_sha256','code_commit','common_sha256','source_sha256'):
        changed=dict(result,**{key:'different'})
        b.write_text(json.dumps(changed))
        assert p.compare(a,b)['comparison']=='比較不能'
    b.write_text(json.dumps(result))
    comparison=p.compare(a,b)
    assert comparison['checks']['X1a']['status']=='pass'
    assert not comparison['p5a_ready']  # two local reports cannot pass the GPU gate


def test_input_guard_and_noncontiguous_table(cpu,local_result):
    _,path=local_result
    b,_=p.load_bundle(path)
    with pytest.raises(ValueError):
        cpu.quant(dict(b,T_eval=np.array([np.nextafter(20.,np.inf)])))
    b=dict(b,T_eval=b['T_eval'][:100])
    padded=np.zeros((len(b['energies']),10))
    padded[:,::2]=b['values']
    assert not padded[:,::2].flags.c_contiguous
    assert all(p.same(a,c) for a,c in zip(cpu.quant(b),cpu.quant(dict(b,values=padded[:,::2]))))


def test_notebook_and_source_contract():
    notebook=json.loads((p.HERE/'p5a0_colab.ipynb').read_text())
    for c in notebook['cells']:
        if c['cell_type']=='code':
            compile(''.join(c['source']),'notebook','exec')
    assert (p.HERE/'p5a0_gpu.cu').read_text().count('//@@COMMON@@')==1
    for name in ('p5a0_common.h','p5a0_gpu.cu'):
        source=(p.HERE/name).read_text()
        for forbidden in ('restrict','malloc(', 'new ', 'throw ', 'std::vector', 'delete '):
            assert forbidden not in source


@pytest.mark.skipif(importlib.util.find_spec('cupy') is None,reason='CuPy/GPU unavailable')
def test_gpu_toy_and_philox(cpu):
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount()==0:
        pytest.skip('GPU unavailable')
    g=p.GPUBackend(cp)
    t=p.rng_inputs()
    assert all(p.same(a,b) for a,b in zip(cpu.rng(t),g.rng(t)))
    assert all(p.same(a,b) for a,b in zip(cpu.toy(1000),g.toy(1000)))
