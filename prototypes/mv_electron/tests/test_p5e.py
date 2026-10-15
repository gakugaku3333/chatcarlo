"""P5e local contracts; no GPU execution."""
import json
import subprocess
import sys
import numpy as np
import pytest
from prototypes.mv_electron.gpu import p5e_core as e, p5b_core as core, p5d_core as d


def test_variants():
    assert len(e.VARIANTS)==12
    assert core.sha(e.source('V0').encode())==d.SOURCE_SHA
    assert e.options('V0')==core.OPTIONS
    for v,x in e.VARIANTS.items():
        assert f'-DP5_STACK_CAP={x["stack"]}' in e.options(v)
        assert ('--maxrregcount='+str(x['registers']) in e.options(v))==(x['registers'] is not None)
        if v!='V0':
            s=e.source(v)
            assert s.count('while(p5e_next(')==3
            assert 'p5b_store(&L,i,' in s
            assert 'start+uint64_t(i)' in s


@pytest.mark.parametrize('n,start',[(2,0),(127,0),(128,0),(129,0),(401,0),(129,7000),(129,7000)])
def test_visits(n,start):
    v,h,r=e.simulate(n,128,start)
    assert e.visits_check(v,h,r,n,start)['status']=='pass'
    v[0]=2
    assert e.visits_check(v,h,r,n,start)['status']=='fail'
    v[0]=0
    assert e.visits_check(v,h,r,n,start)['status']=='fail'
    v[0]=1; h[0]+=np.uint64(1)
    assert e.visits_check(v,h,r,n,start)['status']=='fail'


def test_order():
    assert e.ideal_order(np.array([2,3,3,1])).tolist()==[1,2,0,3]
    for order in (np.array([0,0],np.int32), np.array([1,2],np.int32),np.array([0,1],np.int64)):
        with pytest.raises(core.Stop): e.order_check(order,2)


@pytest.mark.parametrize('bad',['bit','zero','shape','dtype','nan','inf','integer'])
def test_e1(bad):
    a=core.empty_output(core.history_shapes(2))
    b={k:v.copy() for k,v in a.items()}
    assert e.e1(a,b)['status']=='pass'
    if bad=='bit': b['batches'].view(np.uint64)[0,0]=1
    elif bad=='zero': b['batches'][0,0]=-0.
    elif bad=='shape': b['batches']=np.zeros((1,300))
    elif bad=='dtype': b['batches']=b['batches'].astype(np.float32)
    elif bad in ('nan','inf'): b['batches'][0,0]=float(bad)
    else: b['counts'][0,0]=1
    assert e.e1(a,b)['status']=='fail'


def test_e2():
    out=core.empty_output(core.run_shapes(20)); out['nspec'][0,0]=100000
    out['batches'][:]=1
    ref={'c4_'+k:out[k][None].copy() for k in ('nspec','counts','step_batches')}
    ref.update(conversion=np.array(1.),c4_blocks=d.blocks(out,100000,1.)[None])
    assert e.e2(out,ref)['status']=='pass'
    for k in ('nspec','counts','step_batches'):
        b={k:v.copy() for k,v in out.items()}; b[k].flat[0]+=1
        assert e.e2(b,ref)['status']=='fail'
    out['batches']*=1.000001
    assert e.e2(out,ref)['status']=='fail'


def test_selection():
    assert e.select({'V0':[10,10.5,11],'V1':[8.5,9,10.5]})=='V0'
    assert e.select({'V0':[10,10.5,11],'V3':[8,8.5,9],'V1':[8.5,9,10.5]})=='V3'
    assert e.select({'V0':[10,11,12],'V1':[7,8,9],'V3':[7,8,9],'V6':[7,8,9]})=='V1'
    assert e.select({'V0':[10,11,12],'V3':[7,8,9],'V6':[7,8,9]})=='V3'
    with pytest.raises(core.Stop): e.select({'V0':[10]})


def test_identity(tmp_path):
    path=tmp_path/'identity.json'
    e.match_identity(path,dict(a=1))
    with pytest.raises(core.Stop): e.match_identity(path,dict(a=2))


def test_cpu16(tmp_path):
    from prototypes.mv_electron.gpu import p5a
    from prototypes.mv_electron.gpu.p5e import CPU16
    data=p5a.load_inputs(full=True)
    cpu=CPU16(tmp_path,data)
    baseline=d.CPU(tmp_path,data)
    actual=cpu.run(2000,core.SEED)
    assert cpu.flags==d.FLAGS[:-1]+['-DP5_STACK_CAP=16']
    assert p5a.compare_outputs(baseline.run(2000,core.SEED),actual)['status']=='pass'
    assert d.cpu_health(actual,2000)['status']=='pass'


def test_isolation_notebook():
    subprocess.run([sys.executable,'-c',"from prototypes.mv_electron.gpu import p5e_core,p5e_colab; import sys; assert not any(k in ('numba','cupy') or k.endswith('.p5a') for k in sys.modules)"],check=True)
    book=json.loads((core.HERE/'p5e_colab.ipynb').read_text())
    for cell in book['cells']:
        if cell['cell_type']=='code':
            compile(''.join(cell['source']),'<notebook>','exec')
            assert cell['outputs']==[]
    first=''.join(book['cells'][1]['source'])
    assert first.index("'fetch'")<first.index("'checkout'")
    assert 'CODE_COMMIT = ""' in first
