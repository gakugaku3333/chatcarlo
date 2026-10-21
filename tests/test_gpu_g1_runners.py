"""Small request-D execution, registered decisions and history reduction."""
import json
from pathlib import Path
import numpy as np
import pytest
from chatcarlo.gpu import validation,run_detector_image,bundle,core
from chatcarlo.detector import DetectorPlane
from chatcarlo.scene import load_scene

DOC=Path(__file__).resolve().parents[1]/'docs/validation/gpu/g1'


def test_gpu_g1_detector_runner_10000(tmp_path,capsys):
    run_detector_image.main(['--n','1e4','--seed','101','--workers','1','--out',str(tmp_path),'--device','cpu'])
    meta=json.loads((tmp_path/'N000010000_A.json').read_text());assert meta['engine']=='gpu' and meta['gpu_device']=='cpu' and meta['seed']==101 and isinstance(meta['source_seed'],int) and meta['n_histories']==10000 and meta['n_batches']==10
    with np.load(tmp_path/'N000010000_A.npz') as z:
        assert set(z.files)=={'category_fluence','photon_count','energy_sum_keV','energy_sum2_keV2','total_rel_err','n_batches_hit','n_histories','n_batches','pixel_area_cm2'}
        assert z['category_fluence'].shape==(4,100,120) and z['photon_count'].sum()>0
        assert z['n_histories']==10000 and z['n_batches']==10


def test_gpu_g1_independent_decision():
    x=np.arange(1,11,dtype=float);zero=np.zeros(10)
    report=validation.independent_judge(dict(same=x,zeros=zero),dict(same=x,zeros=zero))
    assert report['status']=='pass' and report['K']==1 and report['quantities']['zeros']['status']=='both_zero'
    assert validation.independent_judge(dict(x=x),dict(x=zero))['status']=='fail'
    assert validation.independent_judge(dict(x=np.ones(10)),dict(x=np.full(10,2.)))['status']=='fail'
    assert validation.independent_judge(dict(x=np.ones(10)),dict(x=np.ones(10)))['status']=='pass'
    sem=np.hypot(x.std(ddof=1)/np.sqrt(10),x.std(ddof=1)/np.sqrt(10))
    assert validation.independent_judge(dict(x=x),dict(x=x+sem*3.5))['status']=='retry'
    assert validation.independent_judge(dict(x=x),dict(x=x+sem*4.01))['status']=='fail'
    assert validation.independent_judge(dict(x=x,y=x),dict(x=x+sem*3.5,y=x+sem*3.5))['status']=='fail'


def test_gpu_g1_paired_decision():
    a=np.ones((10,3));b=a.copy();bounds=np.full_like(a,1e-6);b[:,0]+=1e-7;b[:,1]+=np.linspace(-1,1,10)
    assert validation.paired_judge(a,b,bounds)['status']=='pass'
    b[:,2]+=1e-3;assert validation.paired_judge(a,b,bounds)['status']=='fail'


def test_gpu_g1_gate2a_failure_writes_diagnostics(tmp_path,monkeypatch):
    probes=dict(status='fail',probes=dict(physics0=dict(status='pass'),geometry1=dict(status='fail',integer_match=False,float_match=0.5)))
    monkeypatch.setattr(bundle,'load',lambda path:({},None,None,{},{}))
    monkeypatch.setattr(core,'Core',lambda arrays,device:object())
    monkeypatch.setattr(validation,'gate2a',lambda cpu,gpu:probes)
    out=tmp_path/'results'
    with pytest.raises(RuntimeError,match='^gate 2a failed; stop before paired runs$'):
        validation.gate2(tmp_path/'bundle','s2',out,'test-job')
    assert json.loads((out/'gate2a_failed.json').read_text())==probes
    assert sorted(p.name for p in out.iterdir())==['gate2a_failed.json']


def test_gpu_g1_record_reconstruction(tmp_path):
    scene=load_scene(str(DOC/'scenes/s2.yaml'));plane=DetectorPlane([0,0,0],[0,0,1],[1,0,0],(30,30),(60,60));bundle.create(scene,plane,tmp_path)
    a,src,spec,settings,_=bundle.load(tmp_path)
    from chatcarlo.gpu.source_host import sample_source_photons
    o,d,e=sample_source_photons(src,3000,np.random.default_rng(8001),spec);engine=core.Core(a,build_dir=tmp_path/'build');r=engine.run(o,d,e,seed=8000,record=True);rr=validation.record_reduce(r,a)
    bounds=validation.reconstruction_bounds(r,rr,3000)
    for k in rr:assert np.all(np.abs(np.asarray(rr[k])-r[k])<=bounds[k])
    # Global history numbers are independent of the device launch size.
    r0=engine.run(o[:1000],d[:1000],e[:1000],seed=8000,record=True);r1=engine.run(o[1000:],d[1000:],e[1000:],seed=8000,start=1000,record=True)
    for k in ('d','i'):assert np.concatenate((r0[k],r1[k])).tobytes()==r[k].tobytes()


def test_gpu_g1_2a_physics0_grid_index_tolerated():
    from chatcarlo.gpu.validation import physics0_judge
    me=2;cpu=np.zeros((4,8+5*me));cpu[:,:4]=[3.,1.,1.5,.5];cpu[:,8:8+3*me]=.25;cpu[:,8+3*me:8+4*me]=10.
    gpu=cpu.copy();gpu[3,8+3*me:8+4*me]=9.;gpu[3,:4]*=1+1e-15
    r=physics0_judge(cpu,gpu,me,n_random=2)
    assert r['value_pass'] and r['grid_index_shifted']==1 and r['grid_index_shifted_random']==0
    bad=gpu.copy();bad[3,0]=cpu[3,0]*(1+1e-9)
    assert not physics0_judge(cpu,bad,me)['value_pass']
    cpu[0,8]=0.;nonzero=cpu.copy();nonzero[0,8]=1e-300
    assert not physics0_judge(cpu,nonzero,me)['value_pass']
