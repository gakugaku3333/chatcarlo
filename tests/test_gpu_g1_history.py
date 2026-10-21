"""Gate 0b chains, adapter constraints and batch health."""
import numpy as np
import pytest
from chatcarlo import kernel
from chatcarlo.scene import load_scene
from chatcarlo.detector import DetectorPlane
from chatcarlo.transport import run_transport
from chatcarlo.gpu import bundle,core
from chatcarlo.gpu.engine import gpu_engine_compatible
from test_gpu_g1_functions import bitwise
from pathlib import Path

DOC=Path(__file__).resolve().parents[1]/'docs/validation/gpu/g1'

@pytest.mark.parametrize('case',range(1,7))
def test_gpu_g1_0b_history(case,tmp_path):
    scene=load_scene(str(DOC/f'scenes/s{1 if case==1 else 2}.yaml'))
    bundle.create(scene,None,tmp_path);a,src,spec,settings,_=bundle.load(tmp_path)
    t=kernel.bake_scene_materials(settings['material_names']);g=kernel.bake_box_scene(scene.raw['geometry'],'air',t)
    origin=[(0,0,100),(4,0,100),(4,-6,100),(4,-6,100),(-3,0,100),(1,2,100)][case-1]
    d=np.array([.3,.2,-1.] if case==6 else [0.,0.,-1.]);d/=np.linalg.norm(d)
    energy=[60,100,100,100,60,80][case-1];n=10000;seed=2026102100+case
    ref=kernel.run_batch(t,g,energy,origin,tuple(d),n,seed,n_chunks=1,fluorescence_enabled=case!=4)
    mt_seed=int(kernel._chunk_plan(n,1,seed)[0][0])
    twin=core.Core(a,build_dir=tmp_path/'build',rng='mt')
    r=twin.run(np.tile(origin,(n,1)),np.tile(d,(n,1)),np.full(n,energy),seed=mt_seed,fluorescence=case!=4,record=True)
    bitwise(r['i'][:,0],ref.n_scatter,'interaction count')
    np.testing.assert_array_equal(r['i'][:,3]==0,ref.absorbed)
    np.testing.assert_array_equal(r['i'][:,3]==1,ref.escaped)
    bitwise(r['d'][:,0],ref.final_energy,'final energy')
    bitwise(r['d'][:,1],ref.energy_deposited,'deposit')
    bitwise(r['i'][:,2],ref.n_fluorescence,'fluorescence count')

@pytest.mark.parametrize('feature',['cylinder','sphere','heel_effect','mas','ctdi_vol_mGy','rotation','dose_grid','workers'])
def test_gpu_g1_compatible(feature):
    scene=load_scene(str(DOC/'scenes/s1.yaml'));kw={}
    if feature in ('cylinder','sphere'):scene.raw['geometry'][0]['shape']=feature
    elif feature=='dose_grid':kw['dose_grid']=True
    elif feature=='workers':kw['n_workers']=2
    else:scene.raw['source'][feature]=True
    ok,reason=gpu_engine_compatible(scene,**kw)
    assert not ok and reason
    if feature not in ('cylinder','sphere','workers'):assert feature in reason

@pytest.mark.parametrize('uncertainty',[True,False])
def test_gpu_g1_adapter(tmp_path,uncertainty):
    scene=load_scene(str(DOC/'scenes/s1.yaml'));plane=DetectorPlane([0,0,0],[0,0,1],[1,0,0],(30,30),(60,60))
    r=run_transport(scene,n_histories=1001,seed=17,source_seed=18,engine='gpu',gpu_device='cpu',batch_size=500,detector=plane,track_uncertainty=uncertainty)
    assert r.n_histories==1001 and r.detector.photon_count.sum()>0
    assert r.n_batches==(3 if uncertainty else 0)
    assert r.detector.n_batches==(3 if uncertainty else 0)
    assert r.gpu_metadata['source_seed']==18 and r.gpu_metadata['n_workers']==1
    assert r.energy_deposited_sem_MeV if uncertainty else not r.energy_deposited_sem_MeV


def test_gpu_g1_seed_required():
    with pytest.raises(ValueError,match='seed'):
        run_transport(load_scene(str(DOC/'scenes/s1.yaml')),engine='gpu')
