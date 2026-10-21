"""Gates 0c priority, 0c' artificial aggregation, and 0f fault builds."""
import ctypes
from pathlib import Path
import numpy as np
import pytest
from chatcarlo.scene import load_scene
from chatcarlo.detector import DetectorPlane,DetectorTally
from chatcarlo.tally import ScalarMoments
from chatcarlo import kernel
from chatcarlo.gpu import bundle,core
from test_gpu_g1_functions import FIELDS,physics_reference,bitwise

DOC=Path(__file__).resolve().parents[1]/'docs/validation/gpu/g1'

def data(tmp_path):
    scene=load_scene(str(DOC/'scenes/s2.yaml'));bundle.create(scene,None,tmp_path);return bundle.load(tmp_path)[0]

@pytest.mark.parametrize('fault',[1,2,3,4])
def test_gpu_g1_0f_faults(tmp_path,fault):
    t=kernel.bake_scene_materials(['air','water','lead']);a=core.pack(t)
    a.update(centers=np.array([[0.,0.,0.],[0.,0.,0.]]),halves=np.array([[10.,10.,10.],[5.,5.,5.]]),materials=np.array([1,2],np.int64),background=np.array(0),world=np.array([0.,0.,0.,60.,60.,60.]),detector=np.array([]))
    good=core.Core(a,build_dir=tmp_path,rng='mt');bad=core.Core(a,build_dir=tmp_path,rng='mt',fault=fault)
    if fault==4:
        x=np.zeros((1,10));out=[]
        g=core.Geometry(a['centers'].ctypes.data,a['halves'].ctypes.data,a['materials'].ctypes.data,2,0,a['world'].ctypes.data,None)
        for driver in (good,bad):
            y=np.zeros((1,4));fn=driver.lib.g1_geometry_probe;fn.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p];fn(1,ctypes.byref(g),x.ctypes.data,1,y.ctypes.data);out.append(y)
        assert out[0][0,0]==2 and out[1][0,0]==1;return
    if fault==3:
        # Artificial input reaches the otherwise unreachable sub-5-keV branch.
        t.k_edge[2,0]=1.;t.k_omega[2,0]=1.;t.k_frac[2,0,:]=1.;t.n_lines[2,0]=1;t.k_line_e[2,0,:]=4.;t.k_line_p[2,0,:]=0.;t.k_line_p[2,0,0]=1.
        aa=core.pack(t);a['data']=aa['data'];a['offsets']=aa['offsets']
    x=np.zeros((1000,8));x[:,0]=100.;x[:,1]=2
    which={1:2,2:1,3:3}[fault];outputs=[]
    for driver in (good,bad):
        y=np.zeros((len(x),8+5*driver.me));state=np.zeros(626+len(x),np.uint64);nxt=np.empty(16)
        fn=driver.lib.g1_physics_probe;fn.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
        fn(which,a['data'].ctypes.data,a['offsets'].ctypes.data,driver.me,driver.ng,driver.nm,x.ctypes.data,len(x),2026102100,y.ctypes.data,state.ctypes.data,nxt.ctypes.data);outputs.append((y,state,nxt))
    assert any(aa.tobytes()!=bb.tobytes() for aa,bb in zip(*outputs))
    if fault==3:
        assert np.all(outputs[0][0][:,:2]==0)
        assert np.all(outputs[1][0][:,0]==1) and np.all(outputs[1][0][:,1]==4)
        expected=physics_reference(3,x,2026102100,*(getattr(t,f) for f in FIELDS))
        bitwise(outputs[0][0],expected,'artificial fluorescence')

@pytest.mark.parametrize('coincident',['boundary','interaction'])
def test_gpu_g1_0c_detection_first(tmp_path,coincident):
    scene=load_scene(str(DOC/'scenes/s2.yaml'));plane=DetectorPlane([0,0,0],[0,0,1],[1,0,0],(30,30),(60,60))
    bundle.create(scene,plane,tmp_path);a,*_=bundle.load(tmp_path)
    if coincident=='boundary':
        # The S2 aluminum bottom coincides with z=0. Force a flight to that face.
        a['data'][int(a['offsets'][19]):]=0.
        o=np.array([[0.,6.,.5]]);energy=60.;seed=42
    else:
        t=kernel.bake_scene_materials(['air']);a=core.pack(t)
        a.update(centers=np.empty((0,3)),halves=np.empty((0,3)),materials=np.empty(0,np.int64),background=np.array(0),world=np.array([0.,0.,0.,100.,100.,100.]),detector=np.r_[0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,1.,0.,30.,30.,60.,60.])
        # mu is computed by the unchanged reference; exact t=ds, not a tolerance.
        p=np.zeros(t.zs.shape[1]);c=p.copy();r=p.copy();idx=np.zeros(len(p),np.int64);f=p.copy()
        mu,*_=kernel._mu_and_parts_scalar(60.,0,t.n_elem,t.fracs,t.log_e,t.step,t.n_grid,t.photo,t.compt,t.rayl,t.density_g_cm3,p,c,r,idx,f)
        seed=42;tau=-np.log(np.random.RandomState(seed).random_sample());distance=tau/mu
        a['world'][3:]=max(100.,distance*2);a['detector'][2]=-distance;o=np.zeros((1,3));energy=60.
    driver=core.Core(a,build_dir=tmp_path/'build',rng='mt');result=driver.run(o,np.array([[0.,0.,-1.]]),np.array([energy]),seed=seed,record=True)
    assert result['i'][0,3]==2 and result['i'][0,0]==0 and result['i'][0,7]==1
    assert result['d'][0,1]==0 and result['images'][4].sum()==1

@pytest.mark.parametrize('uncertainty',[True,False])
@pytest.mark.parametrize('sizes',[[4,4,2],[10]])
def test_gpu_g1_0c_artificial_tally(tmp_path,uncertainty,sizes):
    # Explicit hand values: 10 histories, energies 1..10, alternating pixels,
    # categories 0..3 repeated; last two are escape/absorption, third batch zero.
    plane=DetectorPlane([0,0,0],[0,0,1],[1,0,0],(2,1),(2,1));scene=load_scene(str(DOC/'scenes/s1.yaml'))
    bundle.create(scene,plane,tmp_path);a,*_=bundle.load(tmp_path);driver=core.Core(a,build_dir=tmp_path/'build')
    # Expose the actual device reducer via a tiny CPU-only probe.
    fn=driver.lib.g1_reduce_probe
    fn.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
    d=np.zeros((10,driver.nm+2));d[:,0]=np.arange(1,11);d[:,2]=2.;d[:,1]=2.;d[8:,1:]=0.
    i=np.zeros((10,9),np.int64);i[:8,3]=2;i[:8,4]=np.arange(8)%2;i[:8,6]=np.arange(8)%4;i[8,3]=1
    tally=DetectorTally(plane,uncertainty);moments=ScalarMoments();start=0;totals=np.zeros(driver.nm+6+11*driver.npix)
    for n in sizes:
        agg=np.zeros_like(totals);dd=np.ascontiguousarray(d[start:start+n]);ii=np.ascontiguousarray(i[start:start+n]);fn(driver.nm,a['detector'].ctypes.data,dd.ctypes.data,ii.ctypes.data,n,agg.ctypes.data);totals+=agg
        images=agg[driver.nm+5:driver.nm+5+7*driver.npix].reshape(7,2,1);tally.category_fluence+=images[:4];tally.photon_count+=images[4];tally.energy_sum_keV+=images[5];tally.energy_sum2_keV2+=images[6];tally.end_batch(n)
        moments.add_batch({'air':float(agg[0])},n);start+=n
    expected=np.zeros((4,2,1));expected[0,0,0]=6;expected[1,1,0]=8;expected[2,0,0]=10;expected[3,1,0]=12
    np.testing.assert_array_equal(tally.category_fluence,expected)
    np.testing.assert_array_equal(tally.photon_count[:,0],[4,4]);np.testing.assert_array_equal(tally.energy_sum_keV[:,0],[16,20]);np.testing.assert_array_equal(tally.energy_sum2_keV2[:,0],[84,120])
    np.testing.assert_array_equal(totals[driver.nm:driver.nm+3],[1,1,8]);assert totals[0]==16 and moments.sums['air']==16
    if uncertainty:
        assert tally.n_batches==len(sizes)
        expected_q=expected**2/10 if len(sizes)==1 else np.array([6.5,10.,14.5,20.])[:,None,None]*np.array([[[1.],[0.]],[[0.],[1.]],[[1.],[0.]],[[0.],[1.]]])
        np.testing.assert_array_equal(tally.category_sum2,expected_q)
        np.testing.assert_array_equal(tally.total_sum2[:,0],[25.6,40.] if len(sizes)==1 else [40.,58.])
        assert moments.sum2s['air']==(25.6 if len(sizes)==1 else 32.)
        np.testing.assert_array_equal(tally.n_batches_hit[:,0],[2,2] if len(sizes)>1 else [1,1])
        if len(sizes)==1:assert np.isnan(tally.total_relative_error()).all()
    else:assert tally.category_sum2 is None and tally.n_batches==0
