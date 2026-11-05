"""A4u S2 deterministic tables, direct relaxation and photon contracts (CPU)."""
import ctypes as ct
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import xraylib as x
from chatcarlo.gpu import core,physics_sets,therapy_scene,validation
from chatcarlo.gpu.generate_photon_tables import element,material,measured
from chatcarlo.kernel import bake_scene_materials


def arrays(names=('air','lead')):
    a=core.pack(bake_scene_materials(list(names)))
    a.update(centers=np.array([[0.,0.,1.]]),halves=np.array([[1.,1.,.1]]),materials=np.array([1],np.int64),background=np.array(0),world=np.array([0.,0.,0.,50.,50.,50.]),detector=np.array([]))
    return a


def test_partial_probabilities_and_domains():
    for z in (1,6,8,29,82):
        e,total,p,edges,y,ck,lines,q,sf,rx,area,upper,report=element(z)
        np.testing.assert_allclose(p.sum(axis=1),1,rtol=1e-14)
        np.testing.assert_allclose((p*total[0,:,None]).sum(axis=1),total[0],rtol=1e-14)
        assert len(report['ranges'])>=4 and all(report['ranges'][k]['high']>=150 for k in ('CS_Photo','CS_Compt','CS_Rayl'))
        assert upper[1]<1000 and len(q)==2000
        for j,fn in enumerate((x.CS_Photo,x.CS_Compt,x.CS_Rayl)):
            for energy in (5.,20.,60.,150.):
                got=np.exp(np.interp(np.log(energy),np.log(e),np.log(total[j])))
                assert got==pytest.approx(fn(z,energy),rel=3e-4)
    report=element(82)[-1]
    assert report['partial_vs_total'][1]['partial_sum']!=pytest.approx(report['partial_vs_total'][1]['photo'],rel=.01)
    assert any(5<=line[1]<10 for line in element(29)[6])


def test_generation_stops_on_missing_required_range():
    def missing(e):
        if e>=5:raise ValueError('missing required datum')
        return 1.
    with pytest.raises(ValueError,match='required datum'):measured(missing,5.)


def relaxation(engine,p,shell,n=6000):
    out=np.zeros((n,38));trace=np.zeros((n,28),np.int64)
    fn=engine.lib.g5_relax_probe;fn.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_uint64,ct.c_void_p,ct.c_void_p]
    fn(p.ctypes.data,0,shell,n,9504002,out.ctypes.data,trace.ctypes.data)
    return out,trace


def test_shell_injection_ck_branches_energy_and_faults(tmp_path):
    a=arrays();p,_=material([82],[1.]);edges=element(82)[3];ys=element(82)[4];ck=element(82)[5]
    cpu=core.Core(a,build_dir=tmp_path/'good')
    out,tr=relaxation(cpu,p,1)
    assert np.all(out[:,37]>=0)
    np.testing.assert_allclose(out[:,36]+out[:,0:36:4].sum(axis=1),edges[1],atol=1e-10)
    # First event identifies the actual competing branch, not only the number
    # of photons (L1->L3 must be distinguishable from L1->L2).
    radiative=tr[:,1]>=0
    assert radiative.mean()==pytest.approx(ys[1],abs=.02)
    for dest in (2,3):
        branch=(~radiative)&(tr[:,2]==dest)
        assert branch.mean()==pytest.approx(ck[1,dest],abs=.02)
    good=((~radiative)&(tr[:,2]==3)).mean()
    for fault in (1,2):
        bad=core.Core(a,build_dir=tmp_path/str(fault),_test_photon_fault=fault)
        _,bt=relaxation(bad,p,1)
        assert ((bt[:,1]<0)&(bt[:,2]==3)).mean()<good-.4
    for shell in range(9):
        o,t=relaxation(cpu,p,shell,100)
        np.testing.assert_allclose(o[:,36]+o[:,0:36:4].sum(axis=1),edges[shell],atol=1e-10)


def test_photon_ledger_escape_spectrum_and_faults(tmp_path):
    a=arrays();src=(np.zeros((96,3)),np.tile([0.,0.,1.],(96,1)),np.full(96,100.))
    good=core.Core(a,build_dir=tmp_path/'good').run(*src,seed=9504003,record=True)
    assert good['completed']==96 and good['created']>96
    assert good['created']==good['endings'].sum()==len(good['terminals'])
    assert good['escaped_spectrum'].sum()==good['endings'][1]
    outgoing=good['terminals'][good['terminals'][:,1]>=-1,3].sum()
    assert good['deposits'].sum()+outgoing==pytest.approx(9600.,rel=1e-12)
    with pytest.raises(ValueError,match='invalid history'):
        core.Core(a,build_dir=tmp_path/'bounded',_test_stack_capacity=1).run(*src,seed=9504003)
    for fault in (3,4,5):
        cpu=core.Core(a,build_dir=tmp_path/str(fault),_test_photon_fault=fault)
        with pytest.raises(ValueError,match='invalid history'):cpu.run(*src,seed=9504003)


def test_two_photons_two_pixels_and_history_moments(tmp_path):
    a=arrays();a['halves'][:]=0.;a['materials']=np.empty(0,np.int64);a['centers']=a['halves']=np.empty((0,3));a['world']=np.array([0.,0.,0.,10.,10.,10.])
    # Vacuum geometry with existing air tables scaled to zero density. This
    # artificial source is compile-time-only and cannot reach the scene schema.
    a['data'][int(a['offsets'][19])]=0.
    a['detector']=np.array([0.,0.,1.,0.,0.,-1.,1.,0.,0.,0.,1.,0.,2.,2.,2.,1.])
    cpu=core.Core(a,build_dir=tmp_path,_test_two_photons=True)
    r=cpu.run(np.zeros((1,3)),np.array([[0.,0.,1.]]),np.array([100.]),seed=9504004,record=True)
    assert r['completed']==1 and r['created']==2 and r['endings'].tolist()==[0.,0.,2.]
    np.testing.assert_array_equal(r['images'][4,:,0],[1.,1.])
    np.testing.assert_allclose(r['images'][5,:,0],[50.,50.]);np.testing.assert_allclose(r['images'][6,:,0],[2500.,2500.])
    assert r['category_count'][3].sum()==2
    one=copy.deepcopy(a);one['detector'][14]=1
    same=core.Core(one,build_dir=tmp_path/'same',_test_two_photons=True).run(np.zeros((1,3)),np.array([[0.,0.,1.]]),np.array([100.]),seed=9504004)
    assert same['images'][4,0,0]==2 and same['images'][6,0,0]==10000.
    rebuilt=validation.record_reduce(r,a)
    for k,v in rebuilt.items():np.testing.assert_allclose(v,r[k],rtol=1e-14,atol=1e-14)


def scene():
    return dict(domain='therapy_mv',source=dict(energy_keV=20,field_cm=[10,10],ssd_cm=100),phantom=dict(material='water',thickness_cm=30),scoring='pdd_ocr',run=dict(histories=100,seed=1,engine='cpu_local'))


def test_entry_source_contract():
    s=scene();assert therapy_scene.validate(s)==s
    for e in (20,60,150,10000):
        s['source']['energy_keV']=e;therapy_scene.validate(s);assert therapy_scene.source_spec(s['source'])['cum']==[1.]
    for e in (19.9,10000.1):
        s['source']['energy_keV']=e
        with pytest.raises(ValueError,match='20'):therapy_scene.validate(s)
    del s['source']['energy_keV']
    s['source']['spectrum']=[dict(low_keV=20,high_keV=150,weight=1),dict(low_keV=150,high_keV=200,weight=2)]
    therapy_scene.validate(s);np.testing.assert_allclose(therapy_scene.source_spec(s['source'])['edges'],[.02,.15,.2])
    s['source']['spectrum'][1]['low_keV']=151
    with pytest.raises(ValueError,match='不連続'):therapy_scene.validate(s)
    s['source']['spectrum']=[dict(low_keV=20+i,high_keV=21+i,weight=1) for i in range(65)]
    with pytest.raises(ValueError,match='64'):therapy_scene.validate(s)
    for name in ('mohan10','mohan4'):
        s['source']['spectrum']=name;therapy_scene.validate(s)
    s['scoring']='H*(10)'
    with pytest.raises(ValueError,match='scoring'):therapy_scene.validate(s)


def test_gate_scaffolds_fail_closed():
    from docs.validation.gpu.g5 import a4u_gates as gates
    old={'s2:deposit:lead':np.ones(10),'s2:category:3:all':np.ones(10)}
    new={k:v.copy() for k,v in old.items()};new['s2:category:3:all']*=2
    assert gates.branches(old,new)['status']=='pass'
    new['s2:deposit:lead']*=2
    assert gates.branches(old,new)['status']=='stop'
    with pytest.raises(ValueError,match='incomplete'):gates.identity({}, {})
    r=np.ones((10,121));bad=r.copy();bad[:,10]*=2
    off=r.copy();off[:,72:86]*=.1
    assert gates.l_bands(r,r)['status']=='pending'
    assert gates.l_bands(r,r,reference_off=off,twin_off=off)['status']=='pass'
    assert gates.l_bands(r,bad,reference_off=off,twin_off=off)['status']=='fail'
    assert all('pending' in v for k,v in gates.pending().items() if k!='divergence_sha256')


def test_electron_driver_low_energy_and_slot_reuse(tmp_path):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'docs/validation/gpu/g5'))
    import a2_common
    from chatcarlo.gpu import therapy
    _,m,a=a2_common.inputs();a=a2_common.no_spectrum(a)
    cpu=therapy.CPU(tmp_path,m,a)
    f=int(sum(a['sizes'][1:4]))
    for e in (.02,.06):
        cpu.g.energy=e
        whole=cpu.histories(24,9504005,slots=3)
        pieces=[cpu.histories(12,9504005,start=start,slots=1) for start in (0,12)]
        np.testing.assert_array_equal(whole['d'],np.concatenate([p['d'] for p in pieces]))
        assert whole['groups'][:,f+6].max()<1e-11
        assert whole['group_ints'][:,:3].sum()==0


def test_primary_transmission_20_and_150(tmp_path):
    a=arrays(('air','water'));a['halves']=np.array([[10.,10.,.05]])
    a['data'][int(a['offsets'][19])]=0.  # air-free deterministic test geometry
    a['detector']=np.array([0.,0.,2.,0.,0.,-1.,1.,0.,0.,0.,1.,0.,40.,40.,1.,1.])
    cpu=core.Core(a,build_dir=tmp_path)
    from chatcarlo.materials import linear_mu
    n=8192
    for energy in (20.,150.):
        r=cpu.run(np.zeros((n,3)),np.tile([0.,0.,1.],(n,1)),np.full(n,energy),seed=9504006,fluorescence=False)
        expected=np.exp(-float(linear_mu('water',energy)[0])*.1)
        observed=r['category_count'][0].sum()/n
        assert abs(observed-expected)<=5*np.sqrt(expected*(1-expected)/n)+.0003


@pytest.mark.parametrize('name', ['lead','water'])
def test_d15_production_fluorescence_direction(name,tmp_path):
    # Fixed before measurement: Pb 100 keV, 0.05 cm foil, 8192 histories.
    # Full shells/CK must increase emitted fluorescence and escaped photons.
    # Water's H/O lines are below 5 keV: both models emit zero transported
    # fluorescence, and toggling fluorescence cannot change production scores.
    a=arrays(('air',name));a['halves']=np.array([[10.,10.,.025]])
    a['data'][int(a['offsets'][19])]=0.
    source=(np.zeros((8192,3)),np.tile([0.,0.,1.],(8192,1)),np.full(8192,100.))
    legacy=core.Core(a,build_dir=tmp_path/'legacy',_test_legacy_physics=True)
    production=core.Core(a,build_dir=tmp_path/'production')
    old=legacy.run(*source,seed=9504015)
    new=production.run(*source,seed=9504015)
    if name=='lead':
        assert new['n_fluorescence']>old['n_fluorescence']
        assert new['endings'][1]>old['endings'][1]
        assert new['created']>new['completed']
    else:
        assert old['n_fluorescence']==new['n_fluorescence']==0
        off=production.run(*source,seed=9504015,fluorescence=False)
        for key in ('deposits','endings','created','escaped_spectrum'):
            np.testing.assert_array_equal(new[key],off[key])
        # Pure oxygen vacancy: all its energy remains local at this cutoff.
        p,_=material([8],[1.]);out,_=relaxation(production,p,0,32)
        assert np.all(out[:,37]==0)
        np.testing.assert_array_equal(out[:,36],np.full(32,x.EdgeEnergy(8,0)))


def test_d15_therapy_photon_cutoff(tmp_path):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'docs/validation/gpu/g5'))
    import a2_common
    from chatcarlo.gpu import therapy
    _,m,a=a2_common.inputs();a=a2_common.no_spectrum(a)
    a['rho']=np.ones(1);a['ids']=np.array([0],np.int64)
    a['faces']=np.array([0.,.00001]);a['bins']=np.array([0.,.00001]);a['owner']=np.array([0],np.int64);a['sizes']=np.array([1,1,150,600,0],np.int64)
    cpu=therapy.CPU(tmp_path,m,a)
    f=int(sum(a['sizes'][1:4]))
    # User decision 2026-11-04 raised the electron-transport cutoff back to 10 keV
    # (G5_ELECTRON_PHOTON_CUT) to match EGS5's AP=0.01 MeV. That reverses the 5 keV
    # value this test was written for, so the probes moved .004/.007 -> .007/.012.
    for e in (.007,.012):
        cpu.g.energy=e;r=cpu.histories(1,9504016,slots=1)
        total=r['groups'][:,f:f+6].sum(axis=0)
        if e==.007:
            assert total[1]==pytest.approx(e) and total[2]==0
        else:
            assert total[2]>0 and total[1]<e
        assert r['groups'][:,f+6].max()<1e-11
