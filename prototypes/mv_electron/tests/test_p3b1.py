"""P3b B1–B6/B9 and review additions; B7/B8 live in run_p3b1."""
import ast
import inspect
import json
from pathlib import Path
import numpy as np
import pytest
from prototypes.mv_electron import photon_transport_p3b as t, photon_transport_p3t as old
from prototypes.mv_electron import run_p3s as s, run_p3a as a
from prototypes.mv_electron import batch_kernels_p3b as k, batch_runner_p3u as runner
from prototypes.mv_electron.scoring_p3b import clip, score_box_bins, score_box_point

REGIONS=[0.,0.,1.,1.,9.,.25,.125,1.,5.,10.,20.]


@pytest.fixture(scope='module')
def mf_inputs():
    from prototypes.mv_electron.p2_common import TABLES
    from prototypes.mv_electron.photon_data_p2 import TABLE
    missing=[str(p) for p in (TABLE,TABLES['mf']) if not p.is_file()]
    if missing:pytest.skip('required photon/mf tables absent: '+', '.join(missing))
    return s.inputs('mf')


@pytest.fixture(autouse=True)
def threads(monkeypatch):
    for key in runner.THREAD_KEYS:monkeypatch.setenv(key,'1')


def healthy(values):
    assert not np.any(values[7])
    assert not np.any(values[8][5:7])
    assert not np.any(values[13][2:4])
    assert values[5][3]==0
    if len(values)>15:assert not np.any(values[20])


def execute(inputs,n=200,seed=20291730,**kw):
    options=dict(annihilate_in_flight=True,score_kerma=False,score_pair=False,**s.rule_kwargs('own'))
    options.update(kw)
    values=t.run_coupled(10.,n,seed,.05,.025,30.,*inputs,**options)
    healthy(values)
    return values


def same(left,right):
    assert len(left)==len(right)
    for x,y in zip(left,right):
        x=np.asarray(x);y=np.asarray(y)
        assert x.shape==y.shape and x.dtype==y.dtype and x.tobytes()==y.tobytes()


@pytest.mark.parametrize('case',[dict(brems_mode=1),dict(brems_mode=2),
    dict(charged_source_T=10.,charged_source_charge=-1),
    dict(charged_source_T=10.,charged_source_charge=1),dict(spectrum=True)])
def test_b1_legacy_bytes(mf_inputs,case):
    case=case.copy()
    if case.pop('spectrum',False):
        case.update(spectrum_edges=np.array([.01,1.,5.,10.]),spectrum_cum=np.array([.2,.6,1.]))
    args=(10.,200,20291730,.05,.025,30.,*mf_inputs)
    kw=dict(annihilate_in_flight=True,score_lateral=False,**s.rule_kwargs('own'),**case)
    baseline=old.run_coupled(*args,**kw);healthy(baseline)
    plain=t.run_coupled(*args,**kw);healthy(plain)
    full=t.run_coupled(*args,regions=REGIONS,record_deposits=2,record_sources=2,**kw);healthy(full)
    same(baseline,plain[:15]);same(baseline,full[:15]);same(plain[15:21],t.run_coupled(*args,**kw)[15:21])
    assert full[22]>0 and full[24]==198
    if case.get('charged_source_charge')==1:assert baseline[8][14]>0


def test_b1_recording_does_not_change_regions(mf_inputs):
    plain=execute(mf_inputs,regions=REGIONS,field_half=5.,ssd=100.)
    recorded=execute(mf_inputs,regions=REGIONS,field_half=5.,ssd=100.,record_deposits=3,record_sources=3)
    same(plain[:21],recorded[:21]);assert recorded[22]>0


def python_run(inputs,n=1,E=1.,**kw):
    from prototypes.mv_electron.bremsstrahlung_p3r import tables
    energies,values=tables()
    options=dict(brems_energies=energies,brems_values=values,record_deposits=2000000,
                 score_kerma=False,score_pair=False)
    options.update(kw)
    out=t._run_coupled.py_func(E,n,20291731,.05,.025,30.,*inputs,**options)
    healthy(out);assert out[22]==0
    return out


def test_b2_completed_children_inherit_parent(mf_inputs,monkeypatch):
    counts={}
    def wrap(function):
        def call(*args,**kwargs):
            frame=inspect.currentframe().f_back
            line=inspect.getframeinfo(frame).code_context[0].strip()
            result=function.py_func(*args,**kwargs)
            if 'stack[0]' not in line and 'rule_value(charged_source_T' not in line:
                parent=frame.f_locals['a']
                np.testing.assert_array_equal(result[:3],parent[:3])
                counts[line]=counts.get(line,0)+1
            return result
        return call
    monkeypatch.setattr(t,'initial',wrap(t.initial));monkeypatch.setattr(t,'photon_state',wrap(t.photon_state))
    python_run(mf_inputs,n=300,E=10.,source_offset=(1.3,-.7),annihilate_in_flight=True)
    # At least once at EACH constructor site, including both pair/annihilation children.
    expected=('photon=photon_state(ep','child=initial(T,','child=initial(E,',
        'first=initial(p[0]','second=initial(p[5]','child=initial(W,',
        'photon_state(photon_k','photon_state(g[0]','photon_state(g[4]',
        'photon_state(M,a[2],ux','photon_state(M,a[2],-ux')
    for site in expected:assert any(site in line and count>0 for line,count in counts.items()),site


def test_b3_translation_and_b3c_scores(mf_inputs):
    base=execute(mf_inputs,regions=REGIONS,field_half=5.,ssd=100.,dz=.25,dz_fine=.1,record_deposits=1000000)
    moved_regions=REGIONS.copy();moved_regions[:2]=[1.3,-.7]
    moved=execute(mf_inputs,regions=moved_regions,field_half=5.,ssd=100.,dz=.25,dz_fine=.1,
                  source_offset=(1.3,-.7),record_deposits=1000000)
    assert base[22]==moved[22]==0
    x=base[21];y=moved[21];assert x.shape==y.shape
    same((x[:,[0,1,4,7,8]],),(y[:,[0,1,4,7,8]],))
    np.testing.assert_allclose(y[:,[2,5]],x[:,[2,5]]+1.3,rtol=0,atol=1e-12)
    np.testing.assert_allclose(y[:,[3,6]],x[:,[3,6]]-.7,rtol=0,atol=1e-12)
    for left,right in zip(base[15:20],moved[15:20]):
        np.testing.assert_allclose(left,right,rtol=0,atol=1e-12*np.max(left))


def test_b3b_fixed_oblique_photon_flight(mf_inputs,monkeypatch):
    direction=np.array([.3,.4,np.sqrt(.75)]);origin=np.array([1.3,-.7,.5]);energy=.0101;length=.125
    source=t.photon_state(energy,origin[2],*direction,0,*origin[:2])
    monkeypatch.setattr(np.random,'random',lambda:np.exp(-length))
    monkeypatch.setattr(t,'coefficients',lambda *args:np.array([1.,0.,0.,0.]))
    monkeypatch.setattr(t,'compton',lambda E,*args:(.0099,*direction,E-.0099,*direction))
    out=python_run(mf_inputs,E=energy,source_state=source)
    points=out[21][out[21][:,1]==1]
    assert len(points)==2
    np.testing.assert_allclose(points[:,2:5],np.tile(origin+length*direction,(2,1)),rtol=0,atol=1e-14)


GEOMETRY=[
    ((-1.,.5,-1.),(3.,.5,3.),2,[1.,1.],(.25,.75)), # diagonal, outside to outside
    ((-1.,.5,.5),(3.,.5,.5),0,[1.,1.],(.25,.75)), # depth displacement zero
    ((3.,.5,.5),(-1.,.5,.5),0,[1.,1.],(.25,.75)), # reverse
    ((-1.,-1.,-1.),(3.,3.,3.),2,[1.,1.],(.25,.75)), # corners
    ((.25,.5,.25),(.75,.5,.75),2,[4.,0.],(0.,1.)), # wholly inside
    ((0.,.5,-1.),(0.,.5,3.),2,[1.,1.],(.25,.75)), # lower face
    ((2.,.5,-1.),(2.,.5,3.),2,[0.,0.],(1.,0.)), # excluded upper face
    ((.5,.5,.5),(.5,.5,.5),2,[4.,0.],(0.,1.)), # point
    ((.5,.5,0.),(.5,.5,1.),2,[4.,0.],(0.,1.)), # exact bin endpoint
    ((3.,3.,0.),(4.,4.,1.),2,[0.,0.],None), # miss
    ((-1.,1.,0.),(1.,-1.,0.),2,[0.,0.],(.5,.5)), # corner contact only
]


@pytest.mark.parametrize('p0,p1,axis,expected,interval',GEOMETRY)
def test_b4_hand_geometry(p0,p1,axis,expected,interval):
    p0=np.array(p0);p1=np.array(p1);lo=np.zeros(3);hi=np.full(3,2.)
    t0,t1=clip(p0,p1,lo,hi)
    if interval is not None:np.testing.assert_allclose((t0,t1),interval,rtol=0,atol=1e-13)
    else:assert t0>=t1
    score=np.zeros(2);total=score_box_bins(score,p0,p1,4.,axis,0.,1.,lo,hi)
    np.testing.assert_allclose(score,expected,rtol=0,atol=1e-13)
    assert abs(total-sum(expected))<=1e-13
    split=np.zeros(2)
    for lower,upper in ((0.,1.),(1.,2.)):
        ll=lo.copy();hh=hi.copy();ll[0]=lower;hh[0]=upper
        score_box_bins(split,p0,p1,4.,axis,0.,1.,ll,hh)
    np.testing.assert_allclose(split,score,rtol=0,atol=1e-13)


@pytest.mark.parametrize('point,expected',[( [0.,0.,0.],[4.,0.]),([1.,.5,1.],[0.,4.]),
    ([2.,.5,1.],[0.,0.]),([.5,.5,2.],[0.,0.]),([-.1,.5,.5],[0.,0.])])
def test_b4_points(point,expected):
    score=np.zeros(2)
    amount=score_box_point(score,np.array(point),4.,2,0.,1.,np.zeros(3),np.full(3,2.))
    np.testing.assert_allclose(score,expected,rtol=0,atol=1e-13);assert amount==sum(expected)


def independent_bins(records,n,lo,hi,axis,origin,width,size):
    """Each bin is an independent box, vectorized over records (no scorer calls)."""
    out=np.zeros((100,size));p=records[:,2:5];q=records[:,5:8];d=q-p
    point=(records[:,1]==1)|np.all(d==0,axis=1)
    batch=np.minimum(99,records[:,0].astype(int)*100//n)
    for i in range(size):
        low=np.array(lo,float);high=np.array(hi,float)
        low[axis]=max(low[axis],origin+i*width);high[axis]=min(high[axis],origin+(i+1)*width)
        tmin=np.zeros(len(records));tmax=np.ones(len(records));valid=np.ones(len(records),bool)
        for j in range(3):
            moving=d[:,j]!=0;still=~moving
            valid[still]&=(p[still,j]>=low[j])&(p[still,j]<high[j])
            v0=(low[j]-p[moving,j])/d[moving,j];v1=(high[j]-p[moving,j])/d[moving,j]
            tmin[moving]=np.maximum(tmin[moving],np.minimum(v0,v1))
            tmax[moving]=np.minimum(tmax[moving],np.maximum(v0,v1))
        fraction=np.where(valid,np.maximum(0.,tmax-tmin),0.)
        inside=np.all((p>=low)&(p<high),axis=1)
        fraction[point]=inside[point].astype(float)
        np.add.at(out[:,i],batch,records[:,8]*fraction)
    return out


def independent_regions(records,n,r,dz=.25,dz_fine=.1):
    cx,cy,half,orth,extent,coarse,fine,layer=r[:8];depths=r[8:]
    axis=[independent_bins(records,n,[cx-half,cy-half,0.],[cx+half,cy+half,30.],2,0.,w,int(round(30/w)))
          for w in (dz,dz_fine)]
    slab=independent_bins(records,n,[-np.inf,-np.inf,0.],[np.inf,np.inf,30.],2,0.,dz,int(round(30/dz)))
    profiles=[]
    for width in (coarse,fine):
        profile=np.zeros((100,2,len(depths),int(round(2*extent/width))))
        for j in range(2):
            for z,depth in enumerate(depths):
                low=[cx-orth,cy-orth,depth-layer/2];high=[cx+orth,cy+orth,depth+layer/2]
                center=(cx,cy)[j];low[j]=center-extent;high[j]=center+extent
                profile[:,j,z]=independent_bins(records,n,low,high,j,center-extent,width,profile.shape[-1])
        profiles.append(profile)
    return axis+[slab]+profiles


@pytest.fixture(scope='module')
def production_record():
    from prototypes.mv_electron.p2_common import TABLES
    from prototypes.mv_electron.run_p3a import SPECTRUM,load_spectrum
    from prototypes.mv_electron.photon_data_p2 import TABLE
    missing=[str(p) for p in (TABLES['elsepa'],SPECTRUM,TABLE) if not p.is_file()]
    if missing:pytest.skip('B5 production inputs absent: '+', '.join(missing))
    edges,prob,cum=load_spectrum()
    return execute(s.inputs('elsepa'),n=300,regions=REGIONS,field_half=5.,ssd=100.,
        dz=.25,dz_fine=.1,spectrum_edges=edges,spectrum_cum=cum,record_deposits=1000000)


def test_b5_production_independent_recalculation(production_record):
    out=production_record;assert out[22]==0
    assert out[21][:,8].sum()==pytest.approx(out[5][1],rel=1e-12)
    for actual,expected in zip(out[15:20],independent_regions(out[21],300,REGIONS)):
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-9*np.max(actual))


def test_b6_finite_box_and_complement(mf_inputs):
    out=execute(mf_inputs,n=300,regions=REGIONS,dz=.25,dz_fine=.1,record_deposits=1000000)
    records=out[21];assert out[22]==0
    half=float(np.max(np.abs(records[:,[2,3,5,6]])))+1.
    # Establish containment before claiming equality with whole-slab scoring.
    assert np.all(np.abs(records[:,[2,3,5,6]])<half)
    assert np.all(records[:,[4,7]]>=0) and np.all(records[:,[4,7]]<=30.+1e-12)
    large=REGIONS.copy();large[2]=half
    all_inside=execute(mf_inputs,n=300,regions=large,dz=.25,dz_fine=.1)
    np.testing.assert_allclose(all_inside[15],all_inside[17],rtol=1e-12,atol=0)
    # An axis box plus four disjoint outside boxes partition the slab.
    boxes=[([-1.,-1.,0.],[1.,1.,30.]),
        ([-np.inf,-np.inf,0.],[-1.,np.inf,30.]),([1.,-np.inf,0.],[np.inf,np.inf,30.]),
        ([-1.,-np.inf,0.],[1.,-1.,30.]),([-1.,1.,0.],[1.,np.inf,30.])]
    split=sum(independent_bins(records,300,lo,hi,2,0.,.25,120) for lo,hi in boxes)
    np.testing.assert_allclose(split,out[17],rtol=1e-12,atol=0)


def controlled_charged(monkeypatch,energy=1.,z=.1,charge=-1,direction=(.6,0.,.8)):
    monkeypatch.setattr(t,'collision',lambda T:1.)
    monkeypatch.setattr(t,'hard',lambda T:(0.,0.))
    monkeypatch.setattr(t,'table_rates',lambda *args:(1.,1.))
    monkeypatch.setattr(t,'quantities',lambda *args:(0.,0.,0.,0.))
    monkeypatch.setattr(t,'new_remaining',lambda *args:1e6)
    source=np.zeros(t.FIELDS);source[:7]=[.3,-.2,z,*direction,energy]
    source[7]=max(.01,energy);source[8]=10.;source[12]=.2;source[14]=.025;source[15]=1e6;source[19]=charge
    return source


def test_b5b_hinge_records_before_rotation(mf_inputs,monkeypatch):
    source=controlled_charged(monkeypatch)
    hinges=[]
    def hinge(state,*args):
        hinges.append(state[:3].copy());state[3:6]=[0.,.6,.8];state[12]=1e6
    monkeypatch.setattr(t,'scattering_hinge',hinge)
    out=python_run(mf_inputs,source_state=source)
    segments=out[21][out[21][:,1]==0]
    assert len(segments)==2 and len(hinges)==1
    np.testing.assert_allclose(segments[0,2:5],source[:3],rtol=0,atol=1e-14)
    np.testing.assert_allclose(segments[0,5:8],source[:3]+.2*source[3:6],rtol=0,atol=1e-14)
    np.testing.assert_array_equal(segments[0,5:8],hinges[0]);np.testing.assert_array_equal(segments[1,2:5],hinges[0])
    np.testing.assert_allclose(segments[1,5:8],hinges[0]+.8*np.array([0.,.6,.8]),rtol=0,atol=1e-14)
    assert out[21][:,8].sum()==pytest.approx(out[5][1],rel=1e-12)


@pytest.mark.parametrize('z,direction',[(29.9,(.6,0.,.8)),(0.,(.6,0.,-.8))])
def test_b5b_boundary_and_escape(mf_inputs,monkeypatch,z,direction):
    source=controlled_charged(monkeypatch,z=z,direction=direction);source[12]=1e6
    out=python_run(mf_inputs,source_state=source)
    records=out[21];assert len(records)==1 and records[0,1]==0
    step=(30.-z)/direction[2] if direction[2]>0 else -z/direction[2]
    raw=source[:3]+step*np.array(direction)
    np.testing.assert_allclose(records[0,5:8],raw,rtol=0,atol=1e-12)
    assert abs(records[0,7]-(30. if direction[2]>0 else 0.))<=1e-12
    assert records[0,8]==pytest.approx(step,abs=1e-14)
    assert records[:,8].sum()==pytest.approx(out[5][1],rel=1e-12,abs=0)
    assert out[5][2]==pytest.approx(1.-step,rel=1e-12)


@pytest.mark.parametrize('charge',[0,-1,1])
def test_b5b_cutoff_points_and_at_rest(mf_inputs,monkeypatch,charge):
    energy=.005;source=controlled_charged(monkeypatch,energy=energy,z=15.,charge=charge)
    # Rest photons escape; only kinetic cutoff is a deposit at the parent position.
    monkeypatch.setattr(t,'coefficients',lambda *args:np.array([1e-30,0.,0.,0.]))
    monkeypatch.setattr(t,'annihilation',lambda:(0.,0.,1.))
    out=python_run(mf_inputs,E=energy,source_state=source)
    assert len(out[21])==1 and out[21][0,1]==1
    np.testing.assert_array_equal(out[21][0,2:5],source[:3])
    assert out[21][0,8]==energy
    if charge==1:assert out[8][12]==1 and out[5][4]==0
    assert out[21][:,8].sum()==pytest.approx(out[5][1],rel=1e-12)


def test_b5b_local_brems_and_endpoint(mf_inputs,monkeypatch):
    source=controlled_charged(monkeypatch);source[12]=1e6
    monkeypatch.setattr(t,'quantities',lambda *args:(1000.,0.,0.,1.))
    monkeypatch.setattr(t,'new_remaining',lambda *args:.1)
    monkeypatch.setattr(t,'sample_k',lambda T,*args:(T-.005,0))
    out=python_run(mf_inputs,source_state=source,brems_mode=1)
    records=out[21];points=records[records[:,1]==1]
    assert out[13][0]==1 and len(points)==2
    expected=source[:3]+.0001*source[3:6]
    np.testing.assert_allclose(points[:,2:5],np.tile(expected,(2,1)),rtol=0,atol=1e-14)
    assert points[0,8]==pytest.approx(1.-.0001-.005,abs=1e-14)
    assert points[1,8]==pytest.approx(.005,abs=1e-14)
    assert records[:,8].sum()==pytest.approx(out[5][1],rel=1e-12)


def field_params():
    return dict(E=10.,backend='mf',brems_mode=2,spectrum=None,field_half=5.,ssd=100.,
                dz=.25,dz_fine=.1,regions=REGIONS.copy())


def batch_run(work,**kw):
    options=dict(kernel='p3b_field',params=field_params(),N=400,seed=20291732,workers=1,nbatch=4)
    options.update(kw);return runner.run_batches(work,**options)


def test_b9_parallel_direct_and_b9b_resume(tmp_path,mf_inputs):
    full=tmp_path/'full';parallel=tmp_path/'parallel';resume=tmp_path/'resume'
    batch_run(full);batch_run(parallel,workers=2)
    assert batch_run(resume,stop_after=2) is None
    batch_run(resume,workers=2)
    assert runner.same(a.load(full/'result.npz'),a.load(parallel/'result.npz'))
    assert runner.same(a.load(full/'result.npz'),a.load(resume/'result.npz'))
    for i,seed in enumerate(runner.batch_seeds(20291732,4)):
        direct=k.reduce(execute(mf_inputs,n=100,seed=seed,field_half=5.,ssd=100.,
            dz=.25,dz_fine=.1,regions=REGIONS))
        row=a.load(full/f'batch_{i:03}.npz');assert k.healthy(row)
        assert runner.same(direct,{key:row[key] for key in k.KEYS},skip=())


@pytest.fixture
def stopped(tmp_path,mf_inputs):
    batch_run(tmp_path,stop_after=1)
    return tmp_path


@pytest.mark.parametrize('name',k.SOURCES)
def test_b9b_hash_reject(stopped,monkeypatch,name):
    original=k.source_hashes
    def changed():return dict(original(),**{name:'changed'})
    monkeypatch.setattr(k,'source_hashes',changed)
    with pytest.raises(RuntimeError,match='config mismatch'):batch_run(stopped)


@pytest.mark.parametrize('key',['regions','dz','dz_fine','field_half','ssd'])
def test_b9b_params_reject(stopped,key):
    params=field_params()
    if key=='regions':params[key]=[.1]+params[key][1:]
    else:params[key]={'dz':.5,'dz_fine':.2,'field_half':4.,'ssd':90.}[key]
    with pytest.raises(RuntimeError,match='config mismatch'):batch_run(stopped,params=params)


def test_b9b_unhealthy_not_saved(tmp_path,mf_inputs,monkeypatch):
    from prototypes.mv_electron import batch_kernels_p3u as registry
    original=registry.KERNELS['p3b_field']
    def unhealthy(row):
        row['region_ledger'][0]=1
        return original.healthy(row)
    monkeypatch.setitem(registry.KERNELS,'p3b_field',original._replace(healthy=unhealthy))
    with pytest.raises(RuntimeError,match='unhealthy output'):batch_run(tmp_path)
    assert not (tmp_path/'batch_000.npz').exists()
    assert json.loads((tmp_path/'terminal.json').read_text())['reason']=='unhealthy output'


@pytest.mark.parametrize('options',[dict(brems_mode=0),dict(field_half=5.),dict(ssd=100.),
    dict(field_half=0.,ssd=100.),dict(field_half=5.,ssd=np.inf),dict(field_half=np.nan,ssd=100.),
    dict(field_half=5.,ssd=100.,charged_source_T=1.),dict(field_half=5.,ssd=100.,positron_source_T=1.),
    dict(field_half=5.,ssd=100.,source_state=np.zeros(23)),dict(record_deposits=-1),dict(record_sources=-1),
    dict(regions=[0.]*8),dict(regions=[0.,0.,1.,1.,9.,.25,.125,1.,np.nan]),
    dict(regions=[0.,0.,0.,1.,9.,.25,.125,1.,5.]),
    dict(regions=[0.,0.,1.,1.,9.,.4,.125,1.,5.]),dict(regions=[0.,0.,1.,1.,9.,.25,.125,1.,30.]),
    dict(dz=.7),dict(dz_fine=0.)])
def test_argument_validation(mf_inputs,options):
    with pytest.raises(ValueError):execute(mf_inputs,n=1,**options)


def test_no_deletion_and_lazy_registration():
    for name in (*k.SOURCES,'run_p3b1.py'):
        tree=ast.parse((a.CODE/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                function=node.func
                assert not isinstance(function,ast.Name) or function.id not in {'unlink','rmtree','remove','rmdir','removedirs'}
                assert not isinstance(function,ast.Attribute) or function.attr not in {'unlink','rmtree','remove','rmdir','removedirs'}
    from prototypes.mv_electron import batch_kernels_p3u as registry
    assert registry.KERNELS['p3b_field'].run is k.run


@pytest.mark.parametrize('key,value',[('field_half',None),('ssd',None),('field_half','invalid')])
def test_batch_source_validation(key,value):
    params=field_params();params[key]=value
    with pytest.raises(ValueError,match='invalid field source'):k.check_params(params,100)
