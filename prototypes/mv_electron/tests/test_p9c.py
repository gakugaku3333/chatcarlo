"""P9c CPU gates and Colab fail-closed orchestration (no cache writes)."""
import json
import numpy as np
import pytest
from prototypes.mv_electron.gpu import p9c,p9c_core as core,p9c_checks as checks
from prototypes.mv_electron.gpu.p9c_stats import Moments,Checkpoint,normalize
from prototypes.mv_electron.gpu.p9c_validation import DOORS,v5,v6b

@pytest.fixture(scope='module')
def build(tmp_path_factory):return tmp_path_factory.mktemp('p9c')
@pytest.fixture(scope='module')
def inputs():
    from prototypes.mv_electron import p9a_data
    return p9a_data.load_bundles()
@pytest.fixture(scope='module')
def probe(build):
    return p9c.CPU(build,densities=[1,1,1,1,2.47],grids=(checks.SMALL,checks.SMALL),cylinders=((2,2,2,.05,.25,4),))

def test_v0(probe):assert checks.v0(probe)['status']=='pass'
def test_v1a(probe):assert checks.v1a(probe)['status']=='pass'
def test_v2a(probe):assert checks.v2a(probe)['status']=='pass'
def test_v3(probe):assert checks.v3(probe)['status']=='pass'
def test_v6(build,inputs):
    _,identity=inputs;e=p9c.CPU(build,densities=identity['densities'],grids=(checks.SMALL,checks.SMALL))
    assert checks.v6(e)['crossings']==18

def test_v5(build,inputs):
    materials,identity=inputs
    assert v5(build,materials,identity['densities'])['cpu_pair'].startswith('exact')

def test_v6b():assert v6b()['cases']==8

@pytest.mark.parametrize('beam,angle', [('rectangle',0),('prostate',45)])
def test_small_transport(build,inputs,beam,angle):
    materials,identity=inputs
    g=core.Grid((-20.,)*3,(2.,)*3,(20,)*3)
    engine=p9c.CPU(build,materials,densities=identity['densities'],grids=(g,g),beam=beam,angle=angle)
    a=engine.run(20,940100,field=core.ANGLES.index(angle))
    np.testing.assert_allclose(a['grids'][1].sum(),a['histories']['dep'].sum(),rtol=1e-12)
    assert not a['source_counts'][:,1].any()
    assert bool(a['source_counts'][:,0].sum())==(beam=='prostate')
    assert (a['sites'].sum(0)>0).sum()>=3
    first=a['grids'][0].copy();second=engine.run(20,940100,field=core.ANGLES.index(angle))
    np.testing.assert_array_equal(first,second['grids'][0])


def test_statistical_formulas():
    rng=np.random.default_rng(1);a=rng.normal(size=(20,3));b=.3*a+rng.normal(size=(20,3))+3
    m=Moments((3,))
    for x,y in zip(a,b):m.add(x,y)
    r=m.result();n=len(a);va=a.var(0,ddof=1)/n;vb=b.var(0,ddof=1)/n;cov=((a-a.mean(0))*(b-b.mean(0))).sum(0)/(n-1)/n;ratio=a.mean(0)/b.mean(0)
    np.testing.assert_allclose(r['ratio_sem'],np.sqrt(va+ratio**2*vb-2*ratio*cov)/abs(b.mean(0)),rtol=1e-12)
    np.testing.assert_allclose(r['correlation'],cov/np.sqrt(va*vb),rtol=1e-12)
    np.testing.assert_allclose(normalize(np.arange(7)[:,None]*np.ones((7,3)),[1,2,3,4,5,6,7]),np.full(3,21/28))


def test_checkpoint_identity_and_atomicity(tmp_path):
    path=tmp_path/'pair.npz';c=Checkpoint(path,{'source':'A'},{'x':(2,)})
    with pytest.raises(ValueError,match='incomplete'):c.commit(0,{'x':(np.ones(2),np.ones(2))},DOORS-{(1,False)})
    assert not path.exists()
    c.commit(0,{'x':(np.ones(2),np.ones(2))},DOORS);digest=c.digest
    with pytest.raises(ValueError,match='shape'):c.commit(1,{'x':(np.ones(3),np.ones(3))},DOORS)
    assert c.digest==digest
    with pytest.raises(ValueError,match='identity'):Checkpoint(path,{'source':'B'},{'x':(2,)})
    with pytest.raises(ValueError,match='replay'):c.commit(0,{'x':(np.ones(2),np.ones(2))},DOORS)


def test_half_open_and_masks():
    assert core.FINE.index((0,-.2,0))==(120,92,100)
    assert core.FINE.index((0,-.9,0))[1]==64
    grid=core.Grid((-1.,)*3,(1.,)*3,(2,)*3)
    assert grid.index((0,0,0))==(1,1,1)
    with pytest.raises(ValueError):grid.index((1,0,0))
    np.testing.assert_array_equal(core.water_mask(grid,cylinders=()),np.ones((2,2,2),bool))
    body=core.water_mask(core.Grid((-18.,-15.,-12.),(36.,30.,24.),(1,1,1)),coarse=True,cylinders=())
    assert not body.any()


def test_invalid_geometry():
    rho=[1,1,1,1,2.47]
    for kw in (dict(beam='invalid'),dict(radii=(0,15,12)),dict(angle=float('nan')),dict(cylinders=((0,0,0,.05,.25,4),(0,0,.1,.05,.25,4))),dict(cylinders=((18,0,0,.05,.25,4),))):
        with pytest.raises(ValueError):core.geometry(rho,**kw)
    for f in range(7):assert p9c.field_seed(1,f)==8+f
    with pytest.raises(ValueError):p9c.field_seed(2**61,0)


def test_colab_modes_and_gate_failure(tmp_path,monkeypatch):
    import sys
    from prototypes.mv_electron.gpu import p9c_colab as c,p9c_io as io,p9c_gates as gates
    with pytest.raises(ValueError,match='production requires'):c.run(tmp_path,tmp_path,'production')
    with pytest.raises(ValueError,match='T0 fixes'):c.run(tmp_path,tmp_path,'t0',n=100)
    monkeypatch.setitem(sys.modules,'cupy',object());monkeypatch.setattr(io,'load',lambda _:({'sources':{},'seed':940100},[]))
    calls=[]
    def reject(*args):raise AssertionError('V4 failed')
    monkeypatch.setattr(c,'initial_gates',reject);monkeypatch.setattr(c,'run_pairs',lambda *a:calls.append(a))
    with pytest.raises(AssertionError,match='V4'):c.run(tmp_path,tmp_path/'output','t0')
    assert not calls and not (tmp_path/'output/job.json').exists()


def test_colab_markers_and_production_gate(tmp_path,monkeypatch,capsys):
    import sys
    from prototypes.mv_electron.gpu import p9c_colab as c,p9c_io as io,p9c_gates as gates
    monkeypatch.setitem(sys.modules,'cupy',object());monkeypatch.setattr(io,'load',lambda _:({'sources':{},'seed':940100},[]))
    monkeypatch.setattr(c,'initial_gates',lambda *a:{name:{'status':'pass'} for name in c.INITIAL_GATES})
    calls=[];monkeypatch.setattr(c,'run_pairs',lambda *a:calls.append(a[4]) or {b:{'batches':100} for b in a[7]})
    assert c.run(tmp_path,tmp_path/'t0','t0')['mode']=='t0'
    assert calls==[100000] and 'P9C_DONE' in capsys.readouterr().out
    def reject(*a):raise AssertionError('V1c failed')
    monkeypatch.setattr(gates,'production_gates',reject)
    with pytest.raises(AssertionError,match='V1c'):c.run(tmp_path,tmp_path/'main','production',n=500,t0=tmp_path/'t0')
    assert calls==[100000]


def test_sections_water_and_centers():
    from types import SimpleNamespace
    from prototypes.mv_electron.gpu.p9c_io import sections
    g=core.Grid((-3.,-2.5,-2.5),(.5,)*3,(12,10,15));m=Moments(g.shape);m.add(np.ones(g.shape),np.full(g.shape,2.));m.add(np.ones(g.shape),np.full(g.shape,2.))
    r=sections(SimpleNamespace(moments={'fine':m}),g)
    assert len(r)==5
    valid=r['yz_x_0']['valid'];assert np.all(r['yz_x_0']['ratio'][valid]==.5)
    assert np.isnan(r['yz_x_0']['ratio'][~valid]).all()
    assert r['yz_x_0']['x_cm']==.25


def test_random_segment_clipping(probe):
    rng=np.random.default_rng(9)
    for _ in range(40):
        a,b=rng.uniform(-4,4,(2,3));probe.clear();probe.score(a,b,loss=.7)
        expected=checks.expected_segment(checks.SMALL,a,b,.7)
        for actual in probe.values:
            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=0)
            assert np.all(actual[expected==0]==0)


@pytest.mark.parametrize('beams',[('rectangle',),('prostate',),('rectangle','prostate')])
def test_batch_commit_markers_and_resume(tmp_path,monkeypatch,capsys,beams):
    import builtins
    from types import SimpleNamespace
    from prototypes.mv_electron.gpu import p9c_colab as c
    fakecp=SimpleNamespace(asnumpy=np.asarray,get_default_memory_pool=lambda:SimpleNamespace(total_bytes=lambda:0),cuda=SimpleNamespace(runtime=SimpleNamespace(memGetInfo=lambda:(1,2),getDevice=lambda:0)))
    fail=[True];calls=[]
    class FakeGPU:
        def __init__(self,cp,materials,**kw):
            self.side=kw['with_seeds'];self.grids=kw['grids'];self.g=SimpleNamespace(angle=0,beam=0)
        def upload(self):pass
        def run(self,n,seed,start=0,field=0):
            calls.append((self.side,field,start,n))
            if fail[0] and field==6 and not self.side:raise RuntimeError('interrupted last door')
            return dict(grids=[np.full(g.shape,n*(2 if self.side else 1),float) for g in self.grids],source_counts=np.zeros((n,2),np.int64),seconds=.1)
    monkeypatch.setattr(c,'GPU',FakeGPU)
    printed=[]
    def logged_print(*args,**kw):
        printed.append((args,kw));builtins.print(*args,**kw)
    monkeypatch.setattr(c,'print',logged_print,raising=False)
    monkeypatch.setattr(c,'MemoryMonitor',lambda _:SimpleNamespace(start=lambda:None,stop=lambda:None,report=lambda:{}))
    from prototypes.mv_electron.gpu import p9c_gates as gates
    monkeypatch.setattr(gates,'source_acceptance',lambda *a:{'status':'pass'})
    g=core.Grid((-3.,-2.5,-2.5),(.5,)*3,(12,10,15));record={'sources':{},'bundles':{},'densities':[1]*5,'edges':[0.,1.],'cum':[1.]}
    with pytest.raises(RuntimeError):c.run_pairs(fakecp,[],record,tmp_path,100,(g,g),1,beams)
    assert not (tmp_path/(beams[0]+'.npz')).exists()
    fail[0]=False;calls.clear();result=c.run_pairs(fakecp,[],record,tmp_path,100,(g,g),1,beams)
    assert len(calls)==len(beams)*100*14
    markers=capsys.readouterr().out.splitlines()
    batches=[x for x in markers if x.startswith('P9C_BATCH ')]
    heartbeats=[x for x in markers if x.startswith('P9C_HEARTBEAT ')]
    assert len(batches)==len(beams)*100 and len(heartbeats)==len(beams)*100*14+13
    assert heartbeats[:2]==[f'P9C_HEARTBEAT {beams[0]} 0 0 1',f'P9C_HEARTBEAT {beams[0]} 0 0 0']
    assert heartbeats[-1]==f'P9C_HEARTBEAT {beams[-1]} 99 6 0'
    assert all(kw.get('flush') is True for args,kw in printed)
    assert set(result)==set(beams)
    for beam in beams:assert result[beam]['aggregate']['ratio']==[2.,2.]
    for beam in set(c.BEAMS)-set(beams):assert not (tmp_path/(beam+'.npz')).exists()
    calls.clear();c.run_pairs(fakecp,[],record,tmp_path,100,(g,g),1,beams)
    assert not calls and not capsys.readouterr().out


def test_rectangle_miss_stops(build,inputs):
    materials,identity=inputs;g=core.Grid((-20.,)*3,(10.,)*3,(4,)*3)
    engine=p9c.CPU(build,materials,densities=identity['densities'],radii=(.1,.1,.1),cylinders=(),grids=(g,g))
    with pytest.raises(AssertionError,match='rectangle body miss'):engine.run(10,940100)


def test_returned_failed_gate_also_stops(tmp_path,monkeypatch):
    import sys
    from prototypes.mv_electron.gpu import p9c_colab as c,p9c_io as io
    monkeypatch.setitem(sys.modules,'cupy',object());monkeypatch.setattr(io,'load',lambda _:({'sources':{},'seed':940100},[]))
    monkeypatch.setattr(c,'initial_gates',lambda *a:{'V1b':{'status':'fail'}})
    calls=[];monkeypatch.setattr(c,'run_pairs',lambda *a:calls.append(a))
    with pytest.raises(AssertionError,match='V1b failed'):c.run(tmp_path,tmp_path/'output','t0')
    assert not calls


def test_coarse_cell_entirely_inside_touching_cap():
    g=core.Grid((-.1,14.8,-.1),(.2,.2,.2),(1,1,1))
    assert core.water_mask(g,coarse=True,cylinders=()).all()
    outside=core.Grid((-.1,15.,-.1),(.2,.2,.2),(1,1,1))
    assert not core.water_mask(outside,coarse=True,cylinders=()).any()


def test_cuda_source_host_syntax(tmp_path):
    import shutil
    import subprocess
    from prototypes.mv_electron.tests.test_p8g2_contracts import STUB
    path=tmp_path/'p9c_cuda_syntax.cpp';path.write_text(STUB+'\nDim gridDim;\n'+core.cuda_source())
    result=subprocess.run([shutil.which('clang++'),'-std=c++11','-ffp-contract=off','-fsyntax-only',str(path)],text=True,capture_output=True)
    assert result.returncode==0,result.stderr


def test_decimal_face_scoring_matches_extraction(build):
    grid=core.Grid((-3.,-2.5,-2.5),(.025,)*3,(1,100,1))
    engine=p9c.CPU(build,densities=[1,1,1,1,2.47],grids=(grid,grid),cylinders=())
    p=(-2.99,-.2,-2.49);q=(-2.98,-.2,-2.49);want=grid.index(p)
    assert want==(0,92,0)
    for site in (1,3):
        engine.clear();engine.score(p,q,site=site)
        for actual in engine.values:
            assert np.count_nonzero(actual)==1 and actual[want]==1.


def test_recorded_grazing_seed(build,inputs):
    _,identity=inputs
    engine=p9c.CPU(build,densities=identity['densities'],grids=(checks.SMALL,checks.SMALL))
    assert checks.grazing_seed(engine)['status']=='pass'


def test_zero_boundary_watchdog(tmp_path):
    import shutil
    import subprocess
    source=tmp_path/'watchdog.cpp'
    source.write_text('''
#include "p5d_gxx_compat.h"
#include "p9c_kernel.h"
#include <cassert>
int main(){
    P9cState q={};long long counts[16]={};
    for(int j=0;j<999;++j)assert(!p9c_zero_boundary(&q,0.,true,counts));
    assert(!q.failed && counts[5]==0);
    assert(p9c_zero_boundary(&q,0.,true,counts));
    assert(q.failed && q.phase==P8G2_H && counts[5]==1);
    q.failed=false;q.phase=P8G2_C;q.zero_boundaries=999;
    assert(!p9c_zero_boundary(&q,1e-17,true,counts));
    assert(q.zero_boundaries==0 && !q.failed);
    q.zero_boundaries=999;
    assert(!p9c_zero_boundary(&q,0.,false,counts));
    assert(q.zero_boundaries==0 && !q.failed);
    for(int j=0;j<999;++j)assert(!p9c_zero_boundary(&q,0.,true,counts));
    assert(p9c_zero_boundary(&q,0.,true,counts));
    assert(q.failed && q.phase==P8G2_H && counts[5]==2);
}
''')
    exe=tmp_path/'watchdog'
    subprocess.run([shutil.which('clang++'),'-std=c++11','-ffp-contract=off','-DP5_RNG_PHILOX','-I',str(core.HERE),str(source),'-o',str(exe)],check=True,capture_output=True,text=True)
    subprocess.run([str(exe)],check=True,timeout=10)


def test_prostate_2000_export_conditions_finishes(build):
    """A subprocess deadline also catches hangs inside the native photon loop."""
    import subprocess
    import sys
    script='''
import json,sys,time
from prototypes.mv_electron import p9a_data
from prototypes.mv_electron.run_p3a import load_spectrum
from prototypes.mv_electron.gpu import p9c,p9c_core as core,p9c_checks as checks
materials,identity=p9a_data.load_bundles()
edges,_,cum=load_spectrum()
e=p9c.CPU(sys.argv[1],materials,densities=identity['densities'],edges=edges,cum=cum,
          beam='prostate',angle=45,grids=(checks.SMALL,core.Grid((-20.,)*3,(10.,)*3,(4,)*3)))
t=time.monotonic()
r=e.run(2000,940100,field=core.ANGLES.index(45))
h=r['histories']
assert core.core.history_health(h)
assert not h['counts'][:,5:7].any()
print(json.dumps(dict(histories=len(h['dep']),seconds=time.monotonic()-t,
                     health_failures=int(h['counts'][:,5:7].sum()),
                     missed=int(r['source_counts'][:,1].sum()))))
'''
    result=subprocess.run([sys.executable,'-c',script,str(build)],text=True,capture_output=True,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads(result.stdout.strip().splitlines()[-1])
    assert report['histories']==2000 and report['health_failures']==report['missed']==0
    print('P9c CPU export-condition regression:',report)


@pytest.fixture
def colab_run(tmp_path,monkeypatch):
    import sys
    from prototypes.mv_electron.gpu import p9c_colab as c,p9c_io as io,p9c_gates as gates
    record={'sources':{'kernel':'sha256'},'seed':940100}
    initial={name:{'status':'pass'} for name in c.INITIAL_GATES}
    production={'V1c':{'status':'pass'},'V2b':{'status':'pass'}}
    monkeypatch.setitem(sys.modules,'cupy',object())
    monkeypatch.setattr(io,'load',lambda _: (record,[]))
    calls=[]
    def first(*args):calls.append('initial');return dict(initial)
    def second(*args):calls.append('production');return dict(production)
    def pairs(*args):
        calls.append(('pairs',args[7]))
        return {b:{'batches':100} for b in args[7]}
    monkeypatch.setattr(c,'initial_gates',first)
    monkeypatch.setattr(gates,'production_gates',second)
    monkeypatch.setattr(c,'run_pairs',pairs)
    t0=tmp_path/'t0'
    c.json_write(t0/'job.json',dict(status='partial-accepted',mode='t0',sources=record['sources'],reason='VM lost after 15 prostate batches'))
    # The real T0 used the old flat gates format.
    c.json_write(t0/'gates.json',initial)
    c.json_write(t0/'rectangle_summary.json',{'batches':100})
    return c,record,initial,production,calls,t0


def test_partial_t0_and_cached_gates(colab_run,tmp_path,capsys):
    c,record,initial,production,calls,t0=colab_run
    out=tmp_path/'production'
    job=c.run(tmp_path,out,'production',n=7142900,t0=t0,beams='rectangle')
    assert job['status']=='complete' and job['beams']==['rectangle']
    assert calls==['initial','production',('pairs',('rectangle',))]
    saved=json.loads((out/'gates.json').read_text())
    assert saved==dict(sources=record['sources'],mode='production',gates={**initial,**production})
    assert capsys.readouterr().out=='P9C_DONE\n'
    calls.clear()
    c.run(tmp_path,out,'production',n=7142900,t0=t0,beams='prostate')
    assert calls==[('pairs',('prostate',))]
    assert capsys.readouterr().out=='P9C_DONE\n'


@pytest.mark.parametrize('fault',['status','mode','sources','reason-empty','reason-space','reason-type','gates-fail','gates-missing','gates-source','summary'])
def test_partial_t0_rejected(colab_run,tmp_path,fault):
    c,record,initial,production,calls,t0=colab_run
    prior=json.loads((t0/'job.json').read_text())
    if fault in ('status','mode','sources'):prior[fault]='wrong'
    elif fault.startswith('reason-'):prior['reason']={'reason-empty':'','reason-space':'  \n','reason-type':17}[fault]
    elif fault=='summary':(t0/'rectangle_summary.json').unlink()
    elif fault=='gates-fail':c.json_write(t0/'gates.json',{**initial,'V0':{'status':'fail'}})
    elif fault=='gates-missing':c.json_write(t0/'gates.json',{})
    elif fault=='gates-source':c.json_write(t0/'gates.json',dict(sources={'kernel':'other'},mode='t0',gates=initial))
    c.json_write(t0/'job.json',prior)
    with pytest.raises((ValueError,AssertionError)):
        c.run(tmp_path,tmp_path/'output','production',n=100,t0=t0)
    assert not calls and not (tmp_path/'output/job.json').exists()


@pytest.mark.parametrize('fault',['sources','mode','failed','empty','missing-initial','missing-production','fallback'])
def test_cached_gates_mismatch_stops(colab_run,tmp_path,fault):
    c,record,initial,production,calls,t0=colab_run
    saved=dict(sources=record['sources'],mode='production',gates={**initial,**production})
    if fault in ('sources','mode'):saved[fault]='wrong'
    elif fault=='failed':saved['gates']['V1c']={'status':'fail'}
    elif fault=='empty':saved['gates']={}
    elif fault=='missing-initial':del saved['gates']['V4_prostate']
    elif fault=='missing-production':del saved['gates']['V1c']
    elif fault=='fallback':saved['gates']['V2b']={'status':'insufficient-statistics','fallback':'wrong'}
    out=tmp_path/'output';c.json_write(out/'gates.json',saved)
    with pytest.raises((ValueError,AssertionError)):
        c.run(tmp_path,out,'production',n=100,t0=t0)
    assert not calls and json.loads((out/'gates.json').read_text())==saved


def test_cached_legacy_and_t0_gates(colab_run,tmp_path):
    c,record,initial,production,calls,t0=colab_run
    # A resumed T0 reuses its flat, source-bound record.
    c.run(tmp_path,t0,'t0',beams='rectangle')
    assert calls==[('pairs',('rectangle',))]
    calls.clear()
    out=tmp_path/'output'
    production['V2b']={'status':'insufficient-statistics','fallback':'V2a'}
    c.json_write(out/'gates.json',{**initial,**production})
    c.json_write(out/'job.json',dict(sources=record['sources'],mode='production'))
    c.run(tmp_path,out,'production',n=100,t0=t0)
    assert calls==[('pairs',c.BEAMS)]


@pytest.mark.parametrize('beams,results,done',[
    ('rectangle',{'rectangle':{'batches':100}},True),
    ('prostate',{'prostate':{'batches':100}},True),
    ('rectangle,prostate',{'rectangle':{'batches':100},'prostate':{'batches':99}},False),
    ('rectangle,prostate',{'rectangle':{'batches':100}},False),
    ('rectangle',{},False),
])
def test_done_requires_all_selected_beams(colab_run,tmp_path,monkeypatch,capsys,beams,results,done):
    c,record,initial,production,calls,t0=colab_run
    monkeypatch.setattr(c,'run_pairs',lambda *a:results)
    job=c.run(tmp_path,tmp_path/'out','t0',beams=beams)
    assert (job['status']=='complete')==done
    assert ('P9C_DONE' in capsys.readouterr().out)==done


@pytest.mark.parametrize('beams',['','rectangle,','other','rectangle,rectangle',()])
def test_invalid_beams_fail_before_gates(colab_run,tmp_path,beams):
    c,record,initial,production,calls,t0=colab_run
    with pytest.raises(ValueError,match='beams'):
        c.run(tmp_path,tmp_path/'out','t0',beams=beams)
    assert not calls


@pytest.mark.parametrize('option',[None,'rectangle','prostate','prostate,rectangle'])
def test_colab_beams_cli(monkeypatch,option):
    import sys
    from prototypes.mv_electron.gpu import p9c_colab as c
    calls=[];monkeypatch.setattr(c,'run',lambda *a:calls.append(a))
    args=['p9c_colab','bundle','output','--mode','t0']
    if option is not None:args+=['--beams',option]
    monkeypatch.setattr(sys,'argv',args);c.main()
    assert calls[0][-1]==(c.BEAMS if option is None else tuple(option.split(',')))
