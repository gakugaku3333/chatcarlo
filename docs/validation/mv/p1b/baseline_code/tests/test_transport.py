import numpy as np
import pytest
from prototypes.mv_electron.transport import *
from prototypes.mv_electron.component_checks import OUT,SEED
@pytest.fixture(scope='module')
def args():
    g=np.load(OUT/'data/gs.npz');return tuple(g[x] for x in ['T','K','u','y'])
def check(result,T,n):
    b,l,c,*_=result
    assert max(abs(l[:,3]))<T*1e-6
    assert abs(sum(l[:,3]))<n*T*1e-6
    assert not np.any(l[:,4:6]) and not c[5] and not c[6]
    assert np.sum(b)==pytest.approx(np.sum(l[:,0]),rel=1e-12)
def test_k5_all_descendants(args):
    check(run(2,1000,SEED,.05,.05,np.array([0.,1.2]),.01,*args),2,1000)
@pytest.mark.parametrize('z0,z1',[ (.003,.047),(.047,.003),(.02,.02), (0,.12)])
def test_k5b_analytic_bin_overlap(z0,z1):
    s=np.zeros(12);score_segment(s,z0,z1,.73,.01)
    expected=np.zeros(12)
    if z0==z1:expected[int(z0/.01)]=.73
    else:
        lo,hi=sorted([z0,z1])
        for i in range(12):expected[i]=.73*max(0,min(hi,(i+1)*.01)-max(lo,i*.01))/(hi-lo)
    assert np.allclose(s,expected,atol=1e-14) and sum(s)==pytest.approx(.73)
def test_k5b_faces_parallel_reverse_and_zero():
    faces=np.array([0.,.005,.01])
    assert boundary(.005,1.,0,faces)==(0.,1)
    assert boundary(.005,-1.,1,faces)==(0.,0)
    assert np.isinf(boundary(.005,0.,1,faces)[0])
    assert boundary(0,1,0,faces)==(.005,1)
@pytest.mark.parametrize('thick',[.00001,.0008,.01,.1,1.2])
def test_k5b_energy_hinge_escape_final_cutoff(args,thick):
    result=run(.03,100,SEED,.25,.05,np.array([0.,thick]),thick/12,*args,True,False,2)
    check(result,.03,100)
    if thick==1.2:assert result[1][:,0]==pytest.approx(np.full(100,.03))
def test_k5b_parent_restore_and_moller_settlement(args):
    a=run(2,1000,SEED,.05,.05,np.array([0.,1.2]),.01,*args)
    b=run(2,1000,SEED,.05,.05,np.array([0.,1.2]),.01,*args,False)
    assert a[-1]>1 and a[2][2]>0
    assert np.array_equal(a[1],b[1]) and np.array_equal(a[2],b[2])
    check(a,2,1000)
def test_limits_are_failure_not_local_deposition(args):
    b,l,c,*_=run(2,100,SEED,.05,.05,np.array([0.,1.2]),.01,*args,maxsteps=1)
    assert c[5]>0 and np.any(l[:,4]>0) and np.any(l[:,5]>0)
def test_k5b_final_hinge_residual_is_scored_on_track(args):
    a=initial(.012,0.,0.,0.,1.,0,1,.05,.05)
    a[7]=.01;a[8]=.002;a[18]=1
    b,l,c,*_=run(.012,100,SEED,.05,.05,np.array([0.,.12]),.001,*args,True,False,2,source_state=a)
    check((b,l,c),.012,100)
    assert np.sum(b[:,0])==pytest.approx(1.2)
def test_k5b_simultaneous_zero_events_terminate(args):
    a=initial(.03,0.,0.,0.,1.,0,1,.05,.05);a[8]=0.;a[12]=0.;a[15]=0.
    r=run(.03,100,SEED,.05,.05,np.array([0.,.000001,.12]),.01,*args,source_state=a)
    check(r,.03,100)
    assert max(r[1][:,6])<10000
def test_stack_overflow_retains_unprocessed_energy(args):
    b,l,c,*_=run(2,100,SEED,.05,.05,np.array([0.,1.2]),.01,*args,maxstack=1)
    assert c[6]>0 and sum(l[:,4])>0
    assert np.allclose(l[:,3],l[:,5],atol=1e-12)
def test_boundary_free_diagnostic_has_no_queries(args):
    b,l,c,*_=run(2,100,SEED,.05,.05,np.array([0.,1.2]),.01,*args,False,True,0,ignore_boundaries=True)
    assert c[3]==c[4]==0 and max(abs(l[:,3]))<1e-12 and not np.any(l[:,4:6])
