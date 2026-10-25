"""Paired-statistics and volume-intersection validation, no cache writes."""
import numpy as np
from . import p9c_core as core
from .p9c_stats import Moments,Checkpoint,normalize

DOORS=set((f,s) for f in range(7) for s in (True,False))

def v5(work,materials=None,rho=None):
    from pathlib import Path
    import tempfile
    from .p9c import CPU
    arrays=[(np.arange(1.,6),np.arange(1.,6)),(np.arange(1.,6),np.arange(5.,0.,-1)),(np.array([-1.,0,1]),np.array([1.,-2,1])),(np.ones(4),np.ones(4)),(np.ones(1),np.ones(1)),(np.ones(4),np.zeros(4))]
    for a,b in arrays:
        m=Moments(())
        for x,y in zip(a,b):m.add(x,y)
        np.testing.assert_allclose(m.values,[a.sum(),b.sum(),a@a,b@b,a@b],rtol=1e-12,atol=0)
        r=m.result();n=len(a);np.testing.assert_allclose(r['difference'],a.mean()-b.mean(),atol=0)
        if n>1:
            np.testing.assert_allclose(r['difference_sem'],np.std(a-b,ddof=1)/np.sqrt(n),rtol=1e-12,atol=0)
            va=np.var(a,ddof=1)/n;vb=np.var(b,ddof=1)/n;cov=np.cov(a,b,ddof=1)[0,1]/n
            if va and vb:np.testing.assert_allclose(r['correlation'],cov/np.sqrt(va*vb),rtol=1e-12,atol=0)
            else:assert np.isnan(r['correlation'])
            if b.mean()!=0:
                rr=a.mean()/b.mean();expect=np.sqrt(max(0.,va+rr*rr*vb-2*rr*cov))/abs(b.mean())
                np.testing.assert_allclose(r['ratio_sem'],expect,rtol=1e-12,atol=0)
        else:assert np.isnan(r['difference_sem'])
        if b.mean()==0:assert np.isnan(r['ratio']) and np.isnan(r['ratio_sem'])
        rev=Moments(())
        for x,y in zip(a[::-1],b[::-1]):rev.add(x,y)
        np.testing.assert_allclose(rev.values,m.values,atol=0,rtol=1e-12)
    np.testing.assert_allclose(normalize(np.arange(1.,8),np.arange(1,8)),1.)
    # Interrupted partial batch cannot alter the last committed state.
    with tempfile.TemporaryDirectory(dir=work) as tmp:
        path=Path(tmp)/'checkpoint.npz';identity={'test':'V5'};shapes={'v':()}
        c=Checkpoint(path,identity,shapes);continuous=Moments(())
        for j in range(4):
            c=Checkpoint(path,identity,shapes);c.commit(j,{'v':(j+1.,2*j+1.)},DOORS);continuous.add(j+1.,2*j+1.)
            try:c.commit(j,{'v':(1.,1.)},DOORS)
            except ValueError:pass
            else:raise AssertionError('replay accepted')
            digest=c.digest
            try:c.commit(j+1,{'v':(2.,2.)},DOORS-{(6,False)})
            except ValueError:pass
            else:raise AssertionError('partial accepted')
            assert c.digest==digest
        np.testing.assert_array_equal(c.moments['v'].values,continuous.values)
    result=dict(status='pass',artificial_cases=len(arrays),resume='exact',replay='rejected',cpu_pair='pending material input')
    if materials is not None:
        g=core.Grid((-20.,)*3,(10.,)*3,(4,)*3)
        sums=[np.zeros(g.shape),np.zeros(g.shape)]
        for angle in core.ANGLES:
            outputs=[]
            for _ in range(2):
                e=CPU(work,materials,densities=rho,angle=angle,grids=(g,g));outputs.append(e.run(2,940100,field=core.ANGLES.index(angle))['grids'][0].copy())
            np.testing.assert_array_equal(*outputs)
            for k in range(2):sums[k]+=outputs[k]
        m=Moments(g.shape);m.add(*sums);m.add(*sums);r=m.result()
        assert np.all(r['difference']==0) and np.all(r['difference_sem']==0)
        assert np.all(r['ratio'][sums[1]!=0]==1)
        result['cpu_pair']='exact, seven fields, same geometry and material'
    return result

def v6b():
    for step in (.025,.2):
        g=core.Grid((0.,)*3,(step,)*3,(4,)*3)
        # Center outside: side overlaps, cap overlaps, exact contact, separated.
        for seed,expected in (((step+.04,step/2,step/2,.05,.01,4),True),((step/2,step+.005,step/2,.001,.01,4),True),((step+.05,step/2,step/2,.05,.01,4),True),((step+.051,step/2,step/2,.05,.01,4),False)):
            assert bool(core.seed_mask(g,(seed,))[0,0,0])==expected
    return dict(status='pass',cases=8,fine_and_coarse=True)
