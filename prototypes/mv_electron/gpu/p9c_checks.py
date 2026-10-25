"""Independent small CPU gates, also replayed against the CUDA entry."""
import json
from pathlib import Path
import numpy as np
from . import p9c_core as core
from .p9c import CPU

SMALL=core.Grid((-2.,)*3,(1.,)*3,(4,)*3)

def state(p,d,r=0):
    a=np.zeros(23);a[:3]=p;a[3:6]=d;a[6]=a[7]=1.;a[16]=r;return a

def expected_segment(grid,a,b,loss):
    """Independent per-cell slab clipping (does not use the DDA)."""
    a=np.array(a);d=np.array(b)-a;out=np.zeros(grid.shape)
    if not np.any(d):
        try:out[grid.index(a)]=loss
        except ValueError:pass
        return out
    for ix in np.ndindex(grid.shape):
        lo=0.;hi=1.
        for k in range(3):
            lower=grid.lo[k]+ix[k]*grid.step[k];upper=lower+grid.step[k]
            if d[k]==0:
                if not lower<=a[k]<upper:hi=-1.;break
            else:
                s,t=sorted(((lower-a[k])/d[k],(upper-a[k])/d[k]));lo=max(lo,s);hi=min(hi,t)
        out[ix]=loss*max(0.,hi-lo)
    return out

def v1a(engine):
    tests=[((-3.,-1.5,-1.5),(3.,-1.5,-1.5)),((-3.,)*3,(3.,)*3),((1.8,1.8,1.8),(-1.8,-1.8,-1.8)),((-2.,-2.,0.),(2.,2.,0.)),((0.,)*3,(0.,)*3),((-3.,0.,0.),(-2.5,0.,0.))]
    for a,b in tests:
        engine.clear();engine.score(a,b,loss=1.25)
        for g,actual in zip(engine.grids,engine.values):
            expected=expected_segment(g,a,b,1.25)
            assert np.array_equal(actual[expected==0],expected[expected==0]),'V1a zero cells'
            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=0)
    for p in ((0.,0.,0.),(-2.,-2.,-2.),(2.,0.,0.)):
        engine.clear();engine.score(p,p,site=1)
        for g,actual in zip(engine.grids,engine.values):
            expected=np.zeros(g.shape)
            try:expected[g.index(p)]=1.
            except ValueError:pass
            np.testing.assert_array_equal(actual,expected)
    return dict(status='pass',segments=len(tests),points=3)

def boundary_cases():
    rows=[];expected=[]
    # Axis sides, caps, seams, tangent, and restarts at surface ± 1 ulp.
    for axis,radius in ((0,18.),(1,15.),(2,12.)):
        for sign in (-1,1):
            for p in (0.,sign*radius,np.nextafter(sign*radius,0.),np.nextafter(sign*radius,sign*np.inf)):
                for sense in (-1,1):
                    x=np.zeros(3);d=x.copy();x[axis]=p;d[axis]=sense
                    t=max(0.,(sense*radius-p)/sense)
                    rows.append(state(x,d));expected.append((t,-1))
    for d in ((1.,1.,0.),(-1.,-1.,0.)):
        d=np.array(d)/np.linalg.norm(d);p=(18.,15.,0.)
        rows.append(state(p,d));expected.append((0. if d[0]>0 else 30/abs(d[1]),-1))
    rows.append(state((18.,0.,0.),(0.,0.,1.)));expected.append((0.,-1))
    # Seed at origin, plus ellipse outer frame in the same geometry.
    for p,d,r,t,nr in (((2,2,1),(0,0,1),0,.95,1),((2,2,2),(0,0,1),1,.05,0),((2,2,2),(0,1,0),1,.25,0),((0,0,1),(0,0,1),0,11.,-1)):
        rows.append(state(p,d,r));expected.append((t,nr))
    return np.array(rows),np.array(expected)

def v0(engine):
    rows,expected=boundary_cases();actual=engine.boundary(rows)
    np.testing.assert_allclose(actual[:,0],expected[:,0],rtol=0,atol=1e-12)
    np.testing.assert_array_equal(actual[:,1],expected[:,1])
    xyz=rows[:,:3]+expected[:,0,None]*rows[:,3:6]
    np.testing.assert_allclose(actual[:,2:5],xyz,rtol=0,atol=1e-12)
    return dict(status='pass',cases=len(rows))

def v2a(engine):
    assert core.ANGLES==(0,45,90,135,225,270,315)
    for angle in core.ANGLES:
        t=angle*np.pi/180;S=[100*np.sin(t),0,-100*np.cos(t)];u=[np.cos(t),0,np.sin(t)];v=[0,1,0]
        frame=core.frame(angle)
        for actual,want in zip(frame,(S,-np.array(S)/100,u,v)):np.testing.assert_allclose(actual,want,atol=1e-12,rtol=0)
        s,d=core.ray(angle,1.,.5)
        # Independent polynomial roots in world coordinates.
        poly=[d[0]**2/18**2+d[2]**2/12**2,2*(s[0]*d[0]/18**2+s[2]*d[2]/12**2),s[0]**2/18**2+s[2]**2/12**2-1]
        distance=min(np.roots(poly));point=s+distance*d
        np.testing.assert_allclose(engine.source(s,d)[2:],point,atol=1e-12,rtol=0)
    return dict(status='pass',angles=list(core.ANGLES))

def v3(engine):
    # The ellipsoid lies within the radius-2.25 sphere. Its perspective
    # projection is bounded by this sphere's tangent cone for every angle.
    projected_bound=2.25/np.sqrt(1.-(2.25/100.)**2)
    assert projected_bound<2.4<3.9
    n=0
    for angle in core.ANGLES:
        for a,b in ((0,0),(.5,.5),(1.,0.),(0,1.),(2.,0.),(0,2.),(3.,0.),(0,3.),(-3.,0.),(0,-3.),(3.95,0),(-3.95,0),(0,3.95),(0,-3.95),(3.95,3.95),(-3.95,-3.95),(1,1),(-1,-1),(2,2),(-2,-2)):
            s,d=core.ray(angle,a,b);x=s/core.AXES;u=d/core.AXES
            closest=x-(x@u)/(u@u)*u;expected=bool(closest@closest<=1.)
            assert core.intersects(s,d)==expected
            assert bool(engine.source(s,d)[0])==expected
            if max(abs(a),abs(b))>=3.9:assert not expected
            n+=1
    return dict(status='pass',known_rays=n,projection_bound_cm=float(projected_bound),statistical_acceptance='pending CUDA T0')

def v6(engine):
    ss=core.seeds();assert len(ss)==58 and len({(s[0],s[2]) for s in ss})==19
    for a,b in zip(core.seeds(),core.seeds(0)):assert a[:5]==b[:5] and a[5]==4 and b[5]==0
    # Independent line/cylinder analytic intervals for five actual rays.
    lines=[(0,-.2),(.5,.5),(-1.,-.9),(1.,1.2),(-.5,-.2)];crossings=0
    for x,y in lines:
        expected=[]
        for j,(cx,cy,cz,r,h,_) in enumerate(ss):
            if abs(y-cy)<h and abs(x-cx)<r:
                dz=np.sqrt(r*r-(x-cx)**2);expected.extend(((cz-dz,1+j),(cz+dz,0)))
        expected.sort();p=np.array([x,y,-11.]);region=0
        for z,nr in expected:
            actual=engine.boundary([state(p,(0,0,1),region)])[0]
            assert int(actual[1])==nr
            np.testing.assert_allclose(actual[2:5],(x,y,z),atol=1e-12,rtol=0)
            p=actual[2:5];region=nr;crossings+=1
    return dict(status='pass',seeds=58,needles=19,lines=5,crossings=crossings,radius_protruding=6)

def grazing_seed(engine):
    """G3 regression: the recorded seed-49 cap/side seam must not re-enter."""
    p=(0.95651418282246681,0.75,1.0246775951908236)
    d=(-0.68154230906564106,-0.54162973744039444,0.49207449484170956)
    # In seed 49 the tiny outward crossing is retained. From the water side
    # its forward chord is negligible and must be skipped, including on restart.
    inside=engine.boundary([state(p,d,50)])[0]
    assert inside[1]==0 and 0<=inside[0]<=1e-12
    np.testing.assert_allclose(inside[2:5],p,rtol=0,atol=1e-12)
    outside=engine.boundary([state(p,d),state(inside[2:5],d) ])
    assert np.all(outside[:,0]>1e-12) and np.all(outside[:,1]!=50)
    # Independent outer ellipse exit; this ray meets no subsequent seeds.
    x,z=p[0]/18.,p[2]/12.;u,w=d[0]/18.,d[2]/12.
    A=u*u+w*w;B=x*u+z*w;C=x*x+z*z-1.
    distance=(-B+np.sqrt(B*B-A*C))/A
    np.testing.assert_allclose(outside[:,0],distance,rtol=0,atol=1e-12)
    np.testing.assert_array_equal(outside[:,1],[-1,-1])
    return dict(status='pass',cases=3)


def local(work,materials=None,densities=None):
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    if materials is None:
        from .. import p9a_data
        materials,identity=p9a_data.load_bundles();densities=identity['densities']
    rho=densities
    common=dict(densities=rho,grids=(SMALL,SMALL))
    e=CPU(work,**common,cylinders=((2,2,2,.05,.25,4),))
    report=dict(V0=v0(e),V1a=v1a(e),V2a=v2a(e),V3=v3(e))
    actual=CPU(work,**common)
    report['V6']=v6(actual);report['G3_grazing']=grazing_seed(actual)
    from .p9c_validation import v5,v6b
    report['V5']=v5(work,materials,rho);report['V6b']=v6b()
    report['status']='pass';work.mkdir(parents=True,exist_ok=True);(work/'local.json').write_text(json.dumps(report,indent=2)+'\n');return report
