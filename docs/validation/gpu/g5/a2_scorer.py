"""Geometry-derived reference for the native physical-deposit sink."""
import ctypes as ct
import numpy as np
import a2_common as common
from chatcarlo.gpu.therapy import D,FieldGeometry


def reference(records,g):
    """Independent slab clipping then overlap with each geometrical bin box."""
    out=np.zeros((len(records),3420))
    for h,row in enumerate(records):
        p=row[2:5];q=row[5:8];delta=q-p;loss=row[8]
        for j in range(15):
            lo=np.array(g.lo[3*j:3*j+3]);hi=np.array(g.hi[3*j:3*j+3]);axis=g.axes[j]
            for b in range(g.sizes[j]):
                low=lo.copy();high=hi.copy()
                low[axis]=max(low[axis],g.origins[j]+b*g.widths[j]);high[axis]=min(high[axis],g.origins[j]+(b+1)*g.widths[j])
                if row[1] or np.all(delta==0):weight=float(np.all(p>=low) and np.all(p<high))
                else:
                    left=0.;right=1.
                    for k in range(3):
                        if delta[k]==0:
                            if not low[k]<=p[k]<high[k]:right=-1.;break
                        else:
                            t=(low[k]-p[k])/delta[k];u=(high[k]-p[k])/delta[k]
                            left=max(left,min(t,u));right=min(right,max(t,u))
                    weight=max(0.,right-left)
                out[h,g.offsets[j]+b]=loss*weight
    return out


def records():
    from prototypes.mv_electron.gpu.p7 import scorer_vectors
    vectors,_=scorer_vectors(0);rows=[]
    for v in vectors:
        for kind in (0,1):rows.append([len(rows),kind,*v[:3],*(v[:3] if kind else v[3:6]),v[6]])
    # Slab/axis-box edges, profile depths, bin crossings and exact outer ends.
    for p,q in (([.3,.6,5.],[.7,.8,5.]),([-1.,.2,5.],[1.,.4,5.]),([.2,-.8,10.],[.4,.8,10.]),([15.,.6,20.],[19.9,.8,20.]),([-20.,0.,5.],[20.,0.,5.]),([20.,0.,5.],[20.,0.,5.]),([0.,0.,.25],[0.,0.,.25])):
        for kind in (0,1):rows.append([len(rows),kind,*p,*(p if kind else q),1.])
    return np.array(rows,float)


def native(cpu,rows):
    rows=np.ascontiguousarray(rows,float);out=np.zeros((len(rows),3420));sums=np.zeros((len(rows),15))
    cpu.lib.t2_score.argtypes=[ct.POINTER(FieldGeometry),D,ct.c_int,D,D]
    g=common.field_geometry();cpu.lib.t2_score(ct.byref(g),rows,len(rows),out,sums)
    return out,sums


def check(cpu):
    rows=records();g=common.field_geometry();actual,sums=native(cpu,rows);expected=reference(rows,g)
    if not np.allclose(actual,expected,rtol=1e-12,atol=2e-14):raise AssertionError('geometry expected bin')
    for j,(off,n) in enumerate(zip(g.offsets,g.sizes)):
        if not np.allclose(actual[:,off:off+n].sum(axis=1),sums[:,j],rtol=1e-12,atol=2e-14):raise AssertionError('scorer input/ledger sum')
    return dict(status='pass',deposits=len(rows),max_abs=float(np.max(np.abs(actual-expected))))
