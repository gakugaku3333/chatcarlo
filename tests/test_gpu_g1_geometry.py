"""Gates 0a/0c: exact analytic geometry and terminal plane intersections."""
import ctypes
import numpy as np
import pytest
from numba import njit
from chatcarlo import kernel
from chatcarlo.geometry import Geometry
from chatcarlo.detector import DetectorPlane
from chatcarlo.scene import load_scene
from test_gpu_g1_functions import probe, bitwise, HERE

class Geo(ctypes.Structure):
    _fields_=[('centers',ctypes.c_void_p),('halves',ctypes.c_void_p),('materials',ctypes.c_void_p),('nb',ctypes.c_int),('bg',ctypes.c_int),('world',ctypes.c_void_p),('det',ctypes.c_void_p)]

@pytest.fixture(scope='module')
def geometry_probe(probe):
    lib=ctypes.CDLL(probe._objects['0']._name);fn=lib.g1_geometry_probe
    fn.argtypes=[ctypes.c_int,ctypes.POINTER(Geo),ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
    return fn

@njit
def reference(which,x,c,h,m,w):
    out=np.zeros((len(x),4))
    for i in range(len(x)):
        a=x[i]
        if which==0:
            aa,bb,cc=kernel._intersect_box_scalar(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8]);out[i,0]=aa;out[i,1]=bb;out[i,2]=cc
        elif which==1:out[i,0]=kernel._material_at_scalar(a[0],a[1],a[2],len(c),c,h,m,0)
        elif which==2:
            aa,bb=kernel._next_boundary_scalar(a[0],a[1],a[2],a[3],a[4],a[5],len(c),c,h,w[:3],w[3:]);out[i,0]=aa;out[i,1]=bb
    return out

@pytest.mark.parametrize('which',range(3),ids=['box','material','boundary'])
def test_gpu_g1_0a_geometry(geometry_probe,which):
    rng=np.random.default_rng(2026102101)
    c=np.array([[0.,0,10],[0,0,10],[1,2,5]]);h=np.array([[10.,10,10],[5,5,5],[2,3,4]])
    m=np.array([1,2,3],np.int64);w=np.array([0.,0,10,60,60,60])
    g=Geo(c.ctypes.data,h.ctypes.data,m.ctypes.data,len(c),0,w.ctypes.data,None)
    x=rng.uniform(-20,20,(100000,10));x[:,6:9]=np.abs(x[:,6:9])
    extra=[]
    for p in (-10.,0.,10.):
        for d in (-1.,0.,1.):extra.append([p,p,p,d,d,d,10,10,10,0])
    x=np.ascontiguousarray(np.r_[x,extra]);out=np.zeros((len(x),4))
    geometry_probe(which,ctypes.byref(g),x.ctypes.data,len(x),out.ctypes.data)
    bitwise(out,reference(which,x,c,h,m,w),'geometry outputs')

@pytest.mark.parametrize('tilted',[False,True])
def test_gpu_g1_0c_detector(geometry_probe,tilted):
    n=np.array([.2,-.1,1.]) if tilted else np.array([0.,0.,1.]);n/=np.linalg.norm(n)
    u=np.array([1.,2.,0.]) if tilted else np.array([1.,0.,0.]);u-=np.dot(u,n)*n;u/=np.linalg.norm(u)
    plane=DetectorPlane([1,-2,3] if tilted else [0,0,0],n,u,(30,30),(60,60))
    p=np.ascontiguousarray(np.r_[plane.center_cm,n,u,plane.v_axis,plane.size_cm,plane.shape],float)
    g=Geo(None,None,None,0,0,None,p.ctypes.data)
    rng=np.random.default_rng(2026102102);x=np.zeros((100000,10))
    x[:,:3]=rng.uniform(-20,20,(len(x),3));x[:,3:6]=rng.normal(size=(len(x),3));x[:,3:6]/=np.linalg.norm(x[:,3:6],axis=1,keepdims=True);x[:,6]=rng.uniform(0,100,len(x))
    extra=[]
    for a in (-15.,-14.5,0.,14.5,15.):
        for b in (-15.,0.,15.):
            point=plane.center_cm+u*a+plane.v_axis*b
            for d in (-n,n,u):extra.append(np.r_[point-d*10,d,10.,0,0,0])
    x=np.ascontiguousarray(np.r_[x,extra]);expected=np.column_stack(plane.intersect_segments(x[:,:3],x[:,3:6],x[:,6]))
    out=np.zeros((len(x),4));geometry_probe(3,ctypes.byref(g),x.ctypes.data,len(x),out.ctypes.data)
    bitwise(out,expected,'detector outputs')
