"""Gate 0a: exact scalar ports, including MT state (no tolerances)."""
import ctypes
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pytest
import numba
from numba import njit, _helperlib
from chatcarlo import kernel

HERE = Path(__file__).resolve().parents[1] / 'chatcarlo/gpu'

@njit
def direction_reference(which,x,seed):
    np.random.seed(seed)
    out=np.empty((len(x),3))
    for i in range(len(x)):
        if which==0:
            out[i]=kernel._scatter_direction_scalar(x[i,0],x[i,1],x[i,2],x[i,3])
        else:
            out[i]=kernel._isotropic_direction_scalar()
    return out

@njit
def next_draws():
    return np.random.random(16)

@pytest.fixture(scope='module')
def probe(tmp_path_factory):
    dest=tmp_path_factory.mktemp('g1-build')/'g1_mt.so'
    cmd=['clang++','-O2','-ffp-contract=off','-std=c++11','-shared','-fPIC','-DP5_RNG_MT',str(HERE/'g1_cpu.cpp'),'-o',str(dest)]
    subprocess.run(cmd,check=True,capture_output=True)
    lib=ctypes.CDLL(str(dest));fn=lib.g1_direction_probe
    fn.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
    return fn

def bitwise(a,b,label):
    assert a.shape==b.shape and a.dtype==b.dtype
    if a.tobytes()!=b.tobytes():
        bad=np.flatnonzero(a.view(np.uint64).ravel()!=b.view(np.uint64).ravel())
        first=int(bad[0]);av=a.ravel()[first];bv=b.ravel()[first]
        raise AssertionError(f'{label}: byte mismatch at flat index {first}: C++={av!r} ({float(av).hex()}), Numba={bv!r} ({float(bv).hex()}); mismatches={len(bad)}/{a.size}')

@pytest.mark.parametrize('which',[0,1],ids=['scatter_direction','isotropic_direction'])
def test_gpu_g1_0a_directions(probe,which):
    rng=np.random.default_rng(2026102100)
    d=rng.normal(size=(100000,3));d/=np.linalg.norm(d,axis=1,keepdims=True)
    x=np.ascontiguousarray(np.column_stack((d,rng.uniform(-1,1,len(d)))))
    x=np.concatenate((x,np.array([[0,0,1,c] for c in (-1.,0.,1.)]+[[0,1,0,c] for c in (-1.,0.,1.)])))
    expected=direction_reference(which,x,2026102100)
    index,words=_helperlib.rnd_get_state(_helperlib.rnd_get_np_state_ptr())
    expected_next=next_draws()
    actual=np.empty_like(expected);state=np.zeros(626+len(x),np.uint64);next_=np.empty(16)
    probe(which,x.ctypes.data,len(x),2026102100,actual.ctypes.data,state.ctypes.data,next_.ctypes.data)
    np.testing.assert_array_equal(state[:624],np.array(words,np.uint64))
    assert state[624]==index
    check=np.random.RandomState(2026102100);check.random_sample(int(state[625]));ref_state=check.get_state()
    np.testing.assert_array_equal(ref_state[1],words);assert ref_state[2]==index
    bitwise(next_,expected_next,'subsequent MT draws')
    bitwise(actual,expected,'direction outputs')

FIELDS='n_elem zs fracs log_e step n_grid photo compt rayl incoh_q incoh_s rayl_x rayl_a k_edge k_omega k_line_e k_line_p n_lines k_frac density_g_cm3'.split()

@pytest.fixture(scope='module')
def tables():
    return kernel.bake_scene_materials(['air','water','soft_tissue','bone','copper','lead','aluminum'])

@pytest.fixture(scope='module')
def physics_probe(probe):
    fn=probe.__self__.g1_physics_probe if hasattr(probe,'__self__') else None
    # ctypes functions keep their shared library alive through _objects.
    lib=ctypes.CDLL(probe._objects['0']._name)
    fn=lib.g1_physics_probe
    fn.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
    return fn

@njit
def physics_reference(which,x,seed,ne,zs,fracs,le,step,ng,photo,compt,rayl,iq,iss,rx,ra,ke,ko,kle,klp,nl,kf,dens):
    np.random.seed(seed);me=zs.shape[1];out=np.zeros((len(x),8+5*me))
    for j in range(len(x)):
        a=x[j];m=int(a[1]);p=np.zeros(me);c=np.zeros(me);r=np.zeros(me);idx=np.zeros(me,np.int64);frac=np.zeros(me)
        tp=0.;tc=0.;tr=0.
        if which<=3:
            mu,tp,tc,tr=kernel._mu_and_parts_scalar(a[0],m,ne,fracs,le,step,ng,photo,compt,rayl,dens,p,c,r,idx,frac)
        if which==0:
            out[j,0]=mu;out[j,1]=tp;out[j,2]=tc;out[j,3]=tr
            for k in range(ne[m]):
                out[j,8+k]=p[k];out[j,8+me+k]=c[k];out[j,8+2*me+k]=r[k];out[j,8+3*me+k]=idx[k];out[j,8+4*me+k]=frac[k]
        elif which==1:
            aa,bb,cc=kernel._sample_compton_bound_scalar(m,a[0],ne[m],fracs,zs,iq,iss,c,tc)
            out[j,0]=aa;out[j,1]=bb;out[j,2]=cc
        elif which==2:
            aa,bb=kernel._sample_rayleigh_cos_theta_scalar(m,a[0],ne[m],fracs,rx,ra,r,tr)
            out[j,0]=aa;out[j,1]=bb
        elif which==3:
            ei=int(a[2]);aa,bb=kernel._sample_fluorescence_scalar(m,ei,a[0],idx[ei],frac[ei],ke,ko,kf,kle,klp,nl)
            out[j,0]=aa;out[j,1]=bb
        elif which==4:out[j,0]=kernel._select_element(ne[m],fracs[m],fracs[m],a[0])
        elif which==5:out[j,0]=kernel._lerp_lookup(le[m,int(a[2]),:ng[m,int(a[2])]],photo[m,int(a[2]),:ng[m,int(a[2])]],a[0])
        elif which==6:
            aa,bb=kernel._element_index_frac(le[m,int(a[2])],step[m,int(a[2])],ng[m,int(a[2])],a[0]);out[j,0]=aa;out[j,1]=bb
    return out

@pytest.mark.parametrize('which',range(7),ids=['mu_parts_work','compton','rayleigh','fluorescence','element','lerp','index_frac'])
def test_gpu_g1_0a_physics(physics_probe,tables,which):
    from chatcarlo.gpu.validation import physics_inputs
    t=tables;x=physics_inputs(t,which)
    expected=physics_reference(which,x,2026102100,*(getattr(t,f) for f in FIELDS))
    index,words=_helperlib.rnd_get_state(_helperlib.rnd_get_np_state_ptr());expected_next=next_draws()
    chunks=[np.asarray(getattr(t,f),float).ravel() for f in FIELDS];offset=np.r_[0,np.cumsum([len(c) for c in chunks])[:-1]].astype(np.int64);data=np.concatenate(chunks)
    actual=np.zeros_like(expected);state=np.zeros(626+len(x),np.uint64);nxt=np.empty(16)
    physics_probe(which,data.ctypes.data,offset.ctypes.data,t.zs.shape[1],t.log_e.shape[2],len(t.material_names),x.ctypes.data,len(x),2026102100,actual.ctypes.data,state.ctypes.data,nxt.ctypes.data)
    np.testing.assert_array_equal(state[:624],np.array(words,np.uint64));assert state[624]==index
    check=np.random.RandomState(2026102100);check.random_sample(int(state[625]));rs=check.get_state()
    np.testing.assert_array_equal(rs[1],words);assert rs[2]==index
    bitwise(nxt,expected_next,'subsequent MT draws')
    bitwise(actual,expected,'physics outputs and work arrays')
