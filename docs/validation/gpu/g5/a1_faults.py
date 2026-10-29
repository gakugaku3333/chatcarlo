"""Philox faults 1--4: independent two-sample KS and fixed-state checks."""
import argparse
import ctypes
from pathlib import Path
import numpy as np
from a1_common import CACHE,SUMMARY,write


def reference(path,n=50000):
    from chatcarlo import kernel
    from numba import njit
    t=kernel.bake_scene_materials(['air','water','lead'])
    @njit
    def samples(which,m,n,ne,zs,fracs,le,step,ng,photo,compt,rayl,iq,iss,rx,ra,dens):
        np.random.seed(11);out=np.zeros(n);me=zs.shape[1]
        for j in range(n):
            p=np.zeros(me);c=np.zeros(me);r=np.zeros(me);idx=np.zeros(me,np.int64);frac=np.zeros(me)
            mu,tp,tc,tr=kernel._mu_and_parts_scalar(30.,m,ne,fracs,le,step,ng,photo,compt,rayl,dens,p,c,r,idx,frac)
            if which==2:out[j]=kernel._sample_rayleigh_cos_theta_scalar(m,30.,ne[m],fracs,rx,ra,r,tr)[0]
            else:out[j]=kernel._sample_compton_bound_scalar(m,30.,ne[m],fracs,zs,iq,iss,c,tc)[1]
        return out
    values={f'{which}_{m}':samples(which,m,n,*(getattr(t,f) for f in
        'n_elem zs fracs log_e step n_grid photo compt rayl incoh_q incoh_s rayl_x rayl_a density_g_cm3'.split()))
        for which in (2,1) for m in (1,2)}
    Path(path).parent.mkdir(parents=True,exist_ok=True);np.savez(path,**values)


def physics_probe(driver,which,x,seed):
    a=driver.a;y=np.zeros((len(x),8+5*driver.me));state=np.zeros(626+len(x),np.uint64);nxt=np.zeros(16)
    if driver.device=='cpu':
        fn=driver.lib.g1_physics_probe
        fn.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
        fn(which,a['data'].ctypes.data,a['offsets'].ctypes.data,driver.me,driver.ng,driver.nm,x.ctypes.data,len(x),seed,y.ctypes.data,state.ctypes.data,nxt.ctypes.data)
    else:
        cp=driver.cp;gx,gy,gs,gn=map(cp.asarray,(x,y,state,nxt))
        driver.module.get_function('g1_physics_probe_kernel')((1,),(1,),
            (np.int32(which),driver.gpu['data'],driver.gpu['offsets'],np.int32(driver.me),np.int32(driver.ng),np.int32(driver.nm),gx,np.int32(len(x)),np.uint64(seed),gy,gs,gn))
        y=cp.asnumpy(gy)
    return y


def geometry_probe(driver):
    from chatcarlo.gpu.core import Geometry
    a=driver.a;x=np.zeros((1,10));y=np.zeros((1,4))
    if driver.device=='cpu':
        g=Geometry(a['centers'].ctypes.data,a['halves'].ctypes.data,a['materials'].ctypes.data,2,0,a['world'].ctypes.data,None)
        fn=driver.lib.g1_geometry_probe
        fn.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
        fn(1,ctypes.byref(g),x.ctypes.data,1,y.ctypes.data)
    else:
        cp=driver.cp;gx,gy=map(cp.asarray,(x,y));g=driver.gpu
        driver.module.get_function('g1_geometry_probe_flat')((1,),(1,),
            (np.int32(1),g['centers'],g['halves'],g['materials'],np.int32(2),np.int32(0),g['world'],np.uint64(0),gx,np.int32(1),gy))
        y=cp.asnumpy(gy)
    return float(y[0,0])


def run(ref,summary=SUMMARY/'faults_cpu.json',device='cpu',n=50000):
    from scipy.stats import ks_2samp
    from chatcarlo import kernel
    from chatcarlo.gpu import core
    t=kernel.bake_scene_materials(['air','water','lead']);a=core.pack(t)
    a.update(centers=np.zeros((2,3)),halves=np.array([[10.]*3,[5.]*3]),materials=np.array([1,2],np.int64),background=np.array(0),world=np.array([0.,0.,0.,60.,60.,60.]),detector=np.array([]))
    rows=[]
    good=core.Core(a,device)
    with np.load(ref) as z:
        for fault,which in ((1,2),(2,1)):
            bad=core.Core(a,device,fault=fault)
            for m in (1,2):
                reference=z[f'{which}_{m}']
                if len(reference)!=n:raise ValueError('independent sample count')
                x=np.zeros((n,8));x[:,0]=30.;x[:,1]=m
                gp=float(ks_2samp(reference,physics_probe(good,which,x,22)[:,0 if which==2 else 1]).pvalue)
                bp=float(ks_2samp(reference,physics_probe(bad,which,x,33)[:,0 if which==2 else 1]).pvalue)
                rows.append(dict(fault=fault,material=m,normal_p=gp,fault_p=bp,passed=gp>=1e-6 and bp<1e-6))
    # Fault 3 reaches the sub-5 keV branch through the historical fixed input.
    t.k_edge[2,0]=1.;t.k_omega[2,0]=1.;t.k_frac[2,0,:]=1.;t.n_lines[2,0]=1
    t.k_line_e[2,0,:]=4.;t.k_line_p[2,0,:]=0.;t.k_line_p[2,0,0]=1.
    aa=dict(a,**core.pack(t));x=np.zeros((1000,8));x[:,0]=100.;x[:,1]=2
    good3=physics_probe(core.Core(aa,device),3,x,22)
    bad3=physics_probe(core.Core(aa,device,fault=3),3,x,33)
    fixed3=bool(np.all(good3[:,:2]==0) and np.all(bad3[:,0]==1) and np.all(bad3[:,1]==4))
    fixed4=geometry_probe(good)==2 and geometry_probe(core.Core(a,device,fault=4))==1
    result=dict(status='pass' if all(r['passed'] for r in rows) and fixed3 and fixed4 else 'fail',
                ks=rows,fault3=fixed3,fault4=fixed4,rng='philox',n=n,production_conditions=n==50000)
    write(summary,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['reference','run'])
    p.add_argument('--reference',type=Path,default=CACHE/'faults_reference.npz')
    p.add_argument('--summary',type=Path,default=None);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args()
    if a.mode=='reference':reference(a.reference)
    else:raise SystemExit(run(a.reference,a.summary or SUMMARY/f'faults_{a.device}.json',a.device)['status']!='pass')
