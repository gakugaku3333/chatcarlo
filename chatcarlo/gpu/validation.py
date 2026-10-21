"""Pre-registered G1 validation and fixed probes; GPU execution belongs to Claude."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
from . import core

ROIS={'s1':((25,35,25,35),(0,10,0,10),(25,35,45,55)),
      's2':((25,35,25,35),(32,44,12,24),(20,30,40,50)),
      's3':((40,60,50,70),(5,20,5,20),(80,95,100,115))}

def quantities(result,names,scene_id,n):
    q={f'deposit:{name}':float(v)/n for name,v in zip(names,result['deposits'])}
    q['escape']=float(result['endings'][1])/n
    for c in range(4):
        image=result['images'][c];q[f'category:{c}:all']=float(image.sum())/n
        for j,(a,b,u,v) in enumerate(ROIS[scene_id]):q[f'category:{c}:roi{j}']=float(image[a:b,u:v].sum())/n
    return q

def quantity_additions(result,names,scene_id,n):
    counts={f'deposit:{name}':n for name in names};counts['escape']=int(result['endings'][1])
    for c in range(4):
        image=result['category_count'][c]
        counts[f'category:{c}:all']=int(image.sum())+image.size-1
        for j,(a,b,u,v) in enumerate(ROIS[scene_id]):
            roi=image[a:b,u:v];counts[f'category:{c}:roi{j}']=int(roi.sum())+roi.size-1
    return counts

def independent_judge(reference,twin):
    if reference.keys()!=twin.keys():raise ValueError('different quantities')
    records={};zs=[];fail=False
    for k in reference:
        a=np.asarray(reference[k]);b=np.asarray(twin[k]);am=float(a.mean());bm=float(b.mean());sa=float(a.std(ddof=1)/np.sqrt(10));sb=float(b.std(ddof=1)/np.sqrt(10));sem=float(np.hypot(sa,sb));z=None;status='pass'
        if a.shape!=(10,) or b.shape!=(10,) or not np.isfinite(a).all() or not np.isfinite(b).all():status='fail'
        elif np.all(a==0) and np.all(b==0):status='both_zero'
        elif np.all(a==0) != np.all(b==0):status='fail'
        elif sem==0:
            if am!=bm:status='fail'
            else:z=0.
        else:z=(bm-am)/sem
        if z is not None:zs.append(z)
        if status=='fail':fail=True
        if z is not None and abs(z)>3:status='outside_3sigma'
        records[k]=dict(reference_mean=am if np.isfinite(am) else None,twin_mean=bm if np.isfinite(bm) else None,reference_SEM=sa if np.isfinite(sa) else None,twin_SEM=sb if np.isfinite(sb) else None,z=z,detectable_difference=3*sem if np.isfinite(sem) else None,status=status)
    z=np.asarray(zs);border=int(np.count_nonzero(np.abs(z)>3))
    status='fail' if fail or np.any(np.abs(z)>=4) or border>1 else 'retry' if border==1 else 'pass'
    return dict(status=status,K=len(z),sum_z2_over_K=float(np.mean(z*z)) if len(z) else None,quantities=records)

def paired_judge(cpu,gpu,bounds):
    a=np.asarray(cpu);b=np.asarray(gpu);bound=np.asarray(bounds)
    if a.shape!=b.shape or a.shape[0]!=10 or bound.shape!=a.shape or not all(np.isfinite(x).all() for x in (a,b,bound)):raise ValueError('10 finite paired batches required')
    delta=b-a;mean=delta.mean(axis=0);sem=delta.std(axis=0,ddof=1)/np.sqrt(10)
    rounding=np.all(np.abs(delta)<=bound,axis=0);passed=rounding|((sem>0)&(np.abs(mean)<=3*sem))
    return dict(status='pass' if passed.all() else 'fail',difference=mean.tolist(),SEM=sem.tolist(),rounding_bound=bound.tolist(),rounding_pass=rounding.tolist(),passed=passed.tolist())

def physics_inputs(t,which):
    rng=np.random.default_rng(2026102100);x=np.zeros((100000,8))
    x[:,0]=np.exp(rng.uniform(np.log(5),np.log(150),len(x)));x[:,1]=rng.integers(0,len(t.n_elem),len(x));x[:,2]=[rng.integers(t.n_elem[int(m)]) for m in x[:,1]]
    if which in (5,6):x[:,0]=np.log(x[:,0])
    extra=[]
    for m in range(len(t.n_elem)):
        for ei in range(t.n_elem[m]):
            vals=t.log_e[m,ei,:t.n_grid[m,ei]] if which in (5,6) else np.r_[np.exp(t.log_e[m,ei,:t.n_grid[m,ei]]),t.k_edge[m,ei]]
            vals=vals[(vals>0) if which not in (5,6) else np.ones(len(vals),bool)]
            for v in np.r_[vals,np.nextafter(vals,-np.inf),np.nextafter(vals,np.inf)]:extra.append([v,m,ei,0,0,0,0,0])
    if which==4:extra.extend([[0.,m,0,0,0,0,0,0] for m in range(len(t.n_elem))])
    return np.ascontiguousarray(np.r_[x,extra])

def unpack_tables(a):
    from types import SimpleNamespace
    nm,me,ng=[int(a[k].item()) for k in ('nm','me','ng')]
    shapes=[(nm,),(nm,me),(nm,me),(nm,me,ng),(nm,me),(nm,me),(nm,me,ng),(nm,me,ng),(nm,me,ng),(nm,me,2000),(nm,me,2000),(nm,me,2000),(nm,me,2000),(nm,me),(nm,me),(nm,me,8),(nm,me,8),(nm,me),(nm,me,ng),(nm,)]
    fields={}
    for name,offset,shape in zip(core.FIELDS,a['offsets'],shapes):
        v=a['data'][int(offset):int(offset)+int(np.prod(shape))].reshape(shape)
        if name in ('n_elem','zs','n_grid','n_lines'):v=v.astype(np.int64)
        fields[name]=v
    return SimpleNamespace(**fields)

def gpu_probe(cpu,gpu,which,x,kind='physics'):
    a=cpu.a;cp=gpu.cp;g=gpu.gpu;n=len(x);seed=2026102100
    if kind=='physics':width=8+5*cpu.me;name='g1_physics_probe';args=(which,a['data'].ctypes.data,a['offsets'].ctypes.data,cpu.me,cpu.ng,cpu.nm,x.ctypes.data,n,seed)
    else:width=3;name='g1_direction_probe';args=(which,x.ctypes.data,n,seed)
    out=np.zeros((n,width));state=np.zeros(626+n,np.uint64);nxt=np.empty(16)
    fn=getattr(cpu.lib,name)
    if kind=='physics':fn.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
    else:fn.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
    fn(*args,out.ctypes.data,state.ctypes.data,nxt.ctypes.data)
    gy=cp.zeros((n,width),cp.float64);gs=cp.zeros(626+n,cp.uint64);gn=cp.zeros(16,cp.float64);gx=cp.asarray(x)
    ga=(np.int32(which),g['data'],g['offsets'],np.int32(cpu.me),np.int32(cpu.ng),np.int32(cpu.nm),gx,np.int32(n),np.uint64(seed)) if kind=='physics' else (np.int32(which),gx,np.int32(n),np.uint64(seed))
    kernel=cp.RawKernel(core.cuda_source(),name+'_kernel',backend='nvrtc',options=core.CUDA_OPTIONS);kernel((1,),(1,),(*ga,gy,gs,gn));cp.cuda.Stream.null.synchronize();actual,actual_state,actual_next=map(cp.asnumpy,(gy,gs,gn))
    integer_cols=[2] if which==1 else [1] if which==2 else [0] if which in (3,4,6) else list(range(8+3*cpu.me,8+4*cpu.me)) if which==0 else []
    integer_ok=True if kind!='physics' else bool(np.array_equal(out[:,integer_cols],actual[:,integer_cols]))
    return dict(status='pass' if integer_ok and np.array_equal(state[625:],actual_state[625:]) else 'fail',integer_match=integer_ok,calls_cpu=int(state[625]),calls_gpu=int(actual_state[625]),float_match=float(np.mean(out.view(np.uint64)==actual.view(np.uint64))),next_match=float(np.mean(nxt.view(np.uint64)==actual_next.view(np.uint64))))

def gate2a(cpu,gpu):
    reports={};t=unpack_tables(cpu.a);cp=gpu.cp
    for which in range(7):reports[f'physics{which}']=gpu_probe(cpu,gpu,which,physics_inputs(t,which))
    # Replay the artificial sub-5-keV fluorescence table from gate 0a too.
    artificial={k:v.copy() for k,v in cpu.a.items()};at=unpack_tables(artificial)
    at.k_edge[0,0]=1.;at.k_omega[0,0]=1.;at.k_frac[0,0,:]=1.
    at.n_lines[0,0]=1;at.k_line_e[0,0,:]=4.;at.k_line_p[0,0,:]=0.;at.k_line_p[0,0,0]=1.
    at.material_names=list(range(len(at.n_elem)));artificial.update(core.pack(at))
    ac=core.Core(artificial,'cpu');ag=core.Core(artificial,'cuda')
    ax=np.zeros((1000,8));ax[:,0]=100.
    reports['artificial_fluorescence']=gpu_probe(ac,ag,3,ax)
    rng=np.random.default_rng(2026102100);d=rng.normal(size=(100000,3));d/=np.linalg.norm(d,axis=1,keepdims=True);x=np.ascontiguousarray(np.column_stack((d,rng.uniform(-1,1,len(d)))));x=np.concatenate((x,np.array([[0,0,1,c] for c in (-1.,0.,1.)]+[[0,1,0,c] for c in (-1.,0.,1.)])))
    for which in (0,1):reports[f'direction{which}']=gpu_probe(cpu,gpu,which,x,'direction')
    # Exact fixed geometry fixture used by gate 0a (including parallel rays).
    rng=np.random.default_rng(2026102101)
    centers=np.array([[0.,0,10],[0,0,10],[1,2,5]]);halves=np.array([[10.,10,10],[5,5,5],[2,3,4]])
    materials=np.array([1,2,3],np.int64);world=np.array([0.,0,10,60,60,60])
    x=rng.uniform(-20,20,(100000,10));x[:,6:9]=np.abs(x[:,6:9]);extra=[]
    for point in (-10.,0.,10.):
        for direction in (-1.,0.,1.):extra.append([point,point,point,direction,direction,direction,10,10,10,0])
    x=np.ascontiguousarray(np.r_[x,extra]);geo=core.Geometry(centers.ctypes.data,halves.ctypes.data,materials.ctypes.data,3,0,world.ctypes.data,None)
    k=cp.RawKernel(core.cuda_source(),'g1_geometry_probe_flat',backend='nvrtc',options=core.CUDA_OPTIONS)
    for which in range(3):
        expected=np.zeros((len(x),4));fn=cpu.lib.g1_geometry_probe;fn.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
        fn(which,ctypes.byref(geo),x.ctypes.data,len(x),expected.ctypes.data);gy=cp.zeros_like(cp.asarray(expected))
        k(((len(x)+127)//128,),(128,),(np.int32(which),cp.asarray(centers),cp.asarray(halves),cp.asarray(materials),np.int32(3),np.int32(0),cp.asarray(world),np.uint64(0),cp.asarray(x),np.int32(len(x)),gy));actual=cp.asnumpy(gy)
        cols=[2] if which==0 else [0] if which==1 else [1]
        integer=bool(np.array_equal(expected[:,cols],actual[:,cols]));reports[f'geometry{which}']=dict(status='pass' if integer else 'fail',integer_match=integer,float_match=float(np.mean(expected.view(np.uint64)==actual.view(np.uint64))))
    # Same line-segment construction as gate 0c, including tilted/moved plane.
    cp=gpu.cp
    for tilted in (False,True):
        normal=np.array([.2,-.1,1.]) if tilted else np.array([0.,0.,1.]);normal/=np.linalg.norm(normal);u=np.array([1.,2.,0.]) if tilted else np.array([1.,0.,0.]);u-=np.dot(u,normal)*normal;u/=np.linalg.norm(u)
        center=np.array([1.,-2.,3.]) if tilted else np.zeros(3);v=np.cross(normal,u);p=np.r_[center,normal,u,v,30.,30.,60.,60.]
        rng=np.random.default_rng(2026102102);x=np.zeros((100000,10));x[:,:3]=rng.uniform(-20,20,(len(x),3));x[:,3:6]=rng.normal(size=(len(x),3));x[:,3:6]/=np.linalg.norm(x[:,3:6],axis=1,keepdims=True);x[:,6]=rng.uniform(0,100,len(x));extra=[]
        for aa in (-15.,-14.5,0.,14.5,15.):
            for bb in (-15.,0.,15.):
                point=center+u*aa+v*bb
                for dd in (-normal,normal,u):extra.append(np.r_[point-dd*10,dd,10.,0,0,0])
        x=np.ascontiguousarray(np.r_[x,extra]);out=np.zeros((len(x),4));a=cpu.a
        geo=core.Geometry(a['centers'].ctypes.data,a['halves'].ctypes.data,a['materials'].ctypes.data,len(a['materials']),int(a['background'].item()),a['world'].ctypes.data,p.ctypes.data)
        fn=cpu.lib.g1_geometry_probe;fn.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
        fn(3,ctypes.byref(geo),x.ctypes.data,len(x),out.ctypes.data)
        gy=cp.zeros_like(cp.asarray(out));g=gpu.gpu;k=cp.RawKernel(core.cuda_source(),'g1_geometry_probe_flat',backend='nvrtc',options=core.CUDA_OPTIONS)
        k(((len(x)+127)//128,),(128,),(np.int32(3),g['centers'],g['halves'],g['materials'],np.int32(len(a['materials'])),np.int32(a['background'].item()),g['world'],cp.asarray(p),cp.asarray(x),np.int32(len(x)),gy));actual=cp.asnumpy(gy)
        integer=bool(np.array_equal(out[:,1:],actual[:,1:]));reports['detector_tilted' if tilted else 'detector']=dict(status='pass' if integer else 'fail',integer_match=integer,float_match=float(np.mean(out[:,0].view(np.uint64)==actual[:,0].view(np.uint64))))
    return dict(status='pass' if all(r['status']=='pass' for r in reports.values()) else 'fail',probes=reports)

def record_reduce(r,a):
    nm=int(a['nm'].item());i=r['i'];d=r['d'];result=dict(deposits=d[:,2:].sum(axis=0),endings=np.bincount(i[:,3],minlength=3).astype(float),scatter_sum=float(i[:,0].sum()),n_fluorescence=float(i[:,2].sum()))
    if len(a['detector']):
        plane=a['detector'];nu,nv=map(int,plane[14:16]);images=np.zeros((7,nu,nv));area=plane[12]*plane[13]/(nu*nv)
        hit=i[:,3]==2;ii=i[hit];ee=d[hit,0]
        np.add.at(images,(ii[:,6],ii[:,4],ii[:,5]),ee/area)
        for cat,weights in ((4,np.ones(len(ee))),(5,ee),(6,ee*ee)):np.add.at(images[cat],(ii[:,4],ii[:,5]),weights)
        result['images']=images
        counts=np.zeros((4,nu,nv));np.add.at(counts,(ii[:,6],ii[:,4],ii[:,5]),1.);result['category_count']=counts
    return result

def reconstruction_bounds(r,rebuilt,n):
    counts=r['category_count'];total=counts.sum(axis=0)
    image_counts=np.concatenate((counts,total[None],total[None],total[None]))
    additions=dict(deposits=2*n-1,endings=2*r['endings'],scatter_sum=2*n-1,n_fluorescence=2*n-1,images=2*image_counts,category_count=2*counts)
    return {k:np.asarray(additions[k])*2.**-52*(np.abs(rebuilt[k])+np.abs(r[k])) for k in rebuilt}

def gate2(path,scene_id,out,job):
    from .bundle import load
    from .source_host import sample_source_photons
    from .colab_driver import identity,read_result,write_result
    a,src,spec,settings,manifest=load(path);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    init=time.perf_counter();cpu=core.Core(a,'cpu');cpu_initialization=time.perf_counter()-init
    init=time.perf_counter();gpu=core.Core(a,'cuda');gpu_initialization=time.perf_counter()-init
    initialization=cpu_initialization+gpu_initialization
    probes=gate2a(cpu,gpu)
    if probes['status']!='pass':raise RuntimeError('gate 2a failed; stop before paired runs')
    o,d,e=sample_source_photons(src,2048,np.random.default_rng(8001),spec)
    cr=cpu.run(o,d,e,seed=8000,record=True,fluorescence=settings['fluorescence']);gr=gpu.run(o,d,e,seed=8000,record=True,fluorescence=settings['fluorescence'])
    bad=np.flatnonzero(np.any(cr['i'][:,:7]!=gr['i'][:,:7],axis=1));differences=[]
    for h in bad:
        traces=[]
        for engine in (cpu,gpu):traces.append(engine.run(o[h:h+1],d[h:h+1],e[h:h+1],seed=8000,start=int(h),fluorescence=settings['fluorescence'],record=True,trace_capacity=100000)['trace'])
        ct,gt=traces;steps=min(len(ct),len(gt));changed=np.flatnonzero(np.any(ct[:steps].view(np.uint64)!=gt[:steps].view(np.uint64),axis=1))
        if len(ct)!=len(gt):changed=np.r_[changed,steps]
        columns=['step','x','y','z','material','rng_calls','energy','tau']
        for first in changed:
            first=int(first);cc=ct[first] if first<len(ct) else None;gg=gt[first] if first<len(gt) else None
            fields=columns if cc is None or gg is None else [columns[j] for j in np.flatnonzero(cc.view(np.uint64)!=gg.view(np.uint64))]
            differences.append(dict(history=int(h),first_step=first,fields=fields,cpu=cc.tolist() if cc is not None else None,gpu=gg.tolist() if gg is not None else None))
            if len(differences)==5:break
        if len(differences)==5:break
    report=dict(gate2a=probes,gate2b=dict(n_histories=2048,integer_diverged=len(bad),float_match=float(np.mean(cr['d'].view(np.uint64)==gr['d'].view(np.uint64))),replays=differences,integer_field_match={name:float(np.mean(cr['i'][:,j]==gr['i'][:,j])) for j,name in enumerate(('interactions','compton_rayleigh','fluorescence','ending','iu','iv','category','rng_calls','error'))},per_history_exact=float(np.mean(np.all(cr['d'].view(np.uint64)==gr['d'].view(np.uint64),axis=1)&np.all(cr['i']==gr['i'],axis=1)))),initialization_s=initialization)
    rng=np.random.default_rng(8001);cq=[];gq=[];bounds=[];cpu_s=gpu_s=source_s=0.
    for batch in range(10):
        t=time.perf_counter();o,d,e=sample_source_photons(src,1000000,rng,spec);source_s+=time.perf_counter()-t
        key=identity(manifest,job,batch,batch*1000000,1000000);name=f'paired_{batch:02d}.npz'
        saved=read_result(out,name,key)
        if saved is None:
            t=time.perf_counter();cr=cpu.run(o,d,e,seed=8000,start=batch*1000000,fluorescence=settings['fluorescence']);ct=time.perf_counter()-t
            t=time.perf_counter();gr=gpu.run(o,d,e,seed=8000,start=batch*1000000,fluorescence=settings['fluorescence']);gt=time.perf_counter()-t
            qc=quantities(cr,settings['material_names'],scene_id,1000000);qg=quantities(gr,settings['material_names'],scene_id,1000000);add_c=quantity_additions(cr,settings['material_names'],scene_id,1000000);add_g=quantity_additions(gr,settings['material_names'],scene_id,1000000)
            saved=dict(cpu=np.array(list(qc.values())),gpu=np.array(list(qg.values())),additions=np.array([add_c[k]+add_g[k] for k in qc]),cpu_s=np.array(ct),gpu_s=np.array(gt),keys=np.array(list(qc)))
            write_result(out,name,saved,key)
        cpu_s+=float(saved['cpu_s']);gpu_s+=float(saved['gpu_s']);cq.append(saved['cpu']);gq.append(saved['gpu'])
        # The signed difference contains CPU and GPU additions. Their exact
        # addition counts and sum of absolute (nonnegative) contributions are saved.
        bounds.append(saved['additions']*2.**-52*(np.abs(saved['cpu'])+np.abs(saved['gpu'])))
    report['gate2c']=paired_judge(cq,gq,bounds);report['gate2c']['keys']=saved['keys'].tolist()
    # Independent reconstruction from 100,000 recorded histories, both devices.
    o,d,e=sample_source_photons(src,100000,np.random.default_rng(8001),spec);reconstruction={}
    for label,engine in (('cpu',cpu),('gpu',gpu)):
        r=engine.run(o,d,e,seed=8000,record=True,fluorescence=settings['fluorescence']);rr=record_reduce(r,a);checks={}
        limits=reconstruction_bounds(r,rr,100000)
        for k in rr:
            diff=np.abs(np.asarray(rr[k])-r[k]);checks[k]=bool(np.all(diff<=limits[k]))
        reconstruction[label]=checks
    report['reconstruction']=reconstruction
    report['timing']=dict(cpu_initialization_s=cpu_initialization,gpu_initialization_s=gpu_initialization,cpu_s=cpu_s,gpu_s=gpu_s,source_s=source_s,cpu_histories_per_s=10000000/cpu_s,gpu_histories_per_s=10000000/gpu_s,including_source_and_initialization_s=gpu_s+source_s+gpu_initialization)
    report.update(scene_id=scene_id,scene_sha256=settings['scene_sha256'],seeds=dict(transport=8000,source=8001),batch_size=1000000,n_batches=10,bundle=manifest)
    report['status']='pass' if report['gate2c']['status']=='pass' and all(all(v.values()) for v in reconstruction.values()) else 'fail'
    report['timing']['request_d_estimate_s_9x2']=2*sum(round(10**x) for x in np.linspace(4,8,9))*(gpu_s+source_s)/10000000+18*gpu_initialization
    write_result(out,'gate2.json',report,identity(manifest,job,-1,0,10000000));return report
