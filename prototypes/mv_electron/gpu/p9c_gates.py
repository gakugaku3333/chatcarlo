"""Colab-only atomic/connection/rotation gates and compact paired reports."""
import numpy as np
from . import p9c_core as core

def v1b(engine):
    cp=engine.cp
    engine.adds=[cp.zeros(g.shape,np.uint64) for g in engine.grids];engine.upload();engine.clear()
    n=100003;loss=.1;p=(-1.5,)*3;engine.score(p,p,loss,site=1,repeats=n)
    for grid,values,adds in zip(engine.grids,engine.values,engine.adds):
        v=cp.asnumpy(values);a=cp.asnumpy(adds);expected=np.zeros(grid.shape);expected[grid.index(p)]=n*loss
        bound=(a+1)*2.**-52*abs(expected)
        if np.any(abs(v-expected)>bound) or a.sum()!=n:raise AssertionError('V1b atomic addition bound')
    return dict(status='pass',histories=n,loss=loss)

def v3_acceptance():
    from scipy.integrate import quad
    report={};rng=np.random.default_rng(940103)
    for angle in core.ANGLES:
        s,_,u,v=core.frame(angle);s=s/core.AXES;u=u/core.AXES;v=v/core.AXES;C=s@s-1
        def discriminant(a,b):
            d=a*u+b*v-s;return (s@d)**2-(d@d)*C
        def width(a):
            c=discriminant(a,0);plus=discriminant(a,1);minus=discriminant(a,-1);A=(plus+minus)/2-c;B=(plus-minus)/2;D=B*B-4*A*c
            if D<=0:return 0.
            roots=sorted(((-B-np.sqrt(D))/(2*A),(-B+np.sqrt(D))/(2*A)))
            return max(0.,min(4.,roots[1])-max(-4.,roots[0]))
        # Adaptive integration of the independent discriminant's projected width.
        area=quad(width,-4.,4.,epsabs=1e-8,points=[-2.5,-2.,-1.5,0.,1.5,2.,2.5],limit=200)[0]
        samples=rng.uniform(-4,4,(100000,2));S,_,U,V=core.frame(angle);d=samples[:,0,None]*U+samples[:,1,None]*V-S
        x=S/core.AXES;dd=d/core.AXES;B=dd@x;accepted=(B<0)&(B*B-(dd*dd).sum(1)*(x@x-1)>=0)
        expected=area/64.;observed=accepted.mean();sigma=np.sqrt(expected*(1-expected)/len(samples))
        if abs(observed-expected)>3*sigma:raise AssertionError('V3 acceptance outside 3 sigma')
        report[str(angle)]=dict(projected_area_cm2=area,expected=expected,observed=float(observed),sigma=float(sigma))
    return dict(status='pass',angles=report)

def source_acceptance(measurements,nb):
    expected=v3_acceptance()['angles'];result={}
    for field,angle in enumerate(core.ANGLES):
        rows=[r for m in measurements.values() for r in m['timings'] if r['field']==field and r['with_seeds']]
        accepted=nb*len(rows);rejected=sum(r['rejected'] for r in rows);trials=accepted+rejected
        if not trials:raise AssertionError('V3 missing source measurements')
        p=expected[str(angle)]['expected'];observed=accepted/trials;sigma=np.sqrt(p*(1-p)/trials)
        if abs(observed-p)>3*sigma:raise AssertionError('V3 CUDA sampler acceptance outside 3 sigma at '+str(angle))
        result[str(angle)]=dict(accepted=accepted,rejected=rejected,expected=p,observed=observed,sigma=float(sigma))
    return dict(status='pass',angles=result)


def summarize(checkpoint,fine,selection,measurements):
    result={}
    for key in ('aggregate','shadow','lines','representative'):
        r=checkpoint.moments[key].result()
        for k in ('mean_with','mean_without','sem_with','sem_without','difference','difference_sem'):r[k]=r[k]/fine.volume
        r['covariance']=r['covariance']/fine.volume**2
        result[key]={k:np.where(np.isfinite(v),v,np.nan).tolist() for k,v in r.items()}
    # JSON null is the undefined marker for zero denominators/zero variances.
    def clean(v):
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,list):return [clean(x) for x in v]
        if isinstance(v,float) and not np.isfinite(v):return None
        return v
    result=clean(result);mask,_,_,lines,ids=selection;x,y,z=fine.centers()
    result.update(batches=len(checkpoint.completed),line_seed_ids=ids,line_cell_centers=[[float(x[i]),float(y[j])] for i,j in lines],line_z_cm=z.tolist(),line_valid=[mask[i,j,:].tolist() for i,j in lines],aggregate_order=['prostate_mean','rectum_mean'],checkpoint_sha256=checkpoint.digest,measurements=measurements,water_density_g_cm3=1.,dose_unit='MeV/g per accepted photon',voxel_volume_cm3=fine.volume)
    return result

def region_grid_sums(values,grid,geometry):
    out=np.zeros(3420)
    for r in range(15):
        axis=geometry.axes[r];origin=geometry.origins[r];width=geometry.widths[r];offset=geometry.offsets[r]
        for b in range(geometry.sizes[r]):
            lo=np.array(geometry.lo[3*r:3*r+3]);hi=np.array(geometry.hi[3*r:3*r+3]);lo[axis]=max(lo[axis],origin+b*width);hi[axis]=min(hi[axis],origin+(b+1)*width)
            lo=np.maximum(lo,grid.lo);hi=np.minimum(hi,grid.hi)
            if np.any(hi<=lo):continue
            start=np.rint((lo-grid.lo)/grid.step).astype(int);stop=np.rint((hi-grid.lo)/grid.step).astype(int)
            out[offset+b]=values[tuple(slice(a,z) for a,z in zip(start,stop))].sum()
    return out

def v1c(cp,materials,record,output):
    from .p9c_colab import GPU,json_write
    # Common divisor of P7's .125/.25 lateral and .1/.25 depth faces.
    g=core.Grid((-20.,-20.,0.),(.125,.125,.05),(320,320,600))
    # Second grid is tiny but covers the body for the independent ledger check.
    coarse=core.Grid((-20.,-20.,0.),(40.,40.,30.),(1,1,1))
    e=GPU(cp,materials,densities=record['densities'],edges=record['edges'],cum=record['cum'],layers=((1,30.,.2),),cylinders=(),half=(20.,20.,30.),score_field=True,field_half=2.5,grids=(g,coarse))
    e.adds=[cp.zeros(x.shape,np.uint64) for x in e.grids];e.upload()
    total=np.zeros(g.shape);adds=np.zeros(g.shape,np.uint64);regions=np.zeros(3420);sites=np.zeros(6,np.int64)
    for start in range(0,200000,1000):
        r=e.run(1000,record['seed']+1,start=start,regions=True)
        total+=cp.asnumpy(r['grids'][0]);adds+=cp.asnumpy(e.adds[0]);regions+=r['regions'].sum(0);sites+=r['sites'].sum(0)
    actual=region_grid_sums(total,g,e.g.scoring)
    # Each term is nonnegative; count all grid additions plus history and chunk
    # reductions, a conservative P7 c3 forward-error bound for each region.
    K=int(adds.sum())+200000+200;bound=(K*2.**-52)*np.maximum(abs(actual),abs(regions))
    if np.any(abs(actual-regions)>bound):raise AssertionError('V1c region/grid connection')
    forced=[]
    for j in np.flatnonzero(sites==0):
        state=np.zeros(23);state[:3]=(0.,0.,10.03125);state[3]=1.;state[6]=state[7]=.03;state[19]=-1;state[8]=100.;state[12]=state[15]=1e100;state[18]=1
        if j==0:state[19]=0;state[6]=state[7]=.007
        if j==1:state[6]=state[7]=.01;state[18]=0
        if j==3:state[8]=0.
        if j==5:state[19]=0;state[6]=state[7]=.03
        fixed=e
        if j==5:
            fixed=GPU(cp,materials,densities=record['densities'],layers=((1,30.,.2),),cylinders=((0.,0.,10.,.05,.25,4),),half=(20.,20.,30.),grids=(g,coarse))
            state[16]=1
        fixed.clear();out=cp.zeros(15)
        fixed.module.get_function('p9c_fixed_kernel')((1,),(1,),fixed.args+(fixed.geometry,cp.asarray(state),np.uint64(42),out))
        values=cp.asnumpy(out)
        if values[j]<1 or values[14]!=0 or abs(values[6:12].sum()-values[12])>1e-12:raise AssertionError('V1c missing fixed site '+str(j+1))
        for grid_values in fixed.values:
            if abs(float(grid_values.sum())-values[12])>1e-9*max(1.,values[12]):raise AssertionError('V1c fixed site grid connection')
        forced.append(dict(site=int(j+1),reached=int(values[j])))
    report=dict(status='pass',histories=200000,sites=sites.tolist(),forced=forced,max_abs=float(abs(actual-regions).max()),addition_bound_K=K)
    json_write(output/'V1c.json',report);return report

def v2b(cp,materials,record,output):
    from .p9c_colab import GPU,json_write
    grid=core.Grid((-12.2,-15.2,-12.2),(.2,)*3,(122,152,122));x,y,z=grid.centers()
    arrays=[];dref=[]
    for f,angle in enumerate((0,90)):
        e=GPU(cp,materials,densities=record['densities'],edges=record['edges'],cum=record['cum'],radii=(12.,15.,12.),cylinders=(),angle=angle,grids=(grid,grid))
        batches=[];refs=[]
        for b in range(100):
            r=e.run(20000,record['seed']+2,start=b*20000,field=f);values=cp.asnumpy(r['grids'][0])/20000
            # +90 source at +x: undo rotation into the 0-degree coordinates.
            if angle==90:values=np.flip(values.transpose(2,1,0),axis=2)
            profile=values[abs(x)<1][:,abs(y)<1,:].mean((0,1));depth=z+12.
            batches.append([profile[(depth>=a)&(depth<a+5)].mean() for a in (5,10,15)])
            refs.append(float(np.interp(10.,depth,profile)))
        arrays.append(np.array(batches));dref.append(np.mean(refs))
    difference=arrays[1].mean(0)-arrays[0].mean(0);sem=np.sqrt(arrays[0].var(0,ddof=1)/100+arrays[1].var(0,ddof=1)/100);ref=dref[0]
    if ref<=0:raise AssertionError('V2b invalid Dref')
    insufficient=bool(np.any(sem>.0033*ref))
    status='insufficient-statistics' if insufficient else ('pass' if np.all(abs(difference)<=.01*ref) else 'fail')
    report=dict(status=status,difference_over_Dref=(difference/ref).tolist(),sem_over_Dref=(sem/ref).tolist(),Dref=ref,fallback='V2a' if insufficient else None,histories_per_angle=2000000)
    json_write(output/'V2b.json',report)
    if status=='fail':raise AssertionError('V2b failed')
    return report

def production_gates(cp,materials,record,output):
    from pathlib import Path
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    return dict(V1c=v1c(cp,materials,record,output),V2b=v2b(cp,materials,record,output))
