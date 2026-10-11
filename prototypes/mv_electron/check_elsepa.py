"""S2/S3 deterministic checks; no parameter fitting or renormalization."""
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from .dcs import Dcslib, DCSLIB, get_backend, moments
from .elsepa import Elsepa
from .generate_elsepa import energies, prepare, loaded
from .gs_backend import single_cdf
from .p1e_common import OUT, WORK, write

class Atomic(Dcslib):
    def __init__(self,mu,values):
        self.rmu=mu
        self.curve=CubicSpline(mu,np.log(values),bc_type='natural')
        self.scale=1.
    def spline(self,T,z=None): return self.curve

def integrated(provider,t,z=None,order=16):
    g,total,_=moments(provider,t,order=order,z=z)
    return np.r_[total,g[1:3]]

def point_data():
    source,exe=prepare();es=energies();nodes=np.unique(np.r_[.01,es[(es>.01)&(es<20)],20.])
    mid=np.sqrt(nodes[:-1]*nodes[1:]);cases=[]
    for z in (1,8):
        switch=.0201*z
        ts=np.unique(np.r_[mid,switch*(1-1e-6),switch*(1+1e-6)])
        for q in (-1,1):
            for t in ts:
                _,r=loaded(source,exe,'midpoints',z,q,float(t))
                cases.append(dict(Z=z,charge=q,T=float(t),seconds=r['seconds']))
            for t in (.01,.02,.1,1.,2.,10.):
                _,r=loaded(source,exe,'ihef0',z,q,t,0)
                cases.append(dict(Z=z,charge=q,T=t,IHEF=0,seconds=r['seconds']))
    write('direct_cases.json',dict(status='complete',rows=cases))

def s2():
    source,exe=prepare();rows=[];interpolation=[];ihef=[];warning_rows=[]
    es=energies();nodes=np.unique(np.r_[.01,es[(es>.01)&(es<20)],20.]);mid=np.sqrt(nodes[:-1]*nodes[1:])
    for z in (1,8):
        for q in (-1,1):
            with np.load(WORK/'tables'/f'z{z}_q{q}.npz') as data:
                a={k:data[k] for k in data.files}
            energy_spline=CubicSpline(np.log(a['energy_MeV']),np.log(a['dcs']),axis=0,bc_type='natural')
            for i,t in enumerate(a['energy_MeV']):
                b=Atomic(a['mu'],a['dcs'][i]);v=integrated(b,t);v2=integrated(b,t,order=32)
                relative=v/a['table_xs'][i]-1;double=v2/v-1
                warnings=json.loads(str(a['warnings'][i]));notices=json.loads(str(a['notices'][i]))
                valid=bool(np.all(np.isfinite(a['dcs'][i])) and np.all(a['dcs'][i]>0) and np.all(np.isfinite(a['error'][i])) and not np.any((a['dcs'][i]==1e-45)&(a['error'][i]==1)))
                good=valid and not warnings and abs(relative[0])<=.003 and max(abs(relative[1:]))<=.001 and max(abs(double))<=1e-6
                rows.append(dict(Z=z,charge=q,T=float(t),valid=valid,numerical_warnings=len(warnings),factorization_notices=len(notices),prototype_xs=v.tolist(),table_xs=a['table_xs'][i].tolist(),series_xs=[float(x) if np.isfinite(x) else None for x in a['series_xs'][i]],relative_table=relative.tolist(),relative_series=[float(x/y-1) if np.isfinite(y) else None for x,y in zip(v,a['series_xs'][i])],doubling_relative=double.tolist(),status='pass' if good else 'fail'))
                warning_rows.append(dict(Z=z,charge=q,T=float(t),warnings=warnings,notices=notices))
            switch=.0201*z
            for t in np.unique(np.r_[mid,switch*(1-1e-6),switch*(1+1e-6)]):
                direct,_=loaded(source,exe,'midpoints',z,q,float(t))
                b=Atomic(a['mu'],np.exp(energy_spline(np.log(t))));d=Atomic(direct['mu'],direct['dcs'])
                v=integrated(b,t);reference=integrated(d,t)
                yi,ci=single_cdf(b,t);yd,cd=single_cdf(d,t);grid=np.unique(np.r_[yi,yd])
                cdf=float(max(abs(np.interp(grid,yi,ci)-np.interp(grid,yd,cd))))
                relative=v/direct['table_xs']-1; energy_only_relative=v/reference-1; valid=bool(np.all(np.isfinite(direct['dcs'])) and np.all(direct['dcs']>0) and not np.any((direct['dcs']==1e-45)&(direct['error']==1)))
                good=valid and not len(direct['warnings']) and max(abs(relative))<=.002 and cdf<=.002
                interpolation.append(dict(Z=z,charge=q,T=float(t),switch_probe=bool(abs(t/switch-1)<2e-6),relative=relative.tolist(),interpolated_xs=v.tolist(),direct_xs=reference.tolist(),relative_same_angular_interpolator=energy_only_relative.tolist(),CDF_max_difference=cdf,status='pass' if good else 'fail'))
                warning_rows.append(dict(Z=z,charge=q,T=float(t),direct=True,warnings=direct['warnings'].tolist(),notices=direct['notices'].tolist()))
            for t in (.01,.02,.1,1.,2.,10.):
                direct,_=loaded(source,exe,'ihef0',z,q,t,0)
                v=integrated(Atomic(a['mu'],np.exp(energy_spline(np.log(t)))),t)
                v0=integrated(Atomic(direct['mu'],direct['dcs']),t)
                ihef.append(dict(Z=z,charge=q,T=t,IHEF1_over_0=(v/v0).tolist(),IHEF0_xs=v0.tolist()))
    result=dict(status='pass' if all(r['status']=='pass' for r in rows+interpolation) else 'fail',rows=rows,interpolation=interpolation,IHEF_record=ihef,
        maxima=dict(table_total_relative=max(abs(r['relative_table'][0]) for r in rows),table_transport_relative=max(max(abs(x) for x in r['relative_table'][1:]) for r in rows),doubling_relative=max(max(abs(x) for x in r['doubling_relative']) for r in rows),interpolation_relative=max(max(abs(x) for x in r['relative']) for r in interpolation),CDF_difference=max(r['CDF_max_difference'] for r in interpolation)),numerical_warning_count=sum(len(r['warnings']) for r in warning_rows),factorization_notice_count=sum(len(r['notices']) for r in warning_rows))
    write('warnings.json',warning_rows);write('s2.json',result);return result

def read_comparator(z,q):
    lines=(DCSLIB/f'{"eel" if q==-1 else "pel"}dx{z:03}.tab').read_text().splitlines();es=[];heads=[];values=[];i=0
    while i<len(lines):
        h=lines[i].split();i+=1
        if int(h[0])!=q or int(h[1])!=z:raise ValueError('comparator identity')
        es.append(float(h[2])*1e-6);heads.append(list(map(float,h[3:6])));v=[]
        while len(v)<606:v.extend(map(float,lines[i].split()));i+=1
        if len(v)!=606:raise ValueError('comparator row count')
        values.append(v)
    return np.array(es),np.array(values),np.array(heads)

def s3():
    rows=[];water=[];ratios=[]
    from .dcs import inigrd
    for z in (1,8):
        for q in (-1,1):
            with np.load(WORK/'tables'/f'z{z}_q{q}.npz') as a:
                es,ds,hs=read_comparator(z,q);s=CubicSpline(np.log(a['energy_MeV']),np.log(a['dcs']),axis=0,bc_type='natural')
                for e,d,h in zip(es,ds,hs):
                    if not .01<=e<=20:continue
                    ours=Atomic(a['mu'],np.exp(s(np.log(e))));other=Atomic(inigrd(),d)
                    xs=integrated(ours,e);oxs=integrated(other,e)
                    ratio=np.exp(ours.curve(a['mu'])-other.curve(a['mu']))
                    rows.append(dict(Z=z,charge=q,T=float(e),xs_ratio=(xs/oxs).tolist(),dcslib_header_ratio=(xs/h).tolist(),DCS_ratio_min=float(min(ratio)),DCS_ratio_max=float(max(ratio))))
                for e in (.01,.02,.1,1.,2.,10.):
                    d=CubicSpline(np.log(es),np.log(ds),axis=0,bc_type='natural')(np.log(e));other=Atomic(inigrd(),np.exp(d));ours=Atomic(a['mu'],np.exp(s(np.log(e))))
                    ratios.append(dict(Z=z,charge=q,T=e,theta_deg=a['theta_deg'].tolist(),ratio=np.exp(ours.curve(a['mu'])-other.curve(a['mu'])).tolist()))
    from numpy.polynomial.legendre import leggauss
    for name in ('elsepa','dcslib','mf'):
        p=get_backend(name)
        for e in (.01,.02,.1,1.,2.,10.):
            g,total,_=moments(p,e);edges=np.unique(np.r_[p.edges(e),1.]);edges=edges[edges>=1];x,w=leggauss(32)
            y=(edges[:-1,None]+edges[1:,None])/2+(edges[1:,None]-edges[:-1,None])*x/2
            back=float(np.sum(p.density(e,y)*w*(edges[1:,None]-edges[:-1,None])/2))
            water.append(dict(backend=name,T=e,G1=float(g[1]),G2=float(g[2]),back_hemisphere=back,total=total))
    result=dict(status='not-evaluated',reason='記録項目。合否規則なし。計算済み。',rows=rows,water=water,DCS_ratios=ratios)
    write('s3.json',result);return result
