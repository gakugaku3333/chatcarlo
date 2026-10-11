"""Preregistered T2/T3: angular consistency separately from energy interpolation."""
import json
import numpy as np
from scipy.interpolate import CubicSpline
from .check_elsepa import Atomic, integrated, read_comparator
from .dcs import get_backend, moments
from .gs_backend import single_cdf
from .generate_elsepa_p1f import energies, prepare, loaded
from .p1f_common import OUT, WORK, CACHE, write, sha, display

def point_data():
    source,exe=prepare();es=energies();mid=np.sqrt(es[:-1]*es[1:]);cases=[]
    for z in (1,8):
        for q in (-1,1):
            for group,ts in (('midpoints',mid),('probes',.0201*z*np.array([.99,1.01]))):
                for t in ts:
                    _,r=loaded(source,exe,group,z,q,float(t),0)
                    cases.append(dict(Z=z,charge=q,T=float(t),group=group,IHEF=0,seconds=r['seconds'],raw_hashes=r['outputs']))
                    write('direct_progress.json',dict(completed=cases))
                    print('direct case saved',group,z,q,float(t),round(r['seconds'],3),flush=True)
    write('direct_cases.json',dict(status='complete',rows=cases))

def inspect(a,z,q,t,kind,seconds=None):
    b=Atomic(a['mu'],a['dcs']);v=integrated(b,t);v2=integrated(b,t,order=32)
    relative=v/a['table_xs']-1;double=v2/v-1
    valid=bool(np.all(np.isfinite(a['dcs'])) and np.all(a['dcs']>0) and np.all(np.isfinite(a['error'])) and np.all(a['error']>=0))
    replacement=int(np.sum((a['dcs']==1e-45)&(a['error']==1)))
    series_relative=a['table_xs']/a['series_xs']-1
    raw_warnings=a['warnings'].tolist();warnings=[];accepted=[];rejected=[]
    # P1e parser records the same multiline warning at its heading and continuation.
    # Retain all raw blocks, count each warning event once before any judgments.
    for w in raw_warnings:
        if w.lstrip().lower().startswith('the dcs table is not consistent') and warnings and 'the dcs table is not consistent' in warnings[-1].lower():continue
        warnings.append(w)
    for w in warnings:
        consistency='the dcs table is not consistent' in w.lower()
        ok=consistency and np.all(np.isfinite(series_relative)) and abs(series_relative[0])<=.005 and abs(series_relative[1])<=.001
        (accepted if ok else rejected).append(w)
    reasons=[]
    if not valid:reasons.append('finite-positive DCS / finite-nonnegative ERROR')
    if replacement:reasons.append('invalid replacement')
    if max(a['error'])>.001:reasons.append('ERROR maximum')
    if rejected:reasons.append('rejected numerical warning')
    if not np.all(np.isfinite(a['series_xs'])):reasons.append('missing printed series moments')
    if abs(relative[0])>.003:reasons.append('angular total')
    if max(abs(relative[1:]))>.001:reasons.append('angular transport')
    if max(abs(double))>1e-6:reasons.append('quadrature doubling')
    return dict(Z=z,charge=q,T=float(t),kind=kind,seconds=seconds,valid=valid,invalid_replacements=replacement,
        ERROR_max=float(max(a['error'])),prototype_xs=v.tolist(),table_xs=a['table_xs'].tolist(),series_xs=a['series_xs'].tolist(),
        relative_table=relative.tolist(),relative_series=(v/a['series_xs']-1).tolist(),table_over_series_relative=series_relative.tolist(),
        doubling_relative=double.tolist(),raw_warning_blocks=raw_warnings,accepted_consistency_warnings=accepted,rejected_warnings=rejected,notices=a['notices'].tolist(),
        convergence=json.loads(str(a['convergence'])),status='fail' if reasons else 'pass',failure_quantities=reasons)

def s2():
    source,exe=prepare();rows=[];interpolation=[];probes=[];ihef=[];es=energies();mid=np.sqrt(es[:-1]*es[1:])
    for z in (1,8):
        for q in (-1,1):
            with np.load(WORK/'tables'/f'z{z}_q{q}.npz') as data:a={k:data[k] for k in data.files}
            assert int(a['IHEF'])==0
            s=CubicSpline(np.log(es),np.log(a['dcs']),axis=0,bc_type='natural')
            for i,t in enumerate(es):
                atom={k:a[k][i] for k in ('dcs','error','series_xs','table_xs','convergence')};atom['mu']=a['mu']
                for k in ('warnings','notices'):atom[k]=np.array(json.loads(str(a[k][i])),dtype=str)
                receipt=json.loads((WORK/'cases/pass1'/f'z{z}_q{q}_e{float(t).hex()}_h0/complete.json').read_text())
                rows.append(inspect(atom,z,q,t,'node',receipt['seconds']))
            for group,ts in (('midpoints',mid),('probes',.0201*z*np.array([.99,1.01]))):
                pr=[]
                for t in ts:
                    direct,r=loaded(source,exe,group,z,q,float(t),0)
                    rows.append(inspect(direct,z,q,t,group,r['seconds']))
                    b=Atomic(a['mu'],np.exp(s(np.log(t))));d=Atomic(direct['mu'],direct['dcs'])
                    v=integrated(b,t);reference=integrated(d,t)
                    if group=='probes':pr.append(dict(T=float(t),xs=reference.tolist(),printed_xs=direct['table_xs'].tolist()));continue
                    yi,ci=single_cdf(b,t);yd,cd=single_cdf(d,t);grid=np.unique(np.r_[yi,yd])
                    cdf=float(max(abs(np.interp(grid,yi,ci)-np.interp(grid,yd,cd))))
                    relative=v/reference-1;reasons=[]
                    for i,k in enumerate(('total','G1','G2')):
                        if abs(relative[i])>.002:reasons.append('energy '+k)
                    if cdf>.002:reasons.append('energy CDF')
                    interpolation.append(dict(Z=z,charge=q,T=float(t),relative=relative.tolist(),interpolated_xs=v.tolist(),direct_xs=reference.tolist(),
                        relative_printed=(v/direct['table_xs']-1).tolist(),CDF_max_difference=cdf,status='fail' if reasons else 'pass',failure_quantities=reasons))
                if group=='probes':probes.append(dict(Z=z,charge=q,points=pr,relative_after_over_before=(np.array(pr[1]['xs'])/pr[0]['xs']-1).tolist(),note='±1%の2点の記録。連続性の判定ではない。'))
            old_path=CACHE/'p1e/tables'/f'z{z}_q{q}.npz'
            old_manifest=json.loads((OUT.parent/'p1e/manifest.json').read_text())['files']
            if sha(old_path)!=old_manifest[display(old_path)]:raise RuntimeError('P1e comparison table no longer matches preserved manifest')
            with np.load(old_path) as old:so=CubicSpline(np.log(old['energy_MeV']),np.log(old['dcs']),axis=0,bc_type='natural')
            for t in (.01,.02,.1,1.,2.,10.):
                v=integrated(Atomic(a['mu'],np.exp(s(np.log(t)))),t);vo=integrated(Atomic(a['mu'],np.exp(so(np.log(t)))),t)
                ihef.append(dict(Z=z,charge=q,T=t,IHEF0_over_1=(v/vo).tolist(),IHEF0_xs=v.tolist(),IHEF1_xs=vo.tolist()))
    maxima=dict(table_total_relative=max(abs(r['relative_table'][0]) for r in rows),table_transport_relative=max(max(abs(x) for x in r['relative_table'][1:]) for r in rows),
        doubling_relative=max(max(abs(x) for x in r['doubling_relative']) for r in rows),ERROR_max=max(r['ERROR_max'] for r in rows),
        interpolation_relative=max(max(abs(x) for x in r['relative']) for r in interpolation),CDF_difference=max(r['CDF_max_difference'] for r in interpolation))
    warnings=[{k:r[k] for k in ('Z','charge','T','kind','table_over_series_relative','accepted_consistency_warnings','rejected_warnings','notices')} for r in rows]
    result=dict(status='pass' if all(r['status']=='pass' for r in rows+interpolation) else 'fail',rows=rows,interpolation=interpolation,probes=probes,IHEF_record=ihef,maxima=maxima,
        invalid_replacements=sum(r['invalid_replacements'] for r in rows),accepted_consistency_warning_count=sum(len(r['accepted_consistency_warnings']) for r in rows),
        rejected_warning_count=sum(len(r['rejected_warnings']) for r in rows),table_hashes={p.name:sha(p) for p in (WORK/'tables').glob('*.npz')})
    write('warnings.json',warnings);write('t2.json',result);return result

def s3():
    rows=[];water=[];ratios=[]
    from .dcs import inigrd
    from .p1f_common import provider
    for z in (1,8):
        for q in (-1,1):
            with np.load(WORK/'tables'/f'z{z}_q{q}.npz') as a:
                es,ds,hs=read_comparator(z,q);s=CubicSpline(np.log(a['energy_MeV']),np.log(a['dcs']),axis=0,bc_type='natural')
                for e,d,h in zip(es,ds,hs):
                    if not .01<=e<=10:continue
                    ours=Atomic(a['mu'],np.exp(s(np.log(e))));other=Atomic(inigrd(),d)
                    xs=integrated(ours,e);oxs=integrated(other,e)
                    ratio=np.exp(ours.curve(a['mu'])-other.curve(a['mu']))
                    rows.append(dict(Z=z,charge=q,T=float(e),xs_ratio=(xs/oxs).tolist(),dcslib_header_ratio=(xs/h).tolist(),DCS_ratio_min=float(min(ratio)),DCS_ratio_max=float(max(ratio))))
                for e in (.01,.02,.1,1.,2.,10.):
                    d=CubicSpline(np.log(es),np.log(ds),axis=0,bc_type='natural')(np.log(e));other=Atomic(inigrd(),np.exp(d));ours=Atomic(a['mu'],np.exp(s(np.log(e))))
                    ratios.append(dict(Z=z,charge=q,T=e,theta_deg=a['theta_deg'].tolist(),ratio=np.exp(ours.curve(a['mu'])-other.curve(a['mu'])).tolist()))
    from numpy.polynomial.legendre import leggauss
    for name in ('elsepa','dcslib','mf'):
        p=provider() if name=='elsepa' else get_backend(name)
        for e in (.01,.02,.1,1.,2.,10.):
            g,total,_=moments(p,e);edges=np.unique(np.r_[p.edges(e),1.]);edges=edges[edges>=1];x,w=leggauss(32)
            y=(edges[:-1,None]+edges[1:,None])/2+(edges[1:,None]-edges[:-1,None])*x/2
            back=float(np.sum(p.density(e,y)*w*(edges[1:,None]-edges[:-1,None])/2))
            water.append(dict(backend=name,T=e,G1=float(g[1]),G2=float(g[2]),back_hemisphere=back,total=total))
    result=dict(status='not-evaluated',reason='記録項目。合否規則なし。計算済み。',rows=rows,water=water,DCS_ratios=ratios)
    write('t3.json',result);return result
