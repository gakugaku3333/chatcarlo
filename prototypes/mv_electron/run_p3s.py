"""P3s preregistered checks, resumable runs and fixed-region analysis."""
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import time
import numpy as np
from numba import njit
from . import run_p3a as base
from .p2_common import ROOT, sha
from .run_coupled_p2 import inputs as load_inputs, MEV_TO_GY_G
from .photon_transport_p3s import run_coupled, rule_value, hinge_transition, scattering_hinge
from .sampling import gs_mu_cached
from .backend_sampling import table_rates

OUT = ROOT/'docs/validation/mv/p3s'
CACHE = Path(os.environ.get('CHATCARLO_P3S_CACHE','~/.cache/chatcarlo/p3s')).expanduser()
KFILE = ROOT/'docs/egs5_crosscheck/egs5/data/K1.dat'
PLAN = ROOT/'docs/ai/plans/2026-10-11-mv-p3s-step-speedup.md'
KEYS = base.KEYS+('lateral','steps')
PHYSICS = ('coarse','fine','totals','maxima','violations','counts','maxdepth','nspec')
RUNS = {'p3s_own':('own',20291530),'p3s_egs':('egs',20291531)}
write, save, load, plain = base.write, base.save, base.load, base.plain


@lru_cache(maxsize=2)
def inputs(backend):
    # Read/verify immutable inputs before timed transport; run_saved still hashes
    # files and arrays before every batch, as P3a does.
    return load_inputs(backend)


def source_hashes():
    return {**base.source_hashes(), **{str(p.relative_to(ROOT)):sha(p) for p in
        (Path(__file__), ROOT/'prototypes/mv_electron/photon_transport_p3s.py',
         ROOT/'prototypes/mv_electron/tests/test_p3s.py', PLAN)}}


def rules(name):
    if name=='const':return 0,np.array([.01,10.]),np.array([.025,.025])
    if name=='own':return 1,np.array([.02,1.]),np.array([.1,.05,.025])
    if name!='egs':raise ValueError('unknown rule')
    lines=KFILE.read_text().splitlines()
    i=next(i for i,line in enumerate(lines) if line.split()[-1:] == ['H2O'])
    n=int(lines[i+1].split()[0])
    a=np.array([[float(x) for x in line.split()] for line in lines[i+2:i+2+n]])
    E,slope,b,low,klow,high,khigh=a.T
    density=1.;dimension=.5
    K=np.where(dimension*density>high*density,khigh,
        np.where(dimension*density<low*density,klow,
                 np.exp(slope*np.log(dimension*density)+b-slope*np.log(density))))
    order=np.argsort(E);E=E[order];K=K[order]
    mask=(E>=.01)&(E<=10.)
    return 2,E[mask],K[mask]


def rule_kwargs(name):
    kind,energies,values=rules(name)
    return dict(rule_kind=kind,rule_energies=energies,rule_values=values)


def value(name,T):
    kind,e,k=rules(name)
    return float(rule_value(float(T),.025,kind,e,k))


def execute(E,N,seed,backend='elsepa',edges=None,cum=None,rule='const',original=False,**kw):
    args=inputs(backend)
    common=dict(annihilate_in_flight=True,spectrum_edges=edges,spectrum_cum=cum)
    if original:
        return dict(zip(base.KEYS,base.run_coupled(float(E),N,seed,.05,.025,30.,*args,**common)))
    return dict(zip(KEYS,run_coupled(float(E),N,seed,.05,.025,30.,*args,
                         **common,**rule_kwargs(rule),**kw)))


def equality(a,b,keys):
    return {k:bool(np.array_equal(a[k],b[k])) for k in keys}


def make_config(N,seed,edges,cum,rule,backend,E,spectrum_sha,score_kerma,score_pair,
                score_lateral,charged_source_T):
    args=inputs(backend);kind,e,k=rules(rule)
    return dict(N=N,seed=seed,E=E,backend=backend,fe=.05,k=.025,thickness=30.,
        annihilate_in_flight=True,batches=100,unit='Gy cm2',rule=rule,rule_kind=kind,
        rule_nodes_sha256=base.array_hash(e,k),K1_sha256=sha(KFILE) if rule=='egs' else None,
        score_kerma=score_kerma,score_pair=score_pair,score_lateral=score_lateral,
        charged_source_T=charged_source_T,charged_source_charge=-1,charged_source_z=0.,
        source_hashes=source_hashes(),table_hashes=base.table_hashes(backend,args),
        **base.spec_config(edges,cum,spectrum_sha))


def run_saved(name,N,seed,edges,cum,rule='own',backend='elsepa',E=10.,
              spectrum_sha=base.SPECTRUM_SHA,stop_after=None,score_kerma=False,
              score_pair=False,score_lateral=False,charged_source_T=None,check=False):
    base.name_check(name,N)
    work=CACHE/'prototype'/name;dest=OUT/('check_runs' if check else 'runs')/name
    config=make_config(N,seed,edges,cum,rule,backend,E,spectrum_sha,score_kerma,
                       score_pair,score_lateral,charged_source_T)
    args=inputs(backend)
    with base.job_lock(work):
        def verify():
            if source_hashes()!=config['source_hashes']:raise RuntimeError('code hashes changed')
            base.verify_spec(config,edges,cum)
            if base.table_hashes(backend,args)!=config['table_hashes']:raise RuntimeError('table hashes changed')
            if base.array_hash(*rules(rule)[1:])!=config['rule_nodes_sha256']:raise RuntimeError('rule changed')
            for folder in (work,dest):
                path=folder/'config.json'
                if path.exists() and json.loads(path.read_text())!=config:
                    raise RuntimeError('saved run config mismatch; no rerun permitted')
                if not path.exists():write(path,config)
        verify()
        if (work/'terminal.json').exists():raise RuntimeError('run previously failed')
        if (dest/'result.npz').exists():
            r=load(dest/'result.npz');validate_result(r,config)
            return dest/'result.npz'
        seeds=[int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(100)]
        for b,bs in enumerate(seeds):
            verify();target=work/f'batch_{b:03}.npz'
            if target.exists():
                r=load(target)
                if int(r['N'])!=N//100 or int(r['seed'])!=bs or not base.healthy(r):
                    raise RuntimeError('invalid saved batch')
                continue
            t=time.perf_counter()
            r=dict(zip(KEYS,run_coupled(float(E),N//100,bs,.05,.025,30.,*args,
                  annihilate_in_flight=True,spectrum_edges=edges,spectrum_cum=cum,
                  **rule_kwargs(rule),score_kerma=score_kerma,score_pair=score_pair,
                  score_lateral=score_lateral,charged_source_T=charged_source_T)))
            if not base.healthy(r):
                write(work/'terminal.json',dict(batch=b,**base.ledger(r)))
                raise RuntimeError('unhealthy batch')
            save(target,**{k:r[k].sum(axis=0) for k in KEYS[:5]},
                 **{k:r[k] for k in base.KEYS[5:]},lateral=r['lateral'].sum(),steps=r['steps'].sum(),
                 seconds=time.perf_counter()-t,N=N//100,seed=bs)
            if (b+1)%10==0:print(f'{name}: batch {b+1}/100',flush=True)
            if stop_after is not None and b+1==stop_after:return None
        verify();rows=[load(work/f'batch_{b:03}.npz') for b in range(100)]
        v={k:np.stack([r[k] for r in rows]) for k in KEYS[:5]}
        v['coarse']*=MEV_TO_GY_G/(N/100*.2)
        for k in KEYS[1:5]:v[k]*=MEV_TO_GY_G/(N/100*.05)
        for k in ('totals','violations','counts','nspec'):v[k]=np.sum([r[k] for r in rows],axis=0)
        v['maxima']=np.max([r['maxima'] for r in rows],axis=0)
        v['maxdepth']=max(int(r['maxdepth']) for r in rows)
        v['seconds']=sum(float(r['seconds']) for r in rows)
        v['batch_totals']=np.stack([r['totals'] for r in rows])
        v['batch_counts']=np.stack([r['counts'] for r in rows])
        v['lateral']=np.array([r['lateral'] for r in rows]);v['steps']=np.array([r['steps'] for r in rows])
        v.update(N=N,seed=seed,E=E,backend=backend,unit='Gy cm2',rule=rule,
                 rule_nodes_sha256=config['rule_nodes_sha256'],score_kerma=score_kerma,
                 score_pair=score_pair,score_lateral=score_lateral)
        validate_result(v,config);save(dest/'result.npz',**v)
        return dest/'result.npz'


def validate_result(r,c):
    if (not base.healthy(r) or r['coarse'].shape!=(100,150) or r['fine'].shape!=(100,600)
        or r['lateral'].shape!=(100,) or r['steps'].shape!=(100,) or r['batch_totals'].shape!=(100,6)
        or r['batch_counts'].shape!=(100,16)
        or any(str(r[k])!=str(c[k]) for k in ('N','seed','E','backend','unit','rule',
                   'rule_nodes_sha256','score_kerma','score_pair','score_lateral'))
        or r['nspec'].shape!=(c['nb'],10) or (c['nb'] and r['nspec'].sum()!=c['N'])
        or not np.array_equal(r['batch_totals'].sum(axis=0),r['totals'])
        or not np.array_equal(r['batch_counts'].sum(axis=0),r['counts'])):
        raise RuntimeError('invalid completed result')
    if not c['score_kerma'] and (np.any(r['kerma']) or np.any(r['collision_kerma'])):
        raise RuntimeError('disabled kerma nonzero')
    if not c['score_pair'] and np.any(r['pair']):raise RuntimeError('disabled pair nonzero')


@njit
def angle_moments(T,K,n,seed,ts,ks,us,ys,rates):
    np.random.seed(seed);total,G1=table_rates(T,ts,rates);s=0.;s2=0.
    for i in range(n):
        x=1-gs_mu_cached(T,K,ts,ks,us,ys,total,G1);s+=x;s2+=x*x
    mean=s/n;sem=np.sqrt(max(0.,(s2/n-mean*mean)/(n-1)))
    return mean,sem


def hinge_checks():
    rows=[]
    for old,tail,new,u in ((.025,.015,.1,.3),(.1,.07,.025,.3)):
        a=np.zeros(22);a[12]=.99;a[13]=tail;a[14]=old
        passed_k=hinge_transition(a,new,u)
        expected=np.array([tail+u*new,(1-u)*new,new])
        rows.append(dict(old=old,tail=tail,new=new,u=u,updated=a[12:15],
                         angle_K=passed_k,expected=expected,
                         passed=bool(passed_k==old and np.all(abs(a[12:15]-expected)<=1e-15))))
    # Exercise the actual sampling path against an independent old-K draw, same RNG.
    args=inputs('elsepa');ts,ks,us,ys,rates=args[:5]
    @njit
    def actual():
        a=np.zeros(22);a[5]=1.;a[13]=.015;a[14]=.025
        total,G1=table_rates(2.,ts,rates)
        np.random.seed(219);mu=gs_mu_cached(2.,.025,ts,ks,us,ys,total,G1)
        np.random.seed(219);scattering_hinge(a,2.,.1,ts,ks,us,ys,total,G1)
        return mu,a[5]
    mu,z=actual()
    return all(r['passed'] for r in rows) and mu==z,dict(transitions=rows,old_K_sampling_equal=mu==z)


def r50(curve):
    peak=int(np.argmax(curve));half=curve[peak]/2;z=(np.arange(len(curve))+.5)*.05
    for i in range(peak+1,len(curve)):
        if curve[i]<=half:return float(z[i-1]+(half-curve[i-1])/(curve[i]-curve[i-1])*.05)
    raise RuntimeError('R50 not bracketed')


def electron_comparison(a,b):
    ca=b['fine'].mean(axis=0);dref=float(ca.max());mask=ca>=.01*dref
    delta=(a['fine'].mean(axis=0)-ca)/dref
    sem=np.sqrt(a['fine'].var(axis=0,ddof=1)/100+b['fine'].var(axis=0,ddof=1)/100)/dref
    depth_status=('統計不足' if np.any(3*sem[mask]>.0025) else
                  '可' if np.all(abs(delta[mask])<=.005+3*sem[mask]) else '不合格')
    ar=base.ratio_sem(a['lateral'],a['batch_totals'][:,1])
    br=base.ratio_sem(b['lateral'],b['batch_totals'][:,1])
    q=base.quotient(ar,br);diff=q['value']-1;s=q['SEM']
    lateral_status='統計不足' if 3*s>.01 else '可' if abs(diff)<=.02+3*s else '不合格'
    return dict(Dref_e=dref,bins=np.flatnonzero(mask),delta_pct=delta[mask]*100,
        SEM_pct=sem[mask]*100,max_abs_delta_pct=float(abs(delta[mask]).max()*100),
        max_3SEM_pct=float(3*sem[mask].max()*100),depth_status=depth_status,
        mean_r2=ar,const_mean_r2=br,lateral_delta_pct=diff*100,lateral_SEM_pct=s*100,
        lateral_3SEM_pct=3*s*100,lateral_status=lateral_status,
        R50_cm=r50(a['fine'].mean(axis=0)),const_R50_cm=r50(ca),
        R50_difference_cm=r50(a['fine'].mean(axis=0))-r50(ca))


def checks():
    attempt=time.strftime('%Y%m%d_%H%M%S')+f'_{os.getpid()}'
    report=dict(status='実行中',source_hashes=source_hashes(),checks={},attempt=attempt,
                spectrum_sha256=base.SPECTRUM_SHA,K1_sha256=sha(KFILE))
    health={};active='S1'
    def record(key,ok,v):
        report['checks'][key]=dict(passed=bool(ok),**plain(v));write(OUT/'checks.json',report)
        print(key+': '+('合格' if ok else '不合格'),flush=True)
        if not ok and key!='S4':raise RuntimeError(key+' physical check failed; thresholds unchanged')
    def observe(name,r):health[name]=base.ledger(r);return r
    try:
        print('S1: starting regression',flush=True)
        edges,prob,cum=base.load_spectrum()
        refs={};reg=[]
        for backend,E,N,seed,ed,cu in (('mf',6.,20000,20291501,None,None),
                                      ('elsepa',10.,20000,20291502,edges,cum)):
            a=observe('S1_'+backend+'_original',execute(E,N,seed,backend,ed,cu,original=True))
            b=observe('S1_'+backend,execute(E,N,seed,backend,ed,cu))
            eq=equality(a,b,base.KEYS);reg.append(dict(backend=backend,N=N,seed=seed,equality=eq))
            refs[backend]=b
        record('S1',all(all(x['equality'].values()) for x in reg),dict(runs=reg))
        active='S2';rows=[]
        for kerma,pair in ((False,False),(False,True),(True,False)):
            r=observe(f'S2_{kerma}_{pair}',execute(10.,20000,20291502,'elsepa',edges,cum,
                      score_kerma=kerma,score_pair=pair))
            eq=equality(refs['elsepa'],r,PHYSICS)
            zeros=(kerma or not (np.any(r['kerma']) or np.any(r['collision_kerma']))) and (pair or not np.any(r['pair']))
            rows.append(dict(score_kerma=kerma,score_pair=pair,equality=eq,disabled_zero=bool(zeros)))
        # One primary per batch exposes zero-after-hit and wide-to-narrow reuse.
        small_a=observe('S2_reuse_original',execute(10.,100,20291502,'elsepa',edges,cum,original=True))
        small_b=observe('S2_reuse',execute(10.,100,20291502,'elsepa',edges,cum))
        widths=np.array([np.count_nonzero(x) for x in small_a['fine']]);hit=widths>0
        zero_after=np.flatnonzero(hit[:-1]&~hit[1:])+1
        narrow_after=np.flatnonzero((widths[:-1]>widths[1:])&(widths[1:]>0))+1
        eq=equality(small_a,small_b,base.KEYS)
        record('S2',all(all(x['equality'].values()) and x['disabled_zero'] for x in rows)
               and all(eq.values()) and len(zero_after)>0 and len(narrow_after)>0,
               dict(flags=rows,reuse=dict(N=100,batch_size=1,equality=eq,
                    zero_after_hit=zero_after,wide_then_narrow=narrow_after,widths=widths)))
        active='S3';print('S3: starting rules and 27 angle moments',flush=True)
        own_points=[0.,.01,np.nextafter(.02,0.),.02,np.nextafter(.02,1.),
                    np.nextafter(1.,0.),1.,np.nextafter(1.,2.),10.,100.]
        own_vals=[value('own',x) for x in own_points]
        own_ok=own_vals==[.1,.1,.1,.1,.05,.05,.05,.025,.025,.025]
        # Numerical anchors are read from the approved plan, not copied data tables.
        text=PLAN.read_text();anchor=re.search(r'0.01 MeVで(.*?)。計算の手順',text).group(1)
        # Parse each energy/value phrase explicitly (first energy precedes the capture).
        anchors=[(.01,float(anchor.split('、')[0]))]
        anchors += [(float(e),float(k)) for e,k in re.findall(r'(\d+(?:\.\d+)?)\s*(?:MeV)?で([0-9.]+)',anchor)]
        ten=[dict(T=e,value=value('egs',e),expected=k,relative_error=value('egs',e)/k-1) for e,k in anchors]
        mids=[dict(T=e,value=value('egs',e),expected=k,relative_error=value('egs',e)/k-1)
              for e,k in zip((.04,1.5,6.,8.),(.077556,.042736,.017484,.010557))]
        # Low/high clamping and linear interpolation at actual nodes.
        _,en,kn=rules('egs');clamped=value('egs',0.)==kn[0] and value('egs',100.)==kn[-1]
        a=observe('S3_const',execute(10.,2000,20291502,'elsepa',edges,cum))
        args=inputs('elsepa');flat=[]
        for kind,en,kn in ((1,np.array([.02,1.]),np.full(3,.025)),
                           (2,np.array([.01,10.]),np.full(2,.025))):
            b=observe('S3_flat_'+str(kind),dict(zip(KEYS,run_coupled(10.,2000,20291502,.05,.025,30.,*args,
              annihilate_in_flight=True,spectrum_edges=edges,spectrum_cum=cum,
              rule_kind=kind,rule_energies=en,rule_values=kn))))
            flat.append(equality(a,b,KEYS))
        hok,hv=hinge_checks();moments=[]
        for ri,name in enumerate(('const','own','egs')):
            for j,T in enumerate((.01,.02,.05,.1,.5,1.,2.,5.,10.)):
                K=value(name,T);mean,sem=angle_moments(T,K,1000000,20291540+ri*9+j,*args[:5])
                expected=-np.expm1(-K);relative=mean/expected-1;relsem=sem/expected
                moments.append(dict(rule=name,T=T,K=K,N=1000000,seed=20291540+ri*9+j,
                       moment=mean,SEM=sem,expected=expected,relative_difference=relative,
                       relative_SEM=relsem,limit=.02+3*relsem,passed=abs(relative)<=.02+3*relsem))
        write(OUT/'k_rules.json',dict(egs_representative=[dict(T=x['T'],K=x['value']) for x in ten],
                                  K1_sha256=sha(KFILE),rules=['const','own','egs']))
        record('S3',own_ok and len(ten)==10 and all(abs(x['relative_error'])<=1e-6 for x in ten)
          and all(abs(x['relative_error'])<=1e-4 for x in mids) and clamped
          and all(all(x.values()) for x in flat) and hok and all(x['passed'] for x in moments),
          dict(own_points=own_points,own_values=own_vals,egs_ten=ten,egs_intermediate=mids,
               endpoint_clamping=bool(clamped),flat_equality=flat,hinge=hv,moments=moments))
        active='S4';print('S4: starting nine 500000-history electron runs',flush=True);s4={name:dict(energies={}) for name in ('own','egs')}
        for ti,T in enumerate((2.,6.,10.)):
            runs={}
            for ri,name in enumerate(('const','own','egs')):
                seed=20291510+ti*3+ri
                path=run_saved(attempt+f'_s4_{int(T)}_{name}',500000,seed,None,None,
                     rule=name,E=T,spectrum_sha=None,score_lateral=True,charged_source_T=T,check=True)
                runs[name]=observe(f'S4_{T}_{name}',load(path))
                print(f'S4 T={T:g} rule={name} complete',flush=True)
            for name in s4:
                s4[name]['energies'][str(T)]=electron_comparison(runs[name],runs['const'])
        for name,r in s4.items():
            statuses=[v[k] for v in r['energies'].values() for k in ('depth_status','lateral_status')]
            r['status']='合格' if all(x=='可' for x in statuses) else '統計不足' if '統計不足' in statuses else '不合格'
        record('S4',all(r['status']=='合格' for r in s4.values()),dict(rules=s4))
        active='S5';timings={name:[] for name in ('original','const','own','egs')}
        # Warm each distinct signature outside the timed intervals.
        for name in timings:
            observe('S5_warm_'+name,execute(10.,100,777,'elsepa',edges,cum,
                    **(dict(original=True) if name=='original' else
                       dict(rule=name,score_kerma=False,score_pair=False))))
        for rep in range(2):
            for name in timings:
                start=time.perf_counter()
                r=execute(10.,50000,777,'elsepa',edges,cum,
                    **(dict(original=True) if name=='original' else
                       dict(rule=name,score_kerma=False,score_pair=False)))
                sec=time.perf_counter()-start;observe(f'S5_{name}_{rep}',r)
                if name=='original':original_result=r
                if name=='const':
                    if not all(equality(original_result,r,PHYSICS).values()):
                        raise RuntimeError('S5 constant/original trajectories differ')
                    # The untouched original has no loop counter. Recover its exact
                    # count from the bit-identical constant trajectory, without a rerun.
                    timings['original'][-1]['counted_total_steps_per_history']=float(r['steps'].sum()/50000)
                # Boundary calls count all charged steps, including terminal loop entries;
                # add photon interactions/escapes, subcutoff photons have no separate counter.
                timings[name].append(dict(seconds=sec,us_per_history=sec/50000*1e6,
                    charged_steps_per_history=float(r['counts'][0]/50000),
                    counted_total_steps_per_history=float((r['steps'].sum() if 'steps' in r else r['counts'][4]+r['counts'][7]+sum(r['counts'][8:12]))/50000),
                    scattering_per_history=float(r['counts'][1]/50000),counts=r['counts']))
        timing={k:dict(repeats=v,us_per_history=np.mean([r['us_per_history'] for r in v]),
                 us_per_history_SEM=np.std([r['us_per_history'] for r in v],ddof=1)/np.sqrt(2),
                 steps_per_history=np.mean([r['counted_total_steps_per_history'] for r in v])) for k,v in timings.items()}
        record('S5',True,dict(N=50000,seed=777,timing=timing,
            step_definition='Exact transport loop iterations. Original counter recovered from const after common physics arrays and tallies match bit-for-bit' ))
        active='S6'
        record('S6',all(v['finite'] and not any(v['violations']) and not any(v['failures']) for v in health.values()),dict(runs=health))
        active='S7';kwargs=dict(N=2000,seed=20291520,edges=edges,cum=cum,rule='own',check=True)
        name=attempt+'_s7_resume';run_saved(name,**kwargs,stop_after=3)
        count=len(list((CACHE/'prototype'/name).glob('batch_*.npz')))
        part=load(run_saved(name,**kwargs));whole=load(run_saved(attempt+'_s7_whole',**kwargs))
        eq=equality(part,whole,[k for k in part if k!='seconds'])
        changed=False
        try:run_saved(name,**{**kwargs,'N':2100})
        except RuntimeError as exc:changed='config mismatch' in str(exc)
        c1=json.loads((OUT/'check_runs'/name/'config.json').read_text())
        c2=json.loads((OUT/'check_runs'/(attempt+'_s7_whole')/'config.json').read_text())
        record('S7',count==3 and all(eq.values()) and c1==c2 and changed and base.healthy(part) and base.healthy(whole),
               dict(N=2000,seed=20291520,stopped_batches=count,equality=eq,
                    config_equal=c1==c2,changed_conditions_rejected=changed,
                    resumed_ledger=base.ledger(part),continuous_ledger=base.ledger(whole)))
        report['status']='合格' # S4 eligibility is separately required, even if neither rule passes.
    except Exception as exc:
        report['status']='不合格';report['stopped_check']=active;report['reason']=str(exc)
        if active not in report['checks']:report['checks'][active]=dict(passed=False,completed=False,error=str(exc))
        report['not_executed']=[f'S{i}' for i in range(1,8) if f'S{i}' not in report['checks']]
        write(OUT/'checks.json',report);write(OUT/'attempts'/f'{attempt}.json',report)
        raise
    write(OUT/'checks.json',report);write(OUT/'attempts'/f'{attempt}.json',report)
    return report


def require_checks(rule):
    r=json.loads((OUT/'checks.json').read_text())
    if (r.get('status')!='合格' or r.get('source_hashes')!=source_hashes()
        or set(r.get('checks',{}))!={f'S{i}' for i in range(1,8)}
        or not all(r['checks'][f'S{i}']['passed'] for i in (1,2,3,5,6,7))
        or r.get('spectrum_sha256')!=sha(base.SPECTRUM) or r.get('K1_sha256')!=sha(KFILE)):
        raise RuntimeError('S1–S7 checks with current hashes required')
    if r['checks']['S4']['rules'][rule]['status']!='合格':raise RuntimeError('S4 rule not eligible')


def production(name):
    rule,seed=RUNS[name];require_checks(rule);edges,prob,cum=base.load_spectrum()
    return run_saved(name,10000000,seed,edges,cum,rule=rule)


def inspect_run(name):
    rule,seed=RUNS[name];require_checks(rule);edges,prob,cum=base.load_spectrum()
    config=make_config(10000000,seed,edges,cum,rule,'elsepa',10.,base.SPECTRUM_SHA,False,False,False,None)
    dest=OUT/'runs'/name
    if json.loads((dest/'config.json').read_text())!=config:raise RuntimeError('production config mismatch')
    r=load(dest/'result.npz');validate_result(r,config);return r


def block_comparison(a,b,lock):
    rows=[]
    for bins in lock['blocks']:
        row=base.region_difference(a,b,'coarse',bins,lock['Dref'])
        row['status']='統計不足' if row['SEM_pct']>1/3 else '1%以内' if abs(row['delta_pct'])<=1 else '1%を超える'
        row['abs_delta_plus_1_96_SEM_pct']=abs(row['delta_pct'])+1.96*row['SEM_pct'];rows.append(row)
    return dict(blocks=rows,all_within=all(r['status']=='1%以内' for r in rows))


def analyse():
    lock=json.loads((base.OUT/'pilot_lock.json').read_text());base.require_regions(lock)
    paths=[base.OUT/'runs/p3a_proto/result.npz',base.OUT/'egs5/p3a_egs/result.npz',
           base.OUT/'egs5/p3a_egs_seed2/result.npz',base.OUT/'pilot_lock.json']
    current,r1,r2=map(load,paths[:3])
    if not base.healthy(current) or not base.finite(r1) or not base.finite(r2):raise RuntimeError('unhealthy reference')
    for name in ('p3a_egs','p3a_egs_seed2'):
        summary=json.loads((base.OUT/'egs5'/name/'summary.json').read_text())
        if summary['status']!='合格' or not all(summary['tests'].values()):raise RuntimeError('unhealthy EGS5 reference')
    ref={k:(r1[k]+r2[k])/2 for k in ('coarse','fine')}
    ref_in=(r1['batchin']+r2['batchin'])/2
    ref_dep=ref['coarse'].sum(axis=1)*.2*(int(r1['N'])/100)/MEV_TO_GY_G
    ref_ratio=base.ratio_sem(ref_dep,ref_in)
    checks_report=json.loads((OUT/'checks.json').read_text())
    runs={name:inspect_run(name) for name,(rule,_) in RUNS.items()
          if checks_report['checks']['S4']['rules'][rule]['status']=='合格'}
    if not runs:raise RuntimeError('no S4 eligible production')
    comparisons={};summaries=[];blocks={};timing=checks_report['checks']['S5']['timing']
    pairs=[('現行 − EGS5平均',current,ref)]
    for name,r in runs.items():
        rule=RUNS[name][0];pairs.append((rule+' − EGS5平均',r,ref))
        comparisons[rule]=dict(vs_EGS5=block_comparison(r,ref,lock),vs_current=block_comparison(r,current,lock))
    if 'p3s_egs' in runs:pairs.append(('egs − 現行',runs['p3s_egs'],current))
    def dep_ratio(r):
        return ref_ratio if r is ref else base.ratio_sem(r['batch_totals'][:,1],r['batch_totals'][:,0])
    for label,a,b in pairs:
        ar,br=dep_ratio(a),dep_ratio(b);q=base.quotient(ar,br)
        summaries.append(dict(label=label,buildup=base.region_difference(a,b,'fine',lock['build'],lock['Dref']),
            surface_to_15cm=base.region_difference(a,b,'coarse',list(range(75)),lock['Dref']),
            deposited_over_incoming_relative=dict(delta_pct=(q['value']-1)*100,SEM_pct=q['SEM']*100)))
    predictions=[dict(quantity='採点なし時間',prediction='const 約300、own 約220、egs 約200 µs/履歴',
         result={k:v['us_per_history'] for k,v in timing.items()},basis='計画0節の時間と採点を切る効果'),
        dict(quantity='全ステップ/履歴',prediction='const 約1200、own 約700、egs 約650',
         result={k:v['steps_per_history'] for k,v in timing.items()},basis='計画0節の実測'),
        dict(quantity='電子ビームの深さ・横の広がり',prediction='両規則とも可（仮説）',
         result=checks_report['checks']['S4'],basis='粗くする領域では短い刻み。ただし精度の保証ではない'),
        dict(quantity='各規則 − 現行の深部ブロック',prediction='±0.2% Dref以内（統計の範囲）',
         result={k:v['vs_current'] for k,v in comparisons.items()},basis='P2の半刻み比較'),
        dict(quantity='egs − EGS5平均のビルドアップ',prediction='ユーザー: 0に近づく。Claude: 変化0.1%未満、SEM約0.15%では識別困難の可能性',
         result=summaries,basis='ビルドアップ差の原因は未確定')]
    result=dict(source_hashes=source_hashes(),input_hashes={str(p.relative_to(ROOT)):sha(p) for p in
         paths+[OUT/'runs'/name/'result.npz' for name in runs]},comparisons=comparisons,
         summaries=summaries,predictions=predictions,S5=timing,
         production_timing={name:dict(seconds=float(r['seconds']),us_per_history=float(r['seconds']/int(r['N'])*1e6)) for name,r in runs.items()})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    z=(np.arange(150)+.5)*.2
    for target,b in (('egs5',ref),('current',current)):
        fig,ax=plt.subplots(figsize=(9,5))
        for name,a in runs.items():
            d=(a['coarse'].mean(axis=0)-b['coarse'].mean(axis=0))/lock['Dref']*100
            s=np.sqrt(a['coarse'].var(axis=0,ddof=1)/100+b['coarse'].var(axis=0,ddof=1)/100)/lock['Dref']*100
            ax.errorbar(z,d,yerr=s,label=RUNS[name][0],lw=.8,elinewidth=.5)
        ax.axhline(0,color='black',lw=.5);ax.legend();ax.set(xlabel='Depth (cm)',ylabel='Difference (% Dref, ±SEM)')
        fig.tight_layout();fig.savefig(OUT/f'difference_{target}.png',dpi=160);plt.close(fig)
    write(OUT/'analysis.json',result)
    lines=['# P3s 結果','','## 予想','',
           '| 量 | 計算前の予想 | 結果 ± SEM | 根拠 |','|---|---|---|---|']
    for name in ('const','own','egs'):
        t=timing[name]
        predicted={'const':300,'own':220,'egs':200}[name]
        lines.append(f"| {name} 採点なし時間 | 約{predicted} µs/履歴 | {t['us_per_history']:.3f} ± {t['us_per_history_SEM']:.3f} µs/履歴 | 計画0節の実測と採点を切る効果 |")
        steps={'const':1200,'own':700,'egs':650}[name]
        lines.append(f"| {name} 全ステップ/履歴 | 約{steps} | {t['steps_per_history']:.3f} ± 0（同じseedで2回） | 計画0節の実測 |")
    for rule,data in checks_report['checks']['S4']['rules'].items():
        for energy,row in data['energies'].items():
            lines.append(f"| 電子 {energy} MeV {rule} 深さ | 可（仮説） | {row['depth_status']}、最大絶対差 {row['max_abs_delta_pct']:.6f}% Dref、最大3SEM {row['max_3SEM_pct']:.6f}% | 粗くする領域で刻みが短いが、保証ではない |")
            lines.append(f"| 電子 {energy} MeV {rule} 横 | 可（仮説） | {row['lateral_delta_pct']:+.6f} ± {row['lateral_SEM_pct']:.6f}%、{row['lateral_status']} | 計画5.2の事前仮説 |")
    for rule,comp in comparisons.items():
        for i,row in enumerate(comp['vs_current']['blocks']):
            lines.append(f"| {rule} − 現行 深部ブロック{i+1} | ±0.2% Dref以内（統計の範囲） | {row['delta_pct']:+.6f} ± {row['SEM_pct']:.6f}% Dref | P2の半刻み比較 |")
    for row in summaries:
        if row['label']=='egs − EGS5平均':
            q=row['buildup']
            lines.append(f"| egs − EGS5平均 ビルドアップ | ユーザー: 現行より0に近づく。Claude: 変化0.1%未満で識別困難の可能性 | {q['delta_pct']:+.6f} ± {q['SEM_pct']:.6f}% Dref | 差の原因は未確定、事前のSEM約0.15% |")
    lines += ['']
    lines += ['## 結果','','### 結果を変えない高速化の効果','',
       f"S1・S2: 完全一致。元の採点あり {timing['original']['us_per_history']:.3f}、const採点なし {timing['const']['us_per_history']:.3f} µs/履歴。",'',
       '### 刻みの変更による線量の変化とEGS5との差','',
       '| 規則・比較 | 深部bin | 差 ± SEM (% Dref) | 判定 | |差|+1.96 SEM |','|---|---|---:|---|---:|']
    for rule,c in comparisons.items():
        for target,comp in c.items():
            for row in comp['blocks']:
                lines.append(f"| {rule} {target} | {row['bins'][0]}〜{row['bins'][-1]} | {row['delta_pct']:+.6f} ± {row['SEM_pct']:.6f} | {row['status']} | {row['abs_delta_plus_1_96_SEM_pct']:.6f} |")
            lines += ['',(f'{rule}: 固定した深部のブロック平均の差が1%以内。' if target=='vs_EGS5' else f'{rule}: 刻みの変更による差が1%以内。') if comp['all_within'] else f'{rule} {target}: 全ブロック1%以内の条件を満たさない。','']
    lines += ['| 比較 | ビルドアップ (% Dref) | 表面〜15 cm (% Dref) | 全沈着/入射の相対差 (%) |','|---|---:|---:|---:|']
    for r in summaries:
        cells=[f"{r[k]['delta_pct']:+.6f} ± {r[k]['SEM_pct']:.6f}" for k in ('buildup','surface_to_15cm','deposited_over_incoming_relative')]
        lines.append('| '+r['label']+' | '+' | '.join(cells)+' |')
    lines += ['', '| 条件 | µs/履歴 | ステップ/履歴 |','|---|---:|---:|']
    for name,t in timing.items():lines.append(f"| S5 {name} | {t['us_per_history']:.3f} | {t['steps_per_history']:.3f} |")
    for name,r in runs.items():lines.append(f"| 本計算 {name} | {r['seconds']/int(r['N'])*1e6:.3f} | {r['steps'].sum()/int(r['N']):.3f} |")
    lines += ['', '![各規則 − EGS5平均](difference_egs5.png)','![各規則 − 現行](difference_current.png)', '', '## 考察','',
        '採点整理の効果はS1・S2の完全一致とS5の時間で確認する。刻み変更の効果はS4と現行との差で、EGS5との差は独立に読む。',
        '予想と結果は値とSEMを併せて読む。差の原因を刻みだけに帰することはできず、ビルドアップの仮説は結果の不確かさに制限される。',
        '深部の判定は固定したブロック平均についてのもので、ブロック内部の正負の差が打ち消し合っても合格になる。',
        '判定は点推定と±1%の比較であり、真の差が95%の確度で1%以内という意味ではない。',
        'S4は現行との変更前後の比較であり、より細かい刻みへの収束検査ではない。光子由来電子の初期状態や陽電子を直接には検査していない。',
        'egsはK1.datに基づく研究用の対照であり、EGS5の電子輸送全体の再現ではない。',
        'S5と本計算のステップ定義はchecks.jsonを参照。厚さ30 cmの水スラブでの検証である。','']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--checks',action='store_true');g.add_argument('--run',choices=tuple(RUNS));g.add_argument('--analyse',action='store_true')
    a=parser.parse_args()
    if a.checks:checks()
    elif a.run:production(a.run)
    else:analyse()


if __name__=='__main__':main()
