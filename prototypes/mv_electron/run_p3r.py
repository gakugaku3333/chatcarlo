"""P3r fixed checks → gated, resumable production → fixed-region analysis.

EGS builds/work files and private spectrum inputs stay in the external cache.
No plan participates in source hashing. No generated numerical table is saved.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import numpy as np
from numba import njit
from scipy.integrate import quad
from scipy.stats import chisquare, binomtest
from . import run_p3a as base
from . import run_p3s as p3s
from .p2_common import ROOT, EGS, sha
from .reference_p2 import pegs_input, FLAG_NAMES
from .run_coupled_p2 import MEV_TO_GY_G
from .photon_transport_p3r import run_coupled
from .bremsstrahlung_p3r import tables, quantities, sample_k, direction
from .physics import M, RE, NE, collision, hard, LOGET, ER

CODE=ROOT/'prototypes/mv_electron'
OUT=ROOT/'docs/validation/mv/p3r'
CACHE=Path(os.environ.get('CHATCARLO_P3R_CACHE','~/.cache/chatcarlo/p3r')).expanduser()
SPECTRUM, SPECTRUM_SHA=base.SPECTRUM,base.SPECTRUM_SHA
KEYS=p3s.KEYS+('brems','first_record')
RUNS={'p3r_egs':dict(N=10000000,seed=20291620),
      'p3r_proto':dict(N=10000000,seed=20291621,brems_mode=2),
      'p3r_proto_local':dict(N=10000000,seed=20291622,brems_mode=1)}
# An explicit dependency list, including the deterministic/independent tests.
SOURCES=tuple(dict.fromkeys(base.SOURCES+('run_p3s.py','photon_transport_p3s.py',
    'run_p3r.py','bremsstrahlung_p3r.py','photon_transport_p3r.py',
    'p3r_egs.f','p3r_egs_common.inc','tests/test_p3r.py')))
write,save,load,plain=base.write,base.save,base.load,base.plain
finite=base.finite
spec_config,verify_spec,spectrum_text=base.spec_config,base.verify_spec,base.spectrum_text
name_check,job_lock=base.name_check,base.job_lock
inputs=p3s.inputs

def source_hashes():
    return {str((CODE/x).relative_to(ROOT)):sha(CODE/x) for x in SOURCES}

def equality(a,b,keys):
    return {k:bool(np.array_equal(a[k],b[k])) for k in keys}

def healthy(r):
    return base.healthy(r) and not np.any(r.get('brems',np.zeros(10))[2:4]) and r['totals'][3]==0

def execute(E,N,seed,backend='elsepa',edges=None,cum=None,brems_mode=2,**kw):
    keys=p3s.KEYS if brems_mode==0 else KEYS
    return dict(zip(keys,run_coupled(float(E),N,seed,.05,.025,30.,*inputs(backend),
        brems_mode=brems_mode,annihilate_in_flight=True,spectrum_edges=edges,
        spectrum_cum=cum,**p3s.rule_kwargs('own'),**kw)))

def verify_config(work,dest,config):
    if source_hashes()!=config['source_hashes']:
        raise RuntimeError('code hashes changed; stop')
    for p in (work/'config.json',dest/'config.json'):
        if p.exists() and json.loads(p.read_text())!=config:
            raise RuntimeError('saved run config mismatch; no rerun permitted')
    if not (work/'config.json').exists():
        write(work/'config.json',config)
    if not (dest/'config.json').exists():
        write(dest/'config.json',config)


def run_saved(name,N,seed,edges,cum,backend='elsepa',E=10.,spectrum_sha=SPECTRUM_SHA,stop_after=None,brems_mode=2):
    name_check(name,N)
    work = CACHE/'prototype'/name; dest = OUT/'runs'/name
    with job_lock(work):
        args = inputs(backend)
        config = dict(N=N,seed=seed,E=E,backend=backend,fe=.05,k=.025,thickness=30.,
                      annihilate_in_flight=True,batches=100,unit='Gy cm2',brems_mode=brems_mode,
                      rule='own',score_kerma=False,score_pair=False,
                      source_hashes=source_hashes(),table_hashes=base.table_hashes(backend,args),
                      **spec_config(edges,cum,spectrum_sha))
        def verify():
            verify_config(work,dest,config); verify_spec(config,edges,cum)
            if base.table_hashes(backend,args)!=config['table_hashes']:
                raise RuntimeError('input table hashes changed')
        verify()
        if (work/'terminal.json').exists():
            raise RuntimeError('run previously failed; no rerun permitted')
        if (dest/'result.npz').exists():
            v = load(dest/'result.npz')
            if (not healthy(v) or v['coarse'].shape!=(100,150) or v['fine'].shape!=(100,600) or
                int(v['N'])!=N or int(v['seed'])!=seed or float(v['E'])!=E or
                str(v['backend'])!=backend or str(v['unit'])!='Gy cm2' or
                v['nspec'].shape!=(config['nb'],10) or
                (config['nb'] and v['nspec'].sum()!=N)):
                raise RuntimeError('invalid completed result')
            return dest/'result.npz'
        seeds = [int(x.generate_state(1)[0]) for x in np.random.SeedSequence(seed).spawn(100)]
        for b,bs in enumerate(seeds):
            verify(); target = work/f'batch_{b:03}.npz'
            if target.exists():
                r = load(target)
                if int(r['N'])!=N//100 or int(r['seed'])!=bs or not healthy(r):
                    raise RuntimeError('saved batch invalid')
                continue
            t = time.perf_counter()
            r = dict(zip(KEYS,run_coupled(float(E),N//100,bs,.05,.025,30.,*args,
                         annihilate_in_flight=True,spectrum_edges=edges,spectrum_cum=cum,
                         brems_mode=brems_mode,**p3s.rule_kwargs('own'),
                         score_kerma=False,score_pair=False)))
            if not finite(r):
                write(work/'terminal.json',dict(reason='non-finite output',batch=b))
                raise RuntimeError('non-finite batch; stop')
            save(target,**{k:np.sum(r[k],axis=0) for k in KEYS[:5]},
                 **{k:r[k] for k in KEYS[5:]},seconds=time.perf_counter()-t,N=N//100,seed=bs)
            if not healthy(r):
                write(work/'terminal.json',dict(batch=b,**ledger(r)))
                raise RuntimeError('ledger violation or failure; stop')
            print(f'{name}: batch {b+1}/100',flush=True)
            if stop_after is not None and b+1==stop_after:
                return None
        verify()
        rows = [load(work/f'batch_{b:03}.npz') for b in range(100)]
        v = {k:np.stack([r[k] for r in rows]) for k in KEYS[:5]}
        v['coarse'] *= MEV_TO_GY_G/(N/100*.2)
        for k in KEYS[1:5]:
            v[k] *= MEV_TO_GY_G/(N/100*.05)
        for k in ('totals','violations','counts','nspec','brems','first_record'):
            v[k] = np.sum([r[k] for r in rows],axis=0)
        v['maxima'] = np.max([r['maxima'] for r in rows],axis=0)
        v['maxdepth'] = max(int(r['maxdepth']) for r in rows)
        v['seconds'] = sum(float(r['seconds']) for r in rows)
        v['batch_totals'] = np.stack([r['totals'] for r in rows])
        v['batch_counts'] = np.stack([r['counts'] for r in rows])
        v['batch_brems'] = np.stack([r['brems'] for r in rows])
        for k in ('lateral','steps'):v[k]=np.array([r[k].sum() for r in rows])
        if not healthy(v):
            raise RuntimeError('unhealthy aggregate')
        save(dest/'result.npz',**v,N=N,seed=seed,E=E,backend=backend,unit='Gy cm2',brems_mode=brems_mode)
        return dest/'result.npz'


def parse_egs(work,config):
    text = (work/'egs5job.out').read_text()
    if 'TIME_LIMIT' in text:
        raise RuntimeError('EGS5 time limit')
    def one(tag):
        matches = re.findall(r'^\s*'+tag+r'\s+(.+)',text,re.M)
        if len(matches)!=1:
            raise RuntimeError('incomplete EGS5 output: '+tag)
        a = np.array([float(x) for x in matches[0].split()])
        if not np.all(np.isfinite(a)):
            raise RuntimeError('non-finite EGS5 output: '+tag)
        return a
    def rows(tag,shape,indices):
        matches = re.findall(r'^\s*'+tag+r'\s+(.+)',text,re.M)
        if len(matches)!=int(np.prod(shape[:indices])):
            raise RuntimeError('missing EGS5 rows: '+tag)
        a = np.zeros(shape); seen = set()
        for row in matches:
            values = row.split(); loc = tuple(int(x)-1 for x in values[:indices])
            if loc in seen or any(i<0 or i>=shape[j] for j,i in enumerate(loc)):
                raise RuntimeError('invalid/duplicate EGS5 index: '+tag)
            seen.add(loc); a[loc] = np.array([float(x) for x in values[indices:]]).squeeze()
        if not np.all(np.isfinite(a)):
            raise RuntimeError('non-finite EGS5 rows: '+tag)
        return a
    c = rows('BATCH',(100,150),2)*MEV_TO_GY_G
    f = rows('FINE',(100,600),2)*MEV_TO_GY_G
    sinb = rows('BATCHIN',(100,),1)
    brb = rows('BATCHBR',(100,2),1)
    cbatches = rows('BATCHCB',(100,3),1)
    nb = int(one('SPECTRUM')[0])
    if nb!=config['nb']:
        raise RuntimeError('EGS5 spectrum bin count mismatch')
    rawspec = rows('SPEC',(nb,10),1) if nb else np.zeros((0,10))
    if np.any(rawspec<0) or np.any(rawspec!=rawspec.astype(np.int64)):
        raise RuntimeError('invalid spectrum counts')
    spec = rawspec.astype(np.int64)
    led = one('LEDGER'); cb = one('CALLBACKS'); flags = dict(zip(FLAG_NAMES,one('FLAGS')))
    cut = one('CUTOFFS'); cfg = one('CONFIG'); maxrel = float(one('MAXREL')[0])
    pegs = (work/'pgs5job.pegs5dat').read_text()
    rflags = list(map(int,pegs.splitlines()[5].split()))
    effective = dict(IBOUND=rflags[8],INCOH=rflags[9],ICPROF=rflags[10],IRAYL=rflags[7],**flags)
    rebin = float(np.max(abs(f.reshape(100,150,4).mean(axis=2)-c)))
    scale = max(1e-25,float(np.max(abs(c))))
    kinetic = one('KINETIC_CUTOFF')
    tests = dict(config=np.array_equal(cfg,np.array([config['E'],config['N'],config['seed'],0.])),
                 noflight=np.array_equal(one('NOFLIGHT'),[0.]),
                 balance=maxrel<=1e-9 and led[4]==led[5]==0,
                 score=led[6]/config['E']<=1e-9 and led[7]/config['E']<=1e-9,
                 callbacks=cb[0]>=0 and cb[0]==int(cb[0]) and cb[1]==0 and cb[2]>=0 and cb[6]==0,
                 cutoffs=abs(cut[0]-.521)<1e-12 and abs(cut[2]-.521)<1e-12 and
                         abs(cut[1]-.01)<1e-12 and abs(cut[3]-.01)<1e-12,
                 kinetic_cutoff=abs(kinetic[0]-(.521-kinetic[1]))<1e-12,
                 flags=all(v==0 for k,v in effective.items() if k not in ('iausfl8','usegsd')) and
                       flags['iausfl8']==flags['usegsd']==1,
                 IUNRST='IUNRST=0' in pegs,
                 dual_scoring=rebin<=scale*1e-9,
                 spectrum_total=(int(spec.sum())==config['N'] if nb else spec.size==0),
                 batchin=np.isclose(sinb.sum(),led[0],rtol=1e-12,atol=0),
                 batchcb=np.array_equal(cbatches.sum(axis=0),cb[[5,4,3]]),
                 mono_input=(nb>0 or np.isclose(led[0],config['N']*config['E'],rtol=1e-12,atol=0)))
    # Relative ledgers and normalization are independently reconstructed from both grids.
    for key,a,dz in (('coarse',c,.2),('fine',f,.05)):
        tests[key+'_integral'] = np.isclose(a.sum()*dz*(config['N']/100)/MEV_TO_GY_G,
                                           led[1],rtol=1e-9,atol=0)
    v = dict(coarse=c,fine=f,nspec=spec,batchin=sinb,batchcb=cbatches,
             totals=led,callbacks=cb,N=config['N'],seed=config['seed'],E=config['E'],
             seconds=one('WALL')[0],maxrel=maxrel,escapes=one('ESCAPES')[0],unit='Gy cm2')
    v['batch_brems']=brb
    tests['brems_batches']=np.all(brb>=0) and np.all(brb[:,0]==brb[:,0].astype(np.int64)) and np.allclose(brb.sum(axis=0),cb[[0,2]],rtol=1e-12,atol=1e-12)
    tests['brems_bounds']=cb[0]*.01*(1-1e-10)<=cb[2]<=cb[0]*20.
    tests['finite'] = finite(v)
    summary = dict(status='合格' if all(tests.values()) else '不合格',tests=tests,
                   ledger=led,callbacks=dict(zip(('brems_created','brems_discarded','brems_generated_MeV',
                     'flight_annihilation','rest_annihilation','pairs','wrong_tags'),cb)),
                   effective_flags=effective,cutoffs=cut,max_history_relative_balance=maxrel,
                   rebin_max_Gy_cm2=rebin,CPU_seconds=one('CPU')[0],wall_seconds=one('WALL')[0],
                   effective_electron_kinetic_cutoff_keV=kinetic[0]*1000)
    return v,summary


def egs_hashes():
    # Freeze inputs of the established EGS5 build in addition to the user code.
    from .p2_common import verify
    verify('egs5_inputs')
    baseline = json.loads((ROOT/'docs/validation/mv/p2/manifest.json').read_text())['egs5_inputs']
    return baseline


def run_egs(name,N,seed,edges,cum,E=10.,spectrum_sha=SPECTRUM_SHA,stop_after=None):
    name_check(name,N); work = CACHE/'egs5'/name; dest = OUT/'egs5'/name
    config = dict(N=N,seed=seed,E=E,IUNRST=0,noflight=0,thickness=30.,batches=100,
                  input=pegs_input(0),egs_input_hashes=egs_hashes(),source_hashes=source_hashes(),
                  **spec_config(edges,cum,spectrum_sha))
    with job_lock(work) as fd:
        def verify():
            verify_config(work,dest,config); verify_spec(config,edges,cum)
            if egs_hashes()!=config['egs_input_hashes']:
                raise RuntimeError('EGS5 inputs changed')
        verify()
        for filename in ('p3r_egs.f','p3r_egs_common.inc'):
            p = work/filename
            if p.exists() and sha(p)!=sha(CODE/filename):
                raise RuntimeError('EGS5 work source mismatch')
            if not p.exists():
                shutil.copy2(CODE/filename,p)
        expected = {'config.dat':f'{E:.17g} {N} {seed} 0\n','p3r_egs.inp':config['input']}
        spec = spectrum_text(edges,cum)
        if spec is not None:
            expected['spectrum.dat'] = spec
        elif (work/'spectrum.dat').exists():
            raise RuntimeError('unexpected spectrum.dat in monochromatic job')
        if (work/'noflight.dat').exists():
            raise RuntimeError('unexpected noflight.dat')
        for filename,value in expected.items():
            p = work/filename
            if p.exists() and p.read_text()!=value:
                raise RuntimeError('EGS5 work configuration mismatch: '+filename)
            if not p.exists():
                p.write_text(value)
        epstar = ROOT/'docs/validation/mv/p1/egs5/base_2/epstar.dat'
        if (work/'epstar.dat').exists() and sha(work/'epstar.dat')!=sha(epstar):
            raise RuntimeError('EGS5 stopping input mismatch')
        if not (work/'epstar.dat').exists():
            shutil.copy2(epstar,work/'epstar.dat')
        if not (work/'egs5job.exe').exists():
            p = subprocess.run([str(EGS/'egs5run'),'comp'],cwd=work,input='p3r_egs\n\n\n',
                               capture_output=True,text=True)
            (work/'build.log').write_text(p.stdout+p.stderr)
            if p.returncode or not (work/'egs5job.exe').exists():
                raise RuntimeError('EGS5 build failed: '+str(work/'build.log'))
        binary = sha(work/'egs5job.exe')
        if (work/'build_hash.json').exists() and json.loads((work/'build_hash.json').read_text())!={'sha256':binary}:
            raise RuntimeError('EGS5 executable changed')
        if not (work/'build_hash.json').exists():
            write(work/'build_hash.json',dict(sha256=binary))
        if (work/'gsdist.dat').exists() and not (work/'checkpoint.bin').exists():
            raise RuntimeError('gsdist.dat without checkpoint; stop without deleting files')
        # Compare the PEGS input to the existing P3a reference; never embed its tables.
        old=ROOT/'docs/validation/mv/p3a/egs5/p3a_egs/config.json'
        if json.loads(old.read_text())['input']!=config['input']:
            raise RuntimeError('P3a PEGS input mismatch')
        if stop_after is not None:
            target=work/'stop_batches.dat'; value=str(stop_after)+'\n'
            if target.exists() and target.read_text()!=value:raise RuntimeError('stop config mismatch')
            if not target.exists():target.write_text(value)
        complete = (dest/'result.npz').exists() and (dest/'summary.json').exists()
        if not complete:
            with (work/'run.log').open('a') as log:
                p = subprocess.Popen(['./egs5job.exe'],cwd=work,stdout=log,
                                     stderr=subprocess.STDOUT,pass_fds=(fd,))
                try:
                    while p.poll() is None:
                        time.sleep(.002 if stop_after else .2)
                finally:
                    if p.poll() is None:
                        p.kill(); p.wait()
            if p.returncode:
                raise RuntimeError('EGS5 run failed: '+str(work/'run.log'))
        verify()
        for filename,value in expected.items():
            if (work/filename).read_text()!=value:
                raise RuntimeError('EGS5 work input changed during run')
        if (work/'noflight.dat').exists():
            raise RuntimeError('noflight file appeared during run')
        if 'P3R_PAUSE' in (work/'egs5job.out').read_text():
            done,b,*rest=(work/'batch_progress.txt').read_text().split()
            write(work/'interruption.json',dict(completed_histories=int(done),completed_batches=int(b),N=N,seed=seed))
            raise InterruptedError(name)
        v,s = parse_egs(work,config)
        if complete:
            previous = load(dest/'result.npz')
            if not all(equality(previous,v,previous.keys()).values()):
                raise RuntimeError('saved EGS5 output differs from raw output')
        else:
            save(dest/'result.npz',**v); write(dest/'summary.json',s)
        if s['status']!='合格':
            raise RuntimeError('EGS5 output checks failed: '+json.dumps(plain(s['tests'])))
        return dest/'result.npz'


# ---------------------------------------------------------------------------
# Checks, gated production and analysis. Written by Claude after Codex stopped
# on its usage limit (plan header: takeover is permitted and recorded).
ledger=base.ledger
# Plan amendment 1 (2026-10-12): G4 seeds 20291606–08 replaced; the failed first attempt stays in checks_attempt1.json.
G4_SEEDS=(20291625,20291626,20291627)
# Plan amendment 2: the 2 MeV yield run had relative SEM 1.22% (> 1%) with 200000 histories; resized.
G7_RUNS=((800000,20291628),(200000,20291612),(200000,20291613))
CHECK_KEYS=('G1','G2','G3','G4','G5','G6','G6b','G7','G8','EGS')


def helpers():
    """Independent formulae live in the (hashed) test module; reuse, do not copy."""
    import importlib.util
    spec=importlib.util.spec_from_file_location('p3r_independent',CODE/'tests/test_p3r.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


@njit
def draw_k(T,n,seed,energies,values):
    np.random.seed(seed);a=np.empty(n);excess=0
    for i in range(n):
        a[i],e=sample_k(T,energies,values);excess+=e
    return a,excess


@njit
def draw_azimuth(T,ux,uy,uz,n,seed):
    # Azimuth of the photon about the parent in a fixed frame; polar-angle error.
    np.random.seed(seed);phi=np.empty(n);worst=0.
    if abs(uz)<.9:ax,ay,az=0.,0.,1.
    else:ax,ay,az=1.,0.,0.
    d=ax*ux+ay*uy+az*uz;e1x=ax-d*ux;e1y=ay-d*uy;e1z=az-d*uz
    s=np.sqrt(e1x*e1x+e1y*e1y+e1z*e1z);e1x/=s;e1y/=s;e1z/=s
    e2x=uy*e1z-uz*e1y;e2y=uz*e1x-ux*e1z;e2z=ux*e1y-uy*e1x
    for i in range(n):
        gx,gy,gz=direction(T,ux,uy,uz,2*np.pi*np.random.random())
        worst=max(worst,abs(np.arccos(min(1.,gx*ux+gy*uy+gz*uz))-M/(T+M)),abs(np.sqrt(gx*gx+gy*gy+gz*gz)-1))
        phi[i]=np.arctan2(gx*e2x+gy*e2y+gz*e2z,gx*e1x+gy*e1y+gz*e1z)
    return phi,worst


def mean_sem(total,square,n):
    mean=total/n;return float(mean),float(np.sqrt(max(0.,square/n-mean*mean)/n))


def csda_yield(T0):
    from .physics import ET
    e=np.load(ROOT/'docs/validation/mv/p1/data/estar.npz');ratio=e['radiative']/e['total']
    grid=np.r_[ET[ET<T0],T0];r=np.interp(np.log(grid),LOGET,ratio)
    # Below the first ESTAR node (10 keV) the ratio is taken constant; contribution < 1e-5.
    return float((np.trapezoid(r,grid)+r[0]*grid[0])/T0)


def checks():
    from .bremsstrahlung_p3r import integral,CUT
    h=helpers();energies,values=tables();edges,prob,cum=base.load_spectrum()
    report=dict(status='実行中',source_hashes=source_hashes(),spectrum_sha256=sha(SPECTRUM),checks={})
    def record(key,passed,v):
        report['checks'][key]=dict(passed=bool(passed),**v);write(OUT/'checks.json',report)
        print(key+': '+json.dumps(plain(report['checks'][key]),ensure_ascii=False)[:2000],flush=True)
        if not passed:
            report['status']='不合格';write(OUT/'checks.json',report)
            raise RuntimeError(key+' failed; stop without changing criteria')
    OUT.mkdir(parents=True,exist_ok=True)
    unit=subprocess.run([sys.executable,'-m','pytest',str(CODE/'tests/test_p3r.py'),'-q'],cwd=ROOT,capture_output=True,text=True)
    deterministic=unit.returncode==0;tail=unit.stdout.strip().splitlines()[-1] if unit.stdout.strip() else ''
    ok,v=h.table_check();record('G1',ok,v)
    rows=[]
    for j,T in enumerate((.02,.1,1.,10.)):
        k,excess=draw_k(T,1000000,20291601+j,energies,values)
        be=np.geomspace(CUT,T,201);total=integral(T,CUT,T,True)
        p=np.array([integral(T,be[i],be[i+1],True) for i in range(200)])/total
        obs=np.histogram(k,be)[0].astype(float);ex=p/p.sum()*len(k)
        # Merge neighbours until every expected count is at least 5.
        o2=[];e2=[];ao=ae=0.
        for o,e in zip(obs,ex):
            ao+=o;ae+=e
            if ae>=5:o2.append(ao);e2.append(ae);ao=ae=0.
        if ae>0:o2[-1]+=ao;e2[-1]+=ae
        pv=float(chisquare(o2,e2).pvalue)
        rows.append(dict(T=T,N=len(k),seed=20291601+j,cells=len(o2),p=pv,min_k=float(k.min()),max_k=float(k.max()),
                         envelope_excess=int(excess),passed=bool(pv>.01/4 and excess==0 and k.min()>=CUT and k.max()<=T)))
    record('G2',all(r['passed'] for r in rows),dict(conditions=rows,rule='raw p > 0.01/4'))
    n=200000;r=execute(10.,n,20291605,brems_mode=1,charged_source_T=10.,score_kerma=False,score_pair=False)
    dn=mean_sem(r['brems'][4],r['brems'][5],n);de=mean_sem(r['brems'][6],r['brems'][7],n)
    z=(dn[0]/dn[1],de[0]/de[1])
    record('G3',abs(z[0])<3 and abs(z[1])<3 and healthy(r) and deterministic,
           dict(N=n,seed=20291605,source='10 MeV electron, 30 cm slab, mode 1',events=float(r['brems'][0]),sum_k_MeV=float(r['brems'][1]),
                dN_mean=dn[0],dN_SEM=dn[1],z_N=z[0],dE_mean_MeV=de[0],dE_SEM_MeV=de[1],z_E=z[1],
                k_above_T=float(r['brems'][3]),deterministic_tests=tail,ledger=ledger(r)))
    rows=[]
    for j,u in enumerate(((0.,0.,1.),(0.,0.,-1.),(.3,.4,float(np.sqrt(.75))))):
        phi,worst=draw_azimuth(1.,*u,100000,G4_SEEDS[j])
        pv=float(chisquare(np.histogram(phi,np.linspace(-np.pi,np.pi,21))[0]).pvalue)
        rows.append(dict(parent=u,N=100000,seed=G4_SEEDS[j],p=pv,max_angle_or_norm_error=float(worst),passed=bool(pv>.01/3 and worst<1e-12)))
    first=json.loads((OUT/'checks_attempt1.json').read_text())['checks']['G4']
    record('G4',all(x['passed'] for x in rows) and deterministic,dict(conditions=rows,rule='raw p > 0.01/3; errors < 1e-12',
           amendment_1='seeds replaced after the first attempt failed by chance (plan, amendment 1); first attempt kept verbatim',
           first_attempt=first,diagnosis='docs/validation/mv/p3r/supplementary/g4_diagnosis.json'))
    rows={};old=p3s.execute(10.,20000,20291609,'elsepa',edges,cum,rule='own')
    olde=p3s.execute(10.,10000,20291610,'elsepa',rule='own',charged_source_T=10.)
    runs={}
    for mode in (0,1,2):
        x=execute(10.,20000,20291609,edges=edges,cum=cum,brems_mode=mode);runs[mode]=x
        rows[str(mode)]=dict(ledger=ledger(x),unresolved=float(x['totals'][3]),
                             passed=bool(base.healthy(x) and x['totals'][3]==0 and (mode==0 or healthy(x))))
    record('G5',all(x['passed'] for x in rows.values()),dict(N=20000,seed=20291609,source='Mohan spectrum',modes=rows))
    e0=execute(10.,10000,20291610,brems_mode=0,charged_source_T=10.)
    eq=dict(mohan=equality(old,runs[0],p3s.KEYS),electron=equality(olde,e0,p3s.KEYS))
    record('G6',all(all(x.values()) for x in eq.values()),dict(equality=eq,seeds=[20291609,20291610]))
    d=dict(mohan=(old,execute(10.,20000,20291609,edges=edges,cum=cum,brems_mode=1,disable_brems=True)),
           electron=(olde,execute(10.,10000,20291610,brems_mode=1,disable_brems=True,charged_source_T=10.)))
    v={};okb=True
    for name,(a,b) in d.items():
        exact=equality(a,b,p3s.KEYS);worst=0.
        for key in p3s.KEYS:
            x=np.asarray(a[key],float);y=np.asarray(b[key],float);m=np.abs(x)>0
            if np.any((x==0)!=(y==0)):worst=np.inf
            elif np.any(m):worst=max(worst,float(np.max(np.abs(y[m]/x[m]-1))))
        v[name]=dict(bit_identical=all(exact.values()),equality=exact,max_relative_difference=worst)
        okb=okb and (all(exact.values()) or worst<=1e-12)
    record('G6b',okb,dict(runs=v,rule='bit-identical, or relative 1e-12 if the operation order differs'))
    rows=[]
    for j,T in enumerate((2.,5.,10.)):
        n,seed=G7_RUNS[j];r=execute(T,n,seed,brems_mode=1,infinite_medium=True,charged_source_T=T,score_kerma=False,score_pair=False)
        m,s=mean_sem(r['brems'][8],r['brems'][9],n);y=m/T;ref=csda_yield(T);rel=y/ref-1
        state='統計不足' if s/m>.01 else '異常' if abs(rel)>.1 else '異常なし'
        rows.append(dict(T=T,N=n,seed=seed,yield_MC=y,yield_relative_SEM=s/m,yield_CSDA_ESTAR=ref,relative_difference=rel,
                         status=state,ledger_ok=bool(not np.any(r['violations'][:1]) and not np.any(r['counts'][5:7]) and r['totals'][3]==0)))
    record('G7',all(x['status']=='異常なし' and x['ledger_ok'] for x in rows),dict(conditions=rows,
           definition='sum k + track-length soft radiative loss of the primary electron and its Moller secondaries, mode 1, infinite water',
           amendment_2='2 MeV rerun with 800000 histories and a new seed after the first run was 統計不足 (plan, amendment 2)',
           second_attempt=json.loads((OUT/'checks_attempt2.json').read_text())['checks']['G7']))
    rows=[];args=inputs('elsepa')
    for j,E in enumerate((.0101,.02,.05,.1)):
        c=h.independent_photon_coefficients(E,args);mu=float(c.sum());n=1000000
        # Amendment 3: thickness rounded up to a multiple of the 0.2 cm scoring bin.
        thick=float(np.ceil(1/mu/.2-1e-9)*.2)
        r=dict(zip(KEYS,run_coupled(E,n,20291614+j,.05,.025,thick,*args,annihilate_in_flight=True,brems_mode=2,
                                    record_first=True,score_kerma=False,score_pair=False,**p3s.rule_kwargs('own'))))
        f=r['first_record'];pa=float(binomtest(int(f[0]),n,np.exp(-mu*thick)).pvalue)
        inter=f[1:3];pb=float(chisquare(inter,inter.sum()*c[:2]/c[:2].sum()).pvalue)
        rows.append(dict(E=E,N=n,seed=20291614+j,mu_independent=mu,thickness_cm=thick,expected_transmission=float(np.exp(-mu*thick)),transmitted=int(f[0]),p_transmission=pa,
                         first_compton=int(f[1]),first_photoelectric=int(f[2]),first_pair=int(f[3]+f[4]),p_process=pb,
                         healthy=bool(healthy(r)),passed=bool(pa>.01/8 and pb>.01/8 and f[3]+f[4]==0 and healthy(r))))
    record('G8',all(x['passed'] for x in rows) and deterministic,dict(conditions=rows,rule='8 tests, raw p > 0.01/8',
           subcutoff_deposit_test='tests/test_p3r.py::test_subcutoff_compton_at_scatter_point',deterministic_tests=tail,
           amendment_3='thickness 1/mu replaced by the next multiple of 0.2 cm; expectation exp(-mu t) (plan, amendment 3)',
           third_attempt=json.loads((OUT/'checks_attempt3.json').read_text())['checks']['G8']))
    # EGS5 user code: short continuous run against a stopped-and-resumed run, same seed.
    pc=run_egs('check_continuous',10000,20291618,edges,cum);interrupted=False
    try:pr=run_egs('check_resumed',10000,20291618,edges,cum,stop_after=50)
    except InterruptedError:
        interrupted=True;pr=run_egs('check_resumed',10000,20291618,edges,cum,stop_after=50)
    a,b=load(pc),load(pr);eq=equality(a,b,[k for k in a if k!='seconds'])
    s=json.loads((pc.parent/'summary.json').read_text())
    record('EGS',interrupted and all(eq.values()) and s['status']=='合格',
           dict(N=10000,seed=20291618,interrupted=interrupted,equality=eq,callbacks=s['callbacks'],tests=s['tests']))
    report['status']='合格';write(OUT/'checks.json',report);return report


def require_checks():
    r=json.loads((OUT/'checks.json').read_text())
    if (r.get('status')!='合格' or r.get('source_hashes')!=source_hashes() or set(r.get('checks',{}))!=set(CHECK_KEYS)
        or not all(x['passed'] for x in r['checks'].values()) or r.get('spectrum_sha256')!=sha(SPECTRUM)):
        raise RuntimeError('G1–G8 and EGS checks with current hashes required')


def production(name):
    require_checks();edges,prob,cum=base.load_spectrum();cfg=RUNS[name]
    return run_egs(name,edges=edges,cum=cum,**cfg) if name=='p3r_egs' else run_saved(name,edges=edges,cum=cum,**cfg)


def analyse():
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    require_checks();lock=json.loads((base.OUT/'pilot_lock.json').read_text());base.require_regions(lock)
    paths=dict(egs_local_1=base.OUT/'egs5/p3a_egs/result.npz',egs_local_2=base.OUT/'egs5/p3a_egs_seed2/result.npz',
               proto_continuous=p3s.OUT/'runs/p3s_own/result.npz',lock=base.OUT/'pilot_lock.json',
               egs=OUT/'egs5/p3r_egs/result.npz',proto=OUT/'runs/p3r_proto/result.npz',local=OUT/'runs/p3r_proto_local/result.npz')
    r1,r2,cont,egs,proto,local=(load(paths[k]) for k in ('egs_local_1','egs_local_2','proto_continuous','egs','proto','local'))
    for name in ('p3a_egs','p3a_egs_seed2'):
        if json.loads((base.OUT/'egs5'/name/'summary.json').read_text())['status']!='合格':raise RuntimeError('unhealthy EGS5 reference')
    summary=json.loads((OUT/'egs5/p3r_egs/summary.json').read_text())
    if summary['status']!='合格' or not healthy(proto) or not healthy(local) or not base.healthy(cont):raise RuntimeError('unhealthy run')
    for r,name in ((proto,'p3r_proto'),(local,'p3r_proto_local')):
        if int(r['N'])!=RUNS[name]['N'] or int(r['seed'])!=RUNS[name]['seed'] or int(r['brems_mode'])!=RUNS[name]['brems_mode']:
            raise RuntimeError('production identity mismatch')
    ref={k:(r1[k]+r2[k])/2 for k in ('coarse','fine')}  # batch-wise mean of the two local-deposit EGS5 runs
    dref=lock['Dref'];N=10000000
    def dd(key,bins):
        # (proto − proto_local) − (EGS − EGS local mean); four independent batch sets.
        parts=[np.asarray(x[key])[:,bins].mean(axis=1) for x in (proto,local,egs,ref)]
        delta=(parts[0].mean()-parts[1].mean())-(parts[2].mean()-parts[3].mean())
        var=sum(p.var(ddof=1)/100 for p in parts)
        return dict(bins=list(bins),delta_pct=float(delta/dref*100),SEM_pct=float(np.sqrt(var)/dref*100))
    r2rows=[]
    for bins in lock['blocks']:
        row=dd('coarse',bins);row['status']=base.classify(row['delta_pct'],row['SEM_pct']);r2rows.append(row)
    def regions(a,b):
        c=base.compare(a,b,lock);return c
    def dep(r,egs_run=False):
        if egs_run:
            d=r['coarse'].sum(axis=1)*.2*(int(r['N'])/100)/MEV_TO_GY_G;return base.ratio_sem(d,r['batchin'])
        return base.ratio_sem(r['batch_totals'][:,1],r['batch_totals'][:,0])
    def per_history(col,n=N):
        x=np.asarray(col,float);return dict(value=float(x.sum()/n),SEM=float(x.std(ddof=1)*10/n))
    brems=dict(prototype=dict(count=per_history(proto['batch_brems'][:,0]),energy_MeV=per_history(proto['batch_brems'][:,1])),
               prototype_local=dict(count=per_history(local['batch_brems'][:,0]),energy_MeV=per_history(local['batch_brems'][:,1])),
               EGS5=dict(count=per_history(egs['batch_brems'][:,0]),energy_MeV=per_history(egs['batch_brems'][:,1])))
    def rel(a,b):
        q=a['value']/b['value']-1;return dict(relative_pct=float(q*100),SEM_pct=float(100*np.hypot(a['SEM']/a['value'],b['SEM']/b['value'])))
    result=dict(source_hashes=source_hashes(),input_hashes={k:sha(v) for k,v in paths.items()},Dref=dref,
        R1=regions(proto,egs),R2=dict(blocks=r2rows,near=dd('coarse',lock['near']),buildup=dd('fine',lock['build']),
            verdict=('変化は1%以内で同じ' if all(r['status']=='1%以内' for r in r2rows) else
                     '1%を超えるブロックあり' if any(r['status']=='1%を超える' for r in r2rows) else '統計不足')),
        R3=dict(local_vs_EGS5_local=regions(local,ref),local_vs_continuous=regions(local,cont),continuous_vs_EGS5_local=regions(cont,ref)),
        change=dict(prototype=regions(proto,local),EGS5=regions(egs,ref)),
        deposited_over_incoming=dict(prototype=dep(proto),prototype_local=dep(local),prototype_continuous=dep(cont),
            EGS5=dep(egs,True),EGS5_local_1=dep(r1,True),EGS5_local_2=dep(r2,True)),
        brems_per_history=brems,brems_prototype_vs_EGS5=dict(count=rel(brems['prototype']['count'],brems['EGS5']['count']),
            energy=rel(brems['prototype']['energy_MeV'],brems['EGS5']['energy_MeV'])),
        microseconds_per_history=dict(prototype=float(proto['seconds'])/N*1e6,prototype_local=float(local['seconds'])/N*1e6,
            EGS5=float(egs['seconds'])/N*1e6,prototype_continuous=float(cont['seconds'])/N*1e6))
    write(OUT/'analysis.json',result)
    z=(np.arange(150)+.5)*.2;fig,ax=plt.subplots(2,1,figsize=(8,7),sharex=True)
    def curve(a,b):
        x=np.asarray(a['coarse']);y=np.asarray(b['coarse'])
        return (x.mean(0)-y.mean(0))/dref*100,np.hypot(x.std(0,ddof=1),y.std(0,ddof=1))/10/dref*100
    for label,a,b in (('prototype − EGS5 (photons transported)',proto,egs),('prototype − EGS5 (deposited locally)',local,ref)):
        d,s=curve(a,b);ax[0].errorbar(z,d,s,fmt='.',ms=3,lw=.6,label=label)
    for label,a,b in (('prototype: transported − local',proto,local),('EGS5: transported − local',egs,ref)):
        d,s=curve(a,b);ax[1].errorbar(z,d,s,fmt='.',ms=3,lw=.6,label=label)
    for a in ax:a.axhline(0,color='k',lw=.5);a.legend();a.set_ylabel('difference (% of Dref)')
    ax[0].axhline(1,color='gray',ls=':');ax[0].axhline(-1,color='gray',ls=':');ax[1].set_xlabel('depth (cm)')
    fig.tight_layout();fig.savefig(OUT/'dose_difference.png',dpi=130);plt.close(fig)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--checks',action='store_true');g.add_argument('--run',choices=tuple(RUNS));g.add_argument('--analyse',action='store_true')
    a=parser.parse_args()
    if a.checks:checks()
    elif a.run:production(a.run)
    else:print(json.dumps(plain(analyse()),ensure_ascii=False,indent=1)[:6000])


if __name__=='__main__':main()
