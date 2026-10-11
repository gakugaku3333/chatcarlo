"""Approved P3a spectrum slab workflow. Pilot, production and analysis are explicit.

The private spectrum and generated EGS inputs stay outside tracked artifacts.
--checks stops on the first failed preregistered check; thresholds are immutable.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import numpy as np
from scipy.stats import chisquare
from .p2_common import ROOT, EGS, sha
from .run_coupled_p2 import inputs, KEYS as P2_KEYS, MEV_TO_GY_G
from .run_p2c import table_hashes as base_table_hashes
from .photon_transport_p3 import run_coupled
from .reference_p2 import pegs_input, FLAG_NAMES

CODE = ROOT/'prototypes/mv_electron'
OUT = ROOT/'docs/validation/mv/p3a'
CACHE = Path('~/.cache/chatcarlo/p3a').expanduser()
SPECTRUM = ROOT/'docs/validation/mv/p0/data/mohan10.spectrum'
SPECTRUM_SHA = '06fe2ba471dbfd64ca4305b5f1560479439ea4b565f993ecb21dcda86a6d8065'
MEAN = 2.801397434
KEYS = P2_KEYS+('nspec',)
RUNS = {'p3a_egs': dict(N=10000000, seed=20291420),
        'p3a_proto': dict(N=10000000, seed=20291421)}
# Every local Python dependency of inputs/transport plus the reference program.
SOURCES = ('run_p3a.py','photon_transport_p3.py','p3_egs.f','p3_egs_common.inc',
           'run_p2c.py','photon_transport_p2c.py','flight_annihilation_p2c.py',
           'run_coupled_p2.py','p2_common.py','photon_data_p2.py','photon_sampling_p2.py',
           'physics.py','transport.py','sampling.py','backend_sampling.py',
           'elsepa.py','dcs.py','mf.py','single_table.py','gs_backend.py','reference_p2.py',
           'p1e_common.py','p1d_common.py','workflow_p2.py','compare_photon_p2.py')


def plain(v):
    if isinstance(v, dict):
        return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return plain(v.tolist())
    if isinstance(v, np.generic):
        return v.item()
    return v


def write(path, v):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix=path.name+'.',
                                     suffix='.tmp', delete=False) as f:
        json.dump(plain(v), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); tmp = Path(f.name)
    tmp.replace(path)


def save(path, **v):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('wb', dir=path.parent, prefix=path.stem+'.',
                                     suffix='.pending.npz', delete=False) as f:
        np.savez(f, **v); tmp = Path(f.name)
    tmp.replace(path)


def load(path):
    with np.load(path, allow_pickle=False) as f:
        return {k:f[k] for k in f.files}


def source_hashes():
    return {str(p.relative_to(ROOT)):sha(p) for p in [CODE/x for x in SOURCES]}


def table_hashes(backend,args):
    from .physics import DATA, ET, ED, ER
    value = base_table_hashes(backend,args)
    value['electron_data_hashes'] = {str((DATA/x).relative_to(ROOT)):sha(DATA/x)
                                     for x in ('metadata.json','estar_water.html','estar.npz')}
    value['electron_arrays_sha256'] = array_hash(ET,ED,ER)
    return value


def array_hash(*arrays):
    h = hashlib.sha256()
    for x in arrays:
        a = np.ascontiguousarray(x)
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def validate_spectrum(edges, cum):
    edges = np.asarray(edges, dtype=float); cum = np.asarray(cum, dtype=float)
    if (edges.ndim!=1 or cum.ndim!=1 or not 1<=len(cum)<=64 or len(edges)!=len(cum)+1 or
        not np.all(np.isfinite(edges)) or not np.all(np.isfinite(cum)) or
        edges[0]<0 or edges[-1]>10 or np.any(np.diff(edges)<=0) or
        cum[0]<=0 or np.any(np.diff(cum)<=0) or cum[-1]!=1.):
        raise ValueError('invalid spectrum')
    return edges, cum


def load_spectrum(path=SPECTRUM):
    path = Path(path)
    if sha(path)!=SPECTRUM_SHA:
        raise RuntimeError('Mohan spectrum SHA256 mismatch')
    lines = path.read_text().splitlines()
    header = [float(x) for x in lines[1].replace(',', ' ').split()]
    if header!=[20.,0.,1.]:
        raise RuntimeError('spectrum must have 20 bins, lower edge 0, mode 1')
    rows = np.array([[float(x) for x in s.replace(',', ' ').split()]
                     for s in lines[2:] if s.strip()])
    if rows.shape!=(20,2) or not np.all(np.isfinite(rows)) or np.any(rows[:,1]<=0):
        raise RuntimeError('invalid spectrum rows')
    edges = np.r_[0.,rows[:,0]]
    weights = rows[:,1]*np.diff(edges); prob = weights/weights.sum()
    cum = np.cumsum(prob); cum[-1] = 1.
    validate_spectrum(edges,cum)
    if not np.array_equal(edges,np.arange(21)*.5):
        raise RuntimeError('Mohan spectrum edges differ from preregistration')
    return edges,prob,cum


def spectrum_stats(edges,prob):
    lo,hi = edges[:-1],edges[1:]
    mean = float(prob@((lo+hi)/2))
    variance = float(prob@((lo*lo+lo*hi+hi*hi)/3)-mean*mean)
    return mean,variance


def finite(v):
    return all(np.all(np.isfinite(x)) for x in v.values()
               if np.issubdtype(np.asarray(x).dtype,np.number))


def healthy(v):
    return finite(v) and not np.any(v['violations']) and not np.any(v['counts'][5:7])


def ledger(v):
    return dict(finite=bool(finite(v)),violations=v['violations'].tolist(),
                failures=v['counts'][5:7].tolist(),maxima=v['maxima'].tolist())


def execute(E,N,seed,backend='mf',spectrum_edges=None,spectrum_cum=None):
    return dict(zip(KEYS,run_coupled(float(E),N,seed,.05,.025,30.,*inputs(backend),
                    annihilate_in_flight=True,spectrum_edges=spectrum_edges,spectrum_cum=spectrum_cum)))


@contextmanager
def job_lock(work):
    work.mkdir(parents=True,exist_ok=True)
    fd = os.open(work/'run.lock',os.O_CREAT|os.O_RDWR)
    try:
        try:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('same run is already executing') from exc
        yield fd
    finally:
        os.close(fd)


def name_check(name,N):
    if not name or Path(name).name!=name or name in ('.','..') or N<=0 or N%100:
        raise ValueError('single-component name and equal 100 batches required')


def spec_config(edges,cum,spectrum_sha):
    if edges is None and cum is None:
        return dict(spectrum_sha256=None,spectrum_arrays_sha256=None,nb=0)
    edges,cum = validate_spectrum(edges,cum)
    return dict(spectrum_sha256=spectrum_sha,spectrum_arrays_sha256=array_hash(edges,cum),nb=len(cum))


def verify_spec(config,edges,cum):
    if spec_config(edges,cum,config['spectrum_sha256'])!={k:config[k] for k in
                     ('spectrum_sha256','spectrum_arrays_sha256','nb')}:
        raise RuntimeError('spectrum arrays changed')
    if config['spectrum_sha256']==SPECTRUM_SHA and sha(SPECTRUM)!=SPECTRUM_SHA:
        raise RuntimeError('spectrum file hash changed')


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


def run_saved(name,N,seed,edges,cum,backend='elsepa',E=10.,spectrum_sha=SPECTRUM_SHA,stop_after=None):
    name_check(name,N)
    work = CACHE/'prototype'/name; dest = OUT/'runs'/name
    with job_lock(work):
        args = inputs(backend)
        config = dict(N=N,seed=seed,E=E,backend=backend,fe=.05,k=.025,thickness=30.,
                      annihilate_in_flight=True,batches=100,unit='Gy cm2',
                      source_hashes=source_hashes(),table_hashes=table_hashes(backend,args),
                      **spec_config(edges,cum,spectrum_sha))
        def verify():
            verify_config(work,dest,config); verify_spec(config,edges,cum)
            if table_hashes(backend,args)!=config['table_hashes']:
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
                         annihilate_in_flight=True,spectrum_edges=edges,spectrum_cum=cum)))
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
        for k in ('totals','violations','counts','nspec'):
            v[k] = np.sum([r[k] for r in rows],axis=0)
        v['maxima'] = np.max([r['maxima'] for r in rows],axis=0)
        v['maxdepth'] = max(int(r['maxdepth']) for r in rows)
        v['seconds'] = sum(float(r['seconds']) for r in rows)
        v['batch_totals'] = np.stack([r['totals'] for r in rows])
        v['batch_counts'] = np.stack([r['counts'] for r in rows])
        if not healthy(v):
            raise RuntimeError('unhealthy aggregate')
        save(dest/'result.npz',**v,N=N,seed=seed,E=E,backend=backend,unit='Gy cm2')
        return dest/'result.npz'


def spectrum_text(edges,cum):
    if edges is None:
        return None
    return str(len(cum))+'\n'+''.join('%.17g %.17g %.17g\n'%row
                                      for row in zip(edges[:-1],edges[1:],cum))


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
                 callbacks=cb[0]==cb[1] and cb[6]==0,
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
    tests['finite'] = finite(v)
    summary = dict(status='合格' if all(tests.values()) else '不合格',tests=tests,
                   ledger=led,callbacks=dict(zip(('brems_created','brems_discarded','brems_local_MeV',
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
        for filename in ('p3_egs.f','p3_egs_common.inc'):
            p = work/filename
            if p.exists() and sha(p)!=sha(CODE/filename):
                raise RuntimeError('EGS5 work source mismatch')
            if not p.exists():
                shutil.copy2(CODE/filename,p)
        expected = {'config.dat':f'{E:.17g} {N} {seed} 0\n','p3_egs.inp':config['input']}
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
            p = subprocess.run([str(EGS/'egs5run'),'comp'],cwd=work,input='p3_egs\n\n\n',
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
        complete = (dest/'result.npz').exists() and (dest/'summary.json').exists()
        if not complete:
            with (work/'run.log').open('a') as log:
                p = subprocess.Popen(['./egs5job.exe'],cwd=work,stdout=log,
                                     stderr=subprocess.STDOUT,pass_fds=(fd,))
                try:
                    while p.poll() is None:
                        time.sleep(.002 if stop_after else .2)
                        progress = work/'batch_progress.txt'
                        if stop_after and progress.exists():
                            done,b,*rest = progress.read_text().split()
                            if int(b)>=stop_after:
                                p.kill(); p.wait()
                                write(work/'interruption.json',dict(completed_histories=int(done),
                                      completed_batches=int(b),N=N,seed=seed))
                                raise InterruptedError(name)
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


def ratio_sem(numerator,denominator):
    a = np.asarray(numerator,dtype=float); b = np.asarray(denominator,dtype=float)
    if a.shape!=(100,) or b.shape!=(100,) or not finite({'a':a,'b':b}) or b.mean()<=0:
        raise RuntimeError('ratio SEM requires 100 finite batches and positive denominator')
    q = float(a.sum()/b.sum())
    sem = float(np.std((a-q*b)/b.mean(),ddof=1)/10)
    return dict(value=q,SEM=sem)


def classify(delta_pct,sem_pct):
    if not np.isfinite(delta_pct) or not np.isfinite(sem_pct) or sem_pct<0:
        raise RuntimeError('invalid block statistic')
    return ('統計不足' if sem_pct>1/3 else
            '1%以内' if abs(delta_pct)<=1 else '1%を超える')


def deep_blocks(bins):
    bins = list(bins)
    if not bins:
        raise RuntimeError('領域を定義できない: deep')
    blocks = [bins[i:i+25] for i in range(0,len(bins),25)]
    if len(blocks)>1 and len(blocks[-1])<13:
        blocks[-2].extend(blocks.pop())
    return blocks


def require_regions(lock):
    if (not np.isfinite(lock['Dref']) or lock['Dref']<=0 or
        not lock.get('near') or not lock.get('build') or not lock.get('blocks') or
        any(not b for b in lock['blocks'])):
        raise RuntimeError('領域を定義できない')


def define_regions(ref):
    c = np.mean(ref['coarse'],axis=0); f = np.mean(ref['fine'],axis=0)
    if c.shape!=(150,) or f.shape!=(600,) or not finite({'c':c,'f':f}):
        raise RuntimeError('領域を定義できない: invalid grids')
    smooth = np.convolve(c,np.ones(5)/5,mode='same')
    peak = int(np.argmax(smooth)); dref = float(smooth[peak])
    if dref<=0 or c[peak]<.97*dref:
        raise RuntimeError('領域を定義できない: raw maximum outside window')
    lo = hi = peak
    while lo>0 and c[lo-1]>=.97*dref:
        lo -= 1
    while hi<149 and c[hi+1]>=.97*dref:
        hi += 1
    # First form blocks by physical depth (25 bins = 5 cm); then apply the
    # established P2 exclusion (<20% Dref) within each block, without changing edges.
    physical = deep_blocks(range(hi+1,150))
    blocks = [[i for i in b if c[i]>=.2*dref] for b in physical]
    build = np.where((np.arange(600)<4*lo)&(f>=.2*dref))[0].tolist()
    lock = dict(Dref=dref,window=[lo,hi],near=list(range(lo,hi+1)),build=build,
                deep=[i for b in blocks for i in b],blocks=blocks,physical_blocks=physical,
                excluded_low_dose_coarse=np.where(c<.2*dref)[0].tolist(),
                excluded_low_dose_fine=np.where(f<.2*dref)[0].tolist(),
                rule='five-bin smoothed maximum; contiguous raw >=97%; P2 <20% excluded')
    require_regions(lock)
    return lock


def region_difference(left,right,key,bins,dref):
    if not bins or dref<=0 or not np.isfinite(dref):
        raise RuntimeError('領域を定義できない')
    a = np.asarray(left[key])[:,bins].mean(axis=1)
    b = np.asarray(right[key])[:,bins].mean(axis=1)
    if a.shape!=(100,) or b.shape!=(100,) or not finite({'a':a,'b':b}):
        raise RuntimeError('analysis requires 100 finite batches')
    delta = float((a.mean()-b.mean())/dref*100)
    sem = float(np.hypot(a.std(ddof=1),b.std(ddof=1))/10/dref*100)
    return dict(key=key,bins=list(bins),delta_pct=delta,SEM_pct=sem,
                delta_Gy_cm2=float(a.mean()-b.mean()),SEM_Gy_cm2=sem*dref/100)


def compare(left,right,lock):
    require_regions(lock)
    rows = []
    for bins in lock['blocks']:
        r = region_difference(left,right,'coarse',bins,lock['Dref'])
        r['status'] = classify(r['delta_pct'],r['SEM_pct'])
        r['abs_delta_plus_1_96_SEM_pct'] = abs(r['delta_pct'])+1.96*r['SEM_pct']
        rows.append(r)
    verdict = ('固定した深部のブロック平均の差が1%以内' if all(r['status']=='1%以内' for r in rows)
               else '1%を超えるブロックあり' if any(r['status']=='1%を超える' for r in rows)
               else '統計不足で判定保留')
    return dict(Dref=lock['Dref'],blocks=rows,verdict=verdict,
                near=region_difference(left,right,'coarse',lock['near'],lock['Dref']),
                buildup=region_difference(left,right,'fine',lock['build'],lock['Dref']))


def analysis_check():
    # This entire check is independent of the private spectrum file.
    x = np.arange(100,dtype=float); x -= x.mean()
    a = np.tile((2+.02+x*.001)[:,None],(1,150))
    b = np.tile((2-x*.002)[:,None],(1,150))
    left = dict(coarse=a,fine=np.repeat(a,4,axis=1))
    right = dict(coarse=b,fine=np.repeat(b,4,axis=1))
    r = region_difference(left,right,'coarse',[50,51],2.)
    expected_delta = 1.
    expected_sem = float(np.std(x,ddof=1)*np.hypot(.001,.002)/10/2*100)
    t = dict(known_difference=abs(r['delta_pct']-expected_delta)<1e-12,
             known_SEM=abs(r['SEM_pct']-expected_sem)<1e-12,
             at_one=classify(1.,0.)=='1%以内',at_negative_one=classify(-1.,0.)=='1%以内',
             above_one=classify(np.nextafter(1.,np.inf),0.)=='1%を超える',
             above_SEM=classify(0.,np.nextafter(1/3,np.inf))=='統計不足',
             remainder12=[len(b) for b in deep_blocks(range(62))]==[25,37],
             remainder13=[len(b) for b in deep_blocks(range(63))]==[25,25,13])
    base = dict(Dref=2.,near=[20,21],build=[1,2],blocks=[[50,51]])
    for key in ('near','build','blocks'):
        try:
            compare(left,right,{**base,key:[]})
        except RuntimeError as exc:
            t['empty_'+key] = '領域を定義できない' in str(exc)
        else:
            t['empty_'+key] = False
    den = 10+.01*x; num = .4*den+.02*x+np.sin(np.arange(100))*.01
    q = num.sum()/den.sum()
    direct = np.sqrt((np.var(num,ddof=1)+q*q*np.var(den,ddof=1)-
                      2*q*np.cov(num,den,ddof=1)[0,1])/100)/den.mean()
    ratio = ratio_sem(num,den)
    t['ratio_SEM'] = abs(ratio['SEM']-direct)<1e-12
    t['ratio_value'] = ratio['value']==q
    return all(t.values()),dict(tests=t,known_delta_pct=r['delta_pct'],
              known_SEM_pct=r['SEM_pct'],expected_SEM_pct=expected_sem,
              boundary_statuses=dict(at_one=classify(1.,0.),above_one=classify(np.nextafter(1.,np.inf),0.),
                                    above_SEM=classify(0.,np.nextafter(1/3,np.inf))),
              deep_remainder12=[len(b) for b in deep_blocks(range(62))],
              deep_remainder13=[len(b) for b in deep_blocks(range(63))],
              ratio=ratio,ratio_direct_SEM=direct,correlation=float(np.corrcoef(num,den)[0,1]))


def sampling_result(r,edges,prob,N):
    theoretical_mean,variance = spectrum_stats(edges,prob)
    spec = r['nspec']
    expected = np.repeat(N*prob/10,10)
    observed = spec.reshape(-1)
    chi = chisquare(observed,expected)
    mean = float(r['totals'][0]/N); sem = float(np.sqrt(variance/N))
    ok = (spec.shape==(20,10) and spec.sum()==N and chi.pvalue>=1e-4 and
          abs(mean-MEAN)<=4*sem)
    return bool(ok),dict(N=N,chi2=float(chi.statistic),df=199,p=float(chi.pvalue),
                         mean_MeV=mean,target_mean_MeV=MEAN,theoretical_mean_MeV=theoretical_mean,
                         theoretical_SEM_MeV=sem,mean_delta_MeV=mean-MEAN,
                         mean_delta_in_SEM=(mean-MEAN)/sem,total_samples=int(spec.sum()))


def cutoff_result(r,N,egs=False):
    incoming = float(r['totals'][0]); deposited = float(r['totals'][1]); escape = float(r['totals'][2])
    values = dict(N=N,incoming_MeV=incoming,deposited_MeV=deposited,escaped_MeV=escape,
                  mean_input_MeV=incoming/N,relative_deposit_delta=deposited/incoming-1)
    t = dict(ledger=abs(deposited/incoming-1)<=1e-9,escape=escape==0)
    for key,dz in (('coarse',.2),('fine',.05)):
        a = r[key]
        recovered = float(a.mean(axis=0).sum()*dz/MEV_TO_GY_G)
        delta = recovered/(incoming/N)-1
        t[key+'_location'] = not np.any(a[:,1:]) and np.all(a[:,0]>0)
        t[key+'_normalization'] = abs(delta)<=1e-9
        values[key+'_recovered_MeV_per_history'] = recovered
        values[key+'_relative_delta'] = delta
    t['healthy'] = (bool(finite(r)) and r['totals'][4]==r['totals'][5]==0) if egs else healthy(r)
    return bool(all(t.values())),dict(tests=t,**values)


def equality(left,right,keys):
    return {k:bool(np.array_equal(left[k],right[k])) for k in keys}


def checks():
    report = dict(status='実行中',source_hashes=source_hashes(),spectrum_sha256=SPECTRUM_SHA,checks={})
    # New attempt names preserve evidence when a nonphysics harness bug is repaired.
    attempt = str(time.time_ns()); prefix = 'checks_'+attempt+'_'
    report['attempt'] = attempt
    def record(name,passed,values):
        if source_hashes()!=report['source_hashes']:
            raise RuntimeError('check source hashes changed')
        report['checks'][name] = dict(passed=bool(passed),**values)
        report['status'] = '実行中' if passed else '不合格'
        write(OUT/'checks.json',report)
        write(OUT/'attempts'/('checks_'+attempt+'.json'),report)
        print(name+': '+json.dumps(plain(report['checks'][name]),ensure_ascii=False),flush=True)
        if not passed:
            raise RuntimeError(name+' failed; stop without changing criteria')
    write(OUT/'checks.json',report)
    active = 'D1'
    try:
        from .photon_transport_p2c import run_coupled as original
        args = inputs('mf')
        old = dict(zip(P2_KEYS,original(6.,20000,20291401,.05,.025,30.,*args,annihilate_in_flight=True)))
        d1 = execute(6.,20000,20291401)
        eq = equality(old,d1,P2_KEYS)
        record('D1',all(eq.values()),dict(E=6.,N=20000,seed=20291401,backend='mf',equality=eq))
        active = 'D2'
        p = run_egs(prefix+'d2',10000,20261524,None,None,E=6.,spectrum_sha=None)
        d2 = load(p); old = load(ROOT/'docs/validation/mv/p2/egs5/selfcheck_d2_2_0/result.npz')
        eq = equality(old,d2,('coarse','fine'))
        record('D2',all(eq.values()),dict(E=6.,N=10000,seed=20261524,IUNRST=0,equality=eq))
        active = 'D3'
        edges,prob,cum = load_spectrum(); mean,variance = spectrum_stats(edges,prob)
        record('D3',abs(prob.sum()-1)<=1e-12 and abs(mean/MEAN-1)<=1e-9,
               dict(probability_sum=float(prob.sum()),mean_MeV=mean,target_mean_MeV=MEAN,
                    relative_mean_delta=mean/MEAN-1,variance_MeV2=variance,SHA256=sha(SPECTRUM)))
        active = 'D4'
        d4 = execute(10.,200000,20291402,'elsepa',edges,cum)
        ok,v = sampling_result(d4,edges,prob,200000)
        record('D4',ok,dict(seed=20291402,backend='elsepa',**v))
        active = 'D5'
        p = run_egs(prefix+'d5',200000,20291403,edges,cum)
        d5 = load(p); ok,v = sampling_result(d5,edges,prob,200000)
        record('D5',ok,dict(seed=20291403,**v))
        active = 'D6'
        v = dict(D1=ledger(d1),D4=ledger(d4),D5=json.loads((p.parent/'summary.json').read_text()))
        record('D6',healthy(d1) and healthy(d4) and finite(d5) and v['D5']['status']=='合格',dict(runs=v))
        active = 'D7'
        lowedges = np.array([0.,.01]); lowcum = np.array([1.])
        proto = load(run_saved(prefix+'d7_proto',10000,20291404,lowedges,lowcum,
                              E=.01,spectrum_sha='synthetic-cutoff'))
        ref = load(run_egs(prefix+'d7_egs',10000,20291405,lowedges,lowcum,
                           E=.01,spectrum_sha='synthetic-cutoff'))
        oka,va = cutoff_result(proto,10000); okb,vb = cutoff_result(ref,10000,egs=True)
        record('D7',oka and okb,dict(prototype=va,EGS5=vb,seeds=[20291404,20291405]))
        active = 'D8'
        name = prefix+'d8_resume'
        try:
            run_egs(name,200000,20291403,edges,cum,stop_after=40)
        except InterruptedError:
            pass
        ev = json.loads((CACHE/'egs5'/name/'interruption.json').read_text())
        resumed = load(run_egs(name,200000,20291403,edges,cum))
        eq = equality(d5,resumed,('coarse','fine','totals','callbacks','nspec','batchin','batchcb','maxrel','escapes'))
        record('D8',0<ev['completed_histories']<200000 and all(eq.values()),
               dict(seed=20291403,interruption=ev,equality=eq))
        active = 'D10'
        ok,v = analysis_check(); record('D10',ok,v)
        active = 'D9'
        name = prefix+'d9_resume'
        run_saved(name,2000,20291406,edges,cum,stop_after=3)
        count = len(list((CACHE/'prototype'/name).glob('batch_*.npz')))
        write(CACHE/'prototype'/name/'interruption.json',dict(completed_batches=count,N=2000,seed=20291406))
        part = load(run_saved(name,2000,20291406,edges,cum))
        whole = load(run_saved(prefix+'d9_continuous',2000,20291406,edges,cum))
        eq = equality(part,whole,[k for k in part if k!='seconds'])
        rejected = False
        try:
            run_saved(name,2100,20291406,edges,cum)
        except RuntimeError as exc:
            rejected = 'config mismatch' in str(exc)
        record('D9',count==3 and all(eq.values()) and rejected,
               dict(N=2000,seed=20291406,stopped_batches=count,equality=eq,changed_N_rejected=rejected))
        report['status'] = '合格'; write(OUT/'checks.json',report)
        write(OUT/'attempts'/('checks_'+attempt+'.json'),report)
    except Exception as exc:
        report['status'] = '不合格'; report['stopped_check'] = active; report['stopped_reason'] = str(exc)
        if active not in report['checks']:
            report['checks'][active] = dict(passed=False,error=str(exc),completed=False)
        report['not_executed'] = [f'D{i}' for i in range(1,11) if f'D{i}' not in report['checks']]
        write(OUT/'checks.json',report)
        write(OUT/'attempts'/('checks_'+attempt+'.json'),report)
        raise
    return report


def require_checks():
    v = json.loads((OUT/'checks.json').read_text())
    if (v.get('status')!='合格' or v.get('source_hashes')!=source_hashes() or
        set(v.get('checks',{}))!={f'D{i}' for i in range(1,11)} or
        not all(r['passed'] for r in v['checks'].values()) or sha(SPECTRUM)!=SPECTRUM_SHA):
        raise RuntimeError('D1–D10 pass with current hashes required')


def pilot():
    require_checks(); edges,prob,cum = load_spectrum()
    p = run_egs('pilot_egs',500000,20291410,edges,cum)
    r = load(p)
    a = load(run_saved('pilot_proto',500000,20291411,edges,cum))
    hashes = {str(x.relative_to(ROOT)):sha(x) for x in (p,OUT/'runs/pilot_proto/result.npz')}
    base = dict(source_hashes=source_hashes(),spectrum_sha256=SPECTRUM_SHA,input_hashes=hashes,
                N_pilot=500000,N_main=10000000,seeds=[20291410,20291411])
    lockfile = OUT/'pilot_lock.json'
    if lockfile.exists():
        previous = json.loads(lockfile.read_text())
        if any(previous.get(k)!=v for k,v in base.items()):
            raise RuntimeError('pilot lock/hash mismatch; no repeat permitted')
        return previous
    try:
        lock = define_regions(r)
    except RuntimeError as exc:
        v = dict(**base,status='領域を定義できない',reason=str(exc))
        write(lockfile,v); return v
    comparison = compare(a,r,lock)
    projected = [row['SEM_pct']*np.sqrt(500000/10000000) for row in comparison['blocks']]
    limit = (1/3)/1.5
    v = dict(**base,**lock,status='合格' if max(projected)<=limit else '統計不足の見込み',
             projected_block_SEM_pct=projected,max_projected_SEM_pct=max(projected),
             precision_limit_pct=limit,pilot_dose_not_used_for_verdict=True)
    write(lockfile,v); return v


def require_pilot():
    require_checks()
    v = json.loads((OUT/'pilot_lock.json').read_text())
    if (v.get('status')!='合格' or v.get('source_hashes')!=source_hashes() or
        v.get('spectrum_sha256')!=SPECTRUM_SHA or v.get('N_main')!=10000000):
        raise RuntimeError('passing pilot lock with current hashes required')
    require_regions(v)
    for p,h in v['input_hashes'].items():
        if sha(ROOT/p)!=h:
            raise RuntimeError('pilot results changed')
    return v


def production(name):
    require_pilot(); edges,prob,cum = load_spectrum()
    cfg = RUNS[name]
    return (run_egs(name,**cfg,edges=edges,cum=cum) if name=='p3a_egs'
            else run_saved(name,**cfg,edges=edges,cum=cum))


def inspect_completed(name,egs=False):
    """Read and validate registered production without triggering any transport."""
    dest = OUT/('egs5' if egs else 'runs')/name
    config = json.loads((dest/'config.json').read_text()); r = load(dest/'result.npz')
    expected = dict(**RUNS[name],E=10.,batches=100,thickness=30.,spectrum_sha256=SPECTRUM_SHA,
                    source_hashes=source_hashes(),nb=20)
    if not egs:
        expected.update(backend='elsepa',fe=.05,k=.025,annihilate_in_flight=True)
    else:
        expected.update(IUNRST=0,noflight=0,egs_input_hashes=egs_hashes(),input=pegs_input(0))
    if any(config.get(k)!=v for k,v in expected.items()):
        raise RuntimeError('production config/hash mismatch; no analysis')
    edges,prob,cum = load_spectrum(); verify_spec(config,edges,cum)
    if config['spectrum_arrays_sha256']!=array_hash(edges,cum):
        raise RuntimeError('production spectrum mismatch')
    if (not finite(r) or r['coarse'].shape!=(100,150) or r['fine'].shape!=(100,600) or
        int(r['N'])!=config['N'] or int(r['seed'])!=config['seed'] or
        r['nspec'].shape!=(20,10) or r['nspec'].sum()!=config['N'] or str(r['unit'])!='Gy cm2'):
        raise RuntimeError('invalid production output; no analysis')
    if egs:
        summary = json.loads((dest/'summary.json').read_text())
        if summary['status']!='合格' or not all(summary['tests'].values()) or r['totals'][4]!=0 or r['totals'][5]!=0:
            raise RuntimeError('EGS5 failures or violations; no analysis')
        parsed,again = parse_egs(CACHE/'egs5'/name,config)
        if again['status']!='合格' or not all(equality(r,parsed,r.keys()).values()):
            raise RuntimeError('saved EGS5 result differs from raw output')
    else:
        if not healthy(r) or r['batch_totals'].shape!=(100,6) or r['batch_counts'].shape!=(100,16):
            raise RuntimeError('prototype failures or violations; no analysis')
        args = inputs('elsepa')
        if config['table_hashes']!=table_hashes('elsepa',args):
            raise RuntimeError('prototype input table hash mismatch')
        if not np.array_equal(r['batch_totals'].sum(axis=0),r['totals']):
            raise RuntimeError('prototype batch totals mismatch')
        if not np.array_equal(r['batch_counts'].sum(axis=0),r['counts']):
            raise RuntimeError('prototype batch counts mismatch')
    return r


def peak_summary(r):
    z = (np.arange(150)+.5)*.2
    value = float(z[np.argmax(r['coarse'].mean(axis=0))])
    # Uncertainty of the maximum-of-mean estimator, resampling the saved 100 batches.
    rng = np.random.default_rng(20291430)
    depths = [z[np.argmax(r['coarse'][rng.integers(0,100,100)].mean(axis=0))] for _ in range(1000)]
    return dict(value=value,SEM=float(np.std(depths,ddof=1)),
                method='1000 bootstrap resamples of 100 batches, seed 20291430; 2 mm bin centres')


def summaries(r,egs=False):
    n = int(r['N']); nb = n/100
    if egs:
        inp = r['batchin']; dep = r['coarse'].sum(axis=1)*.2*nb/MEV_TO_GY_G
        pairs,rest,flight = r['batchcb'].T
        escaped = int(r['escapes']); violations = int(r['totals'][5]); failures = int(r['totals'][4])
    else:
        inp = r['batch_totals'][:,0]; dep = r['batch_totals'][:,1]
        c = r['batch_counts']; pairs = c[:,10]+c[:,11]; rest = c[:,12]; flight = c[:,14]
        escaped = int(r['counts'][3]+r['counts'][7])
        violations = r['violations'].tolist(); failures = r['counts'][5:7].tolist()
    return dict(deposited_over_incoming=ratio_sem(dep,inp),flight_over_pairs=ratio_sem(flight,pairs),
                mean_input_MeV=dict(value=float(inp.sum()/n),SEM=float(np.std(inp/nb,ddof=1)/10)),
                pairs=int(pairs.sum()),rest=int(rest.sum()),flight=int(flight.sum()),escapes=escaped,
                violations=violations,failures=failures,seconds=float(r['seconds']),dmax_cm=peak_summary(r))


def quotient(a,b):
    if b['value']<=0:
        raise RuntimeError('nonpositive ratio denominator')
    q = a['value']/b['value']
    s = np.hypot(a['SEM']/b['value'],a['value']*b['SEM']/b['value']**2)
    return dict(value=float(q),SEM=float(s))


def analyse():
    lock = require_pilot()
    p = inspect_completed('p3a_proto'); r = inspect_completed('p3a_egs',egs=True)
    comparison = compare(p,r,lock)
    stats = dict(prototype=summaries(p),EGS5=summaries(r,egs=True))
    a,b = stats['prototype'],stats['EGS5']
    dep = quotient(a['deposited_over_incoming'],b['deposited_over_incoming'])
    fraction = quotient(a['flight_over_pairs'],b['flight_over_pairs'])
    predictions = []
    for label,s in stats.items():
        predictions.append(dict(quantity=label+' dmax',prediction='2〜3 cm',unit='cm',**s['dmax_cm'],
             basis='探索的な予想。臨床の照射野・発散・混入電子の条件とは異なる。'))
    predictions.append(dict(quantity='全沈着/入射の試作−EGS5（相対）',prediction='−0.1〜+0.1%',
                            unit='%',value=(dep['value']-1)*100,SEM=dep['SEM']*100,
                            basis='P2cの6・10 MeVの比。1.25 MeV未満は高エネルギー2点からの外挿。'))
    for i,row in enumerate(comparison['blocks']):
        predictions.append(dict(quantity=f'深部ブロック {i+1}',prediction='±0.3% Dref以内',
                                unit='% Dref',value=row['delta_pct'],SEM=row['SEM_pct'],
                                basis='P2cの6・10 MeVで最大0.25%。'))
    predictions.append(dict(quantity='ビルドアップ 試作−EGS5',prediction='+0.1〜+0.4% Dref',unit='% Dref',
        value=comparison['buildup']['delta_pct'],SEM=comparison['buildup']['SEM_pct'],
        basis='P2cで両エネルギーとも約+0.25%。原因未確定。'))
    predictions.append(dict(quantity='飛行中/対生成の試作÷EGS5',prediction='0.98〜0.99',unit='比',**fraction,
                            basis='P2cで6 MeVが0.984、10 MeVが0.987。'))
    for label,s in stats.items():
        predictions.append(dict(quantity=label+' 入射平均',prediction='2.8014 MeV',unit='MeV',
                                **s['mean_input_MeV'],basis='スペクトルの定義。'))
    caveats = [
        '厚さ30 cmの水スラブ・平行ビームでの比較であり、10×10 cm²の照射野のPDD・側方分布ではない。',
        '制動放射の光子を両コードとも運ばず、その場に沈着させている。',
        'Fano試験、線量/衝突カーマ比の文献値との照合などの独立の検査は未実施。',
        '判定は点推定を±1%と比べる。真の差が95%の確度で±1%以内という意味ではない。',
        '5 cmのブロック平均の一致は、深さ線量の局所的な一致までは示さない。',
        '電子のカットオフは試作が10 keV、EGS5が約10.001 keVで、乱数の生成器も異なる。',
        '陽電子にも電子の阻止能・硬衝突・弾性散乱を使う近似が残る。',
        'スペクトルの値そのものは報告に載せない。',
        '割合のSEMは100バッチの分子と分母の共分散を含む残差式。独立な両コード間の比は誤差伝播。',
    ]
    discussion = []
    for row in predictions:
        sentence = f"{row['quantity']}: 予想 {row['prediction']}、結果 {row['value']:+.6f} ± {row['SEM']:.6f} {row['unit']}。"
        if row['quantity'].startswith('深部') or row['quantity'].startswith('ビルドアップ'):
            sentence += ('スペクトルは低エネルギー成分とエネルギーごとに異なる電子飛程・対生成割合を重ねるため、'
                         '単色の差からの予想には限界がある。陽電子の輸送近似と残ったビルドアップ差の原因は未確定。')
        elif '飛行中' in row['quantity']:
            sentence += '低エネルギーでは対生成と消滅までの競合が変わるため、6・10 MeVの割合の単純な外挿ではない。'
        elif '全沈着' in row['quantity']:
            sentence += '低エネルギー光子の局所沈着、電子飛程、後方脱出の重みが単色から変わり、表面付近の輸送差も寄与しうる。'
        elif 'dmax' in row['quantity']:
            sentence += 'スペクトルの電子飛程の重なりと表面からの脱出がピーク位置に影響する。臨床値との定量的比較はできない。'
        else:
            sentence += '標本平均の揺らぎをSEMと併せて解釈する。'
        discussion.append(sentence)
    paths = [OUT/'runs/p3a_proto/result.npz',OUT/'egs5/p3a_egs/result.npz',OUT/'pilot_lock.json']
    result = dict(source_hashes=source_hashes(),spectrum_sha256=SPECTRUM_SHA,
                  input_hashes={str(x.relative_to(ROOT)):sha(x) for x in paths},
                  comparison=comparison,predictions=predictions,runs=stats,caveats=caveats,discussion=discussion)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    z = (np.arange(150)+.5)*.2; dref = lock['Dref']
    fig,axes = plt.subplots(2,1,figsize=(9,8),sharex=True)
    for label,data in (('Prototype',p),('EGS5',r)):
        axes[0].plot(z,data['coarse'].mean(axis=0)/dref,label=label)
    delta = (p['coarse'].mean(axis=0)-r['coarse'].mean(axis=0))/dref*100
    sem = np.sqrt(p['coarse'].var(axis=0,ddof=1)/100+r['coarse'].var(axis=0,ddof=1)/100)/dref*100
    axes[1].errorbar(z,delta,yerr=sem,lw=.8,elinewidth=.5)
    axes[1].axhline(0,color='black',lw=.5)
    axes[0].set(ylabel='Dose / Dref',title='Mohan nominal 10 MV, water slab'); axes[0].legend()
    axes[1].set(xlabel='Depth (cm)',ylabel='Prototype - EGS5 (% Dref, ±SEM)')
    fig.tight_layout();fig.savefig(OUT/'dose_difference.png',dpi=160);plt.close(fig)
    write(OUT/'analysis.json',result)
    lines = ['# P3a — Mohan公称10 MVスペクトル・水スラブ比較','',
             '## 予想','', '| 量 | 計算前の予想 | 結果 ± SEM | 単位 | 根拠 |', '|---|---|---:|---|---|']
    for row in predictions:
        lines.append(f"| {row['quantity']} | {row['prediction']} | {row['value']:+.6f} ± {row['SEM']:.6f} | {row['unit']} | {row['basis']} |")
    lines += ['', '## 結果','',comparison['verdict'],'',f"Dref = {dref:.12g} Gy cm²（EGS5パイロットの5 bin移動平均の最大値）。",'',
              '| 深部bin | 差 ± SEM (% Dref) | 状態 | |差|+1.96 SEM |', '|---|---:|---|---:|']
    for row in comparison['blocks']:
        lines.append(f"| {row['bins'][0]}〜{row['bins'][-1]} | {row['delta_pct']:+.6f} ± {row['SEM_pct']:.6f} | {row['status']} | {row['abs_delta_plus_1_96_SEM_pct']:.6f} |")
    lines += ['']
    for key in ('near','buildup'):
        row = comparison[key]
        lines.append(f"{key}: {row['delta_pct']:+.6f} ± {row['SEM_pct']:.6f}% Dref（判定なし）。")
    lines += ['', '| コード | 全沈着/入射 ± SEM | 飛行中/対生成 ± SEM | 対生成 | 静止 | 飛行中 | 脱出 | 違反 | 未処理/失敗 | 秒 |',
              '|---|---:|---:|---:|---:|---:|---:|---|---|---:|']
    for label,s in stats.items():
        q = s['deposited_over_incoming']; f = s['flight_over_pairs']
        lines.append(f"| {label} | {q['value']:.9f} ± {q['SEM']:.9f} | {f['value']:.9f} ± {f['SEM']:.9f} | {s['pairs']} | {s['rest']} | {s['flight']} | {s['escapes']} | {s['violations']} | {s['failures']} | {s['seconds']:.3f} |")
    lines += ['', '![深さ線量と差±SEM](dose_difference.png)','', '## 考察','']+discussion+['']+caveats+['']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--checks',action='store_true')
    group.add_argument('--pilot',action='store_true')
    group.add_argument('--run',choices=tuple(RUNS))
    group.add_argument('--analyse',action='store_true')
    a = parser.parse_args()
    if a.checks:
        checks()
    elif a.pilot:
        result = pilot(); print(json.dumps(plain(result),ensure_ascii=False),flush=True)
        if result['status']!='合格':
            raise SystemExit(1)
    elif a.run:
        production(a.run)
    else:
        analyse()


if __name__=='__main__':
    main()
