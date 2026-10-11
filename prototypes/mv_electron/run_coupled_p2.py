"""Atomic batch output. A completed batch is reused only with identical inputs."""
import json
import time
import numpy as np
from scipy.constants import electron_volt
from .p2_common import OUT,CACHE,ROOT,TABLES,ELSEPA,write,save,sha,portable,verify
from .photon_data_p2 import TABLE
from .photon_sampling_p2 import pair_parameters
from .photon_transport_p2 import run_coupled
KEYS=('coarse','fine','pair','kerma','collision_kerma','totals','maxima','violations','counts','maxdepth')
MEV_TO_GY_G=electron_volt*1e6*1000

def inputs(backend):
    verify()
    expected_photon=json.loads((OUT/'photon_generation.json').read_text())['table_sha256']
    if sha(TABLE)!=expected_photon:raise RuntimeError('P2 photon data hash mismatch')
    if backend not in TABLES:raise ValueError('unknown backend')
    if backend=='elsepa':
        from .elsepa import Elsepa
        expected=json.loads((ROOT/'docs/validation/mv/p1f/t2.json').read_text())['table_hashes']
        Elsepa(ELSEPA/'tables',expected_ihef=0,expected_hashes=expected).validate_energy(10.)
    g=np.load(TABLES[backend]);p=np.load(TABLE)
    if str(g['backend'])!=backend:raise ValueError('GS backend identity mismatch')
    if backend=='elsepa' and int(g['IHEF'])!=0:raise ValueError('ELSEPA IHEF mismatch')
    dl,bpar,delcm,delpos=pair_parameters()
    nist=np.loadtxt(ROOT/'chatcarlo/data/nist_xaamdi/water.csv',delimiter=',',comments='#')
    args=tuple(g[x] for x in ('T','K','u','y','rates','single'))+(p['E'],p['water'],p['thresholds'],dl,bpar,delcm,delpos,nist[:,0]/1000,nist[:,2])
    return args

def execute(backend,E,n,seed,fe=.05,k=.025,thickness=30.,source_state=None,maxstack=4096,maxsteps=10000000):
    if n<=0 or n%100:raise ValueError('equal 100 batches required')
    if E<.01 or E>10:raise ValueError('P2 source energy outside 10 keV–10 MeV')
    args=inputs(backend)
    return dict(zip(KEYS,run_coupled(float(E),n,seed,fe,k,thickness,*args,maxstack=maxstack,maxsteps=maxsteps,source_state=source_state)))

def run_saved(name,backend,E,n,seed,fe=.05,k=.025,production=False):
    if n%100:raise ValueError('N must be divisible by 100')
    verify()
    if production:
        if not (OUT/'pilot_lock.json').exists():raise RuntimeError('no pilot lock; production forbidden')
        verify('production_frozen')
        lock=json.loads((OUT/'pilot_lock.json').read_text())['runs'][name]
        if lock!=dict(backend=backend,E=E,N=n,seed=seed,fe=fe,k=k):raise RuntimeError('production config differs from pilot lock')
    args=inputs(backend);dest=OUT/'runs'/name;dest.mkdir(parents=True,exist_ok=True)
    # All code/input hashes are part of the per-run config, even before final freeze.
    config=dict(backend=backend,E=E,N=n,seed=seed,fe=fe,k=k,batches=100,production=production,
                gs_sha256=sha(TABLES[backend]),photon_sha256=sha(TABLE),
                source_hashes={portable(p):sha(p) for p in [ROOT/'prototypes/mv_electron'/x for x in ('photon_transport_p2.py','photon_sampling_p2.py','photon_data_p2.py','run_coupled_p2.py','physics.py','transport.py')]})
    if (dest/'config.json').exists():
        if json.loads((dest/'config.json').read_text())!=config:raise RuntimeError('saved run config mismatch; no rerun permitted')
    else:write(dest/'config.json',config)
    if (dest/'result.npz').exists():return dest/'result.npz'
    batch_seeds=[int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(100)]
    elapsed=sum(float(np.load(p)['seconds']) for p in dest.glob('batch_*.npz'))
    terminal=dest/'terminal.json'
    if terminal.exists():return None
    write(dest/'batch_seeds.json',batch_seeds)
    for b,bs in enumerate(batch_seeds):
        target=dest/f'batch_{b:03}.npz'
        if target.exists():continue
        verify()
        if production:verify('production_frozen')
        write(dest/'progress.json',dict(next_batch=b,N=n,seed=seed,complete=False))
        t=time.perf_counter()
        r=run_coupled(E,n//100,bs,fe,k,30.,*args);seconds=time.perf_counter()-t
        elapsed+=seconds
        v=dict(zip(KEYS,r))
        # The kernel batch subgroups remain descendants grouped by primary;
        # sum these subgroups to the one outer batch used for 100-batch statistics.
        save(target,**{x:np.sum(v[x],axis=0) for x in KEYS[:5]},
             **{x:v[x] for x in KEYS[5:]},seconds=seconds,N=n//100,seed=bs)
        write(dest/'progress.json',dict(next_batch=b+1,N=n,seed=seed,complete=b==99,wall_seconds=elapsed))
        if production and elapsed>=86400 and b<99:
            write(terminal,dict(status='統計不足',reason='24 hour transport wall limit',completed_batches=b+1,wall_seconds=elapsed))
            return None
    rows=[np.load(dest/f'batch_{b:03}.npz') for b in range(100)]
    v={x:np.stack([r[x] for r in rows]) for x in KEYS[:5]}
    for x in ('coarse',):v[x]*=MEV_TO_GY_G/(n/100*.2)
    for x in KEYS[1:5]:v[x]*=MEV_TO_GY_G/(n/100*.05)
    v['totals']=sum(r['totals'] for r in rows);v['maxima']=np.max([r['maxima'] for r in rows],axis=0)
    v['violations']=sum(r['violations'] for r in rows);v['counts']=sum(r['counts'] for r in rows)
    v['maxdepth']=max(int(r['maxdepth']) for r in rows);v['seconds']=sum(float(r['seconds']) for r in rows)
    save(dest/'result.npz',**v,N=n,seed=seed,E=E,fe=fe,k=k,unit='Gy cm2',backend=backend)
    return dest/'result.npz'


def run_budgeted(name,**config):
    """One process for the whole 100-batch run; hard wall bound includes JIT."""
    import sys
    from .process_p2 import run_process
    dest=OUT/'runs'/name
    if (dest/'terminal.json').exists():return None
    if (dest/'result.npz').exists():return run_saved(name,production=True,**config)
    work=CACHE/'prototype'/name;work.mkdir(parents=True,exist_ok=True)
    # Arguments remain in external process metadata; repository records are portable.
    args=[sys.executable,'-m','prototypes.mv_electron.run_coupled_p2',name,json.dumps(config)]
    state=run_process(work/'wall_budget.json',args,ROOT,work/'run.log',86400)
    if state.get('timed_out'):
        write(dest/'terminal.json',dict(status='統計不足',reason='24 hour per-run wall limit including imports/JIT/interrupted attempts',wall_seconds=state['total_seconds']))
        return None
    if state['returncode']:raise RuntimeError('prototype worker failed; external log '+portable(work/'run.log'))
    return dest/'result.npz'

if __name__=='__main__':
    import sys
    run_saved(sys.argv[1],**json.loads(sys.argv[2]),production=True)
