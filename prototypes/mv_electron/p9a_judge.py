"""P9a sealed-result conversion, sequential GPU allocation and block judgement."""
import json
import math
from pathlib import Path
import numpy as np
from . import p9a as ref, p9a_data as data
from .materials_p8a import sha
from .run_p8e import block_statistics, judge_batches
from .gpu import p9a as gpu, p9a_store as store


def normalize_calls(results, histories, masses, key='p8_batches'):
    """Sum matching independent batch IDs across calls, then divide by mass/N."""
    masses=np.asarray(masses,float)
    if (len(results)!=len(histories) or not results or masses.ndim!=1 or
        not np.isfinite(masses).all() or np.any(masses<=0)):
        raise ValueError('normalization inputs')
    raw=np.zeros((100,len(masses)))
    counts=np.zeros(100,np.int64)
    for result,n in zip(results,histories):
        if type(n) is not int or n<100 or n%100:raise ValueError('batch history count')
        a=np.asarray(result[key])
        if a.shape!=raw.shape or not np.isfinite(a).all() or np.any(a<0):
            raise ValueError('raw batch shape/values')
        raw+=a;counts+=n//100
    return raw/counts[:,None]/masses


def fixed_dref(dose,grid):
    block_statistics(dose,grid)
    edges=grid['edges_cm']
    w=np.maximum(0.,np.minimum(edges[1:],5.)-np.maximum(edges[:-1],3.))/2
    value=float(np.mean(dose@w))
    if not math.isfinite(value) or value<=0:raise ValueError('finite positive Dref required')
    return value


def gpu_projection(egs_dose,gpu_pilot,grid,pilot_dref,pilot_n=200000,cap=240000000):
    if not math.isfinite(pilot_dref) or pilot_dref<=0:raise ValueError('finite positive Dref required')
    if type(pilot_n) is not int or pilot_n<=0 or pilot_n%gpu.CALL_N:raise ValueError('pilot N')
    _,_,se=block_statistics(egs_dose,grid)
    _,_,sg=block_statistics(gpu_pilot,grid)
    se=se/pilot_dref;sg=sg/pilot_dref;t=.0033
    if np.any(se>=t):raise ValueError('EGS SEM >= target; stop')
    required=pilot_n*sg**2/(t*t-se*se)
    if not np.isfinite(required).all():raise ValueError('nonfinite history forecast')
    n=max(gpu.CALL_N,math.ceil(1.5*float(required.max())/gpu.CALL_N)*gpu.CALL_N)
    predicted=np.hypot(se,sg*np.sqrt(pilot_n/n))
    return dict(histories=n,required_by_block=required.tolist(),safety_factor=1.5,
                pilot_n=pilot_n,Dref=pilot_dref,EGS_SEM_over_Dref=se.tolist(),
                GPU_pilot_SEM_over_Dref=sg.tolist(),projected_SEM_over_Dref=predicted.tolist(),
                cap=cap,status='ready' if n<=cap else 'stop-consult-user')


def load_egs(out,cache,densities,kind='production'):
    out,cache=Path(out),Path(cache)
    report=data.read(out/f'egs_{kind}.json')
    work=cache/kind;ref.verify_seal(work)
    grid=ref.layout(ref.LAYERS,densities)
    if kind=='pilot':
        n,seed=ref.PILOT_N,ref.PILOT_SEED
    elif kind=='production':
        pilot,pr=load_egs(out,cache,densities,'pilot')
        forecast=ref.projection(pilot,grid,pr['CPU_seconds'])
        if forecast!=pr['projection'] or forecast['exceeds_12_hours']:
            raise ValueError('EGS production allocation differs')
        n,seed=forecast['histories'],ref.P9A_EGS_SEED
    else:raise ValueError('EGS result kind')
    artifact=out/f'egs_{kind}.npz'
    expected=dict(N=n,seed=seed,fluor=1,media_set=5,layers=[list(x) for x in ref.LAYERS],
        densities_g_cm3=list(densities),units='MeV cm2/g/history',
        artifact_sha256=sha(artifact),identity_sha256=sha(work/'identity.json'),
        complete_sha256=sha(work/'complete.json'))
    if any(report.get(k)!=v for k,v in expected.items()):raise ValueError('EGS hash/N/seed/layout mismatch')
    definitions,spectrum=ref.load_inputs()
    inp=ref.pegs_input(definitions)
    ref.check_actual_inputs(work,{ref.CODE.name:ref.CODE.read_text()},inp,
                            ref.config_text(n,seed,ref.LAYERS),spectrum)
    if report['pegs_input_sha256']!=ref.digest(inp):raise ValueError('EGS PEGS input hash')
    ref.check_pegs((work/'pgs5job.pegs5dat').read_text(),5)
    raw,fine,_=ref.parse_egs((work/'egs5job.out').read_text(),n,seed,ref.LAYERS,densities)
    with np.load(artifact,allow_pickle=False) as f:
        for k,v in grid.items():
            if not np.array_equal(f[k],v):raise ValueError('EGS bin mapping')
        if not np.array_equal(f['dose'],raw) or not np.array_equal(f['fine'],fine):
            raise ValueError('EGS raw/analysis mismatch')
        fg=ref.fine_layout(ref.LAYERS,densities)
        if not np.array_equal(f['fine_edges_cm'],fg['edges_cm']) or not np.array_equal(f['fine_mass_g_cm2'],fg['mass_g_cm2']):
            raise ValueError('EGS fine mapping')
    fixed_dref(raw,grid)
    return raw,report


def load_gpu(root,bundle_identity,seed,expected_n=None):
    root=Path(root)
    saved=store.read_json(root/'input.json')
    record,schedule=saved['identity'],saved['schedule']
    if (record['bundles']!=bundle_identity or record['materials']!=bundle_identity['names']
        or record['sources']!=gpu.core.fingerprint() or record['layers']!=[list(x) for x in ref.LAYERS]
        or record['densities']!=bundle_identity['densities'] or schedule!=record['schedule']
        or schedule['seed']!=seed or schedule['mode']!='gpu' or schedule['n']!=gpu.CALL_N):
        raise ValueError('GPU material/kernel/seed/layout/schedule mismatch')
    expected,_=gpu.input_record([None]*len(bundle_identity['names']),bundle_identity,
        record['spectrum']['edges'],record['spectrum']['cum'],ref.LAYERS,
        schedule['n']*schedule['count'],seed)
    expected['phase']='pilot' if seed==gpu.PILOT_SEED else 'production'
    if record != expected:
        raise ValueError('GPU input record differs from reconstructed conditions')
    grid=ref.layout(ref.LAYERS,bundle_identity['densities'])
    fine=ref.fine_layout(ref.LAYERS,bundle_identity['densities'])
    for label,mapping in [('grid',grid),('fine_grid',fine)]:
        if record[label]!={k:v.tolist() for k,v in mapping.items()}:
            raise ValueError('GPU bin mapping')
    sizes=[len(ref.LAYERS),len(grid['mass_g_cm2']),round(sum(t for _,t,_ in ref.LAYERS)/.2),
           len(fine['mass_g_cm2']),len(record['spectrum']['cum'])]
    if record['sizes']!=sizes or schedule['shapes']!=gpu.shapes_from_sizes(sizes):raise ValueError('GPU record sizes')
    from .run_p3a import load_spectrum
    e,_,c=load_spectrum()
    if record['spectrum']!=dict(edges=e.tolist(),cum=c.tolist()):raise ValueError('GPU source spectrum')
    complete=store.verify(root,record,schedule)
    if expected_n is not None and complete['n']!=expected_n:raise ValueError('GPU histories differ from allocation')
    # Stream up to 1200 calls without retaining multi-GB internal records.
    raw=np.zeros((100,len(grid['mass_g_cm2'])))
    raw_fine=np.zeros((100,len(fine['mass_g_cm2'])))
    for j in range(schedule['count']):
        result,_=store.load_call(root,j,record,schedule)
        raw+=result['p8_batches'];raw_fine+=result['fine']
    combined=[dict(p8_batches=raw,fine=raw_fine)]
    dose=normalize_calls(combined,[complete['n']],grid['mass_g_cm2'])
    fine_dose=normalize_calls(combined,[complete['n']],fine['mass_g_cm2'],key='fine')
    return dose,fine_dose,dict(record=record,complete=complete,complete_sha256=sha(root/'complete.json'))


def allocate(pilot_root,out=ref.OUT,cache=data.CACHE,egs_cache=ref.CACHE):
    _,b=data.load_bundles(cache,out)
    grid=ref.layout(ref.LAYERS,b['densities'])
    e,er=load_egs(out,egs_cache,b['densities'])
    ep,epr=load_egs(out,egs_cache,b['densities'],'pilot')
    g,_,gr=load_gpu(pilot_root,b,gpu.PILOT_SEED,gpu.CALL_N)
    from .gpu.p9a_colab import require_trial
    require_trial(pilot_root,gr['record'])
    forecast=gpu_projection(e,g,grid,fixed_dref(ep,grid))
    forecast['inputs']=dict(egs=er['artifact_sha256'],egs_pilot=epr['artifact_sha256'],gpu_pilot=gr['complete_sha256'])
    store.write_json(Path(out)/'gpu_projection.json',forecast,immutable=True)
    return forecast


def judge(gpu_root,pilot_root,out=ref.OUT,cache=data.CACHE,egs_cache=ref.CACHE):
    data.verify_protected(out)
    _,b=data.load_bundles(cache,out)
    allocation=allocate(pilot_root,out,cache,egs_cache)
    if allocation['status']!='ready':raise ValueError('GPU allocation exceeds cap; stop')
    e,er=load_egs(out,egs_cache,b['densities'])
    g,gfine,gr=load_gpu(gpu_root,b,gpu.GPU_SEED,allocation['histories'])
    grid=ref.layout(ref.LAYERS,b['densities'])
    # All hashes, histories, seeds and mappings have been verified before statistics.
    report=judge_batches(g,e,grid)
    from .gpu.p9a_colab import require_trial
    require_trial(pilot_root,gr['record'])
    with np.load(Path(out)/'egs_production.npz',allow_pickle=False) as f:efine=f['fine'].copy()
    dref=report['Dref']
    report['report_only']['fine_005cm']=dict(
        edges_cm=ref.fine_layout(ref.LAYERS,b['densities'])['edges_cm'].tolist(),
        GPU_mean=gfine.mean(axis=0).tolist())
    report['report_only']['fine_005cm'].update(
        EGS_mean=efine.mean(axis=0).tolist(),
        difference_over_Dref=((gfine.mean(axis=0)-efine.mean(axis=0))/dref).tolist())
    report['inputs']=dict(egs_artifact_sha256=er['artifact_sha256'],gpu_complete_sha256=gr['complete_sha256'],
                          egs_N=er['N'],gpu_N=gr['complete']['n'],egs_seed=er['seed'],gpu_seed=gpu.GPU_SEED)
    data.verify_protected(out)
    store.write_json(Path(out)/'judgement.json',report,immutable=True)
    return report
