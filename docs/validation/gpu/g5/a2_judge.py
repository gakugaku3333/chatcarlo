"""Thin adapters to the sealed P7 / P8f judgements; no new statistical test."""
import json
from pathlib import Path
import numpy as np
import a2_common as common
from chatcarlo.gpu import physics_sets,therapy,therapy_bundle
from prototypes.mv_electron import analysis_p7 as p7
from prototypes.mv_electron.gpu import p7_core,p8f_core

NS={'A':20000000,'B':138000000,'C':26000000,'stack':20000000}
SEEDS={'A':(20301831,20301841),'B':(20301832,20301842),'C':(20301833,20301843),'stack':(830101,830111)}


def reference_hashes(gate):
    manifest=json.loads((common.HERE/'a0_reference_hashes.json').read_text())
    for row in manifest['gates'][str(gate)]:
        if therapy_bundle.sha(common.ROOT/row['path'])!=row['sha256']:
            raise ValueError('A0 reference hash mismatch: '+row['path'])
    return {r['path']:r['sha256'] for r in manifest['gates'][str(gate)]}


def metadata(rows,condition,*,retry=False,N=None):
    """Require complete disjoint outer batches, and one production/retry seed."""
    N=NS[condition] if N is None else N
    if type(N) is not int or N<=0 or N%100 or len(rows)!=100:raise ValueError('100 batches / valid N required')
    n=N//100;wantseed=SEEDS[condition][int(retry)]
    expected=physics_sets.declaration(therapy.NAME)
    for b,row in enumerate(rows):
        if row['N']!=n or row['seed']!=wantseed or row['start']!=b*n or row['physics_set']!=expected:
            raise ValueError('N/seed/start/physics_set mismatch')
        if row['health'].get('status')!='pass':raise ValueError('unhealthy batch')
        if not isinstance(row['sha256'],str) or len(row['sha256'])!=64:raise ValueError('missing batch hash')
    return True


def load_batches(paths,rows,condition,*,retry=False):
    metadata(rows,condition,retry=retry)
    if len(paths)!=100:raise ValueError('100 batch files required')
    out=[]
    for path,row in zip(paths,rows):
        if therapy_bundle.sha(path)!=row['sha256']:raise ValueError('batch hash mismatch')
        with np.load(path,allow_pickle=False) as z:out.append({k:z[k] for k in z.files})
    return out


def p7_expand(regions,N):
    x=np.asarray(regions)
    if x.shape!=(100,3420) or not np.isfinite(x).all() or np.any(x<0):raise ValueError('100 x 3420 outer field batches required')
    sizes=(120,300,120,960,1920);names=('axis_coarse','axis_fine','slab_coarse','profile','profile_fine')
    out={'N':np.array(N),'nbatch':np.array(100)};off=0
    for name,size in zip(names,sizes):
        value=x[:,off:off+size];off+=size
        if name.startswith('profile'):value=value.reshape(100,2,3,-1)
        out[name]=value
    return out


def p7_judge(regions,N,condition,reference,*,volumes=None,dref=None):
    reference_hashes(4)
    lock=json.loads((common.ROOT/f'docs/validation/mv/p7/pilot_lock_{condition}.json').read_text())
    if N!=lock['N_main']:raise ValueError('N differs from pilot lock')
    if dref is not None and dref!=lock['Dref']:raise ValueError('Dref differs from pilot lock')
    if volumes is not None and not np.array_equal(volumes,p7_core.layout('p7')['volumes']):raise ValueError('field volume mismatch')
    s=lock['spec'];left=p7.dose(p7_expand(regions,N),s);right=p7.dose(reference,s)
    stats=p7.compare(left,right,lock['Dref'],s);p7.stop_on_missing(stats)
    return dict(statistics=stats,symmetry=p7.symmetry(left['profile'],s))


def stack_dose(deposits,N,mass,grid):
    x=np.asarray(deposits);m=np.asarray(mass)
    if type(N) is not int or N<=0 or N%100:raise ValueError('invalid N')
    if x.shape!=(100,220) or not np.isfinite(x).all() or np.any(x<0):raise ValueError('100 x 220 input required (not 19 blocks)')
    if m.shape!=(220,) or not np.array_equal(m,grid['mass_g_cm2']) or not np.isfinite(m).all() or np.any(m<=0):raise ValueError('mass mismatch')
    return x/(N//100)/m


def stack_judge(deposits,N,mass,grid,reference,*,dref):
    reference_hashes(5)
    if dref!=p8f_core.DREF:raise ValueError('fixed Dref required')
    return p8f_core.judge(stack_dose(deposits,N,mass,grid),reference,grid,dref=dref)


def retry_eligible(statistics,*,kind,healthy=True,retried=False):
    if not healthy or retried:return False
    if kind=='stack':
        d=np.asarray(statistics['difference']);s=np.asarray(statistics['SEM'])
        if d.shape!=(19,) or s.shape!=(19,) or not np.isfinite(d).all() or not np.isfinite(s).all() or np.any(s<0):return False
        failed=np.abs(d)>.01;insufficient=s>.0033
    elif kind=='p7':
        rows=[(v,1.) for k in ('J1','J2') for v in statistics[k]]+[(v,.1) for v in statistics['J3']]
        if len(rows)!=11:return False
        if any(v['delta'] is None or v['SEM'] is None for v,t in rows):return False
        if any(not np.isfinite([v['delta'],v['SEM']]).all() or v['SEM']<0 for v,t in rows):return False
        failed=np.array([abs(v['delta'])>t for v,t in rows]);insufficient=np.array([v['SEM']>t/3 for v,t in rows])
    else:raise ValueError('unknown judgement kind')
    # A mixture with insufficient statistics is not the single accidental failure.
    return bool(failed.sum()==1 and not insufficient.any())


def saved_regression(archive=None):
    """Read only required archived tables; never restore whole history archives."""
    import io
    import subprocess
    import tarfile
    archive=Path.home()/'.cache/chatcarlo/archive' if archive is None else Path(archive)
    reference_hashes(4);reference_hashes(5)
    stored=json.loads((common.ROOT/'docs/validation/mv/p7/analysis.json').read_text())['conditions']
    reports={};tables={};receipts={}
    with subprocess.Popen(['zstd','-dc',str(archive/'p7_results.tar.zst')],stdout=subprocess.PIPE) as process:
        with tarfile.open(fileobj=process.stdout,mode='r|') as tar:
            for member in tar:
                if member.name in [f'results/production_{c}/result.npz' for c in 'ABC']:
                    payload=tar.extractfile(member).read();c=member.name.split('/')[1][-1]
                    import hashlib
                    if hashlib.sha256(payload).hexdigest()!=stored[c]['production_sha256']:raise ValueError('old P7 table SHA')
                    with np.load(io.BytesIO(payload),allow_pickle=False) as z:tables[c]={k:z[k] for k in z.files if k in (*p7.shapes(p7.spec(p7_core.CONDITIONS[c]['field_half'])),'N','nbatch')}
        if process.wait():raise RuntimeError('zstd P7 archive')
    for c,data in tables.items():
        with np.load(common.ROOT/f'docs/validation/mv/p7/egs5/main_egs_{c}/result.npz',allow_pickle=False) as z:ref={k:z[k] for k in z.files}
        raw=np.column_stack([data[k].reshape(100,-1) for k in ('axis_coarse','axis_fine','slab_coarse','profile','profile_fine')])
        actual=p7_judge(raw,int(data['N']),c,ref)['statistics']
        # Existing comparison includes classifications and bootstrap details.
        if actual!=stored[c]['comparison']:raise AssertionError('P7 saved judgement differs: '+c)
        reports[c]=dict(status='pass',quantities=11)
    grid=__import__('prototypes.mv_electron.run_p8e',fromlist=['layout'])
    g=grid.layout(grid.LAYERS,[1.,.26,1.85,11.35]);batches={};hashes={}
    with subprocess.Popen(['zstd','-dc',str(archive/'p8f.tar.zst')],stdout=subprocess.PIPE) as process:
        with tarfile.open(fileobj=process.stdout,mode='r|') as tar:
            for member in tar:
                name=member.name
                if name.startswith('p8f/results/production/batch_') and name.endswith(('.npz','.json')):
                    payload=tar.extractfile(member).read();b=int(Path(name).stem[-3:])
                    if name.endswith('.json'):receipts[b]=json.loads(payload)
                    else:
                        import hashlib
                        hashes[b]=hashlib.sha256(payload).hexdigest()
                        with np.load(io.BytesIO(payload),allow_pickle=False) as z:batches[b]=z['p8_batches'].sum(axis=0)
        if process.wait():raise RuntimeError('zstd P8f archive')
    if set(batches)!=set(range(100)):raise ValueError('old GPU batch missing')
    for b in range(100):
        receipt=receipts[b]
        if hashes[b]!=receipt['sha256'] or receipt['seed']!=820101 or receipt['n']!=200000 or receipt['start']!=200000*b:raise ValueError('old GPU batch seal/range')
    ref,_=grid.load_egs_result(common.ROOT/'docs/validation/mv/p8e',Path.home()/'.cache/chatcarlo/p8e/egs',g,[1.,.26,1.85,11.35])
    actual=stack_judge(np.array([batches[b] for b in range(100)]),20000000,g['mass_g_cm2'],g,ref,dref=p8f_core.DREF)
    expected=json.loads((common.ROOT/'docs/validation/mv/p8f/judgement.json').read_text())['A']
    for k in ('difference','SEM'):
        if not np.allclose(actual[k],expected[k],rtol=1e-12,atol=1e-15):raise AssertionError('old stack judgement '+k)
    if actual['status']!=expected['status']:raise AssertionError('old stack classification')
    reports['stack']=dict(status='pass',blocks=19,max_difference_error=float(np.max(np.abs(np.array(actual['difference'])-expected['difference']))))
    common.write(common.SUMMARY/'judge_regression.json',dict(status='pass',conditions=reports))
    return reports


def analyse_p7(paths,rows,condition,reference,*,retry=False):
    batches=load_batches(paths,rows,condition,retry=retry)
    return p7_judge(np.stack([b['field'] for b in batches]),NS[condition],condition,reference)


def analyse_stack(paths,rows,grid,reference,*,retry=False):
    batches=load_batches(paths,rows,'stack',retry=retry)
    return stack_judge(np.stack([b['bins'] for b in batches]),NS['stack'],grid['mass_g_cm2'],grid,reference,dref=p8f_core.DREF)
