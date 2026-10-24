"""P9b export provenance and thin adapters to the unchanged P9a/P7 judges."""
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
from . import p9b_core as core
from .p9b import CPU,OUT
from .. import p9a_data,p9a as ref

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def transport_cases(record):
    base=dict(layers=[tuple(x) for x in ref.LAYERS],densities=record['densities'],edges=record['edges'],cum=record['cum'],score_field=True)
    return dict(z=base,x=dict(base,axis=0),seed=dict(base,layers=((1,30.,.2),),seeds=((0.,0.,10.,.05,.25,4),),field=True,field_half=2.5,ssd=100.))

def known_scores(engine):
    """Exactly representable point/segment inside the seed, independently placed bins.

    Seven P7 regions contain this deposit. Every path receives exactly 1 MeV.
    The parent water layer owns depth bin 50 despite the seed region number.
    """
    L=core.p7_core.layout('p7');expected=np.zeros(3420)
    for region,bin_id in ((0,40),(1,100),(2,40),(4,80),(7,80),(10,160),(13,160)):
        expected[int(L['offsets'][region])+bin_id]=1.
    p0=[0.,0.,10.015625];point=[0.,0.,10.03125];depth=np.zeros(engine.g.nb);depth[50]=1.
    for site in range(1,7):
        actual=engine.score_probe(p0 if site==3 else point,point,1.,engine.g.nl,site)
        if not np.array_equal(actual['field'],expected) or not np.array_equal(actual['depth'],depth):raise AssertionError('G5 known deposit/parent '+str(site))
        sites=np.zeros(6,np.int64);sites[site-1]=1
        if not np.array_equal(actual['sites'],sites) or not np.array_equal(actual['ledger'],sites):raise AssertionError('G5 forced site '+str(site))
        if not np.array_equal(actual['additions'],expected.astype(np.uint64)):raise AssertionError('G5 actual addition counters')
    from .p9b_cases import state
    forced=[]
    for site in (1,2,4,5,6):
        a=state(point,(1,0,0),engine.g.nl,kind=0 if site in (1,6) else -1)
        if site==1:a[6]=a[7]=.007
        elif site==2:a[6]=a[7]=.01
        elif site==4:a[18]=1;a[8]=0.
        elif site==5:a[18]=1;a[8]=100.
        elif site==6:a[6]=a[7]=.03
        r=engine.injected_probe(a)
        if r[site-1]<1 or r[14]!=0 or abs(r[6:12].sum()-r[12])>1e-12:raise AssertionError('G5 actual fixed-state transport site '+str(site))
        forced.append(dict(site=site,reached=int(r[site-1])))
    return dict(status='pass',sites=6,point_and_segment='exact',parent_bin=50,forced_transport=forced)

def export(work):
    from .p9b import local
    from .p9b_cases import cases
    work=Path(work);dest=work/'export'
    if dest.exists() and any(dest.iterdir()):raise ValueError('refuse export overwrite')
    gates=local(work);materials,identity=p9a_data.load_bundles()
    from ..run_p3a import load_spectrum
    edges,_,cum=load_spectrum();record=dict(densities=identity['densities'],bundles=identity,edges=edges.tolist(),cum=cum.tolist(),sources=core.fingerprint(),local=gates)
    dest.mkdir(parents=True,exist_ok=True);refs={};probes={}
    for name,kw in transport_cases(record).items():
        engine=CPU(work/'export_build',materials,**kw);h=engine.histories(2000,core.core.SEED2,field_history=True)
        refs.update({name+'_'+k:v for k,v in h.items()});refs[name+'_field']=engine.last['field']
        if name=='seed':known_scores(engine)
    np.savez_compressed(dest/'cpu_reference.npz',**refs)
    engines={}
    for j,case in enumerate(cases(record['densities'])):
        key=repr(case['kw'])
        if key not in engines:engines[key]=CPU(work/'export_build',materials,**case['kw'])
        for mode in (0,1):probes[f'{j}_{mode}']=engines[key].probe([case['state']],[case['direction']],mode=mode)[0]
    np.savez(dest/'g3_reference.npz',**probes)
    from ..photon_transport_p8e import Material
    for j,m in enumerate(materials):np.savez(dest/f'material_{j}.npz',**dict(zip(Material._fields,m)))
    (dest/'input.json').write_text(json.dumps(record,indent=2)+'\n')
    root=core.HERE.parents[2]
    with zipfile.ZipFile(dest/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        # Only source, no material values or spectra. Dependencies remain unchanged.
        names=set(core.fingerprint())|{p.name for p in core.HERE.glob('*.py')}
        for name in sorted(names):z.write(core.HERE/name,'prototypes/mv_electron/gpu/'+name)
        for p in core.HERE.parent.glob('*.py'):z.write(p,'prototypes/mv_electron/'+p.name)
        for name in ('metadata.json','estar_water.html','estar.npz'):
            path=root/'docs/validation/mv/p1/data'/name;z.write(path,str(path.relative_to(root)))
        z.writestr('prototypes/__init__.py','')
    manifest={p.name:sha(p) for p in dest.iterdir() if p.is_file()}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return dict(status='exported',folder=str(dest),local=gates)

def load(bundle):
    bundle=Path(bundle);manifest=json.loads((bundle/'manifest.json').read_text())
    if not manifest or any(Path(k).name!=k or sha(bundle/k)!=v for k,v in manifest.items()):raise ValueError('export manifest mismatch')
    record=json.loads((bundle/'input.json').read_text())
    if record['sources']!=core.fingerprint():raise ValueError('kernel source mismatch')
    if record['local']['G0']['status']!='pass' or record['local']['G3']['status']!='pass':raise ValueError('local gates required')
    from ..photon_transport_p8e import Material
    materials=[]
    for j in range(5):
        with np.load(bundle/f'material_{j}.npz',allow_pickle=False) as f:materials.append(Material(*(f[k].item() if f[k].ndim==0 else f[k].copy() for k in Material._fields)))
    return record,materials

def analyse(work):
    from .. import p9a_judge as judge,analysis_p7 as field
    from ..run_p8e import judge_batches
    work=Path(work);record,_=load(work/'export');results=work/'results'
    job=json.loads((results/'job.json').read_text())
    if job['status']!='complete' or job['sources']!=record['sources']:raise ValueError('completed matching CUDA job required')
    for name,digest in job['files'].items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or sha(results/path)!=digest:raise ValueError('job result hash mismatch')
    report={};grid=ref.layout(ref.LAYERS,record['densities'])
    egs,_=judge.load_egs(ref.OUT,ref.CACHE,record['densities'])
    for name,n,seed in (('G1_x',13600000,930101),('G1_y',13600000,930102),('G2',20000000,930103)):
        dest=results/name;config=json.loads((dest/'config.json').read_text())
        if config['n']!=n or config['seed']!=seed or config['call_n']!=200000 or config['source']!=record['sources']:raise ValueError('production schedule')
        paths=sorted(dest.glob('call_*.npz'))
        if [p.name for p in paths]!=[f'call_{j:03}.npz' for j in range(n//200000)]:raise ValueError('production calls incomplete')
        rows=[]
        for path in paths:
            with np.load(path,allow_pickle=False) as f:rows.append({k:f[k].copy() for k in f.files})
        if name!='G2':
            doses=judge.normalize_calls(rows,[200000]*len(rows),grid['mass_g_cm2']);r=judge_batches(doses,egs,grid);r['passed']=all(s=='pass' for s in r['status']);report[name]=r
        else:
            packed=sum((r['field'] for r in rows),np.zeros((100,3420)))
            raw=dict(core.p7_core.unpack_regions(packed),N=np.array(n),nbatch=np.array(100));s=field.spec(2.5)
            root=core.HERE.parents[2]/'docs/validation/mv/p7'
            lock=json.loads((root/'pilot_lock_A.json').read_text())
            from ..run_p7 import read_egs
            ed=field.dose(read_egs('A','main',n),s)
            gd=field.dose(raw,s);r=field.compare(gd,ed,lock['Dref'],s);symmetry=field.symmetry(gd['profile'],s)
            passed=all(x['status'] in ('1%以内','0.1 cm以内') for group in r.values() for x in group) and symmetry['status']=='偏りは検出されず'
            report[name]=dict(passed=passed,comparison=r,symmetry=symmetry,lateral_escape=sum(int(r['lateral'][0]) for r in rows),Dref=lock['Dref'])
    report['status']='pass' if all(r['passed'] for r in report.values()) else 'fail'
    (work/'judgement.json').write_text(json.dumps(report,indent=2)+'\n');return report
