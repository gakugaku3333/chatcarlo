"""Unmodified ELSEPA build, isolated case execution, two independent table passes."""
import argparse
import importlib.metadata
import json
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from prototypes.mv_electron.p1e_common import CACHE, WORK, OUT, display, sha, write, save, sanitize
ARCHIVE_SHA='1066d4d4a008d4e65c087f1bdf5e348e5c0a8b2d0dc088748f403ea9fdbaa7e5'
FLAGS=['-O2','-std=legacy','-ffixed-line-length-none']
READ_ONLY=False
NUMBER=r'[-+]?\d+(?:\.\d*)?(?:[EDed][-+]?\d+)?'

def energies():
    es=list(np.geomspace(.001,100,81))
    for t in (.01,.02,.1,1.,2.,10.):
        es=[e for e in es if not np.isclose(e,t,rtol=1e-13,atol=0)] + [t]
    return np.array(sorted(es))

def input_text(z,q,t,ihef=1):
    entries=dict(IZ=z,MNUCL=3,NELEC=z,MELEC=4,MUFFIN=0,RMUF=0,IELEC=q,MEXCH=1 if q==-1 else 0,MCPOL=0,VPOLA=-1,VPOLB=-1,MABS=0,VABSA=-1,VABSD=-1,IHEF=ihef)
    return ''.join(f'{k:<6} {v:12g}\n' for k,v in entries.items())+f'{"EV":<6} {t*1e6:.6E}\n'

def prepare():
    archive=CACHE/'elsepa-2020.zip'
    if not archive.exists(): raise FileNotFoundError('cached ELSEPA archive missing; no network fetch attempted')
    if sha(archive)!=ARCHIVE_SHA: raise RuntimeError('S1a archive SHA256 mismatch; stop')
    src=WORK/'source'
    if READ_ONLY and (not src.exists() or not (WORK/'build/elscata').exists()):raise RuntimeError('saved ELSEPA source/build missing; no execution allowed')
    if not src.exists():
        src.mkdir(parents=True)
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                target=src/info.filename
                if not target.resolve().is_relative_to(src.resolve()): raise RuntimeError('unsafe archive member')
            z.extractall(src)
    source=src/'elsepa-2020'
    # Check each pristine archive member, not merely a previous extraction.
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if not info.is_dir() and (src/info.filename).read_bytes()!=z.read(info): raise RuntimeError('ELSEPA source changed')
    build=WORK/'build'; build.mkdir(exist_ok=True); exe=build/'elscata'
    manifest=build/'build.json'
    if not exe.exists():
        command=['gfortran',*FLAGS,'-I',str(source),str(source/'elscata.f'),'-o','elscata']
        p=subprocess.run(command,cwd=build,capture_output=True,text=True)
        (build/'build.log').write_text(sanitize(p.stdout+p.stderr))
        if p.returncode: raise RuntimeError('ELSEPA build failed; see external build/build.log')
        write(manifest,dict(compiler=subprocess.check_output(['gfortran','--version'],text=True).splitlines()[0],flags=FLAGS,command=sanitize(' '.join(command)),executable_sha256=sha(exe),archive_sha256=sha(archive),sources={p.name:sha(p) for p in source.glob('*.f')},database={p.name:sha(p) for p in (source/'database').iterdir() if p.is_file()},python=sys.version,packages={n:importlib.metadata.version(n) for n in ('numpy','scipy','numba','pytest')}))
    meta=json.loads(manifest.read_text())
    if meta['executable_sha256']!=sha(exe): raise RuntimeError('ELSEPA executable changed')
    if meta['sources']!={p.name:sha(p) for p in source.glob('*.f')} or meta['database']!={p.name:sha(p) for p in (source/'database').iterdir() if p.is_file()}:raise RuntimeError('ELSEPA sources/database changed')
    return source,exe

def cross_sections(text):
    vals=[]
    for key in ('Total elastic cross section','1st transport cross section','2nd transport cross section'):
        m=re.search(re.escape(key)+r'\s*=\s*('+NUMBER+')',text)
        if not m: raise ValueError('missing printed table cross section')
        vals.append(float(m[1].replace('D','E')))
    return np.array(vals)

def parse(path,stdout):
    text=path.read_text(); lines=text.splitlines()
    a=np.array([[float(v.replace('D','E')) for v in line.split()] for line in lines if line.strip() and not line.lstrip().startswith('#')])
    if a.shape!=(606,6) or a[0,0]!=0 or a[-1,0]!=180 or a[0,1]!=0 or a[-1,1]!=1 or np.any(np.diff(a[:,1])<=0): raise ValueError('ELSEPA angular output mismatch')
    # Series moments are printed by DPWA* to stdout; table-integrated values in DCS header.
    series=[]
    for key in ('Total elastic cross section','1st transport cross section','2nd transport cross section'):
        matches=re.findall(re.escape(key)+r'\s*=\s*('+NUMBER+')',stdout,re.I)
        series.append(float(matches[-1].replace('D','E')) if matches else np.nan)
    warnings=[]; notices=[]
    for source in (text,stdout):
        ls=source.splitlines()
        for i,line in enumerate(ls):
            if any(w in line.lower() for w in ('warning','not consistent','too large','not converg','accuracy not')):
                block='\n'.join(ls[i:i+3]).strip()
                (notices if 'high-energy factorization' in line.lower() else warnings).append(block)
    return dict(theta_deg=a[:,0],mu=a[:,1],dcs=a[:,2],error=a[:,5],series_xs=np.array(series),series_available=np.isfinite(series),table_xs=cross_sections(text),warnings=np.array(warnings,dtype=str),notices=np.array(notices,dtype=str),header=text[:text.index('# Differential cross section:')])

def case(source,exe,group,z,q,t,ihef=1,custom_input=None):
    text=input_text(z,q,t,ihef) if custom_input is None else custom_input
    identity=dict(Z=z,charge=q,energy_MeV=t,IHEF=ihef,input_sha256=__import__('hashlib').sha256(text.encode()).hexdigest(),program_sha256=sha(exe))
    base=WORK/'cases'/group/f'z{z}_q{q}_e{float(t).hex()}_h{ihef}'
    receipt=base/'complete.json'
    if receipt.exists():
        r=json.loads(receipt.read_text())
        if r['identity']!=identity: raise RuntimeError('case identity changed')
        attempt=base/r['attempt']
        if any(sha(attempt/name)!=h for name,h in r['outputs'].items()): raise RuntimeError('saved raw output changed')
        return attempt,r
    if READ_ONLY:raise RuntimeError('saved case missing; no execution allowed: '+display(base))
    base.mkdir(parents=True,exist_ok=True)
    completed=[p for p in sorted(base.glob('attempt*')) if (p/'execution.json').exists() and (p/'input.in').read_text()==text and json.loads((p/'execution.json').read_text())['returncode']==0]
    if completed:
        attempt=completed[-1]; execution=json.loads((attempt/'execution.json').read_text())
        r=subprocess.CompletedProcess([],0,(attempt/'stdout.log').read_text(),(attempt/'stderr.log').read_text()); seconds=execution['seconds']
    else:
        attempt=base/f'attempt{len(list(base.glob("attempt*")))+1:03}'
        attempt.mkdir(); (attempt/'database').symlink_to(source/'database',target_is_directory=True)
        (attempt/'input.in').write_text(text)
        start=time.perf_counter()
        with (attempt/'input.in').open() as inp:
            r=subprocess.run([str(exe)],stdin=inp,cwd=attempt,capture_output=True,text=True)
        seconds=time.perf_counter()-start
        (attempt/'stdout.log').write_text(sanitize(r.stdout));(attempt/'stderr.log').write_text(sanitize(r.stderr))
        write(attempt/'execution.json',dict(returncode=r.returncode,seconds=seconds,identity=identity))
    outputs=list(attempt.glob('dcs_*.dat'))
    if r.returncode or not outputs: raise RuntimeError('ELSEPA case stopped; raw input/output saved: '+display(attempt))
    if custom_input is None and len(outputs)!=1: raise RuntimeError('unexpected ELSEPA output count')
    path=next((p for p in outputs if '1p000e03' in p.name),outputs[0]) if custom_input else outputs[0]
    parsed=parse(path,r.stdout)
    header=parsed['header']; actual_q=-1 if 'Projectile: electron' in header else 1 if 'Projectile: positron' in header else 0
    actual_z=int(re.search(r'Z\s*=\s*(\d+)',header)[1]); ev=float(re.search(r'Kinetic energy\s*=\s*('+NUMBER+')',header)[1])
    if actual_z!=z or actual_q!=q or abs(ev/(t*1e6)-1)>1e-5: raise RuntimeError('ELSEPA requested identity mismatch')
    if custom_input is None:
        factor=ihef==1 and t*1e6>20100*z
        required=['Nuclear model: Fermi distribution',f'./database/z_{z:03}.den']
        required+=['High-energy factorization'] if factor else ['No absorption potential','No correlation-polarization potential']+(['Furness-McCarthy exchange potential'] if q==-1 else [])
        if q==1 and 'exchange potential' in header.lower():raise RuntimeError('unexpected positron exchange model')
        if any(k not in header for k in required) or any(k in header for k in ('LDA-I absorption','LDA-II absorption','Correlation-polarization potential (','Muffin-tin model')): raise RuntimeError('effective ELSEPA model mismatch')
    rr=dict(identity=identity,attempt=attempt.name,dcs_file=path.name,seconds=seconds,printed_energy_eV=ev,outputs={p.name:sha(p) for p in attempt.iterdir() if p.is_file()})
    write(receipt,rr); return attempt,rr

def loaded(source,exe,group,z,q,t,ihef=1):
    attempt,r=case(source,exe,group,z,q,t,ihef)
    a=parse(attempt/r['dcs_file'],(attempt/'stdout.log').read_text())
    a.pop('header'); a['input_sha256']=r['identity']['input_sha256'];a['program_sha256']=r['identity']['program_sha256'];a['printed_energy_eV']=r['printed_energy_eV']
    return a,r

def bundled(source,exe):
    attempt,r=case(source,exe,'bundled',80,-1,.001,2,(source/'elscata.in').read_text())
    generated=attempt/r['dcs_file']; ref=source/'test-run-output/dcs_1p000e03.dat'
    def dcs(p):return np.loadtxt(p,comments='#')[:,2]
    difference=float(np.max(abs(dcs(generated)/dcs(ref)-1)))
    xs=cross_sections(generated.read_text())/cross_sections(ref.read_text())-1
    result=dict(status='pass' if difference<=1e-3 and max(abs(xs))<=1e-4 else 'fail',max_DCS_relative=difference,cross_sections_relative=xs.tolist(),archive_sha256=ARCHIVE_SHA)
    write('s1_bundled.json',result)
    if result['status']!='pass':raise RuntimeError('S1b failed; stop')
    return result

def generate():
    source,exe=prepare();bundled(source,exe)
    es=energies()
    # Separate preliminary timing cases; never mixed into table/check statistics.
    timing=[]
    for z in (1,8):
        for q in (-1,1):
            _,r=case(source,exe,'timing',z,q,2.)
            timing.append(dict(Z=z,charge=q,seconds=r['seconds']))
    write('generation_estimate.json',dict(rows=timing,table_cases=2*4*len(es),estimated_table_seconds=2*len(es)*sum(r['seconds'] for r in timing),method='2 MeV representative per element/charge, before generation; midpoint/IHEF0 work additional'))
    for passno in (1,2):
        for z in (1,8):
            for q in (-1,1):
                rows=[];total=0
                for t in es:
                    a,r=loaded(source,exe,f'pass{passno}',z,q,float(t)); rows.append(a);total+=r['seconds']
                    print('case saved',passno,z,q,float(t),round(r['seconds'],3),flush=True)
                values={k:np.stack([row[k] for row in rows]) for k in ('dcs','error','series_xs','series_available','table_xs','input_sha256','program_sha256','printed_energy_eV')}
                values.update(energy_MeV=es,theta_deg=rows[0]['theta_deg'],mu=rows[0]['mu'],Z=z,charge=q,
                    warnings=np.array([json.dumps(r['warnings'].tolist()) for r in rows]),notices=np.array([json.dumps(r['notices'].tolist()) for r in rows]))
                if any(not np.array_equal(row['mu'],values['mu']) or not np.array_equal(row['theta_deg'],values['theta_deg']) for row in rows): raise RuntimeError('angular grid varies')
                path=WORK/'tables'/f'z{z}_q{q}.npz'
                if passno==1:
                    if not path.exists():save(path,**values)
                if path.exists():
                    with np.load(path) as old:
                        if set(old.files)!=set(values) or any(old[k].dtype!=np.asarray(v).dtype or old[k].shape!=np.asarray(v).shape or old[k].tobytes()!=np.asarray(v).tobytes() for k,v in values.items()): raise RuntimeError('S1c arrays differ; stop')
                write(OUT/f'generation_pass{passno}_z{z}_q{q}.json',dict(status='pass',cases=len(es),seconds=total,table_sha256=sha(path),table_bytes=path.stat().st_size))
    write('s1.json',dict(status='pass',archive_sha256=ARCHIVE_SHA,bundled=json.loads((OUT/'s1_bundled.json').read_text()),two_pass_arrays_identical=True,table_energies=len(es),provenance=display(WORK/'build/build.json')))

def verify_saved():
    global READ_ONLY
    previous=READ_ONLY;READ_ONLY=True
    try:
        with __import__('contextlib').redirect_stdout(__import__('io').StringIO()):generate()
    finally:READ_ONLY=previous
    return json.loads((OUT/'s1.json').read_text())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--generate',action='store_true');a=p.parse_args()
    if a.prepare: print(display(prepare()[1]))
    elif a.generate: generate()
    else:p.error('choose --prepare or --generate')
