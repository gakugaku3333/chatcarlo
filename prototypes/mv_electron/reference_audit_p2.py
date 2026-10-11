"""C2b direct PEGS diagnostics without changing any transport table."""
import json
import shutil
import subprocess
import numpy as np
from .p2_common import ROOT,OUT,CACHE,EGS,write,save,sha,portable,verify,progress,ENERGIES
from .physics import collision,hard,ET,ER
from .reference_p2 import pegs_input
FUNCTIONS=('ESIG','PSIG','EDEDX','PDEDX','EBR1','PBR1','PBR2','TMXS','G1E','G1P','K1E','K1P','CSDAR_E','CSDAR_P','ESTEPMAX')

def diagnostic(mode):
    work=CACHE/'egs5'/f'stopping_audit_{mode}';work.mkdir(parents=True,exist_ok=True)
    code=ROOT/'prototypes/mv_electron/p2_stopping.f'
    config=dict(code_sha256=sha(code),input=pegs_input(mode).split('DECK')[0])
    cp=work/'audit_config.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise RuntimeError('stopping diagnostic inputs changed')
    write(cp,config)
    if not (work/'stopping.txt').exists() or 'DONE' not in (work/'stopping.txt').read_text():
        shutil.copy2(code,work/code.name)
        (work/'p2_stopping.inp').write_text(config['input'])
        shutil.copy2(ROOT/'docs/validation/mv/p1/egs5/base_2/epstar.dat',work/'epstar.dat')
        p=subprocess.run([str(EGS/'egs5run'),'comp'],cwd=work,input='p2_stopping\n\n\n',capture_output=True,text=True)
        (work/'build.log').write_text(p.stdout+p.stderr)
        if p.returncode or not (work/'egs5job.exe').exists():raise RuntimeError('stopping audit build failed: '+portable(work/'build.log'))
        with (work/'run.log').open('w') as f:
            p=subprocess.run(['./egs5job.exe'],cwd=work,stdout=f,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError('stopping audit failed: '+portable(work/'run.log'))
    text=(work/'stopping.txt').read_text()
    data=np.array([list(map(float,l.split()[1:])) for l in text.splitlines() if l.startswith('GRID')])
    audit=np.array([list(map(float,l.split()[1:])) for l in text.splitlines() if l.startswith('AUDIT')])
    meta=list(map(float,text.splitlines()[0].split()[1:]));rlc=meta[3]
    if data.shape!=(401,34) or audit.shape!=(444,31):raise RuntimeError('diagnostic row count mismatch')
    # GRID: T, restricted coll, full rad, restricted rad, 15 direct, 15 interpolated.
    # Transport EDEDX interpolation minus direct radiation separates coll component.
    t=data[:,0];coll_direct=data[:,1];rad_full=data[:,2];rad_used=data[:,3] if mode==0 else rad_full
    coll_transport=data[:,21]/rlc-rad_used
    coll_proto=np.array([collision(float(e))-hard(float(e))[1] for e in t])
    rad_proto=np.interp(np.log(t),np.log(ET),ER)
    coll_delta=coll_transport/coll_proto-1;rad_delta=rad_full/rad_proto-1
    interpolation_delta=data[:,21]/data[:,6]-1
    if not np.all(np.isfinite(data)):raise RuntimeError('nonfinite diagnostic')
    warning=[]
    for k,name in enumerate(FUNCTIONS):
        d=audit[:,1+k];f=audit[:,16+k]
        delta=np.divide(abs(f-d),abs(d),out=np.zeros_like(d),where=d!=0)
        ids=np.where((delta>.01)|((d==0)&(f!=0)))[0]
        if len(ids):warning.append(dict(function=name,function_number=k+1,
            energies_MeV=audit[ids,0].tolist(),relative_errors=delta[ids].tolist(),max_relative=float(delta.max())))
    bad=(abs(coll_delta)>.005)|(abs(rad_delta)>.005)
    raw_warning=[l.strip() for l in (work/'pgs5job.pegs5lst').read_text().splitlines() if 'insufficient' in l or 'maximum relative' in l]
    save(OUT/f't5_stopping_audit_{mode}.npz',T=t,collision_transport=coll_transport,collision_direct=coll_direct,
         radiation_full=rad_full,radiation_restricted=data[:,3],collision_prototype=coll_proto,radiation_prototype=rad_proto,
         collision_relative=coll_delta,radiation_relative=rad_delta,raw_grid=data,raw_audit=audit)
    return dict(IUNRST=mode,status='判定保留' if np.any(bad) else '合格',grid_MeV=t.tolist(),
        collision_relative=coll_delta.tolist(),radiation_relative=rad_delta.tolist(),
        max_collision_relative=float(max(abs(coll_delta))),max_radiation_relative=float(max(abs(rad_delta))),
        max_transport_interpolation_relative=float(max(abs(interpolation_delta))),
        exceeding_energies_MeV=t[bad].tolist(),warning_functions=warning,pegs_warnings=raw_warning,
        direct_file=portable(work/'stopping.txt'),direct_sha256=sha(work/'stopping.txt'),
        RM_PEGS_MeV=meta[0],AE=meta[1],AP=meta[2],
        affected_source_energies=[e for e in ENERGIES if np.any(bad & (t<=e))])

def check():
    if (OUT/'t5_stopping_amendment1.json').exists():return json.loads((OUT/'t5_stopping_amendment1.json').read_text())
    verify();verify('egs5_inputs');rows=[]
    for mode in (0,4):
        progress('C2b','running',IUNRST=mode,phase='direct PEGS stopping and final-QFIT interior-node audit; no transport')
        rows.append(diagnostic(mode));write('t5_stopping_audit_checkpoint.json',dict(rows=rows))
    result=dict(status='判定保留' if any(r['status']=='判定保留' for r in rows) else '合格',rows=rows,
        affected_source_energies=sorted(set(e for r in rows for e in r['affected_source_energies'])),
        criterion='restricted collision and full radiation vs prototype, absolute relative difference <=0.005 at fixed 401 log nodes .01--10 MeV',
        mapping='EFUNS #3=SPTOTE; EDEDX interpolation minus direct restricted radiation (IUNRST=0) or full radiation (4); direct BRMSTM full includes hard photons scored locally in main',
        warning_scope='PEGS final 150 intervals, 15 EFUNS; 3 interior points per 148 physical intervals reproduce final QFIT checks, not a refit',
        prior_reference_caveat='P1–P1f electron EGS5 references also used the same warning-bearing 150-interval tables.')
    write('t5_stopping_amendment1.json',result);progress('C2b','complete',result=result['status'],affected_source_energies=result['affected_source_energies'])
    verify();return result
