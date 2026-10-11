"""P2 EGS5 reference build/run in external work only; raw tables remain external."""
import json
import re
import shutil
import subprocess
import time
import numpy as np
from .p2_common import ROOT,OUT,EGS,CACHE,SEED,ENERGIES,write,save,sha,portable,verify,progress,is_freeze_input
from .run_coupled_p2 import MEV_TO_GY_G

FLAG_NAMES=('incohm','incohr','iprofm','iprofr','iraylm','iraylr','iedgfl','iauger','iprdst','ibrspl','iausfl8','usegsd')

def pegs_input(mode):
    source=(ROOT/'docs/validation/mv/p1/egs5/base_2/base_2.inp').read_text()
    source=source.replace('AP=2,','AP=0.01,').replace('IUNRST=0',f'IUNRST={mode},IBOUND=0,INCOH=0,ICPROF=0,IRAYL=0')
    # Retain the mechanical calls for radiative stopping checks.
    return source

def parse(work,n,E,mode,seed):
    text=(work/'egs5job.out').read_text()
    def line(tag):
        matches=re.findall(r'^\s*'+tag+r'\s+(.+)',text,re.M)
        if len(matches)!=1:raise RuntimeError('incomplete EGS5 output: '+tag)
        return list(map(float,matches[0].split()))
    coarse=np.zeros((100,150));fine=np.zeros((100,600))
    for tag,a,nb in (('BATCH',coarse,150),('FINE',fine,600)):
        rows=re.findall(r'^\s*'+tag+r'\s+(\d+)\s+(\d+)\s+(\S+)',text,re.M)
        if len(rows)!=100*nb:raise RuntimeError('missing EGS5 batch rows')
        for b,j,v in rows:a[int(b)-1,int(j)-1]=float(v)*MEV_TO_GY_G
    ledger=line('LEDGER');cb=line('CALLBACKS');flags=dict(zip(FLAG_NAMES,line('FLAGS')));cut=line('CUTOFFS')
    lst=(work/'pgs5job.pegs5lst').read_text()
    calls=re.findall(r'Function call:\s*(\S+)\s*= BRMSTM OF\s*(\S+)\s+(\S+)',lst)
    if f'IUNRST={mode}' not in (work/'pgs5job.pegs5dat').read_text():raise RuntimeError('PEGS IUNRST mismatch')
    rawpegs=(work/'pgs5job.pegs5dat').read_text().splitlines();rflags=list(map(int,rawpegs[5].split()))
    actual=dict(IBOUND=rflags[8],INCOH=rflags[9],ICPROF=rflags[10],IRAYL=rflags[7],**flags)
    # A zero brems count in sensitivity mode is required; generated/discarded main counts must match.
    rebin=float(np.max(abs(fine.reshape(100,150,4).mean(axis=2)-coarse)))
    balance=ledger[3]/E
    tests=dict(balance=balance<=1e-9 and ledger[4]==ledger[5]==0,score=ledger[6]/E<=1e-9 and ledger[7]/E<=1e-9,
               callbacks=cb[0]==cb[1] and cb[6]==0 and (mode==0 or cb[0]==0),
               flight_annihilation=(mode==0 or cb[3]==0),
               cutoffs=abs(cut[0]-.521)<1e-12 and abs(cut[2]-.521)<1e-12 and abs(cut[1]-.01)<1e-12 and abs(cut[3]-.01)<1e-12,
               flags=all(v==0 for k,v in actual.items() if k not in ('iausfl8','usegsd')) and flags['iausfl8']==flags['usegsd']==1,
               dual_scoring=rebin<=max(1e-25,float(np.max(abs(coarse)))*1e-9))
    summary=dict(status='合格' if all(tests.values()) else '不合格',E=E,N=n,seed=seed,IUNRST=mode,
        tests=tests,ledger=ledger,callbacks=dict(zip(('brems_created','brems_discarded','brems_local_MeV','flight_annihilation','rest_annihilation','pairs','wrong_tags'),cb)),
        effective_flags=actual,cutoffs=dict(zip(('AE','AP','ECUT','PCUT','UE','UP','rho'),cut)),
        max_history_relative_balance=balance,rebin_max_Gy_cm2=rebin,CPU_seconds=line('CPU')[0],
        radiative_stopping_calls=calls,warnings=[s.strip() for s in (text+'\n'+lst).splitlines() if any(x in s.lower() for x in ('warning','not attained','insufficient'))])
    if re.search(r'^\s*KINETIC_CUTOFF',text,re.M):
        kc=line('KINETIC_CUTOFF');summary['effective_electron_kinetic_cutoff_keV']=kc[0]*1000;summary['RM_MeV']=kc[1]
        summary['tests']['kinetic_cutoff']=abs(kc[0]-(.521-kc[1]))<1e-12
        summary['status']='合格' if all(summary['tests'].values()) else '不合格'
    if re.search(r'^\s*WALL',text,re.M):summary['wall_seconds']=line('WALL')[0]
    return coarse,fine,summary

def run(name,E,n,seed,mode,production=False):
    if mode not in (0,4) or n%100:raise ValueError('invalid reference configuration')
    verify();verify('egs5_inputs')
    if production:
        verify('production_frozen')
        lock=json.loads((OUT/'pilot_lock.json').read_text())['reference_runs'][name]
        if lock!=dict(E=E,N=n,seed=seed,mode=mode):raise RuntimeError('reference production differs from pilot lock')
    work=CACHE/'egs5'/name;dest=OUT/'egs5'/name
    config=dict(E=E,N=n,seed=seed,IUNRST=mode,code_sha256=sha(ROOT/'prototypes/mv_electron/p2_egs.f'),include_sha256=sha(ROOT/'prototypes/mv_electron/p2_egs_common.inc'),input=pegs_input(mode),production=production)
    cp=dest/'config.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise RuntimeError('reference config mismatch; no rerun permitted')
    if (dest/'summary.json').exists():return json.loads((dest/'summary.json').read_text())
    def finish():
        c,f,s=parse(work,n,E,mode,seed)
        save(dest/'result.npz',coarse=c,fine=f,E=E,N=n,seed=seed)
        write(dest/'summary.json',s);write(dest/'progress.json',dict(state='complete',work=portable(work)))
        return s
    if work.exists() and (work/'egs5job.out').exists():
        text=(work/'egs5job.out').read_text()
        if re.search(r'^\s*FINE\s+100\s+600\s',text,re.M):return finish()
        if 'TIME_LIMIT' in text:
            result=dict(status='統計不足',reason='24 hour limit',E=E,N=n,seed=seed,IUNRST=mode)
            write(dest/'summary.json',result);return result
    work.mkdir(parents=True,exist_ok=True);dest.mkdir(parents=True,exist_ok=True)
    write(cp,config)
    if not (work/'egs5job.exe').exists():
        for name0 in ('p2_egs.f','p2_egs_common.inc'):shutil.copy2(ROOT/'prototypes/mv_electron'/name0,work/name0)
        (work/'p2_egs.inp').write_text(config['input']);(dest/'p2_egs.inp').write_text(config['input'])
        shutil.copy2(ROOT/'docs/validation/mv/p1/egs5/base_2/epstar.dat',work/'epstar.dat')
        (work/'config.dat').write_text(f'{E:.17g} {n} {seed} {mode}\n')
        p=subprocess.run([str(EGS/'egs5run'),'comp'],cwd=work,input='p2_egs\n\n\n',capture_output=True,text=True)
        (work/'build.log').write_text(p.stdout+p.stderr)
        if p.returncode or not (work/'egs5job.exe').exists():raise RuntimeError('EGS5 build failed; external log '+portable(work/'build.log'))
    write(dest/'progress.json',dict(state='running',config=config,work=portable(work)))
    from .process_p2 import run_process
    timer=work/'wall_budget_d2.json'
    # Fortran enforces elapsed wall internally; the supervisor additionally accounts
    # initialization, interrupted attempts and orphaned work, per reference run.
    timing=json.loads(timer.read_text()) if timer.exists() else dict(total_seconds=0.)
    remaining=max(0.,86400-timing['total_seconds']) if production else 86400
    (work/'budget.dat').write_text(f'{remaining+_checkpoint_wall(work):.17g}\n')
    state=run_process(timer,['./egs5job.exe'],work,work/'run.log',86400 if production else None)
    if state.get('timed_out') or 'TIME_LIMIT' in (work/'egs5job.out').read_text():
        s=dict(status='統計不足',reason='24 hour per-run cumulative wall limit',E=E,N=n,seed=seed,IUNRST=mode,wall_seconds=state['total_seconds'])
        write(dest/'summary.json',s);return s
    if state['returncode']:raise RuntimeError('EGS5 run failed; external log '+portable(work/'run.log'))
    s=finish();verify();return s

def _checkpoint_wall(work):
    p=work/'batch_progress.txt'
    return float(p.read_text().split()[2]) if p.exists() else 0.

def selfcheck():
    progress('T5','running',phase='selfcheck only; no pilot or production',N=10000)
    # Explicit extra source/input list, frozen before any EGS5 build.
    m=json.loads((OUT/'manifest.json').read_text())
    if 'egs5_inputs' not in m:
        files=[p for folder in ('egs','pegs','include','pegscommons','auxcode','auxcommons','data') for p in (EGS/folder).rglob('*') if p.is_file() and is_freeze_input(p)]
        files += [EGS/'egs5run',ROOT/'docs/validation/mv/p1/egs5/base_2/epstar.dat']
        m['egs5_inputs']={portable(p):sha(p) for p in sorted(set(files))}
        write('manifest.json',m)
    verify('egs5_inputs')
    rows=[]
    for j,E in enumerate(ENERGIES):
        for mode in (0,4):
            progress('T5','selfchecks_running',current_run=f'selfcheck_d2_{j}_{mode}',production=False)
            from .report_p2 import report
            report()
            r=run(f'selfcheck_d2_{j}_{mode}',E,10000,SEED+500+2*j+(mode==4),mode);rows.append(r)
            write('t5_checkpoint_d2.json',dict(rows=rows))
    result=dict(status='合格' if all(r['status']=='合格' for r in rows) else '不合格',phase='selfcheck',rows=rows,
          manual='PEGS5 User Manual pp.27–28: IUNRST nonstandard tables should not be prepared for EGS runs; P2 uses IUNRST=4 only as separately self-checked sensitivity.',
          reference_ready=all(r['status']=='合格' for r in rows),
          cutoff_decision='C2: AE=ECUT=.521 accepted; effective kinetic cutoff 10.001095 keV; no cutoff tuning',
          historical_selfcheck='t5_selfcheck.json retained unchanged')
    write('t5_selfcheck_d2.json',result);progress('T5','selfcheck_complete',result=result['status'],reference_ready=result['reference_ready'],active_selfcheck='t5_selfcheck_d2.json');return result
