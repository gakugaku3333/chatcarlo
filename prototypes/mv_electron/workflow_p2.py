"""D2/C3 one-command, one-production workflow; every completed run is immutable."""
import json
import numpy as np
from .p2_common import OUT,ROOT,TABLES,SEED,ENERGIES,write,sha,portable,verify,progress
from .photon_data_p2 import TABLE
from .run_coupled_p2 import run_saved,run_budgeted
from .reference_p2 import run as reference_run
from .compare_photon_p2 import pilot_window,run_requirement,required_n,step_check,compare,curve_stats
BACKENDS=('dcslib','mf','elsepa')
N0=1000000

def native(config):
    """Public lock files use N; both transport APIs accept lower-case n."""
    return {('n' if key=='N' else key):value for key,value in config.items()}

def load(path):
    with np.load(path) as v:return {k:v[k] for k in v.files}

def checkpoint(stage,state,**kwargs):
    progress(stage,state,**kwargs)
    from .report_p2 import report
    report()

def schedule():
    """Seed series is registered before pilots; it never depends on a dose value."""
    p=OUT/'pilot_schedule.json'
    refs={};runs={}
    for j,e in enumerate(ENERGIES):
        for mode in (0,4):
            refs[f'pilot_ref_{j}_{mode}']=dict(E=e,N=N0,seed=SEED+1000+10*j+(mode==4),mode=mode)
        for ib,b in enumerate(BACKENDS):
            runs[f'pilot_{b}_{j}']=dict(backend=b,E=e,N=N0,seed=SEED+2000+100*j+10*ib,fe=.05,k=.025)
            if e in (2.,10.):
                for iv,(variant,fe,k) in enumerate((('fe',.025,.025),('k',.05,.0125)),1):
                    runs[f'pilot_{b}_{j}_{variant}']=dict(backend=b,E=e,N=N0,seed=SEED+2000+100*j+10*ib+iv,fe=fe,k=k)
    value=dict(reference_runs=refs,runs=runs,main_seed_rule='reference SEED+10000+10*j+(mode==4); prototype SEED+20000+100*j+10*backend_index+variant_index',bootstrap_seed_rule='SEED+30000+100*j+backend_index; final SEED+40000+100*j+backend_index')
    if p.exists() and json.loads(p.read_text())!=value:raise RuntimeError('pilot schedule changed')
    if not p.exists():write(p,value)
    return value

def pilots():
    s=schedule()
    for name,c in s['reference_runs'].items():
        checkpoint('T6','pilots_running',current_run=name,config=c,production=False)
        r=reference_run(name,**native(c))
        if r['status']!='合格':raise RuntimeError('reference pilot selfchecks failed: '+name)
        checkpoint('T6','pilots_running',last_completed=name,production=False)
    for name,c in s['runs'].items():
        checkpoint('T6','pilots_running',current_run=name,config=c,production=False)
        run_saved(name,**native(c))
        checkpoint('T6','pilots_running',last_completed=name,production=False)
    checkpoint('T6','pilots_complete',production=False)


def capped_run(config,desired,pilot_seconds):
    # Per individual energy/code/backend run, not a shared 24h budget.
    per_history=pilot_seconds/N0
    cap=min(200000000,max(100,int(86400/per_history)//100*100))
    n=min(desired,cap)
    return dict(**config,N=n),dict(required_N=desired,chosen_N=n,N_cap=cap,
        pilot_seconds=float(pilot_seconds),predicted_seconds=float(n*per_history),
        projected_statistics='統計不足' if n<desired else '要求を満たす見込み',wall_limit_seconds=86400)

def lock_pilots():
    target=OUT/'pilot_lock.json'
    if target.exists():
        lock=json.loads(target.read_text())
        for path,h in lock['pilot_hashes'].items():
            from .p2_common import resolve
            if sha(resolve(path))!=h:raise RuntimeError('locked pilot changed')
        return lock
    schedule_value=schedule();windows={};requirements={};runs={};refs={};estimates={};hashes={}
    for name in schedule_value['reference_runs']:
        p=OUT/'egs5'/name/'result.npz';hashes[portable(p)]=sha(p)
    for name in schedule_value['runs']:
        p=OUT/'runs'/name/'result.npz';hashes[portable(p)]=sha(p)
    for j,e in enumerate(ENERGIES):
        reference=load(OUT/'egs5'/f'pilot_ref_{j}_0/result.npz')
        sensitivity=load(OUT/'egs5'/f'pilot_ref_{j}_4/result.npz')
        window=pilot_window(reference['coarse'],reference['fine']);windows[str(e)]=window
        req={};desired=N0
        for ib,b in enumerate(BACKENDS):
            proto=load(OUT/'runs'/f'pilot_{b}_{j}/result.npz')
            halves=[load(OUT/'runs'/f'pilot_{b}_{j}_{v}/result.npz') for v in ('fe','k')] if e in (2.,10.) else []
            r=run_requirement(proto,reference,halves,window,N0,SEED+30000+100*j+ib)
            req[b]=r;desired=max(desired,r['required_N'])
        if window['status']=='合格':desired=max(desired,required_n(sensitivity['coarse'],sensitivity['fine'],window,N0))
        requirements[str(e)]=dict(by_backend=req,desired_common_N=desired)
        for mode in (0,4):
            name=f'main_ref_{j}_{mode}'
            summary=json.loads((OUT/'egs5'/f'pilot_ref_{j}_{mode}/summary.json').read_text())
            # WALL includes initialization (CPU excludes it).
            seconds=summary['wall_seconds']
            refs[name],estimates[name]=capped_run(dict(E=e,seed=SEED+10000+10*j+(mode==4),mode=mode),desired,seconds)
        for ib,b in enumerate(BACKENDS):
            for iv,(variant,fe,k) in enumerate((('',.05,.025),('_fe',.025,.025),('_k',.05,.0125))):
                if iv and e not in (2.,10.):continue
                name=f'main_{b}_{j}{variant}'
                pilot=load(OUT/'runs'/f'pilot_{b}_{j}{variant}/result.npz')
                runs[name],estimates[name]=capped_run(dict(backend=b,E=e,seed=SEED+20000+100*j+10*ib+iv,fe=fe,k=k),desired,float(pilot['seconds']))
    value=dict(windows=windows,requirements=requirements,runs=runs,reference_runs=refs,estimates=estimates,pilot_hashes=hashes,
        rule='One main run per fixed config; no extra histories, seed selection or dose scaling; same 100 batch partition; per-run 2e8/24h limits',schedule_sha256=sha(OUT/'pilot_schedule.json'))
    write(target,value)
    checkpoint('T6','pilot_lock_complete',pilot_lock='pilot_lock.json',production=False)
    return value

# Explicit declaration, not a directory scan. All documents/plans are excluded.
PRODUCTION_CODE=(
 'amendment_p2.py','workflow_p2.py','run_checks_p2.py','report_p2.py','p2_common.py',
 'reference_p2.py','reference_audit_p2.py','run_coupled_p2.py','compare_photon_p2.py',
 'photon_data_p2.py','photon_sampling_p2.py','photon_transport_p2.py',
 'check_sampling_p2.py','check_coupling_p2.py','check_transfer_p2.py',
 'p2_egs.f','p2_egs_common.inc','p2_stopping.f','physics.py','transport.py',
 'sampling.py','backend_sampling.py','process_p2.py','verify_resume_p2.py')

def freeze():
    verify();verify('egs5_inputs')
    m=json.loads((OUT/'manifest.json').read_text())
    if m['production_frozen']:
        verify('production_frozen');return
    # Third freeze only after full implementation, pilots and immutable lock.
    if not (OUT/'pilot_lock.json').exists():raise RuntimeError('freeze requires pilot lock')
    files=[ROOT/'prototypes/mv_electron'/name for name in PRODUCTION_CODE]
    files += [OUT/name for name in ('pilot_lock.json','pilot_schedule.json','q1_amendment1.json','q2.json','q3.json','q4.json','t5_cutoffs_amendment1.json','t5_stopping_amendment2.json','t5_selfcheck_d2.json','d2_verification.json','d2_reference_restart.json')]
    files += list(TABLES.values())+[TABLE]
    m['production_frozen']={portable(p):sha(p) for p in files}
    m['production_freeze_note']='Explicit inputs only; no plan, MODEL, PREREGISTRATION, RESULTS, or other document.'
    write('manifest.json',m);verify('production_frozen')
    regression_result=json.loads((OUT/'q0_regression.json').read_text())
    write('q0.json',dict(status=regression_result['status'],protected=verify(),egs5_inputs=verify('egs5_inputs'),production=verify('production_frozen'),regression=regression_result))
    checkpoint('T0','production_frozen',result=regression_result['status'],files=len(files))


def mark_main(name,c,kind):
    p=json.loads((OUT/'progress.json').read_text());started=p.setdefault('production_run_names',[])
    if name not in started:started.append(name)
    p['production_runs_started']=len(started);write('progress.json',p)
    checkpoint('T6','production_running',current_run=name,kind=kind,config=c,freeze='manifest.json production_frozen')


def production(lock):
    verify('production_frozen')
    for name,c in lock['reference_runs'].items():
        mark_main(name,c,'reference');reference_run(name,**native(c),production=True)
        checkpoint('T6','production_running',last_completed=name)
    for name,c in lock['runs'].items():
        mark_main(name,c,'prototype');run_budgeted(name,**native(c))
        checkpoint('T6','production_running',last_completed=name)
    checkpoint('T6','production_complete')


def result_at(folder,name):
    target=OUT/folder/name/'result.npz'
    return load(target) if target.exists() else None

def analyse(lock):
    q7=[];rows=[];q5=[]
    for b in BACKENDS:
        steps=[]
        for j in (1,3):
            base=result_at('runs',f'main_{b}_{j}')
            for v in ('fe','k'):
                half=result_at('runs',f'main_{b}_{j}_{v}')
                r=step_check(base,half,lock['windows'][str(ENERGIES[j])]) if base is not None and half is not None else dict(status='統計不足',reason='run terminated at per-run limit')
                steps.append(dict(backend=b,E=ENERGIES[j],half=v,**r))
        state='合格' if all(r['status']=='合格' for r in steps) else next(s for s in ('未評価','統計不足','不合格') if any(r['status']==s for r in steps))
        q7.append(dict(backend=b,status=state,rows=steps))
        for j,e in enumerate(ENERGIES):
            window=lock['windows'][str(e)];p=result_at('runs',f'main_{b}_{j}')
            r=result_at('egs5',f'main_ref_{j}_0');s=result_at('egs5',f'main_ref_{j}_4')
            reference_states=[json.loads((OUT/'egs5'/f'main_ref_{j}_{mode}/summary.json').read_text())['status'] for mode in (0,4)]
            if p is None or r is None or s is None:
                value=dict(status='統計不足',engineering='統計不足',statistical='未評価',reason='run terminated at per-run limit')
            elif any(status!='合格' for status in reference_states):
                value=dict(status='未評価',engineering='未評価',statistical='未評価',reason='reference production selfcheck not met')
            elif np.any(p['violations']) or p['counts'][5] or p['counts'][6]:
                value=dict(status='未評価',engineering='未評価',statistical='未評価',reason='production history ledger or resource-limit violation')
            else:
                value=compare(p,r,s,window,SEED+40000+100*j+BACKENDS.index(b),True,state=='合格')
                if e in json.loads((OUT/'t5_stopping_amendment2.json').read_text())['affected_source_energies']:
                    value['status']='判定保留';value['stopping_reference_hold']=True
            if p is not None:
                dose=curve_stats(p['fine'])[0];kerma=curve_stats(p['kerma'])[0];coll=curve_stats(p['collision_kerma'])[0]
                z=(np.arange(len(dose))+.5)*.05;ids=np.where((z>=5)&(z<=25))[0]
                q5.append(dict(backend=b,E=e,status='記録',depth_cm=z[ids].tolist(),dose_over_total_kerma=(dose[ids]/kerma[ids]).tolist(),dose_over_collision_kerma=(dose[ids]/coll[ids]).tolist()))
                pair=curve_stats(p['pair'])[0]
                value['pair_deposition_fraction']=np.divide(pair,dose,out=np.zeros_like(dose),where=dose>0).tolist()
            rows.append(dict(backend=b,E=e,**value))
    def aggregate(values):
        return next((s for s in ('判定保留','未評価','統計不足','不合格') if s in values),'合格')
    write('q7.json',dict(status=aggregate([r['status'] for r in q7]),rows=q7))
    write('q6.json',dict(status=aggregate([r['status'] for r in rows]),rows=rows,scope='read each backend/energy verdict; not an overall-pass claim'))
    write('q5.json',dict(status='記録' if len(q5)==12 else '未評価',rows=q5,rule='5–25 cm; diagnostic only, no expected-behavior gate'))
    plots(lock)
    verify();verify('egs5_inputs');verify('production_frozen')
    checkpoint('T6','complete',result='q5.json q6.json q7.json; no overall-pass claim')


def plots(lock):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for j,e in enumerate(ENERGIES):
        r=result_at('egs5',f'main_ref_{j}_0');s=result_at('egs5',f'main_ref_{j}_4')
        fig,axs=plt.subplots(3,1,figsize=(9,11))
        if r is not None:
            d,se=curve_stats(r['coarse']);z=(np.arange(len(d))+.5)*.2
            axs[0].plot(z,d,label='EGS5 main');axs[0].fill_between(z,d-se,d+se,alpha=.15)
            if s is not None:axs[1].plot(z,(curve_stats(s['coarse'])[0]-d)/lock['windows'][str(e)]['Dref'],label='EGS5 sensitivity - main')
            for b in BACKENDS:
                p=result_at('runs',f'main_{b}_{j}')
                if p is None:continue
                a,ae=curve_stats(p['coarse']);axs[0].plot(z,a,label=b)
                axs[1].plot(z,(a-d)/lock['windows'][str(e)]['Dref'],label=b+' - EGS5')
                fine=curve_stats(p['fine'])[0];zf=(np.arange(len(fine))+.5)*.05
                axs[2].plot(zf,fine,label=b)
            axs[2].plot((np.arange(600)+.5)*.05,curve_stats(r['fine'])[0],label='EGS5 main')
        axs[2].set_xlim(0,6);axs[0].set_ylabel('Dose [Gy cm2]');axs[1].set_ylabel('Difference / pilot Dref');axs[2].set_ylabel('Dose [Gy cm2]')
        for ax in axs:ax.set_xlabel('Depth [cm]');ax.legend();ax.grid(alpha=.2)
        fig.suptitle(f'P2 {e:g} MeV, K1=0.025, fE=0.05 (low-dose bins excluded from verdict)')
        fig.tight_layout();fig.savefig(OUT/f'dose_{e:g}MeV.png',dpi=160);plt.close(fig)


def continue_calculations():
    pilots();lock=lock_pilots();freeze();production(lock);analyse(lock)
