"""D2 active-verdict resume through pilots, freeze, one production and report."""
import sys
from pathlib import Path
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]));__package__='prototypes.mv_electron'
import argparse
import json
import fcntl
from .p2_common import OUT,CACHE,write,verify,regression,progress,sanitize


def from_saved():
    from .report_p2 import report
    rows=report()
    print(json.dumps({k:v['status'] for k,v in rows.items() if k.startswith('Q')},ensure_ascii=False))
    return rows


def prepare():
    verify();verify('egs5_inputs');regression('after')
    from .amendment_p2 import q1_amended,stopping_amended
    q1=q1_amended();stopping=stopping_amended()
    cutoff=json.loads((OUT/'t5_cutoffs_amendment1.json').read_text())
    for i in (2,3,4):
        if not (OUT/f'q{i}.json').exists():raise RuntimeError('saved prerequisite missing')
    statuses=[q1['status']]+[json.loads((OUT/f'q{i}.json').read_text())['status'] for i in (2,3,4)]
    if any(s!='合格' for s in statuses):
        progress('T6','not_started',reason='active Q1–Q4 prerequisite not met')
        from_saved();return False
    if cutoff['status']!='合格':raise RuntimeError('C2 accepted-cutoff check not met')
    progress('D2','prepared',active_Q1='q1_amendment1.json',active_stopping='t5_stopping_amendment2.json',stopping_status=stopping['status'],historical='q1.json, t5_stopping_amendment1.json, t5_selfcheck.json retained')
    progress('T1','complete',result=q1['status'],historical_result='不合格',active_verdict='q1_amendment1.json',historical_verdict='q1.json')
    progress('C2b','complete',result=stopping['status'],historical_result='判定保留',active_verdict='t5_stopping_amendment2.json',historical_verdict='t5_stopping_amendment1.json')
    state=json.loads((OUT/'progress.json').read_text()).get('stages',{}).get('T6',{}).get('state')
    if state=='not_started':progress('T6','ready_for_pilots',reason='active prerequisites met; D2 selfcheck/verification first')
    from_saved();return True


def resume(selfcheck_only=False):
    if not prepare():return
    from .reference_p2 import selfcheck
    from .workflow_p2 import checkpoint,continue_calculations
    checkpoint('D2','selfchecks_running',reason='new selfcheck_d2 case names; old configs unchanged')
    r=selfcheck()
    if r['status']!='合格':
        checkpoint('D2','stopped',reason='new-code reference selfcheck failed');return
    from .verify_resume_p2 import check as restart_check
    restart_evidence=restart_check()
    if restart_evidence['status']!='合格':raise RuntimeError('D2 reference RNG checkpoint replay failed')
    evidence=json.loads((OUT/'d2_verification.json').read_text()) if (OUT/'d2_verification.json').exists() else {}
    if evidence.get('status')!='合格':raise RuntimeError('D2 route/batch verification has not passed')
    checkpoint('D2','complete',active_selfchecks='t5_selfcheck_d2.json',verification='d2_verification.json')
    if not selfcheck_only:continue_calculations()
    from_saved()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--t0',action='store_true');p.add_argument('--from-saved',action='store_true');p.add_argument('--verify-untouched',action='store_true');p.add_argument('--prepare',action='store_true');p.add_argument('--selfcheck-only',action='store_true');a=p.parse_args()
    if a.t0:
        from .p2_common import t0
        t0()
    elif a.verify_untouched:
        from .photon_data_p2 import TABLE
        from .p2_common import sha
        r=dict(status='合格',protected=verify(),egs5_inputs=verify('egs5_inputs'))
        if sha(TABLE)!=json.loads((OUT/'photon_generation.json').read_text())['table_sha256']:raise RuntimeError('P2 generated photon table changed')
        m=json.loads((OUT/'manifest.json').read_text())
        if m['production_frozen']:r['production']=verify('production_frozen')
        regression('after');write('untouched.json',r);print(json.dumps(r,ensure_ascii=False))
    elif a.from_saved:from_saved()
    elif a.prepare:prepare()
    elif a.resume or a.selfcheck_only:
        CACHE.mkdir(parents=True,exist_ok=True)
        with (CACHE/'driver.lock').open('w') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                print('P2 resume is already running; waiting for it. Completed runs will be reused.',flush=True)
                fcntl.flock(lock,fcntl.LOCK_EX)
            lock.write(str(__import__('os').getpid())+'\n');lock.flush()
            try:
                progress('execution','running',command='.venv/bin/python prototypes/mv_electron/run_checks_p2.py --resume',driver_pid=__import__('os').getpid())
                evidence_path=OUT/'d2_verification.json'
                if evidence_path.exists():
                    evidence=json.loads(evidence_path.read_text())
                    if evidence.get('needs_fresh_import'):
                        from .p2_common import ROOT,sha
                        if not evidence.get('final_tests_passed'):raise RuntimeError('final D2 tests are pending; no pilots')
                        if any(sha(ROOT/'prototypes/mv_electron'/name)!=h for name,h in evidence['code_hashes'].items()):
                            raise RuntimeError('D2 tested implementation changed before fresh resume')
                        evidence.update(status='合格',needs_fresh_import=False)
                        write(evidence_path,evidence)
                state=json.loads((OUT/'progress.json').read_text()).get('stages',{}).get('T6',{}).get('state')
                if state=='complete':
                    verify();verify('egs5_inputs');verify('production_frozen');from_saved()
                else:resume(a.selfcheck_only)
            except BaseException as exc:
                progress('execution','interrupted',error=sanitize(str(exc)),resume='same command; completed runs/batches retained')
                from_saved();raise
    else:p.error('select a checkpoint action')
