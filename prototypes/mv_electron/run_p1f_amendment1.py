"""P1f amendment 1: sealed inputs, prechecks, one K5b retry and saved-only judgment.

No transport production runs, sampling, ELSEPA execution or table generation here.
"""
import argparse
import copy
import difflib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from prototypes.mv_electron.p1f_common import (
    OUT, ROOT, WORK, CONFIGS, aggregate, display, resolve, sanitize, sha, targets, write,
)

STOP = OUT / 'stop1'
FREEZE2 = OUT / 'amendment1_hashes.json'
TEST = 'prototypes/mv_electron/tests/test_p1f.py'
SCRIPT = 'prototypes/mv_electron/run_p1f_amendment1.py'
MODIFIED = (TEST, 'prototypes/mv_electron/p1f_common.py',
            'prototypes/mv_electron/run_checks_p1f.py')
K5B = TEST + '::test_actual_p1f_transport_and_k5b'
NEGATIVE = TEST + '::test_k5b_guard_detects_rutherford'


def read(path):
    return json.loads(Path(path).read_text())


def record_command(command, log):
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    output = sanitize(result.stdout + result.stderr)
    (OUT / log).write_text(output)
    row = dict(command=' '.join(command), exit_code=result.returncode,
               seconds=time.perf_counter()-started, log=log, log_sha256=sha(OUT/log))
    p = OUT / 'amendment1_commands.json'
    rows = read(p) if p.exists() else []
    rows.append(row)
    write(p, rows)
    print(row['command'], 'exit', result.returncode, output.splitlines()[-1:] , flush=True)
    return row, output


def verify_immutable():
    frozen = read(STOP/'immutable_hashes.json')
    changed = [p for p, h in frozen.items() if not resolve(p).is_file() or sha(resolve(p)) != h]
    result = dict(status='fail' if changed else 'pass', files=len(frozen), changed=changed)
    if changed:
        raise RuntimeError('B0 immutable hash mismatch: '+str(changed))
    return result


def freeze2_targets():
    now = targets()
    now.update({p:sha(resolve(p)) for p in read(STOP/'immutable_hashes.json')})
    now.update({display(p):sha(p) for p in STOP.rglob('*') if p.is_file()})
    return now


def verify_freeze2():
    frozen = read(FREEZE2)
    if sha(OUT/'freeze_hashes.json') != frozen['freeze1_sha256']:
        raise RuntimeError('freeze1 record changed')
    now = freeze2_targets()
    changed = [p for p,h in frozen['files'].items() if now.get(p) != h]
    added = sorted(set(now)-set(frozen['files']))
    result = dict(status='fail' if changed or added else 'pass',
                  files=len(frozen['files']), changed=changed, added=added, freeze=2)
    if changed or added:
        raise RuntimeError('freeze2 mismatch: '+str(result))
    verify_immutable()
    return result


def precheck():
    if FREEZE2.exists() or (OUT/'k5b_amendment1_attempt.json').exists():
        raise RuntimeError('B1 must precede freeze2 and formal execution')
    if (OUT/'amendment1_prechecks.json').exists():
        raise RuntimeError('prechecks already recorded; no result-driven revision')
    before = sha(ROOT/TEST)
    rows = []
    for command, log in (
        (['.venv/bin/python','-m','pytest',K5B,'-q'], 'amendment1_single.log'),
        (['.venv/bin/python','-m','pytest',NEGATIVE,'-q'], 'amendment1_negative.log'),
        (['.venv/bin/python','-m','pytest','prototypes/mv_electron/tests','-q','-rA'], 'amendment1_prototype.log'),
    ):
        row, output = record_command(command, log)
        ok = row['exit_code'] == 0
        if command[3] in (K5B, NEGATIVE):
            ok = ok and re.search(r'\b1 passed\b',output) is not None and 'skipped' not in output
        else:
            ok = ok and 'PASSED '+K5B in output and 'PASSED '+NEGATIVE in output
        row['status'] = 'pass' if ok else 'fail'
        rows.append(row)
    assert sha(ROOT/TEST) == before
    diff = ''.join(difflib.unified_diff(
        (STOP/'baseline_code/test_p1f.py').read_text().splitlines(keepends=True),
        (ROOT/TEST).read_text().splitlines(keepends=True),
        fromfile='stop1/baseline_code/test_p1f.py', tofile=TEST))
    (OUT/'amendment1_diff.txt').write_text(diff)
    write('amendment1_prechecks.json',dict(status=aggregate(r['status'] for r in rows),
          test_sha256=before, test_diff_sha256=sha(OUT/'amendment1_diff.txt'),
          i='単独と全体の双方で同じK5bテストが合格することを確認',
          ii='同じ隔離JITをrutherfordで実行し、mode 0/1とも禁止経路のRuntimeErrorを検出', rows=rows))
    verify_immutable()


def freeze():
    if FREEZE2.exists():
        raise RuntimeError('freeze2 already exists; refuse overwrite')
    check=read(OUT/'amendment1_prechecks.json')
    if check['status']!='pass' or check['test_sha256']!=sha(ROOT/TEST):
        raise RuntimeError('B1 prechecks missing or test subsequently changed')
    if check['test_diff_sha256']!=sha(OUT/'amendment1_diff.txt'):
        raise RuntimeError('B1 diff changed')
    for r in check['rows']:
        if r['log_sha256']!=sha(OUT/r['log']):raise RuntimeError('B1 log changed')
    verify_immutable()
    old=read(OUT/'freeze_hashes.json');now=targets()
    changed=sorted(p for p,h in old.items() if now.get(p)!=h)
    added=sorted(set(now)-set(old))
    if changed!=sorted(MODIFIED) or added!=[SCRIPT]:
        raise RuntimeError('unauthorized freeze1 difference: '+str((changed,added)))
    # Bind successful prechecks to freeze2; these exact files become stop1 contents.
    for name in ('amendment1_prechecks.json','amendment1_diff.txt',
                 'amendment1_single.log','amendment1_negative.log','amendment1_prototype.log'):
        dest=STOP/'prechecks'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((OUT/name).read_bytes())
    now=freeze2_targets()
    write(FREEZE2,dict(freeze=2,freeze1_sha256=sha(OUT/'freeze_hashes.json'),
          freeze1_files=len(old),changed_from_freeze1=changed,added_code=added,
          added_artifacts=sorted(set(now)-set(old)-set(added)),files=now,
          scope='凍結1の対象 + B0の固定32ファイル + stop1/全ファイル。計画や他の文書は対象外。'))
    print(json.dumps(verify_freeze2(),ensure_ascii=False))


def execute():
    verify_freeze2()
    target=OUT/'k5b_amendment1.json'
    attempt=OUT/'k5b_amendment1_attempt.json'
    # Exclusive creation claims the single formal attempt before starting pytest.
    with attempt.open('x') as f:
        json.dump(dict(stage='B3',attempt=1,test_sha256=sha(ROOT/TEST),
                       freeze2_sha256=sha(FREEZE2)),f,indent=2)
        f.write('\n')
    if target.exists():raise RuntimeError('formal result already exists')
    row,output=record_command(['.venv/bin/python','-m','pytest',K5B,'-q'],
                              'k5b_amendment1.log')
    # Original K5b exit-code/skip judgment, unchanged.
    row.update(status='pass' if row['exit_code']==0 and 'skipped' not in output else 'fail',
               attempt=1,previous_record='k5b.json',test_sha256=sha(ROOT/TEST),
               freeze2_sha256=sha(FREEZE2))
    write(target,row)
    # B4 cannot run until all B0 hashes and freeze2 have been checked after B3.
    write('amendment1_pre_b4_hashes.json',dict(immutable=verify_immutable(),freeze2=verify_freeze2()))
    if row['status']!='pass':
        print('K5b不合格。そのまま停止。',flush=True)
        return
    evaluate_saved()


def evaluate_saved():
    frozen=verify_freeze2()
    boundary=read(OUT/'amendment1_pre_b4_hashes.json')
    if boundary['immutable']['status']!='pass' or boundary['freeze2']['status']!='pass':
        raise RuntimeError('B4 precondition hash check missing')
    k5b=read(OUT/'k5b_amendment1.json')
    if k5b['test_sha256']!=sha(ROOT/TEST) or k5b['freeze2_sha256']!=sha(FREEZE2):
        raise RuntimeError('formal receipt mismatch')
    if sha(OUT/k5b['log'])!=k5b['log_sha256']:raise RuntimeError('formal log changed')
    from prototypes.mv_electron.run_checks_p1f import (
        comparison, regression_saved, validate_run,
    )
    # Receipts/conditions and regressions are read-only. Never call r2() or generators.
    for name in CONFIGS:validate_run(name)
    regression=regression_saved()
    old=read(STOP/'summary.json');summary=copy.deepcopy(old)
    summary['T0']['freeze1']=old['T0']['freeze']
    summary['T0']['freeze']=frozen
    summary['T0']['regression']=regression
    summary['T0']['status']='pass'
    summary['T5']['R2']['K5b']=k5b
    summary['T5']['R2']['status']=aggregate(summary['T5']['R2'][k]['status']
                                            for k in ('K5','K5b','K7','K8','K9ii'))
    summary['T5']['status']=aggregate(summary['T5'][k]['status'] for k in ('R1','R2'))
    summary['T6']=dict(status='not-evaluated',reason='T0,T1,T2,T4,T5合格時のみ判定')
    if k5b['status']=='pass' and all(summary[k]['status']=='pass' for k in ('T0','T1','T2','T4','T5')):
        reference=read(ROOT/'docs/validation/mv/p1/egs5_checks.json')['K4']['status']
        if reference=='pass':summary['T6']=comparison(2)
    summary['amendment1']=dict(
        B0=read(STOP/'preservation.json'),B1=read(OUT/'amendment1_prechecks.json'),
        freeze2_sha256=sha(FREEZE2),formal_attempts=1,
        pre_B4_immutable=verify_immutable(),T6_known_before_amendment=0.007502233691723832,
        saved_values_reused=['T1','T2','T3','T4','K5','K7','K8','K9(i)','K9(ii)','T7','auxiliary'],
        no_new_production_runs=True,no_new_K9_samples=True,no_new_tables=True,EGS5_rerun=False)
    write('summary.json',summary)
    print(json.dumps({k:summary[k]['status'] for k in ('T0','T1','T2','T3','T4','T5','T6','T7')},ensure_ascii=False))
    verify_freeze2()
    return summary


def main():
    p=argparse.ArgumentParser()
    options=p.add_mutually_exclusive_group(required=True)
    for flag in ('precheck','freeze','execute','from-saved','verify-untouched','production-tests'):
        options.add_argument('--'+flag,action='store_true')
    a=p.parse_args()
    if a.precheck:precheck()
    elif a.freeze:freeze()
    elif a.execute:execute()
    elif a.from_saved:evaluate_saved()
    elif a.production_tests:
        verify_freeze2()
        record_command(['.venv/bin/python','-m','pytest','tests/','-q'], 'amendment1_production.log')
        record_command(['git','status','--short','--','chatcarlo','tests','scripts','examples'], 'amendment1_protected.log')
        verify_freeze2()
    else:
        result=dict(freeze2=verify_freeze2(),immutable=verify_immutable())
        write('amendment1_untouched.json',result)
        print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
