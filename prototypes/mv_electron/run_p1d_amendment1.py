"""Approved amendment A0-A6: explicit freeze, reproduction gate, run checkpoints.
No physics, analysis, seed, history count, threshold, or decision changes.
"""
import argparse
import difflib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from prototypes.mv_electron.p1d_common import *

ENTRY = 'prototypes/mv_electron/run_p1d_amendment1.py'
PLAN = 'docs/ai/plans/2026-10-08-mv-p1d-half-step.md'
RESUME = '.venv/bin/python ' + ENTRY
AUTHORIZED = {'prototypes/mv_electron/p1d_common.py', 'prototypes/mv_electron/report_p1d.py'}

def approved():
    text = (ROOT / PLAN).read_text().split('## 追補1', 1)[1]
    return '状態: approved' in text.split('### 何が起きたか', 1)[0]

def scoped_files(saved_fixed):
    paths = list((ROOT / 'prototypes/mv_electron').rglob('*.py'))
    for name in ('p1', 'p1b', 'p1c'):
        paths += [p for p in (ROOT / 'docs/validation/mv' / name).rglob('*') if p.is_file()]
    paths += [ROOT / p for p in TABLES.values()]
    paths += [OUT / 'PREREGISTRATION.md']
    paths += [ROOT / p for p in saved_fixed]
    paths += [p for p in (OUT / 'stop1').rglob('*') if p.is_file()]
    return sorted(set(p.relative_to(ROOT).as_posix() for p in paths))

def verify_freeze2():
    frozen = json.loads((OUT / 'amendment1_hashes.json').read_text())
    current = set(scoped_files(frozen['saved_R1_files']))
    expected = set(frozen['files'])
    added, removed = sorted(current - expected), sorted(expected - current)
    changed = [p for p,h in frozen['files'].items() if not (ROOT / p).is_file() or sha(ROOT / p) != h]
    r = dict(status='pass' if not changed and not added and not removed else 'fail',
             freeze='freeze2', initial_files=len(frozen['files']), frozen_additional_files=0,
             changed=changed, added_files=added, removed_files=removed)
    if r['status'] != 'pass':
        write('amendment1_freeze_failure.json', r)
        raise RuntimeError('R0 freeze2 mismatch: stop')
    return r

def seal():
    assert approved(), 'Amendment must be approved'
    target = OUT / 'amendment1_hashes.json'
    if target.exists(): verify_freeze2(); return
    stop = OUT / 'stop1'
    for p,h in json.loads((stop / 'sha256.json').read_text()).items():
        assert sha(stop / p) == h, 'A0 preservation mismatch'
    assert json.loads((stop / 'progress.json').read_text())['restart_allowed'] is False
    saved_fixed = sorted(p.relative_to(ROOT).as_posix() for p in OUT.glob('*/fixed/*') if p.is_file())
    assert len([p for p in saved_fixed if p.endswith('.npz')]) == 7
    scope = scoped_files(saved_fixed)
    old = {**json.loads((OUT / 'initial_hashes.json').read_text()), **json.loads((OUT / 'code_hashes.json').read_text())}
    hashes = {p:sha(ROOT / p) for p in scope}
    modified = sorted(p for p in set(old)&set(hashes) if old[p] != hashes[p])
    assert set(modified) <= AUTHORIZED, 'Unauthorized change since freeze1'
    known = sorted(p for p,h in old.items() if p not in scope and (not (ROOT / p).is_file() or sha(ROOT / p) != h))
    assert known == sorted([PLAN, 'docs/ai/plans/2026-10-08-mv-p2-photon-coupling.md'])
    baseline = Path('/tmp/p1d-amendment1-baseline')
    original = {p.relative_to(baseline).as_posix():p.read_text() for p in baseline.rglob('*.py')}
    actual = {p.relative_to(ROOT / 'prototypes/mv_electron').as_posix():p.read_text() for p in (ROOT / 'prototypes/mv_electron').rglob('*.py')}
    changes = sorted(p for p in set(original)|set(actual) if original.get(p) != actual.get(p))
    assert set(changes) == {'p1d_common.py','report_p1d.py','run_p1d_amendment1.py'}
    diff = ''.join(''.join(difflib.unified_diff(original.get(p,'').splitlines(True),actual.get(p,'').splitlines(True),
        fromfile='before/prototypes/mv_electron/'+p,tofile='after/prototypes/mv_electron/'+p)) for p in changes)
    (OUT / 'amendment1_diff.txt').write_text(diff)
    write('amendment1_changes.json', dict(modified=modified, added=sorted(set(hashes)-set(old)),
        excluded_from_freeze2=sorted(set(old)-set(hashes)), known_freeze1_mismatches=known,
        unauthorized_common_changes=[], source_files_changed_or_added=['prototypes/mv_electron/'+p for p in changes],
        diff_sha256=sha(OUT / 'amendment1_diff.txt'), plan_edited_by_this_work=False))
    write(target, dict(freeze='freeze2', files=hashes, saved_R1_files=saved_fixed,
        explicit_scope='all prototype .py; all p1/p1b/p1c files; used GS/EGS5; preregistration; seven saved R1 samples and receipts; stop1',
        additions_and_deletions_checked=True))
    verify_freeze2(); regression_saved()
    print('A3: freeze2 sealed;',len(hashes),'files; authorized differences only',flush=True)

def record_command(command, exit_code, log):
    path = OUT / 'amendment1_commands.json'
    saved = json.loads(path.read_text()) if path.exists() else dict(commands=[])
    saved['commands'].append(dict(command=' '.join(command),exit_code=exit_code,log=log))
    write(path,saved)

def reproduce():
    assert approved()
    verify_freeze2(); regression_saved()
    target = OUT / 'amendment1_restart.json'
    if target.exists() and json.loads(target.read_text()).get('reproduction',{}).get('identical'):
        restart_gate(); print('A4: existing successful reproduction retained',flush=True); return
    from prototypes.mv_electron.reference import fixed
    path = OUT / 'mf/fixed/0.05_True.npz'
    before = sha(path)
    _, args, kw = table('mf')
    with np.load(path) as saved:
        seed, n, k, length = int(saved['seed']), int(saved['N']), float(saved['K']), float(saved['path_cm'])
        assert seed == FIXED_SEED and n == 1000000 and k == .025 and length == .05
        t = time.perf_counter()
        values = fixed(2.,length,n,seed,True,k,*args,**kw)
        identical = np.array_equal(values,saved['values'])
        shape = list(values.shape)
    assert sha(path) == before
    verify_freeze2()
    receipt = dict(amendment_approved=approved(), freeze2_saved=True,
        freeze2_sha256=sha(OUT / 'amendment1_hashes.json'), reproduction=dict(identical=identical,
        sample=path.relative_to(ROOT).as_posix(), sample_sha256=before, seed=seed,N=n,K=k,path_cm=length,
        shape=shape,seconds=time.perf_counter()-t, memory_only=True,saved_file_overwritten=False,pooled_into_statistics=False))
    write(target,receipt)
    assert identical, 'A4 reproduction mismatch: stop'
    print('A4: mf saved sample reproduced bit-for-bit in memory; original not overwritten',flush=True)

def restart_gate():
    assert approved(), 'Amendment no longer approved'
    verify_freeze2(); regression_saved()
    receipt = json.loads((OUT / 'amendment1_restart.json').read_text())
    assert receipt['amendment_approved'] and receipt['freeze2_saved'] and receipt['reproduction']['identical']
    assert receipt['freeze2_sha256'] == sha(OUT / 'amendment1_hashes.json')
    assert receipt['reproduction']['memory_only'] and not receipt['reproduction']['pooled_into_statistics']

def progress(**extra):
    write('amendment1_progress.json',dict(resume_command=RESUME,**extra))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seal',action='store_true'); p.add_argument('--reproduce',action='store_true')
    p.add_argument('--worker',choices=BACKENDS); p.add_argument('--run',choices=tuple(CONFIGS))
    a=p.parse_args()
    if a.seal: seal(); return
    if a.reproduce: reproduce(); return
    restart_gate()
    from prototypes.mv_electron.run_p1d import fixed_runs,transport_run,benchmark
    from prototypes.mv_electron.run_checks_p1d import r1,saved,k5b
    if a.worker:
        assert a.run and r1(a.worker)['status']=='pass', 'R1 gate closed'
        transport_run(a.worker,a.run); return
    assert not a.run
    progress(stage='A5',active='dcslib missing R1 sample')
    fixed_runs('dcslib') # Existing three are validated and skipped; only 0.2_True is computed.
    results={b:r1(b) for b in BACKENDS}
    write('amendment1_R1.json',results)
    eligible=[b for b in BACKENDS if results[b]['status']=='pass']
    progress(stage='A6',eligible_backends=eligible,R1_status={b:results[b]['status'] for b in BACKENDS})
    print('R1 gates:',{b:results[b]['status'] for b in BACKENDS},flush=True)
    for b in eligible: k5b(b,execute=True)
    def worker(backend,name):
        command=['.venv/bin/python',ENTRY,'--worker',backend,'--run',name]
        r=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        output=(r.stdout+r.stderr).replace(str(ROOT),'.').replace(str(Path.home()),'~')
        log=f'{backend}/amendment1_{name}.log'; (OUT/log).write_text(output)
        return command,r.returncode,log,output,backend,name
    jobs=[(b,n) for n in CONFIGS for b in eligible]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(worker,*job) for job in jobs]
        for future in as_completed(futures):
            command,code,log,output,b,n=future.result()
            record_command(command,code,log)
            if code: raise RuntimeError('worker failed: '+log)
            progress(stage='A6',last_completed_backend=b,last_completed_run=n,eligible_backends=eligible)
            print(output.strip(),flush=True)
    # Timing is serialized and excluded from inference statistics.
    for b in eligible: benchmark(b)
    saved(); verify_freeze2()
    progress(stage='complete',eligible_backends=eligible,
             withheld_backends=[b for b in BACKENDS if b not in eligible],
             run_checkpoints='per-run NPZ and SHA256 completion receipt; incomplete run restarts at same seed')
    print('A6 complete',flush=True)

if __name__=='__main__': main()
