"""Real EGS5 intermediate-checkpoint replay, separate from all Q6 runs."""
import json
import shutil
import struct
import threading
import time
import numpy as np
from .p2_common import OUT,CACHE,SEED,write,sha,portable,verify,progress
from .reference_p2 import run,parse
from .process_p2 import run_process


def check():
    target=OUT/'d2_reference_restart.json'
    if target.exists():return json.loads(target.read_text())
    name='selfcheck_d2_3_0';work=CACHE/'egs5'/name
    probe=CACHE/'egs5/restart_probe_d2';snapshot=CACHE/'restart_probe_checkpoint.bin'
    finished=threading.Event();captured=[]
    def watch():
        while not finished.is_set():
            p=work/'checkpoint.bin'
            if p.exists() and not snapshot.exists():
                data=p.read_bytes();done=struct.unpack('=i',data[4:8])[0]
                if 0<done<10000:
                    snapshot.write_bytes(data);captured.append(done);return
            time.sleep(.005)
    if not snapshot.exists() and (OUT/'egs5'/name/'summary.json').exists():
        # Never replay or change an already completed scientific run.
        name='restart_capture_d2';work=CACHE/'egs5'/name
    progress('D2','checkpoint_verification_running',current_run=name,production=False)
    from .report_p2 import report
    report()
    watcher=threading.Thread(target=watch,daemon=True);watcher.start()
    try:r=run(name,10.,10000,SEED+506,0)
    finally:finished.set();watcher.join()
    if not snapshot.exists():raise RuntimeError('no intermediate selfcheck checkpoint captured; scientific runs untouched')
    done=struct.unpack('=i',snapshot.read_bytes()[4:8])[0]
    if not 0<done<10000:raise RuntimeError('invalid checkpoint replay test')
    if not probe.exists():
        shutil.copytree(work,probe)
        shutil.copy2(snapshot,probe/'checkpoint.bin')
        for name0 in ('egs5job.out','wall_budget_d2.json','batch_progress.txt'):
            p=probe/name0
            if p.exists():p.unlink()
        (probe/'budget.dat').write_text('86400\n')
    state=run_process(probe/'restart_test_time.json',['./egs5job.exe'],probe,probe/'restart.log')
    if state['returncode']:raise RuntimeError('reference checkpoint replay failed')
    c,f,s=parse(probe,10000,10.,0,SEED+506)
    with np.load(OUT/'egs5'/name/'result.npz') as full:
        equal=dict(coarse=bool(np.array_equal(c,full['coarse'])),fine=bool(np.array_equal(f,full['fine'])))
    for key in ('ledger','callbacks','effective_flags','cutoffs'):
        equal[key]=r[key]==s[key]
    result=dict(status='合格' if all(equal.values()) and s['status']=='合格' else '不合格',
        checkpoint_completed_histories=done,total_histories=10000,seed=SEED+506,E=10.,exact_equal=equal,
        original_case=name,replayed_work=portable(probe),checkpoint_sha256=sha(snapshot),
        source_sha256=sha(work/'p2_egs.f'),scope='selfcheck only, never a pilot/main replay; RNG and all score/ledger/callback arrays exactly compared')
    write(target,result);verify();return result

if __name__=='__main__':
    print(json.dumps(check(),ensure_ascii=False))
