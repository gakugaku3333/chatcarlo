"""External per-run wall budget; a surviving supervisor can be reattached."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def atomic(path,value):
    target=Path(path);temporary=target.with_suffix('.pending.json')
    temporary.write_text(json.dumps(value));temporary.replace(target)


def alive(pid):
    try:
        reaped,_=os.waitpid(pid,os.WNOHANG)
        if reaped:return False
    except ChildProcessError:pass
    try:os.kill(pid,0)
    except ProcessLookupError:return False
    except PermissionError:return True
    return True


def supervised(state,argv,cwd,log,limit):
    value=json.loads(Path(state).read_text())
    start=time.time();prior=value.get('total_seconds',0.)
    value.update(active=dict(pid=os.getpid(),parent_pid=os.getppid(),started=start),attempts=value.get('attempts',0)+1)
    atomic(state,value)
    with open(log,'a') as output:
        child=subprocess.Popen(argv,cwd=cwd,stdout=output,stderr=subprocess.STDOUT)
        value['active']['child_pid']=child.pid;atomic(state,value)
        remaining=None if limit is None else max(0.,limit-prior)
        timed_out=False
        try:
            try:code=child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                code=child.returncode;timed_out=True
        finally:
            value.update(total_seconds=prior+time.time()-start,active=None,returncode=child.returncode,timed_out=timed_out)
            atomic(state,value)
    return code


def run_process(timer,argv,cwd,log,limit=None):
    """No second process while the prior attempt or its child is still active."""
    timer=Path(timer)
    value=json.loads(timer.read_text()) if timer.exists() else dict(total_seconds=0.,attempts=0)
    if value.get('active'):
        active=value['active']
        while True:
            value=json.loads(timer.read_text())
            if not value.get('active'):return value
            if not (alive(active['pid']) or (active.get('child_pid') and alive(active['child_pid']))):break
            time.sleep(.1)
        # Both died before reporting elapsed: conservative charge, never reset budget.
        value['total_seconds']+=max(0.,time.time()-active['started'])
        value['active']=None;value['interrupted_charge']='conservative elapsed until restart; includes uncertain idle time'
        atomic(timer,value)
    if limit is not None and value['total_seconds']>=limit:
        value.update(timed_out=True,returncode=None);atomic(timer,value);return value
    atomic(timer,value)
    command=[sys.executable,str(Path(__file__).resolve()),str(timer),json.dumps(argv),str(cwd),str(log),str(limit) if limit is not None else 'none']
    p=subprocess.Popen(command,start_new_session=True)
    p.wait()
    value=json.loads(timer.read_text())
    if value.get('active'):raise RuntimeError('external supervisor failed; run will reattach on resume')
    return value

if __name__=='__main__':
    supervised(sys.argv[1],json.loads(sys.argv[2]),sys.argv[3],sys.argv[4],None if sys.argv[5]=='none' else float(sys.argv[5]))
