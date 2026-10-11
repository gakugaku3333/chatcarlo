"""追補3: second worker for the frozen P2 production schedule (execution only).

Takes locked prototype runs from the END of pilot_lock.json and executes them
through the frozen run_budgeted, while the one-command driver walks from the
front. No seed, N, batch partition, limit or criterion is touched. progress.json
keeps a single writer (the driver): this file never calls progress/checkpoint.
"""
import json
import sys
from .p2_common import OUT,CACHE
from .workflow_p2 import native
from .run_coupled_p2 import run_budgeted


def driver_index(names):
    """Index of the run the driver is on (or will start next)."""
    t6=json.loads((OUT/'progress.json').read_text())['stages']['T6']
    if t6.get('state')!='production_running':return None
    if t6.get('current_run') in names:return names.index(t6['current_run'])
    if t6.get('last_completed') in names:return names.index(t6['last_completed'])+1
    return None


def untouched(name):
    dest=OUT/'runs'/name
    return not ((CACHE/'prototype'/name/'wall_budget.json').exists() or dest.exists())


def next_run(lock):
    names=list(lock['runs']);current=driver_index(names)
    if current is None:return None,'driver is not in prototype production'
    for i in range(len(names)-1,-1,-1):
        # Never start a run the driver could start at the same moment.
        if i<=current+1:return None,f'met the driver at {names[current]}'
        if untouched(names[i]):return names[i],None
    return None,'no run left'


def main(execute):
    lock=json.loads((OUT/'pilot_lock.json').read_text())
    if not execute:
        names=list(lock['runs']);current=driver_index(names)
        print(json.dumps(dict(driver_run=None if current is None else names[current],
            would_take=[] if current is None else [n for i,n in reversed(list(enumerate(names))) if i>current+1 and untouched(n)]),ensure_ascii=False))
        return
    while True:
        name,reason=next_run(lock)
        if name is None:
            print('helper stops: '+reason,flush=True);return
        print('helper starts '+name,flush=True)
        result=run_budgeted(name,**native(lock['runs'][name]))
        print(f'helper finished {name}: {"result saved" if result else "terminal (per-run limit)"}',flush=True)


if __name__=='__main__':
    main('--run' in sys.argv)
