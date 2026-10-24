"""P8g-3 schedule, immutable sessions and sealed call pairs (NumPy only).

The transport identity remains P8g-2's identity. Drivers are not transport
fingerprints. A receipt binds the *actual* invocation to its expected interval.
"""
import json
from pathlib import Path
import re
import uuid
import numpy as np
from . import p8g2_core as kernel
from .p8f_colab import run_health
from .p8f_core import check_resume

SEED = 820301
COUNT = 400
N = 200000
LAUNCH = [40, 256]


def spec(count=COUNT, n=N, mode='gpu'):
    if mode not in ('gpu', 'cpu') or count < 1 or not 0 < n <= N:
        raise ValueError('schedule')
    return dict(count=count, n=n, seed=SEED, mode=mode)


def invocation(j, schedule):
    if type(j) is not int or not 0 <= j < schedule['count']:
        raise ValueError('call ID')
    return dict(n=schedule['n'], seed=schedule['seed'], start=j*schedule['n'])


def write_json(path, value, immutable=False):
    path = Path(path)
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_text() != data:
            raise ValueError('immutable record conflict '+str(path))
        return
    temp = path.with_name(path.name+'.pending')
    temp.write_text(data)
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def session_path(root, sid):
    if not re.fullmatch(r'[0-9a-f]{32}', sid):
        raise ValueError('session ID')
    return Path(root)/('session_'+sid+'.json')


def new_session(root, identity, schedule, gpu, gate, resume):
    sid = uuid.uuid4().hex
    record = dict(status='pass', session_id=sid, identity=identity,
                  schedule=schedule, launch=LAUNCH, gpu=gpu, gate=gate,
                  resume=resume)
    validate_session(record, identity, schedule)
    write_json(session_path(root, sid), record, immutable=True)
    return sid


def validate_session(record, identity, schedule):
    if (record['status'] != 'pass' or record['identity'] != identity or
        record['schedule'] != schedule or record['launch'] != LAUNCH or
        record['gate']['status'] != 'pass'):
        raise ValueError('session gate/conditions')
    if schedule['mode'] == 'gpu':
        if ('T4' not in record['gpu'] or record['gate'].get('n') != 20000 or
            record['gate'].get('seed') != kernel.SEED2 or
            record['gate'].get('comparison') != 'P8g1/P8g2 byte-identical'):
            raise ValueError('session GPU gate')
        if schedule == spec():
            trial = record['gate'].get('restart_trial', {})
            if (trial.get('status') != 'pass' or trial.get('identity') != identity or
                trial.get('schedule') != spec(4,2000) or
                trial.get('resume',{}).get('status') != 'pass' or
                not trial.get('previous_session') or
                trial.get('previous_session') == trial.get('new_session') or
                trial.get('complete',{}).get('status') != 'pass' or
                not trial.get('faults')):
                raise ValueError('session mandatory T4 restart trial')
    resume = record['resume']
    if resume['status'] not in ('pass', 'new', 'cpu-schedule-only'):
        raise ValueError('session resume failed')
    if schedule['mode'] == 'gpu' and resume['status'] == 'cpu-schedule-only':
        raise ValueError('CPU resume is not a GPU gate')


def require_session(root, sid, identity, schedule, digest=None):
    path = session_path(root, sid)
    record = read_json(path)
    validate_session(record, identity, schedule)
    if record['session_id'] != sid or (digest and kernel.sha(path) != digest):
        raise ValueError('session seal')
    return record


def call_path(root, j):
    return Path(root)/f'call_{j:03}.npz'


def check_result(result, schedule):
    if not run_health(result):
        raise AssertionError('run_health failed; call not saved')
    shapes = dict(dose=(220,), p8_batches=(100,220), coarse=(100,108),
                  fine=(100,430), additions=(100,220))
    if schedule['mode'] == 'gpu':
        shapes.update(_groups_d=(100,795), _groups_i=(100,299), _absolute=(100,795))
    for key, shape in shapes.items():
        if result[key].shape != shape:
            raise ValueError('result shape '+key)
    if schedule['mode'] == 'gpu' and np.any(result['_absolute'] < 0):
        raise ValueError('negative absolute sum')


def load_call(root, j, identity, schedule):
    path = call_path(root, j)
    receipt = read_json(path.with_suffix('.json'))
    expected = dict(call_id=j, identity=identity, schedule=schedule,
                    **invocation(j, schedule), launch=LAUNCH)
    if any(receipt.get(k) != v for k,v in expected.items()):
        raise ValueError('receipt conditions '+str(j))
    if receipt['actual'] != invocation(j, schedule):
        raise ValueError('actual invocation mismatch '+str(j))
    if receipt['visits'] != dict(all_one=True, count=schedule['n']):
        raise ValueError('visits receipt')
    require_session(root, receipt['session_id'], identity, schedule, receipt['session_sha256'])
    if kernel.sha(path) != receipt['sha256']:
        raise ValueError('call hash '+str(j))
    with np.load(path, allow_pickle=False) as f:
        result = {k:f[k].copy() for k in f.files}
    check_result(result, schedule)
    if float(result['seconds']) != receipt['seconds']:
        raise ValueError('timing receipt')
    return result, receipt


def available(root, identity, schedule):
    """Orphans are incomplete; malformed or conflicting sealed pairs are fatal."""
    root = Path(root)
    for path in (*root.glob('call_*.npz'), *root.glob('call_*.json')):
        if '.pending' in path.name:
            continue
        match = re.fullmatch(r'call_([0-9]{3})\.(npz|json)', path.name)
        if not match or int(match[1]) >= schedule['count']:
            raise ValueError('unexpected call ID '+path.name)
    return {j:load_call(root,j,identity,schedule)[1] for j in range(schedule['count'])
            if call_path(root,j).exists() and call_path(root,j).with_suffix('.json').exists()}


def save_call(root, j, result, visits, actual, identity, schedule, sid):
    if actual != invocation(j, schedule):
        raise ValueError('actual invocation mismatch')
    visits = np.asarray(visits)
    if visits.shape != (schedule['n'],):
        raise ValueError('visits shape')
    kernel.check_visits(visits)
    check_result(result, schedule)
    require_session(root,sid,identity,schedule)
    path = call_path(root,j)
    if path.exists() and path.with_suffix('.json').exists():
        load_call(root,j,identity,schedule)
        raise ValueError('duplicate call save '+str(j))
    # Remove orphan receipt before publishing replacement bytes. Listings expose
    # only pairs; an interrupted write at any earlier point remains incomplete.
    path.with_suffix('.json').unlink(missing_ok=True)
    pending = path.with_name(path.stem+'.pending.npz')
    np.savez(pending, **result)
    receipt = dict(call_id=j, identity=identity, schedule=schedule,
                   **actual, actual=actual, launch=LAUNCH, session_id=sid,
                   session_sha256=kernel.sha(session_path(root,sid)),
                   sha256=kernel.sha(pending), seconds=float(result['seconds']),
                   visits=dict(all_one=True,count=len(visits)))
    pending.replace(path)
    write_json(path.with_suffix('.json'), receipt)
    return receipt


def completion(root, identity, schedule):
    calls = available(root,identity,schedule)
    if set(calls) != set(range(schedule['count'])):
        raise ValueError('incomplete call ID set')
    return dict(status='pass', identity=identity, schedule=schedule,
                seed=schedule['seed'], n=schedule['n']*schedule['count'],
                gpus=sorted({require_session(root,r['session_id'],identity,schedule)['gpu']
                             for r in calls.values()}),
                launch=LAUNCH, calls={str(j):dict(sha256=r['sha256'],
                    receipt_sha256=kernel.sha(call_path(root,j).with_suffix('.json')),
                    session_id=r['session_id'],session_sha256=r['session_sha256'])
                    for j,r in calls.items()})


def verify(root, identity, schedule=None):
    schedule = spec() if schedule is None else schedule
    expected = completion(root,identity,schedule)
    if read_json(Path(root)/'complete.json') != expected:
        raise ValueError('complete manifest mismatch')
    return expected


def drive(root, runner, identity, schedule, sid, limit=None):
    existing = available(root,identity,schedule)
    executed = 0
    for j in range(schedule['count']):
        if j in existing:
            continue
        actual = invocation(j,schedule)
        result, visits = runner(**actual)
        save_call(root,j,result,visits,actual,identity,schedule,sid)
        executed += 1
        print('P8G3 call',j,'sealed',flush=True)
        if limit is not None and executed >= limit:
            return
    write_json(Path(root)/'complete.json',completion(root,identity,schedule),immutable=True)


def resume_check(root, runner, identity, schedule):
    calls = available(root,identity,schedule)
    if not calls:
        return dict(status='new')
    j = min(calls)
    if schedule['mode'] == 'cpu':
        return dict(status='cpu-schedule-only',call_id=j)
    old,_ = load_call(root,j,identity,schedule)
    repeated, visits = runner(**invocation(j,schedule))
    if np.shape(visits) != (schedule['n'],):
        raise ValueError('resume visits shape')
    kernel.check_visits(visits)
    check_result(repeated,schedule)
    check_resume(old,repeated,schedule['n'])
    return dict(status='pass',call_id=j)
