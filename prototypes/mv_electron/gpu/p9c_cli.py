"""Claude-only Colab CLI driver for P9c (supervised start / resume / watch / stop). Never executed by Codex.

`run` is the supervisor (plan §10): it starts a T4 VM, uploads the export and any earlier results (files over
120 MB are split into 50 MB parts and reassembled on the VM, every file SHA256-checked), starts p9c_colab with
nohup, then polls every 10 minutes. Each poll collects the whole result folder (committed checkpoints) and writes
`supervisor.json` locally. The job is judged stalled when the VM process is gone, `colab exec` fails 3 times in a
row, or the heartbeat line has not changed for 15 minutes; then the supervisor collects, stops the VM and resumes on
a new VM from the collected checkpoints (at most --max-restarts times). Completion is the job's own P9C_DONE marker.
Every colab call has a time limit; the VM is always stopped on exit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import zipfile

REMOTE = '/content'
PART = 50*1024*1024
SPLIT_ABOVE = 120*1024*1024


def call(args, timeout=180, input=None):
    return subprocess.run(['colab', *args], input=input, text=True, capture_output=True,
                          check=True, timeout=timeout).stdout


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def check_export(export):
    manifest = json.loads((Path(export)/'manifest.json').read_text())
    for name, digest in manifest.items():
        if Path(name).name != name or sha(Path(export)/name) != digest:
            raise ValueError('export hash '+name)
    return manifest


def upload(session, local, remote):
    """Upload one file; large files go up in parts and are joined and hash-checked on the VM."""
    local = Path(local)
    size = local.stat().st_size
    digest = sha(local)
    if size <= SPLIT_ABOVE:
        for attempt in range(3):
            try:
                call(['upload', '-s', session, str(local), remote], timeout=900)
                break
            except subprocess.SubprocessError:
                if attempt == 2:
                    raise
    else:
        with tempfile.TemporaryDirectory(prefix='p9c-part-') as tmp, open(local, 'rb') as f:
            j = 0
            while True:
                block = f.read(PART)
                if not block:
                    break
                part = Path(tmp)/f'part{j:03}'
                part.write_bytes(block)
                for attempt in range(3):
                    try:
                        call(['upload', '-s', session, str(part), f'{remote}.part{j:03}'], timeout=900)
                        break
                    except subprocess.SubprocessError:
                        if attempt == 2:
                            raise
                part.unlink()
                j += 1
        join = f'''
import pathlib
dest = pathlib.Path({remote!r}); parts = sorted(dest.parent.glob(dest.name+'.part*'))
assert len(parts) == {j}, len(parts)
with open(dest, 'wb') as out:
    for p in parts:
        out.write(p.read_bytes()); p.unlink()
print('JOINED')
'''
        if 'JOINED' not in call(['exec', '-s', session], input=join, timeout=600):
            raise RuntimeError('join failed '+remote)
    check = f'import hashlib;h=hashlib.sha256();f=open({remote!r},"rb")\nfor b in iter(lambda:f.read(1<<20),b""):h.update(b)\nprint("SHA",h.hexdigest())'
    if 'SHA '+digest not in call(['exec', '-s', session], input=check, timeout=600):
        raise RuntimeError('VM hash mismatch '+remote)


STATUS = '''
import json, os, pathlib, zipfile
root = pathlib.Path(%r)
logp = root/'p9c.log'
log = logp.read_text() if logp.exists() else ''
pid = int((root/'p9c.pid').read_text()) if (root/'p9c.pid').exists() else None
alive = False
if pid is not None:
    try:
        os.kill(pid, 0); alive = True
    except OSError:
        alive = False
state = 'done' if 'P9C_DONE' in log else ('running' if alive else 'failed')
beats = [l for l in log.splitlines() if l.startswith('P9C_HEARTBEAT')]
batches = [l for l in log.splitlines() if l.startswith('P9C_BATCH')]
with zipfile.ZipFile(root/'p9c_collect.zip', 'w', zipfile.ZIP_STORED) as z:
    out = root/'p9c_results'
    if out.exists():
        for p in sorted(out.rglob('*')):
            if p.is_file() and not p.name.endswith('.pending'): z.write(p, str(p.relative_to(out)))
    if logp.exists(): z.write(logp, 'p9c.log')
print(json.dumps(dict(state=state, alive=alive, heartbeat=beats[-1] if beats else '', batch=batches[-1] if batches else '', tail=log[-1200:])))
'''


def poll_collect(session, output):
    text = call(['exec', '-s', session], input=STATUS % REMOTE, timeout=300)
    status = json.loads(text.strip().splitlines()[-1])
    with tempfile.TemporaryDirectory(prefix='p9c-collect-') as tmp:
        target = Path(tmp)/'collect.zip'
        call(['download', '-s', session, REMOTE+'/p9c_collect.zip', str(target)], timeout=1800)
        staging = Path(tmp)/'staging'
        with zipfile.ZipFile(target) as z:
            z.extractall(staging)
        output = Path(output)
        for p in sorted(staging.rglob('*')):
            if p.is_file():
                dest = output/p.relative_to(staging)
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmpdest = dest.with_name(dest.name+'.incoming')
                shutil.copy2(p, tmpdest)
                tmpdest.replace(dest)
    return status


def start(session, export, output, job, t0=None):
    manifest = check_export(export)
    print(call(['new', '-s', session, '--gpu', 'T4'], timeout=600), flush=True)
    call(['exec', '-s', session], input=f'import pathlib\nfor n in ("p9c_input","p9c_results"):pathlib.Path({REMOTE!r},n).mkdir(parents=True,exist_ok=True)\nprint("OK")\n')
    for name in (*manifest, 'manifest.json'):
        upload(session, Path(export)/name, f'{REMOTE}/p9c_input/{name}')
    if t0 is not None:
        call(['exec', '-s', session], input=f'import pathlib;pathlib.Path({REMOTE!r},"p9c_t0").mkdir(exist_ok=True)')
        for name in ('job.json', 'gates.json', 'rectangle_summary.json'):
            upload(session, Path(t0)/name, f'{REMOTE}/p9c_t0/{name}')
    # Resume: send everything already collected (committed checkpoints, gates, measurements).
    output = Path(output)
    if output.exists():
        for p in sorted(output.rglob('*')):
            if p.is_file() and p.name not in ('p9c.log', 'supervisor.json') and not p.name.endswith(('.incoming', '.pending')):
                rel = p.relative_to(output)
                call(['exec', '-s', session], input=f'import pathlib;pathlib.Path({REMOTE!r},"p9c_results",{str(rel.parent)!r}).mkdir(parents=True,exist_ok=True)')
                upload(session, p, f'{REMOTE}/p9c_results/{rel}')
    script = f'''
import hashlib, json, pathlib, subprocess, zipfile
root = pathlib.Path({REMOTE!r})
m = json.loads((root/'p9c_input/manifest.json').read_text())
for name, digest in m.items():
    if hashlib.sha256((root/'p9c_input'/name).read_bytes()).hexdigest() != digest: raise SystemExit('VM export hash '+name)
zipfile.ZipFile(root/'p9c_input/source.zip').extractall(root/'p9c_source')
cmd = f'cd {{root}}/p9c_source && PYTHONPATH={{root}}/p9c_source nohup python -u -m prototypes.mv_electron.gpu.p9c_colab {{root}}/p9c_input {{root}}/p9c_results {job} >> {{root}}/p9c.log 2>&1 & echo $! > {{root}}/p9c.pid'
subprocess.run(['bash', '-c', cmd], check=True)
print('started', (root/'p9c.pid').read_text().strip())
'''
    print(call(['exec', '-s', session], input=script), flush=True)


def stop(session):
    try:
        call(['stop', '-s', session], timeout=180)
    except subprocess.SubprocessError as error:
        print('stop failed', error, flush=True)


def write_status(output, record):
    Path(output).mkdir(parents=True, exist_ok=True)
    p = Path(output)/'supervisor.json'
    tmp = p.with_name('supervisor.json.tmp')
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(p)


def run(args):
    record = dict(job=args.job, restarts=0, attempts=[], state='starting')
    for attempt in range(args.max_restarts+1):
        session = f'{args.session}{attempt}'
        record.update(state='starting', session=session, restarts=attempt, updated=time.strftime('%F %T'))
        write_status(args.output, record)
        reason = None
        try:
            start(session, args.export, args.output, args.job, args.t0)
            last_beat, last_change, failures = None, time.time(), 0
            time.sleep(args.first_wait)
            while True:
                try:
                    status = poll_collect(session, args.output)
                    failures = 0
                except (subprocess.SubprocessError, ValueError, IndexError, zipfile.BadZipFile) as error:
                    failures += 1
                    print(time.strftime('%T'), 'poll failure', failures, type(error).__name__, flush=True)
                    if failures >= 3:
                        reason = 'colab exec/download failed 3 times'
                        break
                    time.sleep(60)
                    continue
                if status['heartbeat'] != last_beat:
                    last_beat, last_change = status['heartbeat'], time.time()
                record.update(state=status['state'], heartbeat=status['heartbeat'], batch=status['batch'],
                              stalled_minutes=round((time.time()-last_change)/60, 1), updated=time.strftime('%F %T'))
                write_status(args.output, record)
                print(time.strftime('%T'), status['state'], status['batch'], status['heartbeat'], flush=True)
                if status['state'] == 'done':
                    record.update(state='done')
                    write_status(args.output, record)
                    print('SUPERVISOR_DONE', flush=True)
                    return 0
                if status['state'] == 'failed':
                    reason = 'job process gone: '+status['tail'][-300:]
                    break
                if time.time()-last_change > args.stall_minutes*60:
                    reason = f'heartbeat unchanged for {args.stall_minutes} min'
                    break
                time.sleep(args.interval)
        except (subprocess.SubprocessError, RuntimeError, ValueError, OSError) as error:
            reason = f'start/resume error: {type(error).__name__}: {error}'[:500]
        finally:
            stop(session)
        record['attempts'].append(dict(session=session, reason=reason, at=time.strftime('%F %T')))
        record.update(state='restarting', updated=time.strftime('%F %T'))
        write_status(args.output, record)
        print('RESTART', attempt+1, reason, flush=True)
    record.update(state='gave-up')
    write_status(args.output, record)
    print('SUPERVISOR_GAVE_UP', flush=True)
    return 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('run',))
    p.add_argument('--session', default='p9c')
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--job', required=True, help="arguments passed to p9c_colab, e.g. --job='--mode production ...'")
    p.add_argument('--t0', type=Path, help='local T0 folder; job.json, gates.json and rectangle_summary.json go to /content/p9c_t0')
    p.add_argument('--first-wait', type=int, default=300)
    p.add_argument('--interval', type=int, default=600)
    p.add_argument('--stall-minutes', type=int, default=15)
    p.add_argument('--max-restarts', type=int, default=6)
    args = p.parse_args()
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
