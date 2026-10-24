"""Claude-only Colab CLI driver for P9b (start, ten-minute watch, collect, stop). Never executed by Codex.

start: new T4 VM, upload the export (hash-checked, flat manifest name -> sha256), unpack source.zip, start
p9b_colab with nohup. watch: every 10 minutes read the job state, download the whole result folder, merge it
without overwriting changed files, and always collect and `colab stop` at the end (also after a failure).
Completion is decided from the job's own log marker P9B_DONE, never from process names.
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


def call(args, timeout=180, input=None):
    return subprocess.run(['colab', *args], input=input, text=True, capture_output=True,
                          check=True, timeout=timeout).stdout


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_export(export):
    manifest = json.loads((Path(export)/'manifest.json').read_text())
    for name, digest in manifest.items():
        if Path(name).name != name or sha(Path(export)/name) != digest:
            raise ValueError('export hash '+name)
    return manifest


def merge(staging, output):
    """Copy new files; an existing file must be byte-identical except the growing log."""
    output = Path(output)
    for p in sorted(Path(staging).rglob('*')):
        if not p.is_file():
            continue
        dest = output/p.relative_to(staging)
        if dest.exists() and dest.name != 'p9b.log':
            if dest.read_bytes() != p.read_bytes():
                raise ValueError('collected file differs from local copy: '+str(dest))
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)


STATUS = '''
import json, os, pathlib, zipfile
root = pathlib.Path(%r)
logp = root/'p9b.log'
log = logp.read_text() if logp.exists() else ''
pid = int((root/'p9b.pid').read_text()) if (root/'p9b.pid').exists() else None
alive = False
if pid is not None:
    try:
        os.kill(pid, 0); alive = True
    except OSError:
        alive = False
state = 'done' if 'P9B_DONE' in log else ('running' if alive else 'failed')
with zipfile.ZipFile(root/'p9b_collect.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    out = root/'p9b_results'
    if out.exists():
        for p in sorted(out.rglob('*')):
            if p.is_file(): z.write(p, str(p.relative_to(out)))
    if logp.exists(): z.write(logp, 'p9b.log')
print(json.dumps(dict(state=state, alive=alive, tail=log[-1500:])))
'''


def poll_collect(session, output):
    last = None
    for _ in range(3):
        try:
            text = call(['exec', '-s', session], input=STATUS % REMOTE)
            status = json.loads(text.strip().splitlines()[-1])
            break
        except (subprocess.SubprocessError, ValueError, IndexError) as error:
            last = error
    else:
        raise RuntimeError('status retrieval failed') from last
    with tempfile.TemporaryDirectory(prefix='p9b-collect-') as tmp:
        target = Path(tmp)/'collect.zip'
        call(['download', '-s', session, REMOTE+'/p9b_collect.zip', str(target)], timeout=600)
        staging = Path(tmp)/'staging'
        with zipfile.ZipFile(target) as z:
            z.extractall(staging)
        merge(staging, output)
    print(time.strftime('%H:%M:%S'), status['state'], status['tail'][-400:], flush=True)
    return status


def start(args):
    manifest = check_export(args.export)
    created = False
    try:
        print(call(['new', '-s', args.session, '--gpu', 'T4'], timeout=600), flush=True)
        created = True
        call(['exec', '-s', args.session], input=f'import pathlib\npathlib.Path({REMOTE!r},"p9b_input").mkdir(parents=True,exist_ok=True)\n')
        for name in (*manifest, 'manifest.json'):
            call(['upload', '-s', args.session, str(args.export/name), f'{REMOTE}/p9b_input/{name}'], timeout=900)
        script = f'''
import hashlib, json, pathlib, subprocess, zipfile
root = pathlib.Path({REMOTE!r})
m = json.loads((root/'p9b_input/manifest.json').read_text())
for name, digest in m.items():
    if hashlib.sha256((root/'p9b_input'/name).read_bytes()).hexdigest() != digest: raise SystemExit('VM export hash '+name)
zipfile.ZipFile(root/'p9b_input/source.zip').extractall(root/'p9b_source')
cmd = f'cd {{root}}/p9b_source && nohup python -u -m prototypes.mv_electron.gpu.p9b_colab {{root}}/p9b_input {{root}}/p9b_results > {{root}}/p9b.log 2>&1 & echo $! > {{root}}/p9b.pid'
subprocess.run(['bash', '-c', cmd], check=True)
print('started', (root/'p9b.pid').read_text().strip())
'''
        print(call(['exec', '-s', args.session], input=script), flush=True)
    except BaseException:
        if created:
            try:
                poll_collect(args.session, args.output)
            finally:
                call(['stop', '-s', args.session])
        raise


def watch(args):
    try:
        time.sleep(args.first_wait)
        while True:
            status = poll_collect(args.session, args.output)
            if status['state'] == 'failed':
                raise RuntimeError('VM job failed or PID disappeared')
            if status['state'] == 'done':
                print('WATCH_DONE', flush=True)
                return
            time.sleep(600)
    finally:
        try:
            poll_collect(args.session, args.output)
        finally:
            call(['stop', '-s', args.session])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('start', 'watch'))
    p.add_argument('--session', default='p9b')
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--first-wait', type=int, default=180)
    args = p.parse_args()
    start(args) if args.command == 'start' else watch(args)


if __name__ == '__main__':
    main()
