"""Claude-only Colab CLI driver for P9a (start, ten-minute watch, collect, stop). Never executed by Codex.

start: new T4 VM, upload the export (hash-checked) and any earlier output/trial, start p9a_colab with nohup.
watch: every 10 minutes read the job state, download the whole output folder, merge it without overwriting
sealed files, and always collect and `colab stop` at the end (also after a failure or interrupt).
Completion is decided from the job's own log marker P9A_DONE, never from process names.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import zipfile

REMOTE = '/content/p9a'


def call(args, timeout=180, input=None):
    return subprocess.run(['colab', *args], input=input, text=True, capture_output=True,
                          check=True, timeout=timeout).stdout


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_export(export):
    manifest = json.loads((Path(export)/'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if Path(name).name != name or sha(Path(export)/name) != digest:
            raise ValueError('export hash '+name)
    return manifest


def zip_folder(folder, target):
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(Path(folder).rglob('*')):
            if p.is_file() and p.name != 'run.log.tail':
                z.write(p, str(p.relative_to(folder)))


def merge(staging, output):
    """Copy new files; an existing file must be byte-identical (sealed artefacts never change)."""
    output = Path(output)
    for p in sorted(Path(staging).rglob('*')):
        if not p.is_file() or p.name == 'run.log.tail':
            continue
        dest = output/p.relative_to(staging)
        if dest.exists():
            if dest.read_bytes() != p.read_bytes() and dest.name not in ('run.log', 'trial.json'):
                raise ValueError('collected file differs from local copy: '+str(dest))
            if dest.name not in ('run.log', 'trial.json'):
                continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)


STATUS = '''
import json, os, pathlib, zipfile
root = pathlib.Path(%r)
log = (root/'run.log').read_text() if (root/'run.log').exists() else ''
pid = int((root/'pid').read_text()) if (root/'pid').exists() else None
alive = False
if pid is not None:
    try:
        os.kill(pid, 0); alive = True
    except OSError:
        alive = False
state = 'done' if 'P9A_DONE' in log else ('running' if alive else 'failed')
with zipfile.ZipFile(root/'collect.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted((root/'output').rglob('*')):
        if p.is_file(): z.write(p, str(p.relative_to(root/'output')))
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
    with tempfile.TemporaryDirectory(prefix='p9a-collect-') as tmp:
        target = Path(tmp)/'collect.zip'
        call(['download', '-s', session, REMOTE+'/collect.zip', str(target)], timeout=600)
        staging = Path(tmp)/'staging'
        with zipfile.ZipFile(target) as z:
            z.extractall(staging)
        merge(staging, output)
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output)/'run.log.tail').write_text(status['tail'])
    print(time.strftime('%H:%M:%S'), status['state'], status['tail'][-400:], flush=True)
    return status


def start(args):
    manifest = check_export(args.export)
    created = False
    try:
        print(call(['new', '-s', args.session, '--gpu', 'T4'], timeout=600), flush=True)
        created = True
        call(['exec', '-s', args.session], input=f'import pathlib\nfor n in ("export","output","trial"):pathlib.Path({REMOTE!r},n).mkdir(parents=True,exist_ok=True)\n')
        for name in (*manifest['files'], 'manifest.json'):
            call(['upload', '-s', args.session, str(args.export/name), f'{REMOTE}/export/{name}'], timeout=900)
        with tempfile.TemporaryDirectory(prefix='p9a-upload-') as tmp:
            for label, folder in (('output', args.output), ('trial', args.trial)):
                if folder is not None and Path(folder).exists() and any(Path(folder).rglob('*.json')):
                    z = Path(tmp)/(label+'.zip')
                    zip_folder(folder, z)
                    call(['upload', '-s', args.session, str(z), f'{REMOTE}/{label}.zip'], timeout=900)
        trial = f' --trial {REMOTE}/trial' if args.trial is not None else ''
        script = f'''
import hashlib, json, pathlib, subprocess, zipfile
root = pathlib.Path({REMOTE!r})
m = json.loads((root/'export/manifest.json').read_text())
for name, digest in m['files'].items():
    if hashlib.sha256((root/'export'/name).read_bytes()).hexdigest() != digest: raise SystemExit('VM export hash '+name)
for label in ('output', 'trial'):
    z = root/(label+'.zip')
    if z.exists(): zipfile.ZipFile(z).extractall(root/label)
zipfile.ZipFile(root/'export/source.zip').extractall(root/'code')
cmd = f'cd {{root}}/code && nohup python -m prototypes.mv_electron.gpu.p9a_colab {{root}}/export {{root}}/output{trial} > {{root}}/run.log 2>&1 & echo $! > {{root}}/pid'
subprocess.run(['bash', '-c', cmd], check=True)
print('started', (root/'pid').read_text().strip())
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
    p.add_argument('--session', default='p9a')
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--trial', type=Path)
    p.add_argument('--first-wait', type=int, default=120)
    args = p.parse_args()
    start(args) if args.command == 'start' else watch(args)


if __name__ == '__main__':
    main()
