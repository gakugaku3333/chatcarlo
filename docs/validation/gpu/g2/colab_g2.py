"""Host-side launcher for the bundled G2 runners on a Colab T4 (Claude, 2026-10-22).

`colab_driver.launch` only starts the G1/G4 `worker`; G2 adds the `g2-gate`
action but no launcher. This script reuses the same hash-verified bundle
transfer (`split_bundle`, the `_bootstrap` assembly and `bundle.verify` on the
VM) and only swaps the started action for `g2-gate`. It does not touch the G4
approval layer.

Usage (repository root):
  PYTHONPATH=. .venv/bin/python docs/validation/gpu/g2/colab_g2.py BUNDLE_DIR JOB_JSON OUT_DIR [--session NAME]
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time

from chatcarlo.gpu import colab_driver as cd
from chatcarlo.gpu.bundle import verify


def cli(args, input=None, colab='colab', timeout=300, tries=3):
    # A single Colab CLI call can hang indefinitely (seen 2026-10-22): bound it and retry.
    for attempt in range(tries):
        try:
            result = subprocess.run([colab, *args], input=input, text=True, capture_output=True,
                                    check=True, timeout=timeout)
            break
        except subprocess.TimeoutExpired:
            print('colab call timed out:', args[:2], 'attempt', attempt+1, flush=True)
            if attempt == tries-1:
                raise
    if 'not found' in result.stdout.lower() or 'not found' in result.stderr.lower():
        raise RuntimeError('Colab session lost')
    return result.stdout


def bootstrap(files, job_sha):
    text = cd._bootstrap(files, job_sha)
    old = "'worker','--bundle'"
    if old not in text:
        raise RuntimeError('unexpected bootstrap layout')
    # Same assembly and verification; only the started action differs.
    text = text.replace(old, "'g2-gate','--bundle'")
    return text.replace("print('G1_LAUNCHED')", "print('G2_LAUNCHED')")


def listing():
    return f'''import pathlib,os
base=pathlib.Path({cd.REMOTE!r})
import time
for p in sorted((base/'results').glob('*')):
 if p.is_file():st=p.stat();print('G2_FILE %d %d %s'%(st.st_size,int(time.time()-st.st_mtime),p.name))
try:os.kill(int((base/'worker.pid').read_text()),0);print('G2_RUNNING')
except (OSError,ValueError):print('G2_STOPPED')
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bundle', type=Path)
    ap.add_argument('job', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--session', default='chatcarlo-g2')
    ap.add_argument('--poll', type=int, default=600)  # 10 min: standing Colab rule (user decision 2026-10-21)
    a = ap.parse_args()
    manifest = verify(a.bundle)
    job = json.loads(a.job.read_text())
    if manifest['files'][job['runner']] != job['runner_sha256']:
        raise ValueError('job runner hash does not match the bundle')
    a.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='chatcarlo-g2-transfer-') as tmp:
        tmp = Path(tmp)
        files = cd.split_bundle(a.bundle, tmp/'parts', part_bytes=cd.PART_BYTES)
        job_path = tmp/'job.json'
        job_path.write_text(json.dumps(job, sort_keys=True)+'\n')
        try:
            cli(['new', '-s', a.session, '--gpu', 'T4'])
            cli(['exec', '-s', a.session], f"import pathlib\nbase=pathlib.Path({cd.REMOTE!r})\n(base/'parts').mkdir(parents=True,exist_ok=True)\n(base/'results').mkdir(exist_ok=True)\n")
            for part in sorted((tmp/'parts').iterdir()):
                cli(['upload', '-s', a.session, str(part), f'{cd.REMOTE}/parts/{part.name}'])
            cli(['upload', '-s', a.session, str(job_path), f'{cd.REMOTE}/job.json'])
            launched = cli(['exec', '-s', a.session], bootstrap(files, cd.digest(job_path)))
            if 'G2_LAUNCHED' not in launched:
                raise RuntimeError('bundle assembly/verification did not launch: '+launched[-500:])
            print('launched', flush=True)
            fetched = {}
            while True:
                time.sleep(a.poll)
                out = cli(['exec', '-s', a.session], listing())
                # Collect finished results at every poll (standing Colab rule): files untouched
                # for 30 s, re-fetched when their size changes; a lost VM loses one interval.
                files = []
                for line in out.splitlines():
                    if line.startswith('G2_FILE '):
                        size, age, name = line.split(' ', 3)[1:]
                        if Path(name).name != name:
                            raise ValueError('invalid remote name')
                        files.append((name, int(size), int(age)))
                for name, size, age in files:
                    if age >= 30 and fetched.get(name) != size:
                        cli(['download', '-s', a.session, f'{cd.REMOTE}/results/{name}', str(a.out/name)])
                        fetched[name] = size
                        print('collected', name, size, flush=True)
                # Heartbeat for the external watchdog (a stale file means this launcher is stuck).
                (a.out/'heartbeat.txt').write_text(time.strftime('%Y-%m-%d %H:%M:%S')+'\n')
                if 'G2_RUNNING' in out:
                    print('running', time.strftime('%H:%M:%S'), flush=True)
                    continue
                # Final collection: everything (any file may have changed since), plus the log.
                names = [name for name, _, _ in files]
                for name in names + ['../worker.log']:
                    cli(['download', '-s', a.session, f'{cd.REMOTE}/results/{name}', str(a.out/Path(name).name)])
                print('stopped; downloaded', names, flush=True)
                return
        finally:
            try:
                cli(['stop', '-s', a.session])
            except (subprocess.CalledProcessError, RuntimeError):
                pass


if __name__ == '__main__':
    main()
