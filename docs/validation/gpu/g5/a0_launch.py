"""Host launcher. Claude runs this with nohup; every Colab call is bounded."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

from a0_speed import HERE, load_configs, sha, write_json, read_results
from a0_bootstrap import verify_files

TIMEOUTS = dict(new=900, exec=300, upload=900, download=900, stop=180)
POLL_SECONDS = 600
ASSEMBLY_SECONDS = 1800


def call(args, heartbeat, *, input=None, timeout=None):
    if not args or args[0] not in TIMEOUTS:
        raise ValueError('unbounded colab operation')
    limit = timeout or TIMEOUTS[args[0]]
    if args[0] == 'exec':
        args = [*args, '--timeout', str(limit - 30)]
    # Like colab_driver.launch: exec is retried (idempotent steps), never 'new'.
    for attempt in range(3 if args[0] == 'exec' else 1):
        write_json(heartbeat, dict(pid=os.getpid(), time=time.time(), operation=args[0], attempt=attempt))
        try:
            return subprocess.run(['colab', *args], input=input, capture_output=True, text=True,
                                  check=True, timeout=limit).stdout
        except subprocess.TimeoutExpired:
            if attempt == 2 or args[0] != 'exec':
                raise


def watchdog(heartbeat, output, stop, interval=10):
    """Independent process: heartbeat age is checked even if the launcher hangs."""
    while not stop.is_set():
        if Path(heartbeat).exists():
            h = json.loads(Path(heartbeat).read_text())
            age = time.time() - h['time']
            if age > 3 * ASSEMBLY_SECONDS + 120:
                write_json(output, dict(status='stale launcher heartbeat', age_s=age, heartbeat=h))
        stop.wait(interval)


def create_diagnostics(destination, configs):
    from chatcarlo.gpu.bundle import create
    from chatcarlo.scene import load_scene
    from chatcarlo.detector import DetectorPlane
    root = HERE.parents[3]
    for c in configs['configurations']:
        if c['kind'] != 'diagnostic':
            continue
        if sha(root / c['scene']) != c['scene_file_sha256']:
            raise ValueError('scene file SHA256 mismatch')
        scene = load_scene(str(root / c['scene']))
        if not scene.ok:
            raise ValueError(scene.errors)
        plane = DetectorPlane(**c['detector']) if c['detector'] else None
        create(scene, plane, Path(destination) / c['id'], g2_tally=bool(c['grid']))


def run(args):
    from chatcarlo.gpu.colab_driver import split_bundle
    cfg, _ = load_configs(args.configs)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    heartbeat = output / 'heartbeat.json'
    write_json(heartbeat, dict(pid=os.getpid(), time=time.time(), operation='preflight'))
    # A distinct process watches the heartbeat; it is not blocked by colab calls.
    watch = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--watchdog',
                              str(heartbeat), '--output', str(output)])
    base = '/content/g5a0-' + args.session
    remote = lambda code: call(['exec', '-s', args.session], heartbeat, input=code)
    new_attempted = False
    try:
        for name, manifest in [('p7', 'p7_manifest.json'), ('p8g2', 'p8g2_manifest.json')]:
            verify_files(getattr(args, name), json.loads((getattr(args, name) / manifest).read_text()))
        with tempfile.TemporaryDirectory(prefix='g5a0-') as temporary:
            tmp = Path(temporary)
            create_diagnostics(tmp, cfg)
            roots = dict(g2b=tmp / 'g2b', g4=tmp / 'g4', p7=args.p7, p8g2=args.p8g2)
            files = {}
            for name, path in roots.items():
                parts = tmp / ('parts_' + name)
                files[name] = split_bundle(path, parts)
            controls = {n: HERE / n for n in ('a0_speed.py', 'a0_bootstrap.py')}
            controls['a0_configs.json'] = args.configs
            job = dict(control_files={n: sha(p) for n, p in controls.items()}, bundles=files)
            write_json(tmp / 'job.json', job)
            write_json(output / 'job.json', job)
            new_attempted = True
            print(call(['new', '-s', args.session, '--gpu', 'T4'], heartbeat), flush=True)
            remote(f'from pathlib import Path\nPath({base!r}).mkdir(parents=True, exist_ok=True)\n')
            for name, path in dict(controls, **{'job.json': tmp / 'job.json'}).items():
                call(['upload', '-s', args.session, str(path), base + '/' + name], heartbeat)
            for name in roots:
                remote(f'from pathlib import Path\nPath({(base + "/parts/" + name)!r}).mkdir(parents=True, exist_ok=True)\n')
                for part in sorted((tmp / ('parts_' + name)).iterdir()):
                    call(['upload', '-s', args.session, str(part), base + '/parts/' + name + '/' + part.name], heartbeat)
            # Standalone SHA verification and assembly precedes all uploaded imports.
            code = f'''import pathlib,hashlib,json,subprocess,sys
base=pathlib.Path({base!r})
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
assert sha(base/'job.json')=={sha(tmp / 'job.json')!r},'job SHA256'
job=json.loads((base/'job.json').read_text())
for name,digest in job['control_files'].items():
 assert pathlib.Path(name).name==name and sha(base/name)==digest,'control SHA256'
for bundle,files in job['bundles'].items():
 target=base/'bundles'/bundle;target.mkdir(parents=True,exist_ok=True)
 for name,item in files.items():
  assert pathlib.Path(name).name==name
  pending=target/(name+'.pending')
  with pending.open('wb') as out:
   for part in item['parts']:
    assert pathlib.Path(part['name']).name==part['name']
    src=base/'parts'/bundle/part['name'];assert sha(src)==part['sha256'],'part SHA256'
    with src.open('rb') as f:
     for block in iter(lambda:f.read(1048576),b''):out.write(block)
  assert sha(pending)==item['sha256'],'assembled SHA256'
  pending.replace(target/name)
subprocess.run([sys.executable,str(base/'a0_bootstrap.py'),'--base',str(base),'--job-sha',{sha(tmp / 'job.json')!r}],check=True,timeout=240)
'''
            print(call(['exec', '-s', args.session], heartbeat, input=code, timeout=ASSEMBLY_SECONDS), flush=True)
        while True:
            deadline = time.monotonic() + POLL_SECONDS
            status_code = f'''import pathlib,json,os
base=pathlib.Path({base!r});r=base/'results';pid=int((base/'worker.pid').read_text())
try:
 os.kill(pid,0);alive=(pathlib.Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
except OSError:alive=False
print(json.dumps(dict(done=(r/'G5A0_DONE').exists(),alive=alive,files=[p.name for p in r.glob('*') if p.is_file() and not p.name.endswith('.pending')])))
'''
            for attempt in range(3):
                try:
                    status = json.loads(remote(status_code).strip().splitlines()[-1])
                    break
                except (ValueError, subprocess.SubprocessError):
                    if attempt == 2:
                        raise
            for name in [*status['files'], '../worker.log', '../bundle_fingerprints.json']:
                local = output / Path(name).name
                call(['download', '-s', args.session, base + '/results/' + name, str(local)], heartbeat)
            if status['done']:
                read_results(output / 'measurements.json', args.configs)
                return
            if not status['alive']:
                raise RuntimeError('VM worker ended without G5A0_DONE; baseline invalid')
            while time.monotonic() < deadline:
                write_json(heartbeat, dict(pid=os.getpid(), time=time.time(), operation='waiting'))
                time.sleep(min(30, max(0, deadline - time.monotonic())))
    finally:
        try:
            if new_attempted:
                print(call(['stop', '-s', args.session], heartbeat), flush=True)
        finally:
            watch.terminate()
            watch.wait(timeout=10)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', default='g5a0')
    p.add_argument('--configs', type=Path, default=HERE / 'a0_configs.json')
    p.add_argument('--p7', type=Path)
    p.add_argument('--p8g2', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--watchdog', type=Path)
    args = p.parse_args()
    if args.watchdog:
        watchdog(args.watchdog, args.output / 'watchdog.json', threading.Event())
    else:
        if not args.p7 or not args.p8g2:
            p.error('--p7 and --p8g2 sealed export directories required')
        run(args)


if __name__ == '__main__':
    main()
