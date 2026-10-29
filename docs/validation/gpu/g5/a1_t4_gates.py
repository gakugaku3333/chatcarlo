"""A1b statistics/fault/EGS5/identity gates on one T4 session (Claude, 2026-10-30).

host:  prepare -> launch (upload payload, run VM part under nohup, poll, download) -> judge
VM:    --vm runs every exported job with the *new* runtime and writes results/ + G5A1_GATES_DONE.
Judging stays on the host with the existing drivers (a1_gate1 / a1_g2vox remote mode,
a1_g2gate3 compare); the VM only produces outputs.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CACHE = Path.home()/'.cache/chatcarlo/g5/a1/t4_gates'
CHUNK = 20*1024*1024


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- host side
def prepare(payload):
    """Fault-probe inputs: the same packed tables a1_faults builds on the host."""
    import numpy as np
    sys.path[:0] = [str(ROOT), str(HERE)]
    from chatcarlo import kernel
    from chatcarlo.gpu import core
    from a1_faults import reference
    t = kernel.bake_scene_materials(['air', 'water', 'lead'])
    a = core.pack(t)
    a.update(centers=np.zeros((2, 3)), halves=np.array([[10.]*3, [5.]*3]), materials=np.array([1, 2], np.int64),
             background=np.array(0), world=np.array([0., 0., 0., 60., 60., 60.]), detector=np.array([]))
    t.k_edge[2, 0] = 1.; t.k_omega[2, 0] = 1.; t.k_frac[2, 0, :] = 1.; t.n_lines[2, 0] = 1
    t.k_line_e[2, 0, :] = 4.; t.k_line_p[2, 0, :] = 0.; t.k_line_p[2, 0, 0] = 1.
    a3 = dict(a, **core.pack(t))
    out = Path(payload)/'faults'; out.mkdir(parents=True, exist_ok=True)
    np.savez(out/'tables.npz', **a); np.savez(out/'tables_fault3.npz', **a3)
    reference(out/'reference.npz')


def pack_payload(payload, tar):
    """Payload + the new runtime (taken from an exported bundle) + the A1 drivers."""
    with tarfile.open(tar, 'w') as t:
        t.add(payload, arcname='payload')
        runtime = Path(payload)/'gate1/bundles/s1'
        for p in sorted(runtime.iterdir()):
            if p.suffix in ('.py', '.h', '.cu', '.cpp'):
                t.add(p, arcname='chatcarlo/gpu/'+p.name)
        t.addfile(tarfile.TarInfo('chatcarlo/__init__.py'), io.BytesIO(b''))
        for name in ('a1_speed.py', 'a1_health.py', 'a1_faults.py', 'a1_common.py', 'a0_speed.py', 'a1_t4_gates.py'):
            t.add(HERE/name, arcname='docs/validation/gpu/g5/'+name)


def launch(args):
    from a1_launch import call, watchdog
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    heartbeat = out/'heartbeat.json'
    tar = out/'payload.tar'; pack_payload(args.payload, tar)
    data = tar.read_bytes(); parts = []
    for i in range(0, len(data), CHUNK):
        p = out/f'payload.part{i//CHUNK:03d}'; p.write_bytes(data[i:i+CHUNK]); parts.append(p)
    manifest = dict(sha256=hashlib.sha256(data).hexdigest(), parts=[(p.name, sha(p)) for p in parts])
    (out/'transfer.json').write_text(json.dumps(manifest, indent=2)+'\n')
    watch = subprocess.Popen([sys.executable, str(HERE/'a1_launch.py'), '--watchdog', str(heartbeat), '--output', str(out)],
                             start_new_session=True)
    base = '/content/g5a1gates-'+args.session; attempted = False
    remote = lambda code, **kw: call(['exec', '-s', args.session], heartbeat, input=code, **kw)
    try:
        attempted = True; call(['new', '-s', args.session, '--gpu', 'T4'], heartbeat)
        remote(f'import pathlib\npathlib.Path({base!r}).mkdir(parents=True,exist_ok=True)\n')
        for p in [*parts, out/'transfer.json']:
            call(['upload', '-s', args.session, str(p), base+'/'+p.name], heartbeat)
        remote(f'''import pathlib,hashlib,json,tarfile,subprocess,sys,os
base=pathlib.Path({base!r});m=json.loads((base/'transfer.json').read_text())
if not (base/'worker.pid').exists():
 data=b''
 for name,digest in m['parts']:
  b=(base/name).read_bytes();assert hashlib.sha256(b).hexdigest()==digest,'part SHA256';data+=b
 assert hashlib.sha256(data).hexdigest()==m['sha256'],'payload SHA256'
 (base/'payload.tar').write_bytes(data)
 with tarfile.open(base/'payload.tar') as t:
  for x in t.getmembers():assert not x.name.startswith('/') and '..' not in x.name.split('/'),'unsafe member'
  t.extractall(base)
 log=(base/'worker.log').open('ab')
 p=subprocess.Popen(['nohup',sys.executable,str(base/'docs/validation/gpu/g5/a1_t4_gates.py'),'--vm',str(base)],cwd=base,stdout=log,stderr=log,start_new_session=True,env=dict(os.environ,PYTHONPATH=str(base)+':'+str(base/'docs/validation/gpu/g5')))
 (base/'worker.pid').write_text(str(p.pid))
print('LAUNCHED')
''', timeout=1800)
        while True:
            status = json.loads(remote(f'''import pathlib,json,os
base=pathlib.Path({base!r});pid=int((base/'worker.pid').read_text())
try:
 os.kill(pid,0);alive=(pathlib.Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
except OSError:alive=False
tail=(base/'worker.log').read_text()[-600:] if (base/'worker.log').exists() else ''
print(json.dumps(dict(done=(base/'G5A1_GATES_DONE').exists(),alive=alive,tail=tail)))
''').strip().splitlines()[-1])
            (out/'progress.json').write_text(json.dumps(status, indent=2)+'\n')
            if status['done'] or not status['alive']:
                remote(f'''import tarfile,pathlib
base=pathlib.Path({base!r})
with tarfile.open(base/'results.tar','w') as t:
 t.add(base/'results',arcname='results');t.add(base/'worker.log',arcname='worker.log')
print('PACKED')
''')
                call(['download', '-s', args.session, base+'/results.tar', str(out/'results.tar')], heartbeat)
                with tarfile.open(out/'results.tar') as t:
                    t.extractall(out)
                if not status['done']:
                    raise RuntimeError('A1 gates worker failed; see worker.log')
                return
            deadline = time.monotonic()+300
            while time.monotonic() < deadline:
                (heartbeat).write_text(json.dumps(dict(pid=os.getpid(), time=time.time(), operation='poll wait'))+'\n')
                time.sleep(10)
    finally:
        try:
            if attempted:
                call(['stop', '-s', args.session], heartbeat)
        finally:
            watch.terminate(); watch.wait(timeout=30)


# ---------------------------------------------------------------- VM side
def vm(base):
    import numpy as np
    base = Path(base); sys.path[:0] = [str(base), str(base/'docs/validation/gpu/g5')]
    os.chdir(base)
    from a1_common import write
    payload = base/'payload'; results = base/'results'; results.mkdir(exist_ok=True)
    log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)
    # A1b-9: identity of the compiled module on NVRTC (normal, wrong ID, wrong tag, G2 module).
    from chatcarlo.gpu import core, physics_sets
    from chatcarlo.gpu.bundle import load
    a, *_ = load(payload/'gate1/bundles/s1')
    ident = dict(normal=None, wrong_id=None, wrong_tag=None, g2=None)
    normal = core.Core(a, 'cuda'); ident['normal'] = dict(status='pass', physics_set=normal.physics_set)
    for key, macro, value in (('wrong_id', 'G5_SET_ID', '99'), ('wrong_tag', 'G5_SET_TAG', '0ULL')):
        cp = normal.cp; real = cp.RawModule
        def bad_module(*args, options=(), **kw):
            options = tuple(f'-D{macro}={value}' if o.startswith(f'-D{macro}=') else o for o in options)
            return real(*args, options=options, **kw)
        cp.RawModule = bad_module
        try:
            core.Core(a, 'cuda'); ident[key] = dict(status='fail', detail='wrong build accepted')
        except ValueError as exc:
            ident[key] = dict(status='pass' if 'ID/tag mismatch' in str(exc) else 'fail', detail=str(exc))
        finally:
            cp.RawModule = real
    from chatcarlo.gpu.g2_tables import pack
    from chatcarlo.gpu.g2_core import Core as G2
    ga, *_ = load(payload/'g2vox/bundles/water')
    g2 = G2(ga, 'cuda'); ident['g2'] = dict(status='pass' if g2.physics_set == normal.physics_set else 'fail',
                                            physics_set=g2.physics_set)
    write(results/'identity.json', ident); log('identity', {k: v['status'] for k, v in ident.items()})
    # Faults 1-4 on T4: raw probe samples; KS/judging on the host.
    from a1_faults import physics_probe, geometry_probe
    with np.load(payload/'faults/tables.npz') as z:
        fa = {k: z[k] for k in z.files}
    with np.load(payload/'faults/tables_fault3.npz') as z:
        fa3 = {k: z[k] for k in z.files}
    samples = {}
    good = core.Core(fa, 'cuda')
    for fault, which in ((1, 2), (2, 1)):
        bad = core.Core(fa, 'cuda', fault=fault)
        for m in (1, 2):
            x = np.zeros((50000, 8)); x[:, 0] = 30.; x[:, 1] = m; col = 0 if which == 2 else 1
            samples[f'good_{which}_{m}'] = physics_probe(good, which, x, 22)[:, col]
            samples[f'fault{fault}_{which}_{m}'] = physics_probe(bad, which, x, 33)[:, col]
    x = np.zeros((1000, 8)); x[:, 0] = 100.; x[:, 1] = 2
    samples['good3'] = physics_probe(core.Core(fa3, 'cuda'), 3, x, 22)[:, :2]
    samples['fault3'] = physics_probe(core.Core(fa3, 'cuda', fault=3), 3, x, 33)[:, :2]
    samples['good4'] = np.array(geometry_probe(good)); samples['fault4'] = np.array(geometry_probe(core.Core(fa, 'cuda', fault=4)))
    np.savez(results/'faults_t4.npz', **samples); log('faults done')
    # Statistics: every exported initial job (retry jobs only on request).
    for gate in ('gate1', 'g2vox'):
        for job in sorted((payload/gate/'jobs').glob('False_*.json')):
            stem = job.stem
            if gate == 'gate1':
                bundle_dir = payload/'gate1/bundles'/stem.split('_')[1]; target = results/gate/stem
            else:
                _, name, k = stem.split('_'); bundle_dir = payload/'g2vox/bundles'/name; target = results/gate/f'False_{name}'/k
            if (target/'done.json').exists():
                continue
            subprocess.run([sys.executable, str(base/'docs/validation/gpu/g5/a1_speed.py'), '--mode', gate, '--bundle', str(bundle_dir),
                            '--job', str(job), '--output', str(target), '--device', 'cuda'], check=True, cwd=base,
                           env=dict(os.environ, PYTHONPATH=str(base)+':'+str(base/'docs/validation/gpu/g5')))
            log(gate, stem)
    # G2 gate 3 (report only): the unchanged historical execute().
    from chatcarlo.gpu import run_egs5_60kev as legacy
    for name in ('pdd', 'thin', 'phantom'):
        legacy.execute(payload/'g2gate3'/name, results/'g2gate3'); log('g2gate3', name)
    (base/'G5A1_GATES_DONE').write_text('done\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vm', type=Path); p.add_argument('--prepare', action='store_true')
    p.add_argument('--payload', type=Path, default=CACHE/'payload')
    p.add_argument('--session', default='g5-a1-gates'); p.add_argument('--output', type=Path, default=CACHE/'run')
    a = p.parse_args()
    if a.vm:
        vm(a.vm)
    elif a.prepare:
        prepare(a.payload)
    else:
        launch(a)


if __name__ == '__main__':
    main()
