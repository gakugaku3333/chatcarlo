"""Bounded A1 launcher. Claude supplies old bundles frozen at 19c281e."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from a0_launch import call as _call,ASSEMBLY_SECONDS,watchdog,POLL_SECONDS
from a0_speed import sha,write_json
from a1_speed import load_configs
from a1_common import CACHE,SUMMARY,space

HERE=Path(__file__).resolve().parent


def call(args,heartbeat,**kw):
    """a0_launch.call plus bounded retries for idempotent upload/download and for stop.

    A network drop (DNS failure, 900 s download timeout) lost a whole run once: the final
    stop must be retried with a back-off so a T4 VM is never leaked."""
    tries={'upload':3,'download':3,'stop':6,'exec':3}.get(args[0] if args else '',1)
    for attempt in range(tries):
        try:return _call(args,heartbeat,**kw)
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:
            detail=getattr(exc,'stderr',None) or getattr(exc,'stdout',None) or str(exc)
            sys.stderr.write(f'colab {args[0]} failed (attempt {attempt+1}/{tries}): {str(detail)[-400:]}\n')
            # a0_launch.call already retries exec timeouts; here only CLI errors (e.g. VM ReadTimeout).
            if attempt==tries-1 or (args[0]=='exec' and isinstance(exc,subprocess.TimeoutExpired)):raise
            time.sleep(60 if args[0]=='stop' else 5)


def validate_transfer(files):
    for name,item in files.items():
        if Path(name).name!=name or name in ('.','..'):raise ValueError('unsafe transfer member')
        for part in item['parts']:
            if Path(part['name']).name!=part['name'] or part['name'] in ('.','..'):raise ValueError('unsafe transfer part')


def assembly_code(base,bundles,job_sha):
    """Reuse colab_driver's split-bundle assembly, before any uploaded imports."""
    from chatcarlo.gpu.colab_driver import _bootstrap
    pieces=[f'import pathlib,hashlib,json,sys,subprocess,fcntl,os\nbase=pathlib.Path({base!r})\n']
    pieces.append("lock=(base/'launch.lock').open('a')\nfcntl.flock(lock,fcntl.LOCK_EX)\n")
    pieces.append(f"""try:
 pid=int((base/'worker.pid').read_text())
 os.kill(pid,0)
 state=(pathlib.Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]
 if state!='Z':
  assert (base/'worker.job.sha256').read_text()=={job_sha!r},'active worker identity mismatch'
  sys.exit(0)
except (OSError,ValueError,IndexError):pass
if (base/'results/measurements.json').exists():raise ValueError('fresh session required')
""")
    pieces.append(f"assert hashlib.sha256((base/'job.json').read_bytes()).hexdigest()=={job_sha!r},'job SHA256'\n")
    pieces.append("job=json.loads((base/'job.json').read_text())\nfor name,digest in job['controls'].items():\n assert pathlib.Path(name).name==name and hashlib.sha256((base/name).read_bytes()).hexdigest()==digest,'control SHA256'\n")
    for label,files in bundles.items():
        validate_transfer(files)
        if len(Path(label).parts)!=2 or any(p in ('.','..') for p in Path(label).parts):raise ValueError('unsafe bundle label')
        source=_bootstrap(files,job_sha)
        # Reuse only verified assembly, not G4's single-engine worker launcher.
        assembly=source[source.index('files=json.loads'):source.index('sys.path.insert')]
        pieces.append(f"parts=base/'parts'/{label!r};target=base/'bundles'/{label!r};target.mkdir(parents=True,exist_ok=True)\n"+assembly)
    pieces.append("sys.path.insert(0,str(base))\nfrom a1_speed import install_runtimes\ninstall_runtimes(base)\n")
    pieces.append(f"with (base/'worker.log').open('ab') as log:\n p=subprocess.Popen(['nohup',sys.executable,str(base/'a1_speed.py'),'--configs',str(base/'a1_speed_configs.json'),'--bundles',str(base/'bundles'),'--output',str(base/'results')],cwd=base,stdout=log,stderr=log,start_new_session=True)\n(base/'worker.pid').write_text(str(p.pid))\n(base/'worker.job.sha256').write_text({job_sha!r})\n")
    return ''.join(pieces)


def create_new(destination,cfg):
    from chatcarlo.gpu.bundle import create
    from chatcarlo.scene import load_scene
    from chatcarlo.detector import DetectorPlane
    root=HERE.parents[3]
    for c in cfg['configurations']:
        if sha(root/c['scene'])!=c['scene_file_sha256']:raise ValueError('scene SHA256 mismatch')
        scene=load_scene(str(root/c['scene']))
        if not scene.ok:raise ValueError(scene.errors)
        create(scene,DetectorPlane(**c['detector']) if c['detector'] else None,
               Path(destination)/c['id'],g2_tally=bool(c['grid']))


def run(args):
    from chatcarlo.gpu.colab_driver import split_bundle
    cfg,_=load_configs(args.configs)
    if not args.session or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.session):raise ValueError('invalid session')
    out=Path(args.output).resolve();space(out);out.mkdir(parents=True,exist_ok=True)
    heartbeat=out/'heartbeat.json';write_json(heartbeat,dict(time=time.time(),operation='preflight'))
    watch=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--watchdog',str(heartbeat),'--output',str(out)],start_new_session=True)
    base='/content/g5a1-'+args.session;attempted=False
    remote=lambda code:call(['exec','-s',args.session],heartbeat,input=code)
    try:
        with tempfile.TemporaryDirectory(prefix='g5-a1-transfer-') as temp:
            tmp=Path(temp);create_new(tmp/'new',cfg);bundles={}
            for version,root in [('old',args.old_bundles),('new',tmp/'new')]:
                for cid in ('g2b','g4'):
                    src=Path(root)/cid
                    if version=='old':
                        manifest=json.loads((src/'manifest.json').read_text())
                        if manifest['format']!='chatcarlo-g1-v1':raise ValueError('old bundle format')
                        for name,digest in manifest['files'].items():
                            if Path(name).name!=name or sha(src/name)!=digest:raise ValueError('old bundle SHA/path mismatch')
                    label=version+'/'+cid;parts=tmp/'parts'/label
                    bundles[label]=split_bundle(src,parts);validate_transfer(bundles[label])
            controls={p.name:p for p in HERE.glob('a1_*.py')}
            controls.update({n:HERE/n for n in ('a0_speed.py','a0_launch.py','a0_bootstrap.py','a0_configs.json')})
            controls['a1_speed_configs.json']=args.configs
            job=dict(controls={n:sha(p) for n,p in controls.items()},bundles=bundles)
            write_json(tmp/'job.json',job);write_json(out/'job.json',job)
            attempted=True;call(['new','-s',args.session,'--gpu','T4'],heartbeat)
            remote(f'import pathlib\npathlib.Path({base!r}).mkdir(parents=True,exist_ok=True)\n')
            for name,path in dict(controls,**{'job.json':tmp/'job.json'}).items():
                call(['upload','-s',args.session,str(path),base+'/'+name],heartbeat)
            for label in bundles:
                remote(f'import pathlib\npathlib.Path({(base+"/parts/"+label)!r}).mkdir(parents=True,exist_ok=True)\n')
                for part in sorted((tmp/'parts'/label).iterdir()):
                    call(['upload','-s',args.session,str(part),base+'/parts/'+label+'/'+part.name],heartbeat)
            call(['exec','-s',args.session],heartbeat,input=assembly_code(base,bundles,sha(tmp/'job.json')),timeout=ASSEMBLY_SECONDS)
        while True:
            status=json.loads(remote(f'''import pathlib,json,os
base=pathlib.Path({base!r});r=base/'results';pid=int((base/'worker.pid').read_text())
try:
 os.kill(pid,0);alive=(pathlib.Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
except OSError:alive=False
print(json.dumps(dict(done=(r/'G5A1_DONE').exists(),alive=alive,files=[p.name for p in r.glob('*') if p.is_file() and not p.name.endswith('.pending')])))
''').strip().splitlines()[-1])
            for name in status['files']:
                if Path(name).name!=name:raise ValueError('unsafe result path')
                call(['download','-s',args.session,base+'/results/'+name,str(out/name)],heartbeat)
            if status['done']:return
            if not status['alive']:raise RuntimeError('A1 worker failed; partial results preserved')
            deadline=time.monotonic()+getattr(args,'poll_seconds',POLL_SECONDS)
            while time.monotonic()<deadline:
                write_json(heartbeat,dict(time=time.time(),operation='poll wait'));time.sleep(min(10,max(0,deadline-time.monotonic())))
    finally:
        try:
            if attempted:call(['stop','-s',args.session],heartbeat)
        finally:
            watch.terminate();watch.wait(timeout=30)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--old-bundles',type=Path)
    p.add_argument('--configs',type=Path,default=HERE/'a1_speed_configs.json')
    p.add_argument('--session',default='g5-a1');p.add_argument('--output',type=Path,default=SUMMARY/'t4')
    p.add_argument('--watchdog',type=Path);a=p.parse_args()
    if a.watchdog:
        import threading
        watchdog(a.watchdog,a.output/'watchdog.json',threading.Event())
    else:
        if a.old_bundles is None:p.error('--old-bundles is required')
        run(a)


if __name__=='__main__':main()
