"""Bounded Colab CLI transport for therapy; reusable client for fake-VM tests."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import zipfile
from . import colab_driver, contracts, therapy_run
from .therapy_bundle import sha


class Client:
    def __init__(self,session):self.session=session;self.heartbeat=None
    def call(self,action,*args,code=None):
        for attempt in range(1 if action=='new' else 3):
            if self.heartbeat is not None:contracts.write(self.heartbeat,dict(timestamp=time.time(),operation=action,attempt=attempt))
            try:
                r=subprocess.run(['colab',action,'-s',self.session,*map(str,args)],input=code,text=True,capture_output=True,timeout={'new':900,'upload':900,'download':900,'stop':180}.get(action,300),check=True)
                return r.stdout
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError):
                if action=='new' or attempt==2:raise RuntimeError('Colab '+action+' failed')
    def new(self):self.call('new','--gpu','T4')
    def stop(self):self.call('stop')
    def execute(self,code):
        text=self.call('exec',code=code)
        for line in text.splitlines():
            if line.startswith('G5_ENTRY '):return json.loads(line[9:])
        raise RuntimeError('missing Colab status')
    def upload(self,src,dst):self.call('upload',src,dst)
    def download(self,src,dst):
        dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True);self.call('download',src,dst)


def build(out):
    """All package runtime members, no prototype sources or repository spectra."""
    out=Path(out);transfer=out/'transfer';transfer.mkdir(exist_ok=True)
    with zipfile.ZipFile(transfer/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('chatcarlo/__init__.py','')
        root=Path(__file__).parent
        for p in sorted(root.iterdir()):
            if p.suffix in ('.py','.json','.h','.cpp','.cu'):z.write(p,'chatcarlo/gpu/'+p.name)
    # flatten the sealed private bundle for the existing split/assembly helpers
    for p in (out/'bundle').iterdir():
        if p.is_file():
            import shutil
            shutil.copyfile(p,transfer/p.name)
    for name in ('condition.json','generation.json','approval.json','preview_snapshot.json','preview.html'):
        import shutil
        shutil.copyfile(out/name,transfer/name)
    return transfer,colab_driver.split_bundle(transfer,transfer/'parts')


def bootstrap(remote,files):
    # Verify transfer before importing uploaded code. Same split manifest as colab_driver.
    return f'''import pathlib,hashlib,json,zipfile,subprocess,sys,fcntl
root=pathlib.Path({remote!r});files=json.loads({json.dumps(files)!r})
lock=(root/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
try:
 pid=int((root/'worker.pid').read_text())
 state=(pathlib.Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]
 if state!='Z':
  assert (root/'worker.condition.sha256').read_text()==files['condition.json']['sha256'],'active worker condition mismatch'
  print('G5_ENTRY '+json.dumps(dict(status='launched')));sys.exit(0)
except (OSError,ValueError,IndexError):pass
for name,info in files.items():
 assert pathlib.Path(name).name==name
 with (root/name).open('wb') as dest:
  for part in info['parts']:
   src=root/'parts'/part['name'];assert hashlib.sha256(src.read_bytes()).hexdigest()==part['sha256'],'transfer part SHA256'
   dest.write(src.read_bytes())
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==info['sha256'],'assembled SHA256'
source=root/'source';source.mkdir(exist_ok=True)
with zipfile.ZipFile(root/'source.zip') as z:
 for name in z.namelist():
  p=pathlib.Path(name);assert not p.is_absolute() and '..' not in p.parts,'unsafe source member'
 z.extractall(source)
(root/'bundle').mkdir(exist_ok=True)
for name in ('manifest.json','catalog.json','settings.json','tables.npz','inputs.npz'):
 import shutil
 shutil.copyfile(root/name,root/'bundle'/name)
sys.path.insert(0,str(source))
from chatcarlo.gpu.therapy_entry import verify
from chatcarlo.gpu.contracts import require
cond=json.loads((root/'condition.json').read_text());require(root,cond,verify)
with (root/'worker.log').open('ab') as log:
 p=subprocess.Popen([sys.executable,'-m','chatcarlo.gpu.therapy_colab','worker',str(root)],cwd=source,stdout=log,stderr=log,start_new_session=True)
(root/'worker.pid').write_text(str(p.pid))
(root/'worker.condition.sha256').write_text(files['condition.json']['sha256'])
print('G5_ENTRY '+json.dumps(dict(status='launched')))
'''


def listing(remote):
    return f'''import pathlib,json
root=pathlib.Path({remote!r});out=root/'results';files=[]
if (out/'gates.json').exists():files.append('gates.json')
for p in sorted(out.glob('batch_*.json')):
 if p.with_suffix('.npz').exists():files.extend([p.with_suffix('.npz').name,p.name])
state='running'
if (out/'done.json').exists():state='complete'
else:
 try:
  pid=int((root/'worker.pid').read_text());stat=(pathlib.Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]
  if stat=='Z':state='failed'
 except (OSError,ValueError,IndexError):state='failed'
log=(root/'worker.log').read_text(errors='replace')[-3000:] if state=='failed' and (root/'worker.log').exists() else ''
print('G5_ENTRY '+json.dumps(dict(status=state,files=files,log=log)))
'''


def run(out,cond,arrays,fg,*,client=None,poll_seconds=60):
    out=Path(out);results=out/'results';results.mkdir(exist_ok=True)
    # Verify local approval/batches before creating a VM.
    from .therapy_entry import verify
    contracts.require(out,cond,verify)
    therapy_run.preflight(results,cond,arrays,fg)
    remote='/content/g5-entry-'+uuid.uuid4().hex
    client=client or Client('g5-entry-'+uuid.uuid4().hex[:10])
    transfer,files=build(out)
    watch=None
    if isinstance(client,Client):
        client.heartbeat=results/'heartbeat.json'
        contracts.write(client.heartbeat,dict(timestamp=time.time(),operation='preflight'))
        watch=subprocess.Popen([sys.executable,'-m','chatcarlo.gpu.therapy_colab','watchdog',str(client.heartbeat)],start_new_session=True)
    collected=set()
    def collect(status):
        names=status.get('files',[])
        allowed={'gates.json',*(f'batch_{b:03d}{s}' for b in range(100) for s in ('.npz','.json'))}
        if len(names)!=len(set(names)) or set(names)-allowed:raise ValueError('remote duplicate/unexpected result')
        for name in names:
            # A batch and its receipt land together, so an interrupted fetch leaves no half pair.
            pair=(name,) if name=='gates.json' else (name,name[:-4]+'.json') if name.endswith('.npz') else ()
            if name in collected or any(q not in names for q in pair):continue
            for q in pair:client.download(remote+'/results/'+q,results/('tmp_'+q))
            for q in pair:os.replace(results/('tmp_'+q),results/q);collected.add(q)
        therapy_run.preflight(results,cond,arrays,fg)
    try:
        client.new()
        client.execute(f"import pathlib,json\np=pathlib.Path({remote!r});(p/'parts').mkdir(parents=True);(p/'results').mkdir()\nprint('G5_ENTRY '+json.dumps(dict(status='ready')))")
        for p in sorted((transfer/'parts').iterdir()):client.upload(p,remote+'/parts/'+p.name)
        # Reuse only locally verified sealed batches and their matching gates.
        for p in sorted(results.iterdir()):
            if p.name=='gates.json' or p.name.startswith('batch_'):client.upload(p,remote+'/results/'+p.name)
        status=client.execute(colab_driver._bootstrap(files,None,therapy_root=remote))
        if status.get('status')!='launched':raise RuntimeError('therapy worker not launched')
        while True:
            status=client.execute(listing(remote));collect(status)
            contracts.write(results/'heartbeat.json',dict(timestamp=time.time(),collected_batches=len(collected)//2))
            if status.get('status')=='complete':
                therapy_run.collect(results,cond,arrays,fg);return
            if status.get('status')!='running':raise RuntimeError('therapy VM failed: '+str(status.get('log','')))
            time.sleep(poll_seconds)
    finally:
        # Interrupted/failed sessions also get a final collection attempt and stop.
        try:collect(client.execute(listing(remote)))
        finally:
            try:client.stop()
            finally:
                if watch is not None:
                    watch.terminate()
                    try:watch.wait(timeout=10)
                    except subprocess.TimeoutExpired:watch.kill();watch.wait(timeout=10)


def worker(root):
    root=Path(root)
    from . import therapy_entry, therapy_inputs, therapy_checks
    from .therapy import CPU
    from .therapy_cuda import T4
    import cupy as cp
    g=therapy_entry.verify(root);cond=g['condition'];contracts.require(root,cond,therapy_entry.verify)
    m,a,fg,_,_=therapy_inputs.load(root/'bundle',cond['spectrum'])
    cpu=CPU(root/'build',m,a,field_geometry=fg,half=(5.,5.))
    check=T4(cp,m,a,fg,half=(5.,5.),check=1,source_check=True)
    # Never reuse a gate record on a fresh VM without rerunning device checks.
    therapy_checks.run(cpu,a,fg,cond['seed'],check)
    gpu=T4(cp,m,a,fg,half=(5.,5.))
    class Driver:
        def __call__(self,n,seed,start):return gpu.run(n,seed,start=start)
        def observation(self):return gpu.last_observation
    therapy_run.run(root/'results',cond,a,fg,Driver(),lambda:None)
    contracts.write(root/'results/done.json',dict(status='complete'))


def watchdog(heartbeat):
    heartbeat=Path(heartbeat)
    while True:
        try:
            age=time.time()-json.loads(heartbeat.read_text())['timestamp']
            if age>1020:contracts.write(heartbeat.with_name('watchdog.json'),dict(status='stale launcher heartbeat',age_s=age))
        except (OSError,ValueError,KeyError):pass
        time.sleep(10)


if __name__=='__main__':
    if len(sys.argv)!=3 or sys.argv[1] not in ('worker','watchdog'):raise SystemExit('worker ROOT | watchdog HEARTBEAT')
    (worker if sys.argv[1]=='worker' else watchdog)(Path(sys.argv[2]))
