"""G1 Colab CLI transfer, verified receipts and restartable VM worker.

No imports of diagnostic packages; all paths and commands are explicit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np

POLL_SECONDS=600
PART_BYTES=50_000_000
REMOTE='/content/chatcarlo-g1'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def identity(manifest,job,batch,start,count):
    # The complete bundle includes source dictionary, source_host and every
    # table/kernel hash. The complete job includes both seeds and batch layout.
    return dict(bundle_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),job=job,batch=batch,start=start,count=count)

def expected_identity(manifest,job,name):
    if job['mode']=='gate2':
        if name=='gate2.json':return identity(manifest,job,-1,0,10000000)
        if name.startswith('paired_') and name.endswith('.npz'):
            b=int(name[7:-4])
            if 0<=b<10 and name==f'paired_{b:02d}.npz':return identity(manifest,job,b,b*1000000,1000000)
    elif job['mode']=='batches' and name=='source_reference.json':
        return identity(manifest,job,-2,0,job['n_histories'])
    elif name.startswith('batch_') and name.endswith('.npz'):
        b=int(name[6:-4]);start=b*job['batch_size']
        if 0<=start<job['n_histories'] and name==f'batch_{b:06d}.npz':return identity(manifest,job,b,start,min(job['batch_size'],job['n_histories']-start))
    raise ValueError('unexpected result name or range: '+name)

def read_result(folder,name,expected):
    folder=Path(folder);path=folder/name;receipt=folder/(name+'.receipt.json')
    if not path.exists() or not receipt.exists():return None
    try:r=json.loads(receipt.read_text())
    except (OSError,ValueError):raise ValueError('invalid receipt: '+name)
    if r.get('identity')!=expected:raise ValueError('result identity mismatch: '+name)
    if digest(path)!=r.get('sha256'):raise ValueError('result SHA256 mismatch: '+name)
    if name.endswith('.npz'):
        with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    return json.loads(path.read_text())

def write_result(folder,name,value,key,*,observed=None,elapsed_s=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);path=folder/name;pending=folder/(name+'.pending')
    if name.endswith('.npz'):
        with pending.open('wb') as f:np.savez_compressed(f,**value)
    else:pending.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    receipt=dict(sha256=digest(pending),identity=key)
    if observed is not None:receipt['observed']=observed
    if elapsed_s is not None:receipt['elapsed_s']=elapsed_s
    pending.replace(path)
    rp=folder/(name+'.receipt.json.pending');rp.write_text(json.dumps(receipt,sort_keys=True)+'\n');rp.replace(folder/(name+'.receipt.json'))

def split_bundle(folder,parts,part_bytes=PART_BYTES):
    folder=Path(folder);parts=Path(parts);parts.mkdir(parents=True,exist_ok=True);files={}
    for path in sorted(folder.iterdir()):
        if not path.is_file():continue
        chunks=[]
        with path.open('rb') as stream:
            i=0
            while True:
                data=stream.read(part_bytes)
                if not data:break
                name=f'{path.name}.part{i:06d}';(parts/name).write_bytes(data);chunks.append(dict(name=name,sha256=digest(parts/name)));i+=1
        files[path.name]=dict(sha256=digest(path),parts=chunks)
    return files

def assemble(parts,target,files):
    parts=Path(parts);target=Path(target);target.mkdir(parents=True,exist_ok=True)
    for name,item in files.items():
        if Path(name).name!=name:raise ValueError('invalid transfer member')
        pending=target/(name+'.pending')
        with pending.open('wb') as dest:
            for part in item['parts']:
                if Path(part['name']).name!=part['name']:raise ValueError('invalid transfer part')
                src=parts/part['name']
                if digest(src)!=part['sha256']:raise ValueError('transfer part SHA256 mismatch')
                with src.open('rb') as stream:
                    while True:
                        block=stream.read(1024*1024)
                        if not block:break
                        dest.write(block)
        if digest(pending)!=item['sha256']:raise ValueError('assembled SHA256 mismatch')
        pending.replace(target/name)

def source_reference(bundle_path,job_path,out):
    """Replay only approved source inputs, without Core or worker receipts."""
    from .bundle import load
    from .source_host import sample_source_photons
    _,src,spec,_,manifest=load(bundle_path)
    job=json.loads(Path(job_path).read_text())
    if job['mode']!='batches':raise ValueError('source reference requires batches mode')
    rng=np.random.default_rng(job['source_seed']);batches={}
    for batch,start in enumerate(range(0,job['n_histories'],job['batch_size'])):
        n=min(job['batch_size'],job['n_histories']-start)
        arrays=sample_source_photons(src,n,rng,spec)
        batches[f'batch_{batch:06d}.npz']={
            name:hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
            for name,array in zip(('origins','directions','energies'),arrays)}
    name='source_reference.json'
    write_result(out,name,batches,expected_identity(manifest,job,name))

def worker(bundle_path,job_path,out):
    from .bundle import load
    from .core import Core
    from .source_host import sample_source_photons
    a,src,spec,settings,manifest=load(bundle_path);job=json.loads(Path(job_path).read_text());out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if job['mode']=='gate2':
        from .validation import gate2
        gate2(bundle_path,job['scene_id'],out,job)
    else:
        import os
        fault_path=Path(job_path).parent/'fault.json'
        if fault_path.exists():
            fault=json.loads(fault_path.read_text()).get('fault')
            if fault not in ('seed','start','g2_cfg','g2_cfg_inplace','g2_table','g2_swap'):raise ValueError('invalid fault injection')
            os.environ['CHATCARLO_G4_FAULT']=fault
        rng=np.random.default_rng(job['source_seed'])
        if 'grid' in job:
            from .g2_core import Core, grid_from
            driver=Core(a,job['device'],grid=grid_from(job['grid']))
        else:
            driver=Core(a,job['device'])
        for batch,start in enumerate(range(0,job['n_histories'],job['batch_size'])):
            n=min(job['batch_size'],job['n_histories']-start)
            # Always advance source, including when a verified batch is reused.
            o,d,e=sample_source_photons(src,n,rng,spec);key=identity(manifest,job,batch,start,n);name=f'batch_{batch:06d}.npz'
            if read_result(out,name,key) is not None:continue
            t0=time.perf_counter()
            result,observed=driver.run(o,d,e,seed=job['seed'],start=start,fluorescence=settings['fluorescence'],observe=True)
            elapsed=time.perf_counter()-t0
            write_result(out,name,result,key,observed=observed,elapsed_s=elapsed)
    if job['mode']=='batches' and job['device']=='cuda':
        # A fresh process replays from the bundle/job, even if every batch was reused.
        subprocess.run([sys.executable,'-m','chatcarlo.gpu.colab_driver','source-reference',
                        '--bundle',str(Path(bundle_path).resolve()),'--job',str(Path(job_path).resolve()),
                        '--out',str(out.resolve())],cwd=Path(__file__).resolve().parents[2],check=True)
    (out/'done.json.pending').write_text(json.dumps(dict(status='complete')));(out/'done.json.pending').replace(out/'done.json')

def _bootstrap(files,job_sha):
    # Standalone verification before importing any uploaded source code.
    return f'''import pathlib,hashlib,json,subprocess,sys,fcntl
base=pathlib.Path({REMOTE!r});parts=base/'parts';target=base/'chatcarlo'/'gpu';target.mkdir(parents=True,exist_ok=True)
launch_lock=(base/'launch.lock').open('a')
fcntl.flock(launch_lock,fcntl.LOCK_EX)
try:
 pid=int((base/'worker.pid').read_text())
 state=(pathlib.Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]
 if state != 'Z':
  assert (base/'worker.job.sha256').read_text()=={job_sha!r}, 'worker job mismatch'
  print('G1_LAUNCHED')
  sys.exit(0)
except (OSError,ValueError,IndexError):pass
(base/'chatcarlo'/'__init__.py').write_text('')
files=json.loads({json.dumps(files)!r})
for name,item in files.items():
 pending=target/(name+'.pending')
 with pending.open('wb') as dest:
  for part in item['parts']:
   src=parts/part['name'];assert hashlib.sha256(src.read_bytes()).hexdigest()==part['sha256'],'part SHA256'
   with src.open('rb') as stream:
    while True:
     data=stream.read(1048576)
     if not data:break
     dest.write(data)
 assert hashlib.sha256(pending.read_bytes()).hexdigest()==item['sha256'],'assembled SHA256'
 pending.replace(target/name)
assert hashlib.sha256((base/'job.json').read_bytes()).hexdigest()=={job_sha!r},'job SHA256'
sys.path.insert(0,str(base))
from chatcarlo.gpu.bundle import verify
verify(target)
(base/'results').mkdir(exist_ok=True)
with (base/'worker.log').open('ab') as log:
 process=subprocess.Popen(['nohup',sys.executable,'-m','chatcarlo.gpu.colab_driver','worker','--bundle',str(target),'--job',str(base/'job.json'),'--out',str(base/'results')],cwd=base,stdout=log,stderr=log,start_new_session=True)
(base/'worker.pid').write_text(str(process.pid))
(base/'worker.job.sha256').write_text({job_sha!r})
print('G1_LAUNCHED')
'''

def _listing():
    return f'''import pathlib,json,os
root=pathlib.Path({REMOTE!r})/'results'
for receipt in sorted(root.glob('*.receipt.json')):
 name=receipt.name[:-13]
 if name.endswith('.pending'):continue
 if (root/name).is_file():print('G1_RESULT '+name)
if (root/'done.json').exists():print('G1_DONE')
else:
 try:
  pid=int((root.parent/'worker.pid').read_text())
  state=pathlib.Path('/proc')/str(pid)/'stat'
  status=state.read_text().rsplit(')',1)[1].split()[0]
  print('G1_FAILED' if status=='Z' else 'G1_RUNNING')
 except (OSError,ValueError,IndexError):print('G1_FAILED')
'''

def launch(bundle_path,job,out,*,gate1,gate2=None,session='chatcarlo-g1',colab='colab',poll_seconds=POLL_SECONDS,fault_path=None):
    from .bundle import verify
    manifest=verify(bundle_path);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if json.loads(Path(gate1).read_text()).get('status')!='pass':raise ValueError('gate 1 must pass before Colab')
    if job['mode']!='gate2' and (gate2 is None or json.loads(Path(gate2).read_text()).get('status')!='pass'):raise ValueError('gate 2 must pass before detector production')
    if not session or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in session):raise ValueError('invalid session')
    def cli(args,input=None,*,retry=True):
        for attempt in range(3 if retry else 1):
            try:
                result=subprocess.run([colab,*args],input=input,text=True,capture_output=True,check=True,timeout=300)
                if 'not found' in result.stdout.lower() or 'not found' in result.stderr.lower():raise RuntimeError('Colab session lost')
                return result.stdout
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                if not retry or attempt == 2: raise
    with tempfile.TemporaryDirectory(prefix='chatcarlo-g1-transfer-') as tmp:
        tmp=Path(tmp);files=split_bundle(bundle_path,tmp/'parts',part_bytes=PART_BYTES);job_path=tmp/'job.json';job_path.write_text(json.dumps(job,sort_keys=True)+'\n')
        try:
            try:
                cli(['new','-s',session,'--gpu','T4'], retry=False)
            except subprocess.TimeoutExpired:
                # Creation may have succeeded remotely; probe the session, never
                # create a second session on an ambiguous timeout.
                cli(['exec','-s',session], "print('G1_SESSION')")
            cli(['exec','-s',session],f"import pathlib\nbase=pathlib.Path({REMOTE!r})\n(base/'parts').mkdir(parents=True,exist_ok=True)\n(base/'results').mkdir(exist_ok=True)\nfor name in ('done.json','source_reference.json','source_reference.json.receipt.json'):\n (base/'results'/name).unlink(missing_ok=True)\n")
            for part in sorted((tmp/'parts').iterdir()):cli(['upload','-s',session,str(part),f'{REMOTE}/parts/{part.name}'])
            cli(['upload','-s',session,str(job_path),f'{REMOTE}/job.json'])
            if fault_path is not None:cli(['upload','-s',session,str(fault_path),f'{REMOTE}/fault.json'])
            else:cli(['exec','-s',session],f"from pathlib import Path\nPath({REMOTE+'/fault.json'!r}).unlink(missing_ok=True)\n")
            # Restore only validated result+receipt pairs, preserving source replay.
            for receipt in sorted(out.glob('*.receipt.json')):
                name=receipt.name[:-13]
                if name=='source_reference.json':continue  # Always regenerate on the current VM.
                key=expected_identity(manifest,job,name)
                if read_result(out,name,key) is None:continue
                for p in (out/name,receipt):cli(['upload','-s',session,str(p),f'{REMOTE}/results/{p.name}'])
            job_sha=digest(job_path)
            probe = f"""import pathlib
base=pathlib.Path({REMOTE!r})
try:
 pid=int((base/'worker.pid').read_text())
 state=(pathlib.Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]
 if state != 'Z':
  assert (base/'worker.job.sha256').read_text()=={job_sha!r}, 'worker job mismatch'
  print('G1_LAUNCHED')
 else: print('G1_ABSENT')
except (OSError,ValueError,IndexError): print('G1_ABSENT')
"""
            try:
                launched=cli(['exec','-s',session],_bootstrap(files,job_sha),retry=False)
            except subprocess.TimeoutExpired:
                launched=cli(['exec','-s',session],probe)
                if 'G1_ABSENT' in launched:
                    launched=cli(['exec','-s',session],_bootstrap(files,job_sha),retry=False)
            if 'G1_LAUNCHED' not in launched:raise RuntimeError('bundle assembly/verification did not launch worker')
            reference_received=False
            polls=0
            while True:
                listing=cli(['exec','-s',session],_listing())
                polls += 1
                (out/'heartbeat.json').write_text(json.dumps(dict(timestamp=time.time(), polls=polls,
                    collected_batches=len(list(out.glob('batch_*.npz.receipt.json')))))+'\n')
                if not any(status in listing for status in ('G1_RUNNING','G1_DONE','G1_FAILED')):
                    raise RuntimeError('invalid Colab status or lost session')
                for line in listing.splitlines():
                    if not line.startswith('G1_RESULT '):continue
                    name=line.split(' ',1)[1]
                    if Path(name).name!=name or name.endswith('.pending') or not name.endswith(('.npz','.json')):raise ValueError('invalid remote result name')
                    local_receipt=out/(name+'.receipt.json')
                    if local_receipt.exists() and name!='source_reference.json':
                        key=expected_identity(manifest,job,name)
                        if read_result(out,name,key) is not None:continue
                    with tempfile.TemporaryDirectory(dir=tmp) as stage:
                        stage=Path(stage)
                        for part in (name,name+'.receipt.json'):cli(['download','-s',session,f'{REMOTE}/results/{part}',str(stage/part)])
                        receipt=json.loads((stage/(name+'.receipt.json')).read_text());key=receipt['identity']
                        expected=expected_identity(manifest,job,name)
                        read_result(stage,name,expected)
                        (stage/name).replace(out/name);(stage/(name+'.receipt.json')).replace(local_receipt)
                        if name=='source_reference.json':reference_received=True
                (out/'heartbeat.json').write_text(json.dumps(dict(timestamp=time.time(), polls=polls,
                    collected_batches=len(list(out.glob('batch_*.npz.receipt.json')))))+'\n')
                if 'G1_FAILED' in listing:
                    # Keep the VM's traceback: the session is stopped below and its files are lost.
                    try:
                        tail=cli(['exec','-s',session],f"import pathlib\nprint(pathlib.Path({REMOTE!r},'worker.log').read_text(errors='replace')[-20000:])")
                        (out/'worker_tail.log').write_text(tail)
                    except (subprocess.CalledProcessError,subprocess.TimeoutExpired,RuntimeError,OSError):pass
                    raise RuntimeError('G1_FAILED: worker exited without done.json (see worker_tail.log)')
                if 'G1_DONE' in listing:
                    if job['mode']=='gate2':
                        required=[('gate2.json',-1,0,10000000)]+[(f'paired_{b:02d}.npz',b,b*1000000,1000000) for b in range(10)]
                    else:
                        required=[(f'batch_{b:06d}.npz',b,start,min(job['batch_size'],job['n_histories']-start)) for b,start in enumerate(range(0,job['n_histories'],job['batch_size']))]
                    if job['mode']=='batches' and job['device']=='cuda':
                        name='source_reference.json'
                        if not reference_received or read_result(out,name,expected_identity(manifest,job,name)) is None:raise ValueError('completed VM is missing verified result '+name)
                    for name,b,start,count in required:
                        if read_result(out,name,identity(manifest,job,b,start,count)) is None:raise ValueError('completed VM is missing verified result '+name)
                    return
                time.sleep(poll_seconds)
        finally:
            # A vanished session must not discard already verified local results.
            try:cli(['stop','-s',session])
            except (subprocess.CalledProcessError,subprocess.TimeoutExpired,RuntimeError):pass

def g2_gate(bundle_path, job_path, out):
    """Execute only the explicitly hashed, bundled G2 runner."""
    from .bundle import verify
    manifest = verify(bundle_path)
    job = json.loads(Path(job_path).read_text())
    name = job['runner']
    if name not in ('run_gates.py', 'run_egs5_60kev.py'):
        raise ValueError('invalid G2 runner')
    if manifest['files'][name] != job['runner_sha256']:
        raise ValueError('G2 runner SHA256 mismatch')
    if name == 'run_gates.py' and job.get('gate') != 2:
        raise ValueError('bundled G2 gate must be 2')
    args = ['--gate', '2'] if name == 'run_gates.py' else []
    # The runner is a separate process: put the bundle's package root on its path.
    env = dict(os.environ, PYTHONPATH=str(Path(bundle_path).resolve().parents[1]))
    subprocess.run([sys.executable, str(Path(bundle_path)/name),
                    '--bundle', str(bundle_path), '--out', str(out), *args], check=True, env=env)


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['worker','launch','source-reference','g2-gate']);ap.add_argument('--bundle',required=True,type=Path);ap.add_argument('--job',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);ap.add_argument('--gate1',type=Path);ap.add_argument('--gate2',type=Path);ap.add_argument('--session',default='chatcarlo-g1');a=ap.parse_args(argv)
    if a.action=='g2-gate':g2_gate(a.bundle,a.job,a.out)
    elif a.action=='worker':worker(a.bundle,a.job,a.out)
    elif a.action=='source-reference':source_reference(a.bundle,a.job,a.out)
    else:launch(a.bundle,json.loads(a.job.read_text()),a.out,gate1=a.gate1,gate2=a.gate2,session=a.session)

if __name__=='__main__':main()
