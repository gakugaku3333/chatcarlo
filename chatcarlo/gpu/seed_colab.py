"""Seed-specific 50-batch Colab transfer; shared bounded CLI/bootstrap/watchdog."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
import zipfile
from . import colab_driver,therapy_colab,contracts,seed_condition,seed_run,seed_analysis
from .therapy_bundle import sha

MEMBERS={'gates.json','prostate.npz','summary.json','batch_table.npz'}


def build(out):
    out=Path(out);transfer=out/'transfer';transfer.mkdir(exist_ok=True)
    # Link sealed inputs rather than replicating the large tables for each run.
    for p in (out/'bundle').iterdir():
        if p.is_file():
            target=transfer/p.name
            if target.is_symlink():target.unlink()
            elif target.exists():raise ValueError('unexpected existing transfer copy: '+p.name)
            target.symlink_to(p.resolve())
    for name in ('condition.json','generation.json','approval.json','preview_snapshot.json','preview.html'):
        shutil.copyfile(out/name,transfer/name)
    with zipfile.ZipFile(transfer/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('chatcarlo/__init__.py','')
        root=Path(__file__).parent
        for p in sorted(root.iterdir()):
            if p.suffix in ('.py','.json','.h','.cpp','.cu'):z.write(p,'chatcarlo/gpu/'+p.name)
        for path in (seed_analysis.MASK_PATH,seed_analysis.MASK_PATH.with_suffix('.json')):
            z.write(path,'docs/validation/gpu/g5/'+path.name)
    from .therapy import space
    space(out,sum(p.stat().st_size for p in transfer.iterdir() if p.is_file()))
    return transfer,colab_driver.split_bundle(transfer,transfer/'parts_seed')


def bootstrap(remote,files):
    # The seed condition check rebuilds the scene (~40 s on Colab), past the `colab exec` reply
    # timeout. worker() runs the same verify/require before any transport, so launch skips it.
    code=therapy_colab.bootstrap(remote,files)
    check="from chatcarlo.gpu.therapy_entry import verify\nfrom chatcarlo.gpu.contracts import require\ncond=json.loads((root/'condition.json').read_text());require(root,cond,verify)\n"
    worker="'chatcarlo.gpu.therapy_colab','worker'"
    if code.count(check)!=1 or code.count(worker)!=1:raise ValueError('therapy bootstrap layout changed')
    return code.replace(check,'').replace(worker,"'chatcarlo.gpu.seed_colab','worker'")


def listing(remote):
    return f'''import pathlib,json,hashlib
root=pathlib.Path({remote!r});p=root/'results/snapshot.tar'
state='running'
if (root/'results/done.json').exists():state='complete'
else:
 try:
  pid=int((root/'worker.pid').read_text());stat=(pathlib.Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]
  if stat=='Z':state='failed'
 except (OSError,ValueError,IndexError):state='failed'
log=''
if state=='failed' and (root/'worker.log').exists():
 # Keep the FIRST traceback: later CUDA teardown errors can fill the tail (A2b 2026-11-02).
 text=(root/'worker.log').read_text(errors='replace');first=text.find('Traceback')
 log=(text[first:first+4000]+'\\n...\\n' if first>=0 else '')+text[-1500:]
h=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
print('G5_ENTRY '+json.dumps(dict(status=state,sha256=h,bytes=p.stat().st_size if p.exists() else 0,log=log)))
'''


def publish(out):
    out=Path(out);tmp=out/'snapshot.pending'
    with tarfile.open(tmp,'w') as tar:
        for name in sorted(MEMBERS):
            if (out/name).exists():tar.add(out/name,arcname=name)
    tmp.replace(out/'snapshot.tar')


def unpack(path,out):
    out=Path(out)
    with tempfile.TemporaryDirectory(dir=out) as temp:
        temp=Path(temp)
        with tarfile.open(path,'r') as tar:
            names=[]
            for member in tar.getmembers():
                if member.name not in MEMBERS or not member.isfile() or member.name in names:raise ValueError('unexpected/duplicate snapshot member')
                names.append(member.name)
                with (temp/member.name).open('wb') as f:shutil.copyfileobj(tar.extractfile(member),f)
        if 'prostate.npz' in names and 'gates.json' not in names:raise ValueError('checkpoint without gates')
        for name in names:os.replace(temp/name,out/name)


def run(out,cond,materials,inputs,*,client=None,poll_seconds=60):
    out=Path(out);results=out/'results';results.mkdir(exist_ok=True)
    contracts.require(out,cond,seed_condition.verify);seed_run.preflight(results,cond)
    remote='/content/g5-seed-'+uuid.uuid4().hex;client=client or therapy_colab.Client('g5-seed-'+uuid.uuid4().hex[:10])
    transfer,files=build(out);watch=None;collected=None
    if isinstance(client,therapy_colab.Client):
        client.heartbeat=results/'heartbeat.json';contracts.write(client.heartbeat,dict(timestamp=time.time()))
        watch=subprocess.Popen([sys.executable,'-m','chatcarlo.gpu.therapy_colab','watchdog',str(client.heartbeat)],start_new_session=True)
    def collect(status):
        nonlocal collected
        if not status.get('sha256') or status['sha256']==collected:return
        from .therapy import space
        space(results,2*status.get('bytes',0))
        target=results/'snapshot.download';client.download(remote+'/results/snapshot.tar',target)
        if sha(target)!=status['sha256']:raise ValueError('snapshot SHA256 mismatch; retry same condition')
        unpack(target,results);seed_run.preflight(results,cond);target.unlink();collected=status['sha256']
    try:
        client.new();client.execute(f"import pathlib,json\nr=pathlib.Path({remote!r});(r/'parts').mkdir(parents=True);(r/'results').mkdir()\nprint('G5_ENTRY '+json.dumps(dict(status='ready')))")
        for p in sorted((transfer/'parts_seed').iterdir()):client.upload(p,remote+'/parts/'+p.name)
        shutil.rmtree(transfer/'parts_seed')  # uploaded bytes can be regenerated on resume
        for name in sorted(MEMBERS):
            if (results/name).exists():client.upload(results/name,remote+'/results/'+name)
        status=client.execute(colab_driver._bootstrap(files,None,seed_root=remote))
        if status.get('status')!='launched':raise RuntimeError('seed worker not launched')
        while True:
            status=client.execute(listing(remote));collect(status)
            contracts.write(results/'heartbeat.json',dict(timestamp=time.time(),checkpoint_sha256=collected))
            if status.get('status')=='complete':return seed_run.collect(results,cond)
            if status.get('status')!='running':raise RuntimeError('seed VM failed: '+str(status.get('log','')))
            time.sleep(poll_seconds)
    finally:
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
    from . import seed_bundle,seed_cuda,seed_t4_checks
    import cupy as cp
    cond=seed_condition.verify(root)['condition'];contracts.require(root,cond,seed_condition.verify)
    materials,inputs,_=seed_bundle.load(root/'bundle')
    engines=[seed_cuda.GPU(cp,materials,constants=inputs['constants'],strength=inputs['strength'],densities=inputs['densities'],edges=inputs['mohan10_edges'],cum=inputs['mohan10_cum'],with_seeds=s,beam='prostate') for s in (True,False)]
    def driver(batch,field,side,start,n):
        e=engines[not side];e.g.angle=float(__import__('numpy').deg2rad(cond['angles'][field]));e.upload()
        return e.run(n,cond['seed'],start=start,field=field)
    seed_run.run(root/'results',cond,materials,inputs,driver,lambda:seed_t4_checks.run(cp,materials,inputs,root/'checks'),progress=lambda:publish(root/'results'))
    publish(root/'results');contracts.write(root/'results/done.json',dict(status='complete'))

if __name__=='__main__':
    if len(sys.argv)!=3 or sys.argv[1]!='worker':raise SystemExit('worker ROOT')
    worker(Path(sys.argv[2]))
