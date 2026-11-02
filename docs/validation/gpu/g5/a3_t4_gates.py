"""A3 remaining H2/H5/H6 gates. prepare is local; run owns exactly one T4.

No writes to ~/.cache, no changes to engine or frozen files, no silent retries of
valid speed measurements. See A3_T4_RUNBOOK.md for inputs and preview approval.
"""
import argparse
from contextlib import ExitStack
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path[:0]=[str(ROOT),str(HERE)]
from a3_entry_speed import HISTORIES, POLL_SECONDS, SITE_CUSTOMIZE, SharedClient, cache_used, cli_measure, ratio


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.tolist() if hasattr(x,'tolist') else x.item())+'\n')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def writable(path):
    path=Path(path).expanduser().resolve()
    forbidden=(Path.home()/'.cache').resolve()
    if path==forbidden or forbidden in path.parents:raise ValueError('~/.cache is read-only')
    if path==ROOT or ROOT in path.parents:raise ValueError('private outputs must stay outside repository')
    return path


def no_cache_writes(event,args):
    """Host audit guard, also covering frozen helpers called during prepare/run."""
    names=[]
    if event=='open':
        p,mode,flags=args
        if isinstance(p,(str,bytes,os.PathLike)) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):names=[p]
    elif event in ('os.mkdir','os.remove','os.rmdir','os.chmod','os.utime','os.truncate'):names=args[:1]
    elif event in ('os.rename','os.link','os.symlink'):names=args[:2]
    cache=(Path.home()/'.cache').resolve()
    for name in names:
        if isinstance(name,(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(name)).expanduser().resolve()
            if p==cache or cache in p.parents:raise PermissionError('A3 prohibits writes/deletions in ~/.cache')


def identity():
    from chatcarlo.gpu.therapy_entry import source_hashes
    return dict(source_hashes(), **{f'chatcarlo/{n}':sha(ROOT/'chatcarlo'/n) for n in ('__main__.py','scene.py')})


def reserve(path, extra):
    """Keep the approved 3 GiB floor after the next allocation estimate."""
    p=Path(path)
    while not p.exists():p=p.parent
    if shutil.disk_usage(p).free-extra<3*1024**3:
        raise RuntimeError('estimated allocations would cross the 3 GiB free-space floor')


def tree_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def scene():
    return dict(domain='therapy_mv',source=dict(spectrum='mohan10',field_cm=[10,10],ssd_cm=100),
                phantom=dict(material='water',thickness_cm=30),scoring='pdd_ocr',
                run=dict(histories=HISTORIES,seed=20261101,engine='colab_t4'))


def verify_old_diagnostic(root):
    baseline=json.loads((HERE/'a1_run/t4_speed/job.json').read_text())['bundles']
    from a0_bootstrap import verify_files
    for cid in ('g2b','g4'):
        d=Path(root)/cid
        if sha(d/'manifest.json')!=baseline['old/'+cid]['manifest.json']['sha256']:
            raise ValueError('old diagnostic manifest differs from sealed A1 baseline')
        manifest=json.loads((d/'manifest.json').read_text())
        if manifest['format']!='chatcarlo-g1-v1':raise ValueError('old diagnostic format')
        verify_files(d,manifest)


def prepare(a):
    out=writable(a.output)
    if out.exists():raise ValueError('new output directory required')
    verify_old_diagnostic(a.old_diagnostic)
    reserve(out,4*tree_bytes(a.therapy_bundle))
    from a1_common import space
    space(out);out.mkdir(parents=True)
    from chatcarlo.gpu import therapy_inputs, therapy_entry
    from prototypes.mv_electron.entry import cli as old_cli
    import yaml
    # Use caller-supplied sealed inputs; read existing caches only.
    for version in ('old','new'):
        for phase in ('cold','cached'):
            run=out/'entry'/version/phase
            run.parent.mkdir(parents=True,exist_ok=True)
            s=out/f'{version}_{phase}.yaml';s.write_text(yaml.safe_dump(scene()))
            with patch.object(therapy_inputs,'DEFAULT_BUNDLE',a.therapy_bundle.resolve()):
                rc=(old_cli if version=='old' else therapy_entry).main(['preview',str(s),'--out',str(run)])
            if rc:raise RuntimeError('preview failed')
    # G4 real CLI example, with final code and explicit output directory.
    import chatcarlo.__main__ as diagnostic
    with patch.object(sys,'argv',['chatcarlo','preview',str(HERE.parent/'g4/scene_portrait_detector.yaml'),
            '--engine','gpu','--gpu-device','colab','--seed','2040','--run-dir',str(out/'g4')]):
        if diagnostic.main():raise RuntimeError('G4 preview failed')
    prepare_jobs(out,a)
    write(out/'prepared.json',dict(format='g5-a3-t4-v1',sources=identity(),histories=HISTORIES,poll_seconds=POLL_SECONDS,
        inputs={k:str(getattr(a,k).expanduser().resolve()) for k in ('therapy_bundle','p7','p8g2','old_diagnostic','g2_reference')}))
    print('Previews prepared. Review and approve the five previews; see A3_T4_RUNBOOK.md.')


def prepare_jobs(out, inputs):
    """Bake on the host, before VM creation. All generated files live under output."""
    import a1_gate1, a1_g2vox, a2_t4, a2_common
    from a1_launch import create_new
    from a1_speed import load_configs
    from prototypes.mv_electron.gpu import p8g2_colab
    a1_gate1.export(out/'H5_jobs/gate1')
    a1_g2vox.export(inputs.g2_reference,out/'H5_jobs/g2vox')
    generated=out/'generated';generated.mkdir()
    cfg,_=load_configs(HERE/'a1_speed_configs.json')
    create_new(generated/'diagnostic/new',cfg)
    with patch.object(a2_common,'inputs',lambda:p8g2_colab.load(inputs.p8g2)):
        for c in 'ABC':a2_t4.water(generated/f'water_{c}.npz',c)


def judge_h5(out, record):
    """Existing A1 judges, including their restricted statistical retries, on host."""
    import a1_gate1, a1_g2vox
    # Frozen judges use temporary bundles; keep those under --output too.
    scratch=out/'H5/tmp';scratch.mkdir(parents=True)
    with patch('tempfile.tempdir',str(scratch)):
        one=a1_gate1.run(out/'H5/g1',out/'H5/g1.json',device='cuda',remote=out/'H5/gate1')
        if one['status']!='pass' or not one['production_conditions']:raise RuntimeError('H5 G1 failed')
        two=a1_g2vox.run(Path(record['inputs']['g2_reference']),out/'H5/g2',out/'H5/g2.json',
                        device='cuda',remote=out/'H5/g2vox')
        if two['status']!='pass':raise RuntimeError('H5 G2 failed')
    return dict(status='pass',g1=one['status'],g2=two['status'])


def require_prepared(out):
    from chatcarlo.gpu import therapy_entry,contracts,approval
    from prototypes.mv_electron.entry import scene as old_scene,condition,preview
    record=json.loads((out/'prepared.json').read_text())
    if record['sources']!=identity() or record['histories']!=HISTORIES or record['poll_seconds']!=60:
        raise ValueError('prepared code/config changed; regenerate previews')
    for version in ('old','new'):
        for phase in ('cold','cached'):
            d=out/'entry'/version/phase
            if (d/'results').exists():raise ValueError('entry results already exist; no speed rerun')
            if version=='old':
                c,m,_=condition.identities(old_scene.load(out/f'{version}_{phase}.yaml'))
                preview.require_approval(d,c,m)
            else:
                g=therapy_entry.verify(d);contracts.require(d,g['condition'],therapy_entry.verify)
    approval._verify(out/'g4')
    if (out/'g4/results').exists():raise ValueError('G4 results already exist')
    # Verify approval against the generated condition, never generate it here.
    g=contracts.verify(out/'g4',('bundle/manifest.json','job.json','preview_snapshot.json','preview.html'))
    approval.require(out/'g4',g['condition'])
    return record


class Owner:
    def __init__(self,session,out):self.session,self.out=session,out;self.rows=[]
    def call(self,action,*args,code=None,idempotent=False):
        # Exec requires an explicit read/status opt-in. Never replay a launcher.
        retry=(action in ('upload','download','stop') or
               (action=='exec' and idempotent and 'Popen' not in (code or '')))
        tries=6 if retry else 1
        for attempt in range(1,tries+1):
            begin=time.perf_counter();error=detail=None
            try:
                result=subprocess.run(['colab',action,'-s',self.session,*map(str,args)],input=code,
                    capture_output=True,text=True,check=True,timeout=dict(new=900,upload=900,download=900,stop=180).get(action,300))
                return result.stdout
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as exc:
                error=type(exc).__name__;detail=(str(exc.stderr or '')+str(exc.stdout or ''))[-1500:]
                if attempt==tries:raise
            finally:
                self.rows.append(dict(operation=action,attempt=attempt,retry=attempt>1,
                    seconds=time.perf_counter()-begin,error=error,detail=detail))
                write(self.out/'transport_timings.json',self.rows)
            time.sleep(60)
    def raw(self,code,*,idempotent=False):return self.call('exec',code=code,idempotent=idempotent)
    def json(self,code):
        text=self.raw(code,idempotent=True)
        for line in text.splitlines():
            if line.startswith('A3_JSON '):return json.loads(line[8:])
        raise RuntimeError('missing A3 status')


def tar_bytes(files):
    """Exact uncompressed PAX tar size, including extended headers and padding."""
    size=0
    with tarfile.open(fileobj=io.BytesIO(),mode='w') as t:
        for name,p in sorted(files.items()):
            info=t.gettarinfo(str(p),arcname=name)
            if not info.isfile():raise ValueError('payload requires regular files: '+str(p))
            size+=len(info.tobuf(t.format,t.encoding,t.errors))+((info.size+511)//512)*512
    return ((size+1024+tarfile.RECORDSIZE-1)//tarfile.RECORDSIZE)*tarfile.RECORDSIZE


def reusable_archive(archive,files):
    """True only if the tar holds exactly these names with exactly these bytes."""
    if not archive.is_file():return False
    # A later member replaces an earlier one, as it does on extraction.
    held={}
    with tarfile.open(archive) as t:
        for m in t:
            if not m.isfile() or m.name not in files:return False
            h=hashlib.sha256();f=t.extractfile(m)
            while b:=f.read(1<<22):h.update(b)
            held[m.name]=h.hexdigest()
    return held=={name:sha(p) for name,p in files.items()}


def payload(out,record):
    """Stream sealed inputs directly; stage only the small host-generated files."""
    inputs=record['inputs'];verify_old_diagnostic(inputs['old_diagnostic'])
    files={}
    for folder in ('chatcarlo','prototypes/mv_electron','docs/validation/gpu','docs/validation/mv'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.h','.cpp','.cu','.json','.yaml','.yml'):
                files['source/'+str(p.relative_to(ROOT))]=p
    for p in (ROOT/'chatcarlo/data').rglob('*'):
        if p.is_file() and p.suffix in ('.csv','.json'):files['source/'+str(p.relative_to(ROOT))]=p
    # prototypes/mv_electron/physics.py reads these at import (a2_oldnew's layout path).
    for n in ('estar.npz','estar_water.html','metadata.json'):
        p=ROOT/'docs/validation/mv/p1/data'/n
        if p.is_file():files['source/'+str(p.relative_to(ROOT))]=p
    for name,src in (('diagnostic/old',inputs['old_diagnostic']),('bundles/p7',inputs['p7']),
                     ('bundles/p8g2',inputs['p8g2']),('therapy_bundle',inputs['therapy_bundle']),
                     ('h5',out/'H5_jobs')):
        src=Path(src)
        if not src.is_dir():raise ValueError('missing prepared payload input: '+str(src))
        for p in src.rglob('*'):
            if p.is_file():files['inputs/'+name+'/'+str(p.relative_to(src))]=p
    generated=out/'generated'
    if not (generated/'diagnostic/new').is_dir() or not all((generated/f'water_{c}.npz').is_file() for c in 'ABC'):
        raise ValueError('missing host-generated inputs; regenerate prepare')
    small={str(p.relative_to(generated)):p for p in generated.rglob('*') if p.is_file()}
    files.update({'inputs/'+name:p for name,p in small.items()})
    # A tar placed here before the run is used only after every member matches.
    if (out/'payload.tar').exists():
        if not reusable_archive(out/'payload.tar',files):raise ValueError('existing payload.tar differs from prepared inputs')
        write(out/'source_hashes.json',{name[7:]:sha(p) for name,p in files.items() if name.startswith('source/')})
        return out/'payload.tar'
    # Nothing has been allocated in payload yet. Source inputs never get copied.
    expected=tar_bytes(files)
    reserve(out,expected+sum(p.stat().st_size for p in small.values())+2*1024**3)
    staging=out/'payload';staging.mkdir()
    for name,p in small.items():
        dst=staging/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
        files['inputs/'+name]=dst
    archive=out/'payload.tar'
    with tarfile.open(archive,'w') as t:
        for name,p in sorted(files.items()):t.add(p,arcname=name,recursive=False)
    if archive.stat().st_size!=expected:raise RuntimeError('payload tar size changed during packing')
    write(out/'source_hashes.json',{name[7:]:sha(p) for name,p in files.items() if name.startswith('source/')})
    return archive


def upload_payload(owner,archive,remote):
    owner.raw(f"from pathlib import Path\np=Path({remote!r});p.mkdir(parents=True,exist_ok=False)\n")
    parts=[]
    with archive.open('rb') as f:
        while block:=f.read(20*1024*1024):
            p=archive.with_name('upload_part');p.write_bytes(block)
            name=f'upload_{len(parts):04d}'
            owner.call('upload',str(p),remote+'/'+name);parts.append((name,sha(p)))
    # Stream assembly, SHA256 verification and safe extraction on VM.
    # colab exec gives up after about 60 s, so 2 GiB is assembled by a detached process.
    script=f'''import pathlib,hashlib,tarfile
root=pathlib.Path({remote!r})
with (root/'payload.tar').open('wb') as out:
 for name,digest in {parts!r}:
  p=root/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
  with p.open('rb') as src:
   while block:=src.read(1048576):out.write(block)
h=hashlib.sha256()
with (root/'payload.tar').open('rb') as f:
 while block:=f.read(1048576):h.update(block)
assert h.hexdigest()=={sha(archive)!r}
with tarfile.open(root/'payload.tar') as t:
 for m in t.getmembers():
  assert not pathlib.Path(m.name).is_absolute() and '..' not in pathlib.Path(m.name).parts
  assert m.isfile() or m.isdir()
 t.extractall(root,filter='data')
(root/'ASSEMBLED').write_text('done')
'''
    owner.raw(f'''import pathlib,subprocess,sys
r=pathlib.Path({remote!r});(r/'assemble.py').write_text({script!r})
log=(r/'assemble.log').open('wb')
p=subprocess.Popen([sys.executable,str(r/'assemble.py')],stdout=log,stderr=log,start_new_session=True)
(r/'assemble.pid').write_text(str(p.pid))
''')
    while True:
        state=owner.json(f'''import pathlib,json
r=pathlib.Path({remote!r});pid=(r/'assemble.pid').read_text()
try:alive=(pathlib.Path('/proc')/pid/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
except OSError:alive=False
print('A3_JSON '+json.dumps(dict(done=(r/'ASSEMBLED').exists(),alive=alive,log=(r/'assemble.log').read_text()[-1000:])))
''')
        if state['done']:return
        if not state['alive']:raise RuntimeError('payload assembly failed: '+state['log'])
        time.sleep(10)


def collect_task(owner,remote,task,out):
    """Start once; poll/collect every 60 s, including a final failed-worker fetch."""
    root=remote+'/tasks/'+task;source=remote+'/source';base=remote+'/inputs'
    owner.raw(f'''import pathlib,subprocess,sys,os
r=pathlib.Path({root!r});r.parent.mkdir(parents=True,exist_ok=True)
log=r.with_suffix('.log').open('wb')
env=dict(os.environ,PYTHONPATH={source!r}+':'+{source+'/docs/validation/gpu/g5'!r},NUMBA_CACHE_DIR={remote+'/numba'!r},MPLCONFIGDIR={remote+'/mpl'!r},CUPY_CACHE_DIR={remote+'/gate_cache'!r},PYTHONDONTWRITEBYTECODE='1')
p=subprocess.Popen([sys.executable,{source+'/docs/validation/gpu/g5/a3_t4_worker.py'!r},{task!r},{base!r},str(r)],cwd={source!r},env=env,stdout=log,stderr=log,start_new_session=True)
r.with_suffix('.pid').write_text(str(p.pid))
''')
    out.mkdir(parents=True)
    while True:
        state=owner.json(f'''import pathlib,json,tarfile
r=pathlib.Path({root!r});pid=int(r.with_suffix('.pid').read_text())
try:alive=(pathlib.Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
except OSError:alive=False
with tarfile.open(r.with_suffix('.tar'),'w') as t:
 if r.exists():
  for p in r.rglob('*'):
   if p.is_file() and p.suffix not in ('.pending','.tmp'):t.add(p,arcname=str(p.relative_to(r)))
 t.add(r.with_suffix('.log'),arcname='worker.log')
print('A3_JSON '+json.dumps(dict(done=(r/'DONE').exists(),alive=alive)))
''')
        owner.call('download',root+'.tar',str(out/'collected.tar'))
        with tarfile.open(out/'collected.tar') as t:t.extractall(out,filter='data')
        if state['done']:return json.loads((out/'verdict.json').read_text())
        if not state['alive']:raise RuntimeError(task+' failed; see '+str(out/'worker.log'))
        time.sleep(POLL_SECONDS)


def evidence(owner,client):
    return owner.json(f'''import pathlib,json,hashlib
p=pathlib.Path({client.cache!r});log=pathlib.Path({client.remote+'/compile.json'!r})
files={{str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file() and f.suffix=='.cubin'}}
print('A3_JSON '+json.dumps(dict(files=files,compiler=json.loads(log.read_text()) if log.exists() else {{}})))
''')


def entry_speed(owner,remote,out):
    from chatcarlo.gpu import therapy_inputs
    rows={};proof={}
    for phase in ('cold','cached'):
        for version in ('old','new'):
            d=out/'entry'/version/phase;r=f'{remote}/entry/{version}/{phase}'
            cache=f'{remote}/compile_cache/{version}'
            client=SharedClient(owner,r,cache)
            owner.raw(f"from pathlib import Path\np=Path({r!r});(p/'instrument').mkdir(parents=True,exist_ok=False)\n(p/'instrument/sitecustomize.py').write_text({SITE_CUSTOMIZE!r})\n")
            before=evidence(owner,client);begin_ops=len(owner.rows)
            elapsed=cli_measure(version,out/f'{version}_{phase}.yaml',d,client)
            cli_operations=owner.rows[begin_ops:]
            # New CLI has receipt/collection but no EGS5 analysis in main. Include
            # the same analysis definition used by the old main in the total.
            analysis_s=0.
            if version=='new':
                begin=time.perf_counter()
                import a3_judge
                import numpy as np
                with np.load(d/'results/dose.npz',allow_pickle=False) as z:rpt=a3_judge.judge10(z['field_MeV'],HISTORIES)
                write(d/'speed_analysis.json',rpt);analysis_s=time.perf_counter()-begin
            after=evidence(owner,client)
            row=dict(main_s=elapsed,analysis_s=analysis_s,total_s=elapsed+analysis_s,
                cache_dir=cache,cache_before=before['files'],cache_after=after['files'],compiler=after['compiler'],
                histories=HISTORIES,poll_seconds=60,local_results=str(d/'results'),instrumentation_remote=r,
                worker_roots=sorted(client.worker_roots),
                transport=cli_operations,transport_retried=any(op.get('retry',False) for op in cli_operations),
                transport_retry_attempts=sum(op.get('retry',False) for op in cli_operations))
            rows[version+'_'+phase]=row
            if phase=='cached':proof[version]=cache_used(rows[version+'_cold'],row)
            write(out/'entry_speed.json',dict(status='running',rows=rows,cache=proof))
    ratios={phase:ratio(rows['old_'+phase]['total_s'],rows['new_'+phase]['total_s']) for phase in ('cold','cached')}
    status=('fail' if ratios['cold']['status']!='pass' else 'undetermined' if any(p['status']!='pass' for p in proof.values())
        else 'pass' if ratios['cached']['status']=='pass' else 'fail')
    result=dict(status=status,rows=rows,cache=proof,ratios=ratios)
    write(out/'entry_speed.json',result);return result


def g4(owner,remote,out):
    """Exercise the real diagnostic CLI/Colab path on the owned T4."""
    import chatcarlo.__main__ as cli
    from chatcarlo.gpu import colab_driver
    real=subprocess.run;call=owner.call
    def routed(cmd,*args,**kw):
        if isinstance(cmd,(list,tuple)) and cmd and cmd[0]=='colab':
            action=cmd[1]
            if action in ('new','stop'):return subprocess.CompletedProcess(cmd,0,'','')
            # run through the owner, with the frozen session argument removed
            tail=list(cmd[2:]);i=tail.index('-s');del tail[i:i+2]
            # owner.call itself uses real subprocess.run, avoid recursion
            try:
                with patch.object(subprocess,'run',real):
                    text=call(action,*tail,code=kw.get('input'),
                              idempotent=action=='exec' and kw.get('input')==colab_driver._listing())
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as exc:
                # Prevent the frozen driver's outer loop from replaying this call.
                raise RuntimeError('A3 transport exhausted: '+action) from exc
            return subprocess.CompletedProcess(cmd,0,text,'')
        return real(cmd,*args,**kw)
    with patch.object(colab_driver,'REMOTE',remote+'/g4'),patch.object(subprocess,'run',routed),patch.object(sys,'argv',
        ['chatcarlo','run',str(HERE.parent/'g4/scene_portrait_detector.yaml'),'--engine','gpu','--gpu-device','colab',
         '--seed','2040','--run-dir',str(out/'g4')]):
        # launch's default is bound at definition; inject explicit poll_seconds.
        launch=colab_driver.launch
        with patch.object(colab_driver,'launch',lambda *a,**kw:launch(*a,**dict(kw,poll_seconds=60))):
            if cli.main():raise RuntimeError('G4 CLI failed')
    return dict(status='pass',run_dir=str(out/'g4'))


def run(a):
    out=writable(a.output);record=require_prepared(out)
    if (out/'STARTED.json').exists():raise ValueError('single valid run only; STARTED exists')
    archive=payload(out,record)
    owner=Owner('g5-a3-'+uuid.uuid4().hex[:12],out);remote='/content/'+owner.session
    write(out/'STARTED.json',dict(session=owner.session,remote=remote,sources=identity()))
    # Partial runs inherit nothing: the skipped gates must be cited from the earlier run.
    result=dict(status='running',session=owner.session,scope='entry-only' if getattr(a,'entry_only',False) else 'remaining' if getattr(a,'remaining',False) else 'all')
    attempted=False
    try:
        attempted=True;owner.call('new','--gpu','T4')
        upload_payload(owner,archive,remote)
        if getattr(a,'entry_only',False):
            pass
        elif getattr(a,'remaining',False):
            result['H6_oldnew']=collect_task(owner,remote,'remaining',out/'H6')
        else:
            result['H2']=collect_task(owner,remote,'h2',out/'H2')
            collect_task(owner,remote,'h5',out/'H5')
            result['H5_statistics']=judge_h5(out,record)
            result['H5_g4']=g4(owner,remote,out)
            result['H6_kernels']=collect_task(owner,remote,'speed',out/'H6')
        with patch('chatcarlo.gpu.therapy_inputs.DEFAULT_BUNDLE',Path(record['inputs']['therapy_bundle'])):
            result['H6_entry']=entry_speed(owner,remote,out)
        states=[x['status'] for x in result.values() if isinstance(x,dict)]
        result['status']='fail' if 'fail' in states else 'undetermined' if 'undetermined' in states else 'pass'
        if identity()!=record['sources']:raise RuntimeError('runtime changed during gates')
    except Exception as exc:
        result.update(status='fail',error=str(exc));raise
    finally:
        try:
            if attempted:owner.call('stop')
            result['vm_stopped']=attempted
        except Exception as exc:
            result['vm_stopped']=False;result['stop_error']=str(exc)
            raise
        finally:
            write(out/'summary.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--remaining',action='store_true',help='after a run whose H2/H5/kernel speed are recorded: old/new report and entry speed only')
    p.add_argument('--entry-only',action='store_true',help='repeat all four entry timings in one new session after a lost connection')
    p.add_argument('--therapy-bundle',type=Path);p.add_argument('--p7',type=Path);p.add_argument('--p8g2',type=Path)
    p.add_argument('--old-diagnostic',type=Path);p.add_argument('--g2-reference',type=Path)
    a=p.parse_args()
    if a.remaining and a.entry_only:p.error('choose one of --remaining and --entry-only')
    # Host caches for third party libraries must also stay in the new output tree.
    work=writable(a.output);os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
    os.environ['MPLCONFIGDIR']=str(work/'library_cache/mpl');os.environ['NUMBA_CACHE_DIR']=str(work/'library_cache/numba')
    sys.addaudithook(no_cache_writes)
    if a.command=='prepare':
        if any(getattr(a,k) is None for k in ('therapy_bundle','p7','p8g2','old_diagnostic','g2_reference')):p.error('prepare requires all five input paths')
        prepare(a)
    else:
        r=run(a);raise SystemExit(0 if r['status']=='pass' else 2)


if __name__=='__main__':main()
