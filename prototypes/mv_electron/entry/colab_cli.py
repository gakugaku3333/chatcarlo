"""Colab CLI orchestration. No account identifiers or raw CLI output are persisted."""
import json
from pathlib import Path
import subprocess
import time
import uuid
from ..gpu import p5b_core as core
from . import bundle

SESSION = 'mventry'
REMOTE = '/content/mventry'
POLL_SECONDS = 10*60  # user decision 2026-10-21 (was 20 min; the entry VM was lost after 96/100 batches)
SPLIT_THRESHOLD = 150000000
PART_BYTES = 50*1024*1024


class Client:
    def __init__(self, executable='colab'):
        self.executable = executable

    def call(self, *args, code=None):
        result = subprocess.run([self.executable, *args], input=code, capture_output=True, text=True)
        if result.returncode:
            # Do not echo CLI stdout/stderr: authentication errors can include account information.
            raise core.Stop('Colab '+args[0]+': failed (session missing or command rejected)')
        return result.stdout

    def execute(self, code):
        output = self.call('exec', '-s', SESSION, code=code)
        for line in output.splitlines():
            if line.startswith('MV_ENTRY_JSON '):
                return json.loads(line[len('MV_ENTRY_JSON '):])
        raise core.Stop('Colab exec: missing status; session lost or VM failed')

    def upload(self, local, remote):
        for attempt in range(2):
            try:
                self.call('upload', '-s', SESSION, str(local), remote)
                return
            except core.Stop:
                if attempt:
                    raise core.Stop('upload failed after 2 attempts: '+Path(local).name)

    def download(self, remote, local):
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        pending = local.with_name(local.name+'.part')
        self.call('download', '-s', SESSION, remote, str(pending))
        pending.replace(local)


def transfer_files(directory, work):
    """Split only large files; carry hashes of parts and the assembled files."""
    directory, work = Path(directory), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    manifest = bundle.read_json(directory/'manifest.json')
    uploads, assembly = [], {}
    for name in [*manifest['files'], 'manifest.json']:
        path = directory/name
        if path.stat().st_size <= SPLIT_THRESHOLD:
            uploads.append((path, REMOTE+'/'+name))
            continue
        parts = []
        with path.open('rb') as source:
            index = 0
            while chunk := source.read(PART_BYTES):
                part = work/f'{name}.part.{index:04}'
                part.write_bytes(chunk)
                parts.append(dict(name=part.name, sha256=core.file_sha(part)))
                uploads.append((part, REMOTE+'/parts/'+part.name))
                index += 1
        assembly[name] = dict(parts=parts, sha256=core.file_sha(path))
    return uploads, assembly


def bootstrap(assembly, session):
    """VM-side verify/assemble/extract before nohup. This string is fake-tested locally."""
    return f'''
import hashlib,json,pathlib,zipfile,subprocess,sys,importlib.metadata
root=pathlib.Path({REMOTE!r})
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
try:
    assembly=json.loads({json.dumps(assembly)!r})
    for name,info in assembly.items():
        for part in info['parts']:
            p=root/'parts'/part['name']
            if sha(p)!=part['sha256']:raise RuntimeError('part SHA256 mismatch: '+part['name'])
        with (root/name).open('wb') as target:
            for part in info['parts']:
                with (root/'parts'/part['name']).open('rb') as src:
                    for b in iter(lambda:src.read(1024*1024),b''):target.write(b)
        if sha(root/name)!=info['sha256']:raise RuntimeError('assembled SHA256 mismatch: '+name)
    manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        p=pathlib.Path(name)
        if p.is_absolute() or '..' in p.parts:raise RuntimeError('unsafe bundle path')
        if sha(root/name)!=digest:raise RuntimeError('bundle SHA256 mismatch: '+name)
    names=[d.metadata.get('Name','').lower().replace('_','-') for d in importlib.metadata.distributions()]
    if 'cupy-cuda12x' in names:raise RuntimeError('cupy-cuda12x conflict; not launched')
    source=root/'source';source.mkdir(exist_ok=True)
    with zipfile.ZipFile(root/'source.zip') as z:
        for name in z.namelist():
            p=pathlib.Path(name)
            if p.is_absolute() or '..' in p.parts:raise RuntimeError('unsafe source member')
        z.extractall(source)
    with (root/'driver.log').open('w') as log:
        p=subprocess.Popen(['nohup',sys.executable,'-u','-m','prototypes.mv_electron.entry.vm_driver',
            '--bundle',str(root),'--results',str(root/'results'),'--session',{session!r}],
            cwd=source,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (root/'driver.pid').write_text(str(p.pid))
    print('MV_ENTRY_JSON '+json.dumps(dict(status='launched')))
except Exception as exc:
    print('MV_ENTRY_JSON '+json.dumps(dict(status='failed',reason=str(exc))))
'''


def poll_code():
    return f'''
import json,os,pathlib
root=pathlib.Path({REMOTE!r});results=root/'results'
files=[]
# Publish a batch only when its metadata (written last) exists.
for p in sorted(results.glob('batch_*.json')):
    stem=p.stem;index=stem.split('_')[1]
    files.extend([stem+'.npz','observation_'+index+'.json',p.name])
for pattern in ('session_*.json','fault_observation.json'):
    files.extend(p.name for p in results.glob(pattern))
state='running'
pid=int((root/'driver.pid').read_text())
try:os.kill(pid,0)
except ProcessLookupError:state='dead'
status=results/'driver_status.json'
if status.exists():
    report=json.loads(status.read_text())
    if report.get('status') in ('complete','failed'):
        state=report['status'];files.append('driver_status.json')
print('MV_ENTRY_JSON '+json.dumps(dict(state=state,files=files)))
'''


def run(directory, results, client=None, sleep=time.sleep):
    client = client or Client()
    directory, results = Path(directory), Path(results)
    data, manifest = bundle.load(directory)
    cond, mat = bundle.read_json(directory/'condition.json'), bundle.read_json(directory/'materials.json')
    completed = bundle.existing(results, cond, mat)
    session = str(uuid.uuid4())
    timings = {}; before = time.perf_counter(); created = False
    try:
        client.call('new', '-s', SESSION, '--gpu', 'T4'); created = True
        setup = client.execute(f"import pathlib,json\nr=pathlib.Path({REMOTE!r})\n[(r/p).mkdir(parents=True,exist_ok=True) for p in ('parts','results')]\nprint('MV_ENTRY_JSON '+json.dumps(dict(status='ready')))\n")
        if setup.get('status') != 'ready':
            raise core.Stop('VM setup failed')
        uploads, assembly = transfer_files(directory, results.parent/'transfer_parts')
        for local, remote in uploads:
            client.upload(local, remote)
        # Only completely verified batches are sent back, so the driver never recalculates them.
        for b in sorted(completed):
            for path in bundle.batch_paths(results, b):
                client.upload(path, REMOTE+'/results/'+path.name)
        timings['upload'] = time.perf_counter()-before
        before = time.perf_counter()
        status = client.execute(bootstrap(assembly, session))
        if status.get('status') != 'launched':
            raise core.Stop('VM verification/launch failed: '+status.get('reason', 'unknown'))
        timings['vm_launch'] = time.perf_counter()-before
        before = time.perf_counter()
        timings['collection'] = 0.
        while True:
            state = client.execute(poll_code())
            download_begin = time.perf_counter()
            files = state.get('files', [])
            for name in files:
                target = bundle.safe_path(results, name)
                # Session records and completed batch artifacts are immutable. Status can change.
                if not target.exists() or name == 'driver_status.json':
                    client.download(REMOTE+'/results/'+name, target)
            timings['collection'] += time.perf_counter()-download_begin
            if state.get('state') in ('complete', 'failed', 'dead'):
                if state['state'] != 'complete':
                    raise core.Stop('VM driver stopped: '+state['state'])
                break
            if state.get('state') != 'running':
                raise core.Stop('Colab session lost; collected batches retained')
            sleep(POLL_SECONDS)
        timings['production_and_collection'] = time.perf_counter()-before
    finally:
        if created:
            try:
                client.call('stop', '-s', SESSION)
            except core.Stop:
                pass
        core.write_json(results.parent/'colab_timings.json', timings)
    return manifest
