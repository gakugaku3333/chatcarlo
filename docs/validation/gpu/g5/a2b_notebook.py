"""Create a single-Run-all T4 notebook and code archive (no private spectra)."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from chatcarlo.gpu import therapy

ROOT=Path(__file__).resolve().parents[4]


def create(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    archive=out/'a2b_source.zip'
    if archive.exists():raise ValueError('archive already exists')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('chatcarlo/__init__.py','')
        for p in sorted(therapy.HERE.iterdir()):
            if p.suffix in ('.py','.h','.cpp','.cu','.json'):z.write(p,'chatcarlo/gpu/'+p.name)
        for p in sorted((ROOT/'prototypes').rglob('*')):
            if p.suffix in ('.py','.h','.cpp','.cu'):z.write(p,str(p.relative_to(ROOT)))
        for name in ('metadata.json','estar_water.html','estar.npz'):
            p=ROOT/'docs/validation/mv/p1/data'/name;z.write(p,str(p.relative_to(ROOT)))
        for name in ('a2b_legacy_adapter.py','a2b_speed.py','a3_entry_speed.py'):
            p=Path(__file__).with_name(name);z.write(p,name)
        for name in ('a2b_masks.npz','a2b_masks.json'):
            p=Path(__file__).with_name(name);z.write(p,'docs/validation/gpu/g5/'+name)
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    code='''# Set this Drive folder before Run all. It must hold a2b_source.zip and
# the four human-approved directories old_cold/new_cold/old_cached/new_cached.
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
DRIVE=Path('/content/drive/MyDrive/viveMonte/g5/a2b')
import hashlib,zipfile,shutil,sys,subprocess,time,json,os
archive=DRIVE/'a2b_source.zip'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==EXPECTED
source=Path('/content/a2b_source');source.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as z:
 for name in z.namelist():
  p=Path(name);assert not p.is_absolute() and '..' not in p.parts
 z.extractall(source)
sys.path.insert(0,str(source))
import cupy as cp
assert 'T4' in cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
from chatcarlo.gpu import seed_bundle,contracts
from a2b_speed import kernel
# B0/B1/B2 run inside every fresh worker, including resume; no inherited old T0.
rows={}
for phase in ('cold','cached'):
 for version in ('old','new'):
  name=version+'_'+phase;root=Path('/content')/('a2b_'+name)
  if not root.exists():shutil.copytree(DRIVE/name,root)
  script=root/'launch.py'
  script.write_text("import sys;sys.path.insert(0,"+repr(str(source))+");from pathlib import Path;from chatcarlo.gpu import seed_colab;import a2b_legacy_adapter as legacy;"+("legacy" if version=='old' else "seed_colab")+".worker(Path("+repr(str(root))+"))")
  env=dict(os.environ,CUPY_CACHE_DIR='/content/a2b_cache/'+version)
  begin=time.perf_counter()
  with (root/'worker.log').open('ab') as log:
   proc=subprocess.Popen([sys.executable,str(script)],cwd=source,stdout=log,stderr=log,env=env,start_new_session=True)
  (root/'worker.pid').write_text(str(proc.pid))
  last=None
  try:
   while True:
    state=proc.poll();snapshot=root/'results/snapshot.tar'
    if snapshot.exists():
     h=hashlib.sha256(snapshot.read_bytes()).hexdigest()
     if h!=last:
      temp=DRIVE/name/'snapshot.pending';shutil.copyfile(snapshot,temp)
      assert hashlib.sha256(temp.read_bytes()).hexdigest()==h
      temp.replace(DRIVE/name/'snapshot.tar');last=h
    if state is not None:break
    time.sleep(60)
   assert state==0,(root/'worker.log').read_text()[-4000:]
  finally:
   if proc.poll() is None:
    import signal
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
  rows[name]=dict(elapsed_on_VM_s=time.perf_counter()-begin,worker_pid=proc.pid,snapshot_sha256=last)
  (DRIVE/'notebook_timings.json').write_text(json.dumps(rows,indent=2))
  if name=='old_cold':
   materials,inputs,_=seed_bundle.load(root/'bundle')
   speed=kernel(cp,materials,inputs);(DRIVE/'kernel_speed.json').write_text(json.dumps(speed,indent=2))
   assert speed['status']=='pass','B5 exceeds 1.0571; stop and report'
print('A2b T4 finished. Retrieve hash-verified snapshots; run local B3 judge after review.')
print('Notebook elapsed_on_VM_s is a VM-only observation. Full CLI B5 timing/cache proof uses a2b_launch.py.')
'''
    code='EXPECTED='+repr(digest)+'\n'+code
    notebook=dict(cells=[dict(cell_type='markdown',metadata={},source=['T4 / gmail Drive / Run all once. Requires four approved previews. This notebook does not switch the public entry.']),dict(cell_type='code',metadata={},source=code.splitlines(True),outputs=[],execution_count=None)],metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),accelerator='GPU',colab=dict(name='ChatCarlo G5 A2b')),nbformat=4,nbformat_minor=5)
    (out/'a2b_t4.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n')
    return dict(source_sha256=digest,notebook=str(out/'a2b_t4.ipynb'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);a=p.parse_args();print(json.dumps(create(a.out),indent=2))
