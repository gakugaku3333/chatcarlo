"""G5 A2 T4 launcher (E4 trial). Same-session old/new; the worker is a2_speed.py."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np
import a2_common as common
from a0_speed import sha,write_json
from a1_launch import call,watchdog

HERE=Path(__file__).resolve().parent
P7=Path.home()/'.cache/chatcarlo/p7/work/export'
GPU=common.ROOT/'chatcarlo/gpu'


def water(path,condition='A'):
    """Host-built water inputs; goes to the VM only (holds the spectrum), never to the repo."""
    _,_,a=common.inputs()
    w,c=common.water_arrays(a,condition)
    np.savez(path,**{k:w[k] for k in ('faces','rho','bins','strength','edges','cum','ids','owner','constants','sizes','mass_g_cm2')},
             field_geometry=np.frombuffer(bytes(common.field_geometry()),np.uint8),half=c['field_half'],ssd=c['ssd'])


def run(a):
    from chatcarlo.gpu.colab_driver import split_bundle
    out=a.output.resolve();common.space(out,3*1024**3);out.mkdir(parents=True,exist_ok=True)
    hb=out/'heartbeat.json';write_json(hb,dict(time=time.time(),operation='preflight'))
    watch=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--watchdog',str(hb),'--output',str(out)],start_new_session=True)
    base='/content/g5a2-'+a.session;attempted=False
    remote=lambda code,**kw:call(['exec','-s',a.session],hb,input=code,**kw)
    try:
        with tempfile.TemporaryDirectory(prefix='g5a2-',dir=out) as t:
            tmp=Path(t);water(tmp/'water.npz')
            files={n:split_bundle(p,tmp/('parts_'+n)) for n,p in (('p7',P7),('p8g2',common.SOURCE))}
            controls=[HERE/n for n in ('a2_speed.py','a0_speed.py','a0_bootstrap.py','a0_configs.json')]+[tmp/'water.npz']
            runtime=[GPU/n for n in ('therapy.py','physics_sets.py','catalog.py','g5_base.h','physics_set.h')]+sorted(GPU.glob('t2_*'))
            job=dict(files=files,controls={p.name:sha(p) for p in controls},runtime={p.name:sha(p) for p in runtime})
            write_json(tmp/'job.json',job);write_json(out/'job.json',job)
            attempted=True;call(['new','-s',a.session,'--gpu','T4'],hb)
            remote(f"import pathlib\nb=pathlib.Path({base!r});(b/'chatcarlo/gpu').mkdir(parents=True,exist_ok=True)\nfor n in ('p7','p8g2'):(b/'parts'/n).mkdir(parents=True,exist_ok=True)\n")
            for p in [*controls,tmp/'job.json']:call(['upload','-s',a.session,str(p),base+'/'+p.name],hb)
            for p in runtime:call(['upload','-s',a.session,str(p),base+'/chatcarlo/gpu/'+p.name],hb)
            for n in files:
                for p in sorted((tmp/('parts_'+n)).iterdir()):call(['upload','-s',a.session,str(p),f'{base}/parts/{n}/{p.name}'],hb)
            code=f'''import pathlib,hashlib,json,subprocess,sys,os
base=pathlib.Path({base!r})
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
job=json.loads((base/'job.json').read_text())
for n,d in job['controls'].items():assert sha(base/n)==d,n
for n,d in job['runtime'].items():assert sha(base/'chatcarlo/gpu'/n)==d,n
(base/'chatcarlo/__init__.py').write_text('');(base/'chatcarlo/gpu/__init__.py').write_text('')
for bundle,files in job['files'].items():
 target=base/'bundles'/bundle;target.mkdir(parents=True,exist_ok=True)
 for name,item in files.items():
  with (target/name).open('wb') as out:
   for part in item['parts']:
    src=base/'parts'/bundle/part['name'];assert sha(src)==part['sha256']
    out.write(src.read_bytes())
  assert sha(target/name)==item['sha256'],name
sys.path.insert(0,str(base))
from a0_bootstrap import extract_sources
extract_sources(base/'bundles/p7/p7_sources.zip',base);extract_sources(base/'bundles/p8g2/p8g2_source.zip',base)
log=(base/'worker.log').open('ab')
p=subprocess.Popen(['nohup',sys.executable,str(base/'a2_speed.py'),'--bundles',str(base/'bundles'),'--water',str(base/'water.npz'),'--output',str(base/'results')],cwd=base,stdout=log,stderr=log,start_new_session=True,env=dict(os.environ,PYTHONPATH=str(base)))
(base/'worker.pid').write_text(str(p.pid));print('LAUNCHED')
'''
            print(remote(code,timeout=1800),flush=True)
        while True:
            s=json.loads(remote(f'''import pathlib,json,os
base=pathlib.Path({base!r});r=base/'results';pid=int((base/'worker.pid').read_text())
try:
 os.kill(pid,0);alive=(pathlib.Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
except OSError:alive=False
print(json.dumps(dict(done=(r/'G5A2_DONE').exists(),alive=alive,files=[p.name for p in r.glob('*') if p.is_file() and not p.name.endswith('.ncu-rep')])))
''').strip().splitlines()[-1])
            call(['download','-s',a.session,base+'/worker.log',str(out/'worker.log')],hb)
            for n in s['files']:call(['download','-s',a.session,base+'/results/'+n,str(out/n)],hb)
            if s['done']:return
            if not s['alive']:raise RuntimeError('worker died; see worker.log')
            end=time.monotonic()+60
            while time.monotonic()<end:
                write_json(hb,dict(time=time.time(),operation='poll wait'));time.sleep(10)
    finally:
        try:
            if attempted:call(['stop','-s',a.session],hb)
        finally:
            watch.terminate();watch.wait(timeout=30)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',default='g5-a2')
    p.add_argument('--output',type=Path,default=common.CACHE/'e4');p.add_argument('--watchdog',type=Path);a=p.parse_args()
    if a.watchdog:
        import threading
        watchdog(a.watchdog,a.output/'watchdog.json',threading.Event())
    else:run(a)


if __name__=='__main__':main()
