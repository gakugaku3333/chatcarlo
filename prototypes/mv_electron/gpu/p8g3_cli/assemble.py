# VM-only script, MODE supplied by start. Not imported locally.
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
root=Path('/content/p8g3')
for name in sorted({p.name.split('.part.')[0] for p in (root/'parts').glob('*.part.*')}):
    with (root/name).open('wb') as output:
        for part in sorted((root/'parts').glob(name+'.part.*')):
            with part.open('rb') as stream:
                while chunk:=stream.read(8*1024*1024):output.write(chunk)
manifest=json.loads((root/'p8g3_manifest.json').read_text())
if manifest['status']!='pass':raise ValueError('export status')
for name,digest in manifest['files'].items():
    if Path(name).name!=name or hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('export hash '+name)
source=Path('/content/p8g3_source')
for name in ('p8g2_source.zip','p8g3_driver.zip'):zipfile.ZipFile(root/name).extractall(source)
packages=subprocess.run(['python','-m','pip','list','--format=json'],text=True,capture_output=True,check=True,timeout=120)
if any(p['name']=='cupy-cuda12x' for p in json.loads(packages.stdout)):raise RuntimeError('CuPy 12x conflict')
driver=root/'driver.py'
driver.write_text('''import sys,traceback
from pathlib import Path
sys.path.insert(0,'/content/p8g3_source')
from prototypes.mv_electron.gpu.p8g3_colab import run
try:
    run(Path('/content/p8g3'),Path('/content/p8g3/GROUP'),mode=MODE,trial=Path('/content/p8g3/trial'))
except BaseException:
    traceback.print_exc();print('P8G3 FAILED',flush=True);raise
print('P8G3 ALL DONE',flush=True)
'''.replace('GROUP','results' if MODE=='production' else 'trial').replace('mode=MODE','mode='+repr(MODE)))
with (root/'run.log').open('w') as log:
    process=subprocess.Popen(['nohup','python','-u',str(driver)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'pid').write_text(str(process.pid));print('launched',process.pid)
