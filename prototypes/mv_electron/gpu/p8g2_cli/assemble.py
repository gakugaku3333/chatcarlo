# Executes on the VM only, after hash-verified upload.
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

root=Path('/content/p8g2');parts=Path('/content/parts')
for name in sorted({p.name.split('.part.')[0] for p in parts.glob('*.part.*')}):
    with (root/name).open('wb') as output:
        for part in sorted(parts.glob(name+'.part.*')):
            with part.open('rb') as stream:
                while chunk:=stream.read(8*1024*1024):output.write(chunk)
manifest=json.loads((root/'p8g2_manifest.json').read_text())
if manifest['status']!='pass':raise ValueError('incomplete export')
for name,digest in manifest['files'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('hash '+name)
source=Path('/content/p8g2_source');zipfile.ZipFile(root/'p8g2_source.zip').extractall(source)
packages=subprocess.run(['python','-m','pip','list','--format=json'],text=True,capture_output=True,check=True,timeout=120)
if any(p['name']=='cupy-cuda12x' for p in json.loads(packages.stdout)):raise RuntimeError('CuPy 12x conflict; environment must be corrected before launch')
# Preserve the log marker as the completion authority, and keep the PID.
driver=Path('/content/p8g2_driver.py')
driver.write_text('''import sys,traceback
from pathlib import Path
sys.path.insert(0,"/content/p8g2_source")
from prototypes.mv_electron.gpu.p8g2_colab import run
try:
    run(Path("/content/p8g2"),Path("/content/p8g2/results"))
except BaseException:
    traceback.print_exc();print("P8G2 FAILED",flush=True);raise
print("P8G2 ALL DONE",flush=True)
''')
with (root/'run.log').open('w') as log:
    process=subprocess.Popen(['nohup','python','-u',str(driver)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'pid').write_text(str(process.pid));print('launched',process.pid)
