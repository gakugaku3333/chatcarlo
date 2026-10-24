# On a VM: re-export the saved SourceCounters report with the compiled source under NVRTC's temporary file name.
import glob, shutil, subprocess, os
from pathlib import Path
name = open('/content/rx/name.txt').read().strip(); p = Path(name); p.parent.mkdir(parents=True, exist_ok=True)
shutil.copy('/content/rx/compiled.cu', p)
ncu = next(x for x in [shutil.which('ncu'), *glob.glob('/usr/local/cuda*/bin/ncu')] if x)
x = subprocess.run([ncu, '--import', '/content/rx/line_profile.ncu-rep', '--page', 'source', '--print-source', 'cuda,sass', '--csv'], capture_output=True, text=True)
open('/content/rx/source2.csv', 'w').write(x.stdout); print('rc', x.returncode, x.stderr[-500:]); print(x.stdout[:300])
