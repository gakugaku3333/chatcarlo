# On the VM: rebuild split files, verify the p8g manifest, launch the driver with nohup.
import glob,hashlib,json,subprocess
for f in sorted({p.split("/")[-1].split(".part.")[0] for p in glob.glob("/content/parts/*.part.*")}):
    open("/content/p8g/"+f,"wb").write(b"".join(open(p,"rb").read() for p in sorted(glob.glob("/content/parts/"+f+".part.*"))))
m=json.load(open("/content/p8g/p8g_manifest.json"))["files"]
bad=[n for n,h in m.items() if hashlib.sha256(open("/content/p8g/"+n,"rb").read()).hexdigest()!=h]
print("files",len(m),"bad",bad)
if not bad:
    open("/content/driver.py","w").write('''
import sys,zipfile
from pathlib import Path
b=Path("/content/p8g");s=Path("/content/p8g_source")
zipfile.ZipFile(b/"p8g_source.zip").extractall(s);sys.path.insert(0,str(s))
from prototypes.mv_electron.gpu.p8g_colab import run
run(b,b/"results")
print("ALL DONE",flush=True)
''')
    r=subprocess.run("pip list 2>/dev/null | grep -i cupy",shell=True,capture_output=True,text=True).stdout;print(r)
    if "12x" in r:print("CUPY CONFLICT; not launched")
    else:subprocess.Popen("nohup python -u /content/driver.py > /content/run.log 2>&1 &",shell=True);print("launched")
