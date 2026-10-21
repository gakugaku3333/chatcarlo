# Runs on the Colab VM: rebuild split files, verify the manifest, write and launch the driver.
import glob,hashlib,json,os,subprocess
for f in ["p8f_ref_gate2.npz","p8f_water_baseline.npz"]:
    open("/content/p8f/"+f,"wb").write(b"".join(open(p,"rb").read() for p in sorted(glob.glob("/content/parts/"+f+".part.*"))))
m=json.load(open("/content/p8f/p8f_manifest.json"))["files"]
bad=[n for n,h in m.items() if hashlib.sha256(open("/content/p8f/"+n,"rb").read()).hexdigest()!=h]
print("files",len(m),"bad",bad)
if not bad:
    script='''
import sys,zipfile
from pathlib import Path
b=Path("/content/p8f");s=Path("/content/p8f_source")
zipfile.ZipFile(b/"p8f_source.zip").extractall(s);sys.path.insert(0,str(s))
from prototypes.mv_electron.gpu.p8f_colab import run
run(b,b/"results")
print("ALL DONE",flush=True)
'''
    open("/content/driver.py","w").write(script)
    r=subprocess.run("pip list 2>/dev/null | grep -i cupy",shell=True,capture_output=True,text=True).stdout
    print(r)
    if "12x" in r: print("CUPY CONFLICT; not launched")
    else:
        subprocess.Popen("nohup python -u /content/driver.py > /content/run.log 2>&1 &",shell=True);print("launched")
