# On the VM: launch vm_prof.py with nohup (its own checks verify hashes).
import subprocess
r = subprocess.run("pip list 2>/dev/null | grep -i cupy", shell=True, capture_output=True, text=True).stdout; print(r)
if "12x" in r: print("CUPY CONFLICT; not launched")
else: subprocess.Popen("nohup python -u /content/vm_prof.py > /content/prof.log 2>&1 &", shell=True); print("launched")
