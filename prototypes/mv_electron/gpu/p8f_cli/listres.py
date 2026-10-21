# Runs on the Colab VM: list finished result files and whether the driver is alive.
import os,subprocess
for root,_,files in os.walk("/content/p8f/results"):
    for f in files:
        if not f.endswith(".pending"):print("FILE",os.path.relpath(os.path.join(root,f),"/content/p8f/results"))
alive=subprocess.run("pgrep -f 'python -u /content/driver'",shell=True,capture_output=True).returncode==0
print("STATE","ALIVE" if alive else "DEAD")
print(subprocess.run("tail -2 /content/run.log",shell=True,capture_output=True,text=True).stdout)
