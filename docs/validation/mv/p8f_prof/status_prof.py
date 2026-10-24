# DO NOT REUSE: the pgrep below matches its own shell, so it always reports ALIVE (2026-10-24).
# Decide completion from a log marker instead (see p8g2_cli/status.py).
import os, subprocess
print(open('/content/prof.log').read()[-1500:] if os.path.exists('/content/prof.log') else 'no log')
print('STATE', 'ALIVE' if subprocess.run('pgrep -f vm_prof.py', shell=True, capture_output=True).stdout else 'DEAD')
print(sorted(os.listdir('/content/p8f/prof')) if os.path.exists('/content/p8f/prof') else '')
