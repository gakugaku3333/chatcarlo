# Completion is read from the log marker, never from pgrep (a pgrep pattern matches its own shell).
import os
t=open('/content/run.log').read() if os.path.exists('/content/run.log') else 'no log'
print(t[-1500:])
for root,_,fs in os.walk('/content/p8g/results'):
    for f in fs:print('FILE',os.path.relpath(os.path.join(root,f),'/content/p8g/results'))
print('FINISHED' if ('ALL DONE' in t or 'Traceback' in t) else 'RUNNING')
