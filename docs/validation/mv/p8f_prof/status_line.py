import os
t = open('/content/line.log').read() if os.path.exists('/content/line.log') else 'no log'
print(t[-1200:]); print('FINISHED' if ('LINE DONE' in t or 'Traceback' in t or 'Error' in t) else 'RUNNING')
