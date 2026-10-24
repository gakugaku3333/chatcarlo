# VM-only snapshot. Markers authorize completion; PID loss detects failure.
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,'/content/p8g3_source')
from prototypes.mv_electron.gpu import p8g3_core as core
root=Path('/content/p8g3');log=(root/'run.log').read_text() if (root/'run.log').exists() else ''
try:
    pid=int((root/'pid').read_text());os.kill(pid,0)
    stat=Path('/proc')/str(pid)/'stat'
    alive=not (stat.exists() and stat.read_text().split(') ',1)[1].startswith('Z'))
except (OSError,ValueError):alive=False
state='done' if 'P8G3 ALL DONE' in log else 'failed' if 'P8G3 FAILED' in log or 'Traceback' in log or not alive else 'running'
identity=core.read_json(root/'p8g3_manifest.json')['identity'];groups=[]
for name,schedule in (('trial',core.spec(4,2000)),('results',core.spec())):
    dest=root/name;calls=core.available(dest,identity,schedule)
    sessions={}
    for path in dest.glob('session_*.json'):
        sid=core.read_json(path)['session_id'];core.require_session(dest,sid,identity,schedule)
        sessions[sid]=core.kernel.sha(path)
    groups.append(dict(name=name,schedule=schedule,sessions=sessions,
        calls={str(j):dict(sha256=r['sha256'],receipt_sha256=core.kernel.sha(core.call_path(dest,j).with_suffix('.json'))) for j,r in calls.items()},
        records={n:core.kernel.sha(dest/n) for n in ('complete.json','trial_qualified.json') if (dest/n).exists()}))
print(json.dumps(dict(state=state,tail=log[-3000:],groups=groups)))
