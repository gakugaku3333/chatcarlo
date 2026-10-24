# VM-side status. Completion is determined from log markers, never pgrep.
import json
from pathlib import Path
root=Path('/content/p8g2');log=(root/'run.log').read_text() if (root/'run.log').exists() else ''
print(json.dumps(dict(state='done' if 'P8G2 ALL DONE' in log else 'failed' if 'P8G2 FAILED' in log or 'Traceback' in log else 'running',tail=log[-3000:],files=[str(p.relative_to(root)) for p in (root/'results').rglob('*') if p.is_file() and '.pending' not in p.name]+(['run.log'] if (root/'run.log').exists() else []))))
