import sys, json, time
from pathlib import Path
import prototypes.mv_electron.run_p3b3 as p
root = Path('~/.cache/chatcarlo/stats10mv').expanduser()
p.CACHE, p.OUT = root/'work', root/'out'
name, N, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
main = json.load(open('docs/validation/mv/p3b/egs5/main_egs/config.json'))
edges, cum = p.spectrum(p.params())
cfg = json.loads(json.dumps(p.egs_config(N, seed, edges, cum)))
diff = [k for k in main if k not in ('N', 'seed') and cfg.get(k) != main[k]]
print('config diff vs main_egs (excluding N, seed):', diff, flush=True)
if diff: sys.exit('config mismatch')
t = time.time()
p.run_reference(name, N, seed)
print('done', name, round(time.time()-t, 1), 's', flush=True)
