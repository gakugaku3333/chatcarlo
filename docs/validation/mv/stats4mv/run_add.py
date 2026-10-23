"""4 MV (P7 condition C) EGS5 additional runs: same config as main_egs_C except N and seed."""
import sys, json, time
from pathlib import Path
import prototypes.mv_electron.run_p7 as p
from prototypes.mv_electron.gpu import p7_core as h
root = Path('~/.cache/chatcarlo/stats4mv').expanduser()
name, N, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
c = h.condition('C'); edges, _, cum = h.spectrum(c['spectrum'])
main = json.load(open('docs/validation/mv/p7/egs5/main_egs_C/config.json'))
with p.egs_scope('C'):
    p.egs.CACHE, p.egs.OUT = root/'work', root/'out'
    cfg = p.egs.r.plain(dict(main, N=N, seed=seed))
    want = json.loads(json.dumps(p.egs_config('C', 'main', N)))
    diff = [k for k in main if k not in ('N', 'seed') and want.get(k) != main[k]]
    print('config diff vs main_egs_C (excluding N, seed):', diff, flush=True)
    if diff: sys.exit('config mismatch')
    t = time.time()
    p.egs.run_egs(name, N, seed, edges, cum, field=[c['field_half'], c['ssd']], regions=h.layout('p7')['regions'])
    print('done', name, round(time.time()-t, 1), 's', flush=True)
