"""G5 A3 section 2e.1(b): embed the saved entry 10 MV batches in P7, crop, and reproduce the recorded verdict.

Usage: a3_crop_regression.py <extracted 9734038057e4 directory>"""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import a3_judge as A
from prototypes.mv_electron.entry import bundle,compare
from prototypes.mv_electron.entry.condition import digest

R=Path(sys.argv[1]);cond=json.loads((R/'condition.json').read_text());mat=json.loads((R/'materials.json').read_text())
if not digest(cond).startswith('9734038057e4'):raise SystemExit('wrong saved run')
rows=bundle.collect(R/'results',cond,mat);old=compare.stacked(rows,cond)
p6=np.stack([r['regions'].sum(0) if r['regions'].ndim==2 else r['regions'] for r in rows])
if p6.shape!=(100,3420):raise SystemExit(f'unexpected table {p6.shape}')
full=np.full((100,3420),7.0);full[:,:540]=p6[:,:540]
for j in range(6):
    full[:,540+j*160+44:540+j*160+116]=p6[:,540+j*72:540+(j+1)*72]
    full[:,1500+j*320+88:1500+j*320+232]=p6[:,972+j*144:972+(j+1)*144]
new=A.stacked_p6(full,46000000)
for k in ('axis_coarse','axis_fine','slab_coarse','profile','profile_fine'):
    if not np.array_equal(new[k],old[k]):raise SystemExit('crop differs: '+k)
d=A.judge10(full)['abstract']['deep_Dref_percent'];p=A.judge10(full)['abstract']['position_cm']
# Recorded in docs/validation/mv/entry/RESULTS.md, not recomputed here.
if round(d['max_abs'],2)!=.42 or round(d['RMS'],2)!=.24 or round(d['SEM_range'][0],2)!=.15 or round(d['SEM_range'][1],2)!=.26 or p['max_abs']>.00105:
    raise SystemExit('recorded verdict not reproduced')
print('PASS',d,p)
