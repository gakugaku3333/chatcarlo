"""Reporting-only 60 keV comparison. Never launches EGS5."""
import argparse
import json
from pathlib import Path
import numpy as np
import a1_common
from chatcarlo.gpu import bundle,physics_sets
from a1_common import HERE,ROOT,CACHE,module,space

if (HERE.parent/'g2/run_egs5_60kev.py').exists():
    legacy=module('a1_g2_60kev',HERE.parent/'g2/run_egs5_60kev.py')
else:
    from chatcarlo.gpu import run_egs5_60kev as legacy
SEALED={
 'pdd_phantom':'ce5ecad01105262099f7fa1dcc7201b22dc833eaa7745183932b619f889fc5c7',
 'bsf_thinslab':'734c6ebee4cf1f674b8d0541acb091c2fe7590a0a5d24391f48b2f99213bba45',
 'bsf_phantom':'3892284ba6f133089edcf8f286ca876042604832519d99b8ad46d6b744fec13b'}
CONDITIONS={'pdd':(3101,5000000),'thin':(3102,500000),'phantom':(3103,500000)}


def verify_inputs(out):
    for name,digest in SEALED.items():
        if bundle.sha(legacy.OLD/'egs5'/name/'egs5job.out')!=digest:raise ValueError('EGS5 SHA256 mismatch: '+name)
    for name,(seed,n) in CONDITIONS.items():
        _,_,_,settings,manifest=bundle.load(Path(out)/name)
        physics_sets.check(settings['physics_set'],physics_sets.declaration())
        with np.load(Path(out)/name/'g2_tables.npz') as z:
            if (int(z['seed']),int(z['n_histories']))!=(seed,n):raise ValueError('scenario configuration mismatch: '+name)
        r=json.loads((Path(out)/(name+'.json')).read_text())
        d=r['results']['diagnostics']
        if (d['seed'],d['n_histories'])!=(seed,n) or r['manifest']['bundle']!=manifest:
            raise ValueError('result identity mismatch: '+name)


def prepare(out):
    space(out);legacy.prepare(out)


def compare(out):
    verify_inputs(out);legacy.compare(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','execute','compare'])
    p.add_argument('--out',type=Path,default=CACHE/'g2gate3');p.add_argument('--bundle',type=Path)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.out)
    elif a.mode=='compare':compare(a.out)
    else:legacy.execute(a.bundle,a.out)
