"""G5 A2b seed_ratio entry (public since B3 passed, 2026-11-03)."""
import argparse
import json
from pathlib import Path
from . import seed_scene,seed_condition,seed_bundle,seed_preview,contracts
from .condition import sha256


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('validate','preview','approve','run','analyse'));p.add_argument('scene',type=Path)
    p.add_argument('--bundle',type=Path);p.add_argument('--out',type=Path);p.add_argument('--preview',type=Path)
    a=p.parse_args(argv);scene=seed_scene.load(a.scene)
    if a.command=='validate':print('OK: A2b seed_ratio / 50 batches');return 0
    if a.bundle is None:p.error('--bundle required')
    inputs=seed_bundle.load(a.bundle);cond=seed_condition.condition(scene,inputs[-1],*inputs[:2])
    out=a.preview if a.command=='approve' else a.out
    if out is None:p.error('--out (or --preview for approve) required')
    if a.command=='preview':seed_preview.prepare(out,cond,a.bundle)
    elif a.command=='approve':
        if seed_condition.verify(out)['condition']!=cond:raise ValueError('condition mismatch')
        contracts.approve(out,seed_condition.verify)
    else:
        contracts.require(out,cond,seed_condition.verify)
        from . import seed_colab,seed_run
        if a.command=='run':seed_colab.run(out,cond,*inputs[:2])
        seed_run.collect(out/'results',cond)
    return 0

if __name__=='__main__':raise SystemExit(main())
