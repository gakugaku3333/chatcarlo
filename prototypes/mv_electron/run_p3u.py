"""P3u fresh one-/two-process timing. Run only after the U1–U10 gate."""
import json
import os
from pathlib import Path
import numpy as np
from . import run_p3a as a
from .batch_runner_p3u import run_batches, same

CACHE=Path(os.environ.get('CHATCARLO_P3U_CACHE','~/.cache/chatcarlo/p3u')).expanduser()
OUT=a.ROOT/'docs/validation/mv/p3u'
N=2000000
SEED=20291630


def main():
    works=[CACHE/f'timing_w{w}' for w in (1,2)]
    if any(work.exists() for work in works):raise SystemExit('timing work folder already exists; no rerun permitted')
    params=dict(E=10.,backend='elsepa',brems_mode=2,spectrum='mohan')
    rows=[a.load(run_batches(work,'p3t_slab',params,N,SEED,w,nbatch=100))
          for work,w in zip(works,(1,2))]
    if not all(bool(row['fresh']) for row in rows):raise SystemExit('timing requires two fresh runs')
    if not same(*rows):raise SystemExit('T1 identity failed')
    ratio=float(rows[0]['wall_seconds']/rows[1]['wall_seconds'])
    result=dict(N=N,seed=SEED,nbatch=100,T1='合格',
                T2=dict(ratio=ratio,limit=1.7,status='合格' if ratio>=1.7 else '不合格'),
                timing={f'w{w}':dict(wall_seconds=float(row['wall_seconds']),
                    us_per_history=float(np.sum(row['batch_seconds'])/N*1e6),
                    max_rss=float(row['max_rss']),
                    config=json.loads((work/'config.json').read_text()))
                    for w,row,work in zip((1,2),rows,works)},
                max_rss_unit='bytes on macOS, KiB on Linux')
    a.write(OUT/'timing.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if ratio<1.7:raise SystemExit('T2 failed; no rerun permitted')


if __name__=='__main__':main()
