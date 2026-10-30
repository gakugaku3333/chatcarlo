"""Replay all 16 registered states with unchanged SEED2/start/cap/maxsteps."""
import argparse
import tempfile
from pathlib import Path
import numpy as np
import a2_common as common
from chatcarlo.gpu.therapy import CPU
from prototypes.mv_electron.gpu import p8g2_core as core
import a2_health


def run(work,source=common.SOURCE):
    _,m,a=common.inputs(source);a=common.no_spectrum(a)
    builds={};rows=[];total=np.zeros(80,np.uint64)
    for case in core.fixed_cases():
        cap=case['cap']
        if cap not in builds:builds[cap]=CPU(work,m,a,cap=cap)
        cpu=builds[cap];cpu.g.energy=case['state'][6]
        r=cpu.histories(1,core.SEED2,start=case['start'],maxsteps=case['maxsteps'],injected=np.array([case['state']]))
        core.check_visits(r['visits'])
        for target in case['targets']:
            if r['hits'][target]==0:raise AssertionError(case['name']+': target '+core.PATHS[target])
        failure=cap<18 or case['maxsteps']==0
        h,_=a2_health.unpack(r,cpu.g,a)
        if failure and not np.any(h['counts'][:,5:7]):
            raise AssertionError('failure not counted: '+case['name'])
        total+=r['hits'];rows.append(dict(case,hits=r['hits'].tolist(),failures=int(h['counts'][:,5:7].sum()),unresolved=float(h['totals'][:,3].sum())))
    for name,index in core.TRANSITIONS.items():
        if total[index]==0:raise AssertionError('missing transition '+name)
    for i,name in enumerate(core.PATHS):
        if total[i]==0:raise AssertionError('missing path '+name)
    report=dict(status='pass',cases=rows,paths={name:int(total[i]) for i,name in enumerate(core.PATHS)},transitions={n:int(total[i]) for n,i in core.TRANSITIONS.items()})
    common.write(common.SUMMARY/'cases_cpu.json',report);return report


if __name__=='__main__':
    print(run(Path(tempfile.mkdtemp(prefix='a2-cases-',dir='/private/tmp')))['status'])
