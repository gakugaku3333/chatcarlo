"""Entry verification build: source replay, split accumulation and fixed scoring."""
import ctypes as ct
import numpy as np
from . import contracts, therapy_health
from .therapy import D, FieldGeometry


def source_pair(cpu,gpu):
    if cpu['sources'].shape!=gpu['sources'].shape or not np.allclose(cpu['sources'],gpu['sources'],rtol=2e-14,atol=2e-14):raise ValueError('CPU/GPU source state')
    # The energy sample and spectrum index do not depend on transcendental math.
    f=cpu['d'].shape[1]-37
    if not np.array_equal(cpu['d'][:,f+21],gpu['d'][:,f+21]):raise ValueError('CPU/GPU source energy')
    if not np.array_equal(cpu['i'][:,75],gpu['i'][:,75]):raise ValueError('CPU/GPU RNG calls')


def scorer(driver,fg):
    rows=[];wanted=[]
    # One point at each scoring-region centre, with an independently binned oracle.
    for j in range(15):
        p=np.array([(fg.lo[3*j+k]+fg.hi[3*j+k])/2 if np.isfinite(fg.lo[3*j+k]+fg.hi[3*j+k]) else 0. for k in range(3)])
        if j<3:p[2]=1.125
        rows.append([0.,1.,*p,*p,.5])
        expected=np.zeros(3420)
        for q in range(15):
            if all(fg.lo[3*q+k]<=p[k]<fg.hi[3*q+k] for k in range(3)):
                index=int(np.floor((p[fg.axes[q]]-fg.origins[q])/fg.widths[q]))
                if 0<=index<fg.sizes[q]:expected[fg.offsets[q]+index]=.5
        wanted.append(expected)
    rows=np.array(rows);want=np.array(wanted)
    if hasattr(driver,'score'):actual,sums=driver.score(rows)
    else:
        actual=np.zeros_like(want);sums=np.zeros((len(rows),15))
        driver.lib.t2_score.argtypes=[ct.POINTER(FieldGeometry),D,ct.c_int,D,D]
        driver.lib.t2_score(ct.byref(fg),rows,len(rows),actual,sums)
    if not np.array_equal(actual,want):raise ValueError('fixed-deposit scorer')
    for j,(o,n) in enumerate(zip(fg.offsets,fg.sizes)):
        if not np.array_equal(actual[:,o:o+n].sum(axis=1),sums[:,j]):raise ValueError('fixed-deposit ledger')


def run(cpu,arrays,fg,seed,gpu=None):
    whole=cpu.histories(2048,seed,slots=256)
    therapy_health.check(whole,arrays,fg)
    parts=[cpu.histories(256,seed,start=b*256,slots=256) for b in range(8)]
    therapy_health.split_check(whole,parts);scorer(cpu,fg)
    if gpu is not None:
        r=gpu.run(2048,seed,history=1)
        therapy_health.check(r,arrays,fg);source_pair(whole,r)
        parts=[gpu.run(256,seed,start=b*256) for b in range(8)]
        therapy_health.split_check(r,parts);scorer(gpu,fg)
    return dict(status='pass')
