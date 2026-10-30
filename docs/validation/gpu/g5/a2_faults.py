"""CPU fault witnesses. Numerical/geometrical detectors, never output byte diffs."""
import argparse
import ctypes as ct
import tempfile
from pathlib import Path
import numpy as np
import a2_common as common
import a2_health,a2_scorer
from chatcarlo.gpu import therapy
from prototypes.mv_electron.gpu import p8g2_core as core


def reject(action):
    try:action()
    except AssertionError as exc:return str(exc)
    raise AssertionError('fault escaped detector')


def source_check(r,half):
    v=r['sources'];norm=np.sqrt((v[:,:2]**2).sum(axis=1)+100.**2)
    if not np.allclose(v[:,3:5],v[:,:2]/norm[:,None],rtol=1e-15,atol=0):raise AssertionError('source coordinate/direction')
    if np.any(np.abs(v[:,0])>half[0]) or np.any(np.abs(v[:,1])>half[1]):raise AssertionError('source coordinate range')


def run(work,source=common.SOURCE):
    _,m,a=common.inputs(source);a=common.no_spectrum(a);rows=[]
    low=next(c for c in core.fixed_cases() if c['name']=='low-photon')
    flight=next(c for c in core.fixed_cases() if c['name']=='flight')
    kw=dict(field_geometry=common.field_geometry(),half=(2.5,1.))
    normal=therapy.CPU(work,m,a,**kw);normal.g.energy=.007;normal.half=(0.,0.)
    score=a2_scorer.check(normal)
    # All normal detectors first, with the same fixture as faulted builds.
    good=normal.histories(2,core.SEED2,injected=np.tile(low['state'],(2,1)))
    a2_health.check(good,normal.g,a);a2_health.ledger(good,kw['field_geometry'])
    normal.half=kw['half'];source_check(normal.histories(32,core.SEED2,maxsteps=0),kw['half']);normal.half=(0.,0.)
    for fault in (1,2,3,8,11,12,13,14,15,16):
        cpu=therapy.CPU(work,m,a,fault=fault,**kw);cpu.g.energy=.007
        if fault not in (3,8):cpu.half=(0.,0.)
        if fault==1:
            # Scorer returns a reachable wrong bin; verify numeric output differs.
            msg=reject(lambda:a2_scorer.check(cpu));target=int(np.count_nonzero(np.abs(a2_scorer.native(cpu,a2_scorer.records())[0]-a2_scorer.reference(a2_scorer.records(),kw['field_geometry']))>1e-14))
        elif fault in (3,8):
            r=cpu.histories(32,core.SEED2,maxsteps=0);target=int(r['hits'][63 if fault==3 else 68])
            msg=reject(lambda:source_check(r,kw['half']))
        elif fault==11:
            args=dict(start=flight['start'],injected=np.array([flight['state']]))
            r=cpu.histories(1,core.SEED2,**args);goodflight=normal.histories(1,core.SEED2,**args)
            gh,_=a2_health.unpack(goodflight,normal.g,a);h,_=a2_health.unpack(r,cpu.g,a)
            def one_annihilation(h):
                if int(h['counts'][0,12]+h['counts'][0,14])!=1:raise AssertionError('annihilation must occur once')
            one_annihilation(gh);msg=reject(lambda:one_annihilation(h));target=int(r['hits'][30])
            # Fixed two-photon kinematics independently checks E and momentum.
            out=np.zeros(8);normal.lib.t2_flight.argtypes=[therapy.D,ct.c_uint64,ct.c_double,therapy.D]
            normal.lib.t2_flight(normal.constants,core.SEED2,10.,out)
            M=normal.constants[0]
            if abs(out[0]+out[4]-(10.+2*M))>1e-12:raise AssertionError('annihilation energy')
            if not np.allclose(out[0]*out[1:4]+out[4]*out[5:8],[0,0,np.sqrt(10.*(10.+2*M))],rtol=1e-12,atol=1e-12):raise AssertionError('annihilation momentum')
        else:
            r=cpu.histories(2,core.SEED2,injected=np.tile(low['state'],(2,1)))
            target=int(r['hits'][{2:62,12:31,13:32,14:33,15:34,16:35}[fault]])
            msg=reject(lambda:a2_health.check(r,cpu.g,a))
        if target<=0:raise AssertionError('fault site not reached '+str(fault))
        rows.append(dict(fault=fault,target=target,detector=msg,build=cpu.identity))
    report=dict(status='pass',normal_scorer=score,faults=rows)
    common.write(common.SUMMARY/'faults_cpu.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cpu',action='store_true',required=True);p.add_argument('--work',type=Path);args=p.parse_args()
    work=args.work or Path(tempfile.mkdtemp(prefix='a2-faults-',dir='/private/tmp'))
    print(run(work)['status'])
