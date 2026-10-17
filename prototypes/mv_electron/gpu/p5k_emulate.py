"""Transition-count model: report-only, not a time bound or acceptance criterion."""
import numpy as np


def utilization(sequences,R,refill=True,grouped=True,histories=None):
    if not sequences or any(not len(s) for s in sequences):raise ValueError('nonempty traces required')
    n=len(sequences) if histories is None else histories
    lengths=np.array([len(sequences[i%len(sequences)]) for i in range(n)],np.int64)
    offsets=np.r_[0,np.cumsum([len(s) for s in sequences])[:-1]]
    data=np.concatenate(sequences).astype(np.int8)
    owner=np.arange(R,dtype=np.int64);pos=np.zeros(R,np.int64);next_id=R
    transitions=warps=0
    while np.any(owner<n):
        valid=owner<n;idx=np.flatnonzero(valid);ids=owner[idx]
        phase=data[offsets[ids%len(sequences)]+pos[idx]]
        if grouped:
            counts=np.bincount(phase,minlength=4);warps+=int(((counts+31)//32).sum())
        else:
            for w in range((R+31)//32):warps+=len(np.unique(phase[idx//32==w]))
        transitions+=len(idx);pos[idx]+=1
        ended=idx[pos[idx]>=lengths[ids]]
        if refill:
            owner[ended]=np.arange(next_id,next_id+len(ended));next_id+=len(ended);pos[ended]=0
        else:owner[ended]=n
    expected=int(lengths.sum()) if refill else int(lengths[:min(n,R)].sum())
    if transitions!=expected:raise AssertionError('lost transitions')
    return dict(transitions=transitions,warp_calls=warps,utilization=transitions/(32*warps),conserved=True)


def fixed(sequences):
    reports=[utilization(sequences[i:i+32],32,False,False) for i in range(0,len(sequences),32)]
    t=sum(r['transitions'] for r in reports);w=sum(r['warp_calls'] for r in reports)
    return dict(transitions=t,warp_calls=w,utilization=t/(32*w),conserved=True)


def reproduce(cpu,n=4096):
    sequences=[];overflow=0;max_records=0
    for start in range(0,n,16):
        _,trace,meta=cpu.histories(min(16,n-start),'P5j',7000+start,capacity=8192)
        overflow+=int(meta[:,1].sum());max_records=max(max_records,int(meta[:,0].max()))
        for tr,m in zip(trace,meta):
            v=tr[:int(m[0])];sequences.append(v[(v[:,0]>=0)&(v[:,0]<4),0].astype(np.int8))
    if overflow:raise ValueError('trace capacity insufficient')
    all_phases=np.concatenate(sequences);shares=np.bincount(all_phases,minlength=4)/len(all_phases)
    C=sum(int(np.count_nonzero(s[:-1]==1)) for s in sequences)
    toC=sum(int(np.count_nonzero((s[:-1]==1)&(s[1:]==1))) for s in sequences)
    toE=sum(int(np.count_nonzero((s[:-1]==1)&(s[1:]==2))) for s in sequences)
    return dict(formal=False,histories=n,start=7000,overflow=overflow,max_trace_records=max_records,
        mean_transitions=len(all_phases)/n,max_transitions=max(map(len,sequences)),phase_share=shares.tolist(),
        C_to_C=toC/C,C_to_E=toE/C,fixed=fixed(sequences),
        refill={str(R):utilization(sequences,R,True,False) for R in (32,256)},
        pools={f'{R}/{per}':utilization(sequences,R,histories=R*per) for R in (256,512,1024) for per in (10,40)},
        note='Repeated cyclic trace sample for pools larger than sample; transitions equally weighted; not a speed bound')


if __name__=='__main__':
    import argparse
    from pathlib import Path
    from . import p5k,p5b_core as core
    from .p5a import load_inputs
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--histories',type=int,default=4096)
    args=parser.parse_args()
    if args.histories<=0:parser.error('positive history count required')
    cpu=p5k.CPU(args.work/'build',load_inputs(full=True))
    core.write_json(args.work/'emulation.json',reproduce(cpu,args.histories))
