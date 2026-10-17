"""Equal-transition-cost warp-pool model, report only; never predicts GPU time."""
import numpy as np


def utilization(sequences,m=1,K=1,histories=None):
    if m not in (1,2,4) or K not in (1,2,4,8):raise ValueError('unsupported m/K')
    if not sequences or any(len(s)==0 for s in sequences):raise ValueError('nonempty traces required')
    n=len(sequences) if histories is None else histories
    if n<=0:raise ValueError('positive history count required')
    R=32*m;owner=list(range(min(R,n)));pos=[0]*len(owner);next_id=len(owner)
    transitions=cost=calls=tails=0
    expected=sum(len(sequences[i%len(sequences)]) for i in range(n))
    while owner:
        snapshot=[int(sequences[h%len(sequences)][k]) for h,k in zip(owner,pos)]
        advance=[1]*len(owner)
        for phase in range(4):
            members=[j for j,p in enumerate(snapshot) if p==phase]
            for first in range(0,len(members),32):
                group=members[first:first+32];runs=[];calls+=1;tails+=len(group)<32
                for j in group:
                    s=sequences[owner[j]%len(sequences)];k=pos[j];count=1
                    if phase==1:
                        while count<K and k+count<len(s) and s[k+count]==1:count+=1
                    runs.append(count);advance[j]=count
                cost+=max(runs)
        new_owner=[];new_pos=[]
        for h,k,a in zip(owner,pos,advance):
            k+=a;transitions+=a
            if k<len(sequences[h%len(sequences)]):new_owner.append(h);new_pos.append(k)
            elif next_id<n:new_owner.append(next_id);new_pos.append(0);next_id+=1
        owner,pos=new_owner,new_pos
    if transitions!=expected:raise AssertionError('lost transitions')
    return dict(transitions=transitions,warp_steps=cost,warp_calls=calls,tail_groups=tails,
                utilization=transitions/(32*cost),conserved=True)


def reproduce(cpu,n=4096):
    sequences=[]
    for first in range(0,n,16):
        _,trace,meta=cpu.histories(min(16,n-first),'P5j',7000+first,capacity=8192)
        if np.any(meta[:,1]) or np.any(meta[:,0]>8192):raise ValueError('trace overflow')
        for tr,mt in zip(trace,meta):
            rows=tr[:int(mt[0])];sequences.append(rows[(rows[:,0]>=0)&(rows[:,0]<4),0].astype(np.int8))
    rng=np.random.default_rng(0);reports={}
    for m in (1,2,4):
        for K in (1,2,4,8):
            samples=[utilization([sequences[i] for i in rng.choice(n,32*m*10,replace=False)],m,K) for _ in range(4)]
            reports[f'{m}/{K}']=dict(mean=float(np.mean([s['utilization'] for s in samples])),
                std=float(np.std([s['utilization'] for s in samples])),samples=samples)
    return dict(formal=False,histories=n,start=7000,repeats=4,seed=0,results=reports,
                note='Equal transition costs, not a speed prediction; numeric reproduction report only.')


if __name__=='__main__':
    import argparse
    from pathlib import Path
    from . import p5l,p5b_core as core
    from .p5a import load_inputs
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--histories',type=int,default=32768);a=parser.parse_args()
    core.write_json(a.work/'emulation.json',reproduce(p5l.CPU(a.work/'build',load_inputs(full=True)),a.histories))
