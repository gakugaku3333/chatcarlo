"""Registered deterministic boundary and scoring cases, shared by CPU and CUDA."""
import numpy as np
from . import p9b_core as core

def state(p,u,r=0,kind=-1,terminal=False):
    a=np.zeros(23);a[:3]=p;a[3:6]=np.asarray(u)/np.linalg.norm(u);a[6]=a[7]=1.;a[8]=1.;a[9]=a[10]=.025;a[11]=.05;a[15]=a[22]=1e100;a[12]=.001;a[13]=.001;a[14]=.025;a[16]=r;a[18]=int(terminal);a[19]=kind
    return a

def cases(densities):
    base=dict(layers=((1,10.,.2),),densities=densities,seeds=((0.,0.,5.,.05,.25,4),))
    result=[]
    def add(name,p,u,r,want,distance,after=None,kw=None,terminal=False,kind=-1):
        a=state(p,u,r,kind,terminal);v=a[3:6] if after is None else np.asarray(after)/np.linalg.norm(after)
        result.append(dict(name=name,kw=base if kw is None else kw,state=a,direction=v,region=want,distance=distance))
    for kind in (-1,0):
        suffix='-electron' if kind else '-photon'
        add('side-entry'+suffix,(-.1,0,5),(1,0,0),0,1,.05,kind=kind)
        add('side-exit'+suffix,(0,0,5),(1,0,0),1,0,.05,kind=kind)
        add('cap-entry'+suffix,(0,-.5,5),(0,1,0),0,1,.25,kind=kind)
        add('cap-exit'+suffix,(0,0,5),(0,1,0),1,0,.25,kind=kind)
        add('tangent'+suffix,(-.05,0,4.9),(0,0,1),0,-1,5.1,kind=kind)
    add('side-entry-return',(-.1,0,5),(1,0,0),0,0,.05,after=(-1,0,0))
    add('side-exit-return',(0,0,5),(1,0,0),1,1,.05,after=(-1,0,0))
    add('cap-entry-return',(0,-.5,5),(0,1,0),0,0,.25,after=(0,-1,0))
    add('cap-exit-return',(0,0,5),(0,1,0),1,1,.25,after=(0,-1,0))
    for u,want in (((1,1,0),1),((1,-1,0),0),((-1,1,0),0)):
        add('rim-entry-'+str(u),(-.1,-.3,5),(1,1,0),0,want,.05*np.sqrt(2),after=u)
    for u,want in (((-1,-1,0),1),((1,-1,0),0),((-1,1,0),0)):
        add('rim-exit-'+str(u),(0,.2,5),(1,1,0),1,want,.05*np.sqrt(2),after=u)
    add('terminal-entry',(-.1,0,5),(1,0,0),0,1,.05,terminal=True)
    # Starting on either side of a boundary must not discard sub-nanometre entries.
    for x in (-.05,np.nextafter(-.05,-np.inf),np.nextafter(-.05,np.inf),-.05-5e-10):
        add('restart-entry-'+repr(x),(x,0,5),(1,0,0),0,1,max(0.,-.05-x))
        add('restart-entry-return-'+repr(x),(x,0,5),(1,0,0),0,0,max(0.,-.05-x),after=(-1,0,0))
    for x in (.05,np.nextafter(.05,-np.inf),np.nextafter(.05,np.inf),.05-5e-10):
        add('restart-exit-'+repr(x),(x,0,5),(1,0,0),1,0,max(0.,.05-x))
        add('restart-exit-return-'+repr(x),(x,0,5),(1,0,0),1,1,max(0.,.05-x),after=(-1,0,0))
    for face in (-.25,.25):
        sign=1 if face>0 else -1
        for y in (face,np.nextafter(face,-np.inf),np.nextafter(face,np.inf),face+sign*5e-10):
            add('cap-restart-entry-'+repr(y),(0,y,5),(0,-sign,0),0,1,max(0.,sign*(y-face)))
            add('cap-restart-entry-return-'+repr(y),(0,y,5),(0,-sign,0),0,0,max(0.,sign*(y-face)),after=(0,sign,0))
        for y in (face,np.nextafter(face,-np.inf),np.nextafter(face,np.inf),face-sign*5e-10):
            add('cap-restart-exit-'+repr(y),(0,y,5),(0,sign,0),1,0,max(0.,sign*(face-y)))
            add('cap-restart-exit-return-'+repr(y),(0,y,5),(0,sign,0),1,1,max(0.,sign*(face-y)),after=(0,-sign,0))
    for k in range(3):
        kw=dict(layers=((1,5.,.2),(5,5.,.2)),densities=densities,axis=k,half=(2.,2.,2.))
        p=np.zeros(3);p[k]=4.;u=np.zeros(3);u[k]=1.
        for kind in (-1,0):add(f'layer-{k}-{kind}',p,u,0,1,1.,kw=kw,kind=kind)
        add(f'layer-return-{k}',p,u,0,0,1.,after=-u,kw=kw)
        pback=np.zeros(3);pback[k]=6.
        for kind in (-1,0):add(f'layer-backward-{k}-{kind}',pback,-u,1,0,1.,kw=kw,kind=kind)
        add(f'layer-backward-return-{k}',pback,-u,1,1,1.,after=u,kw=kw)
        for coord in (5.,np.nextafter(5.,-np.inf),np.nextafter(5.,np.inf),5.-5e-10):
            restart=np.zeros(3);restart[k]=coord;distance=max(0.,5.-coord)
            for kind in (-1,0):add(f'layer-restart-{k}-{coord!r}-{kind}',restart,u,0,1,distance,kw=kw,kind=kind)
            add(f'layer-restart-return-{k}-{coord!r}',restart,u,0,0,distance,after=-u,kw=kw)
        corner=np.ones(3);corner[k]=9.
        for kind in (-1,0):add(f'box-three-face-{k}-{kind}',corner,(1,1,1),1,-1,np.sqrt(3),kw=kw,kind=kind)

        for lateral in range(3):
            if lateral==k:continue
            for sign in (-1,1):
                p=np.zeros(3);p[k]=1.;u=np.zeros(3);u[lateral]=sign
                for kind in (-1,0):add(f'box-exit-{k}-{lateral}-{sign}-{kind}',p,u,0,-1,2.,kw=kw,kind=kind)
        # Simultaneous layer+box edge: any outgoing box face escapes.
        p=np.zeros(3);p[k]=4.;j=(k+1)%3;p[j]=1.;u=np.zeros(3);u[k]=u[j]=1.
        add(f'box-corner-{k}',p,u,0,-1,np.sqrt(2),kw=kw)
        after=np.zeros(3);after[k]=-1.;after[j]=1.
        add(f'box-corner-return-layer-{k}',p,u,0,-1,np.sqrt(2),after=after,kw=kw)
    kw=dict(layers=((1,5.,.2),(1,5.,.2)),densities=densities,seeds=((.5,0,2.,.05,.25,4),(-.5,0,2.,.05,.25,4),(0,0,7.,.05,.25,4)))
    add('nearest-not-registration',(-1,0,2),(1,0,0),0,3,.45,kw=kw)
    add('ignore-other-layer',(0,0,1),(0,0,1),0,1,4.,kw=kw)
    return result

def rejections(densities):
    base=dict(layers=((1,10.,.2),),densities=densities)
    seeds=[((0,0,0,.05,.25,4),),((0,0,5,.05,.25,4),(0,0,5,.05,.25,4)),((0,0,.05,.05,.25,4),),((0,0,5,.05,.25,4),(.10001,0,5,.05,.25,4))]
    trials=[dict(base,seeds=s) for s in seeds]
    trials += [dict(base,layers=((1,5.,.2),(1,5.,.2)),seeds=((0,0,5,.05,.25,4),)),dict(base,layers=((1,33.,.2),)),dict(base,layers=((1,30.,.1),))]
    for kw in trials:
        try:core.geometry(**kw)
        except ValueError:continue
        raise AssertionError('G3 construction accepted invalid geometry')
    return len(trials)

def check_case(case,out):
    if int(out[39])!=case['region']:raise AssertionError(f"{case['name']}: region {out[39]} != {case['region']}")
    if abs(out[46]-case['distance'])>1e-12:raise AssertionError(f"{case['name']}: distance {out[46]} != {case['distance']}")
    pos=case['state'][:3]+case['state'][3:6]*case['distance']
    if np.max(abs(out[23:26]-pos))>1e-12:raise AssertionError(case['name']+': position')
    expected=-1 if case['region']<0 else (4 if case['region']>=len(case['kw']['layers']) else case['kw']['layers'][case['region']][0]-1)
    if int(out[48])!=expected:raise AssertionError(case['name']+': material')
    if case['state'][19]!=0:
        changed=(out[47]>=0 and case['kw']['layers'][int(case['state'][16])][0]-1!= (4 if out[47]>=len(case['kw']['layers']) else case['kw']['layers'][int(out[47])][0]-1)) if int(case['state'][16])<len(case['kw']['layers']) else True
        if out[49]!=int(changed):raise AssertionError(case['name']+': actual property refresh count')
    if not np.array_equal(out[:23],case['state']):raise AssertionError('probe redrew input')

def g3(work,materials,densities,factory=None):
    from .p9b import CPU
    factory=(lambda kw:CPU(work,materials,**kw)) if factory is None else factory
    rows=[];cache={}
    for case in cases(densities):
        key=repr(case['kw'])
        if key not in cache:cache[key]=factory(case['kw'])
        engine=cache[key];out=engine.probe([case['state']],[case['direction']])[0];check_case(case,out)
        # Repeat the actual scattering/RNG path, with identical state and seed.
        a=engine.probe([case['state']],mode=1)[0];b=engine.probe([case['state']],mode=1)[0]
        if not np.array_equal(a,b):raise AssertionError('G3 random probe nondeterministic')
        rows.append(dict(name=case['name'],region=int(out[39]),distance=float(out[46]),calls=int(a[50])))
    # A photon traverses the seed in two successive invocations, no source redraw.
    engine=factory(dict(layers=((1,10.,.2),),densities=densities,seeds=((0.,0.,5.,.05,.25,4),)))
    a=state((-.1,0,5),(1,0,0),kind=0);r1=engine.probe([a])[0];r2=engine.probe([r1[23:46]])[0]
    if (int(r1[39]),int(r2[39]))!=(1,0) or r1[49]+r2[49]!=2:raise AssertionError('photon material refresh')
    return dict(status='pass',cases=len(rows),rejections=rejections(densities),photon_recomputes=int(r1[49]+r2[49]),rows=rows)
