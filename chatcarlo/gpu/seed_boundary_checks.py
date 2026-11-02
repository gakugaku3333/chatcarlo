import numpy as np

def state(p,u,r=0,kind=-1,terminal=False):
    a=np.zeros(23);a[:3]=p;a[3:6]=np.asarray(u)/np.linalg.norm(u);a[6]=a[7]=1.;a[8]=1.;a[9]=a[10]=.025;a[11]=.05;a[15]=a[22]=1e100;a[12]=.001;a[13]=.001;a[14]=.025;a[16]=r;a[18]=int(terminal);a[19]=kind
    return a

def reference_cases(densities):
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

def check(engine_factory,densities):
    rows=[]
    for c in reference_cases(densities):
        if c['name'].startswith(('layer','box','nearest','ignore')):continue
        kw=dict(c['kw']);kw['cylinders']=kw.pop('seeds')
        e=engine_factory(kw);out=e.probe([c['state']],[c['direction']])[0]
        if int(out[39])!=c['region'] or abs(out[46]-c['distance'])>1e-12:raise AssertionError('G3 '+c['name'])
        want=c['state'][:3]+c['distance']*c['state'][3:6]
        np.testing.assert_allclose(out[23:26],want,rtol=0,atol=1e-12)
        if not np.array_equal(out[:23],c['state']):raise AssertionError('probe redrew clocks')
        if c['state'][19]==0 and out[47]>=0 and out[49]!=1:raise AssertionError('photon property rebuild')
        # Electrons rebuild on every crossing (even a return); no crossing (tangent) means no rebuild, as in P9b.
        if c['state'][19]!=0 and out[49]!=(1 if out[47]>=0 else 0):raise AssertionError('electron property rebuild even on return')
        rows.append(c['name'])
    return dict(status='pass',cases=rows)
