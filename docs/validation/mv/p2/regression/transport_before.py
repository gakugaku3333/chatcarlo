"""P1 slab scalar history transport, dual random hinge; units MeV/cm.
Eval energy differs from ledger energy. Residual LOSS and STRENGTH survive surfaces.
A stack frame contains complete parent state; secondaries are processed first.
"""
import numpy as np
from numba import njit
from .physics import *
from .sampling import gs_mu_cached
from .backend_sampling import table_rates,table_single
CUT=.01
FIELDS=19
@njit
def energy_init(a,fe):
    a[7]=a[6];a[11]=min(fe*a[7],max(0.,a[7]-CUT))
    a[10]=np.random.random()*a[11];a[8]=a[10];a[9]=a[11]-a[10];a[18]=0.
@njit
def initial(T,z,ux,uy,uz,region,primary,fe,k):
    a=np.zeros(FIELDS);a[2]=z;a[3]=ux;a[4]=uy;a[5]=uz;a[6]=T;a[17]=primary;a[16]=region
    energy_init(a,fe);a[14]=k;a[12]=np.random.random()*k;a[13]=k-a[12];a[15]=-np.log(np.random.random())
    return a
@njit
def boundary(z,w,region,faces):
    if w>0:return max(0.,(faces[region+1]-z)/w),region+1
    if w<0:return max(0.,(faces[region]-z)/w),region-1
    return np.inf,region
@njit
def score_segment(score,z0,z1,loss,dz):
    if loss==0:return
    if abs(z1-z0)<1e-15:
        b=max(0,min(len(score)-1,int(z0/dz)));score[b]+=loss;return
    lo=min(z0,z1);hi=max(z0,z1);width=hi-lo
    b0=max(0,min(len(score)-1,int(lo/dz)));b1=max(0,min(len(score)-1,int(hi/dz)))
    for b in range(b0,b1+1):
        length=max(0.,min(hi,(b+1)*dz)-max(lo,b*dz))
        score[b]+=loss*length/width
@njit
def local(score,z,loss,dz):
    b=max(0,min(len(score)-1,int(z/dz)));score[b]+=loss
@njit
def run(T,n,seed,fe,k,faces,dz,ts,ks,us,ys,scoring=True,hard_on=True,mode=0,maxstack=4096,maxsteps=10000000,source_state=None,ignore_boundaries=False,backend_rates=None,backend_single=None):
    # mode: 0 GS dual hinge; 1 single elastic collisions, same loss model; 2 no scattering.
    np.random.seed(seed);nb=int(round(faces[-1]/dz));nbatch=100
    batches=np.zeros((nbatch,nb));ledger=np.zeros((n,7));counts=np.zeros(7,np.int64)
    cross_sum=np.zeros(2);cross_sum2=np.zeros(2);cross_n=np.zeros(2,np.int64)
    stack=np.empty((maxstack,FIELDS));maxdepth=0
    for h in range(n):
        stack[0]=initial(T,0.,0.,0.,1.,0,1,fe,k) if source_state is None else source_state
        if mode==1:stack[0,12]=-np.log(np.random.random())
        top=0;dep=0.;back=0.;forward=0.;steps=0;failed=False
        hs=np.zeros(nb);seen=np.zeros(2,np.int64)
        while top>=0:
            a=stack[top].copy();top-=1;cached_ev=-1.
            eta=np.zeros(2);totalj=np.zeros(2)
            while True:
                steps+=1
                if steps>maxsteps:
                    failed=True;counts[5]+=1;break
                region=int(a[16])
                if region<0 or region>=len(faces)-1:
                    if region<0:back+=a[6]
                    else:forward+=a[6]
                    break
                if a[6]<=CUT+1e-14 and a[18]!=1:
                    if scoring:local(hs,a[2],a[6],dz)
                    dep+=a[6];a[6]=0.;break
                ev=max(CUT,a[7])
                if ev!=cached_ev:
                    S=stopping(ev,hard_on);rate=hard(ev)[0] if hard_on else 0.
                    if backend_rates is None:
                        total,G1=elastic_rates(ev)
                    else:
                        total,G1=table_rates(ev,ts,backend_rates)
                    if mode==1:
                        if backend_rates is None:eta,B,totalj,tmp=elastic(ev)
                        G1=total
                    if mode==2:G1=0.
                    cached_ev=ev
                # Recompute all event distances after every direction/energy update.
                eb=a[8]/S
                mb=a[12]/G1 if G1>0 else np.inf
                hb=a[15]/rate if rate>0 else np.inf
                if a[18]==1:mb=np.inf;hb=np.inf
                if ignore_boundaries:bb=np.inf;nr=region
                else:
                    bb,nr=boundary(a[2],a[5],region,faces);counts[4]+=1
                step=min(eb,mb,hb,bb,a[6]/S)
                if not np.isfinite(step) or step<0:
                    failed=True;counts[6]+=1;break
                z0=a[2];a[0]+=step*a[3];a[1]+=step*a[4];a[2]+=step*a[5]
                loss=step*S;a[6]-=loss;dep+=loss
                if scoring:score_segment(hs,z0,a[2],loss,dz)
                a[8]=max(0.,a[8]-loss);a[12]=max(0.,a[12]-step*G1);a[15]=max(0.,a[15]-step*rate)
                counts[0]+=1
                if a[17]==1:
                    for j in range(2):
                        plane=.2 if j==0 else .5
                        if not seen[j] and z0<plane and a[2]>=plane and a[5]>0:
                            r=(plane-z0)/(a[2]-z0);xx=a[0]-(1-r)*step*a[3];yy=a[1]-(1-r)*step*a[4];v=xx*xx+yy*yy
                            cross_sum[j]+=v;cross_sum2[j]+=v*v;cross_n[j]+=1;seen[j]=1
                # Deterministic ties: boundary, energy, scattering, hard. Consume zero-distance events.
                if bb<=step+1e-13:
                    a[2]=faces[region+1] if a[5]>0 else faces[region];a[16]=nr;counts[3]+=1
                    continue
                if eb<=step+1e-13:
                    if a[18]==1:
                        if scoring:local(hs,a[2],a[6],dz)
                        dep+=a[6];a[6]=0.;break
                    a[7]=max(CUT,a[7]-a[11])
                    if a[7]<=CUT+1e-14:
                        a[8]=a[9];a[9]=0.;a[10]=0.;a[18]=1.;continue
                    prev=a[9];a[11]=min(fe*a[7],a[7]-CUT);a[10]=np.random.random()*a[11]
                    a[9]=a[11]-a[10];a[8]=prev+a[10];continue
                if mb<=step+1e-13:
                    if mode==1:
                        mu=single_cached(eta,totalj) if backend_rates is None else table_single(ev,ts,us,backend_single)
                    else:
                        mu=gs_mu_cached(ev,a[14],ts,ks,us,ys,total,G1)
                    a[3],a[4],a[5]=rotate(a[3],a[4],a[5],mu,2*np.pi*np.random.random());counts[1]+=1
                    if mode==1:
                        a[12]=-np.log(np.random.random());a[13]=0.;a[14]=0.
                    else:
                        u=np.random.random();a[12]=a[13]+u*k;a[13]=(1-u)*k;a[14]=k
                    continue
                if hb<=step+1e-13:
                    # Settle energy hinge against the ledger before the hard collision.
                    energy_init(a,fe);a[15]=-np.log(np.random.random())
                    if a[6]<=2*DELTA:continue
                    W=sample_W(a[6]);dp,ds=moller_directions(a[6],W,a[3],a[4],a[5],2*np.pi*np.random.random());a[6]-=W
                    a[3],a[4],a[5]=dp;energy_init(a,fe);counts[2]+=1
                    if top+2>=maxstack:
                        a[6]+=W  # Both daughters remain unresolved on stack overflow.
                        failed=True;counts[6]+=1;break
                    top+=1;stack[top]=a;top+=1
                    child=initial(W,a[2],ds[0],ds[1],ds[2],int(a[16]),0,fe,k);child[0]=a[0];child[1]=a[1]
                    if mode==1:child[12]=-np.log(np.random.random())
                    stack[top]=child;maxdepth=max(maxdepth,top+1);break
                # Actual-energy endpoint: deposit only the remaining ledger energy.
                if scoring:local(hs,a[2],a[6],dz)
                dep+=a[6];a[6]=0.;break
            if failed:break
        unresolved=0.;number=0
        if failed:
            unresolved=a[6];number=1
            for j in range(top+1):unresolved+=stack[j,6];number+=1
        ledger[h,0]=dep;ledger[h,1]=back;ledger[h,2]=forward;ledger[h,3]=T-dep-back-forward;ledger[h,4]=number;ledger[h,5]=unresolved;ledger[h,6]=steps
        batches[min(nbatch-1,h*nbatch//n)]+=hs
    return batches,ledger,counts,cross_sum,cross_sum2,cross_n,maxdepth
