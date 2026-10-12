"""Half-open analytic boxes; losses are fractions of the original segment."""
import numpy as np
from numba import njit


@njit
def clip(p0,p1,lo,hi):
    t0=0.;t1=1.
    for j in range(3):
        d=p1[j]-p0[j]
        if d==0:
            if p0[j]<lo[j] or p0[j]>=hi[j]:return 1.,0.
        else:
            a=(lo[j]-p0[j])/d;b=(hi[j]-p0[j])/d
            t0=max(t0,min(a,b));t1=min(t1,max(a,b))
    return t0,t1


@njit
def score_box_point(score,p,loss,axis,origin,width,lo,hi):
    for j in range(3):
        if p[j]<lo[j] or p[j]>=hi[j]:return 0.
    i=int(np.floor((p[axis]-origin)/width))
    if 0<=i<len(score):
        score[i]+=loss
        return loss
    return 0.


@njit
def score_box_bins(score,p0,p1,loss,axis,origin,width,lo,hi):
    if np.all(p0==p1):return score_box_point(score,p0,loss,axis,origin,width,lo,hi)
    t0,t1=clip(p0,p1,lo,hi)
    if t0>=t1:return 0.
    d=p1[axis]-p0[axis]
    if d==0:
        i=int(np.floor((p0[axis]-origin)/width))
        if 0<=i<len(score):
            amount=loss*(t1-t0);score[i]+=amount
            return amount
        return 0.
    q0=p0[axis]+t0*d;q1=p0[axis]+t1*d
    first=max(0,int(np.floor((min(q0,q1)-origin)/width)))
    last=min(len(score)-1,int(np.floor((max(q0,q1)-origin)/width)))
    total=0.
    for i in range(first,last+1):
        a=(origin+i*width-p0[axis])/d;b=(origin+(i+1)*width-p0[axis])/d
        amount=loss*max(0.,min(t1,max(a,b))-max(t0,min(a,b)))
        score[i]+=amount;total+=amount
    return total
