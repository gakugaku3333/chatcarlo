"""Single common analysis for EGS5, prototype and every sensitivity curve."""
import numpy as np
from .component_checks import SEED
def dref(q):
    return max(np.mean(q[max(0,i-2):min(len(q),i+3)]) for i in range(len(q)))
def ranges(q,z,D):
    peak=int(np.argmax(q))
    r50=np.nan
    for i in range(peak,len(q)-1):
        if q[i]>=.5*D and q[i+1]<.5*D:
            r50=z[i]+(.5*D-q[i])*(z[i+1]-z[i])/(q[i+1]-q[i]);break
    # Descending contiguous 20--80% segment: longest, first in ties.
    runs=[];start=None
    for i in range(peak,len(q)+1):
        eligible=i<len(q) and .2*D<=q[i]<=.8*D
        if eligible and start is None:start=i
        if not eligible and start is not None:runs.append(np.arange(start,i));start=None
    if not runs:return np.array([r50,np.nan])
    idx=max(runs,key=len)
    if len(idx)<2:return np.array([r50,np.nan])
    slope,intercept=np.polyfit(z[idx],q[idx],1)
    return np.array([r50,-intercept/slope if slope<0 else np.nan])
def compare(ref,other,z,tol=.01,range_tol=.01,mask=None,bootstrap=2000,range_indices=(0,1)):
    rq=ref.mean(0);oq=other.mean(0);D=dref(rq)
    if mask is None:mask=np.ones(len(rq),bool)
    r=ranges(rq,z,D);o=ranges(oq,z,D)
    sem=np.sqrt(ref.var(0,ddof=1)/len(ref)+other.var(0,ddof=1)/len(other))
    rng=np.random.default_rng(SEED);diffs=np.empty((bootstrap,2))
    for i in range(bootstrap):
        rb=ref[rng.integers(len(ref),size=len(ref))].mean(0);ob=other[rng.integers(len(other),size=len(other))].mean(0)
        # One common reference realization sets the threshold of both curves.
        Db=dref(rb);diffs[i]=ranges(ob,z,Db)-ranges(rb,z,Db)
    rs=np.nanstd(diffs,axis=0,ddof=1)
    dose=float(max(abs(oq[mask]-rq[mask]))/D);ds=float(max(sem[mask])/D)
    rd=(o-r)/r[0];rss=rs/r[0]
    stats=bool(ds<=tol/3 and np.all(rss[list(range_indices)]<=range_tol/3) and np.all(np.isfinite(rs)))
    passed=bool(dose<=tol and np.all(abs(rd[list(range_indices)])<=range_tol) and stats)
    return {'Dref':float(D),'reference_ranges_cm':r.tolist(),'other_ranges_cm':o.tolist(),'range_difference_over_reference_R50':rd.tolist(),'range_difference_SEM_over_R50':rss.tolist(),'max_dose_difference_over_Dref':dose,'max_dose_difference_SEM_over_Dref':ds,'evaluation_bins':int(sum(mask)),'statistics_ok':stats,'pass':passed}
