"""Q6 preregistered absolute-dose/position comparison (no fitted scale)."""
import numpy as np
from scipy.stats import norm

def curve_stats(b):
    return b.mean(axis=0),b.std(axis=0,ddof=1)/np.sqrt(len(b))

def peak(b,window):
    lo,hi=window;z=(np.arange(b.shape[1])+.5)*.2;ids=np.arange(lo,hi+1)
    if len(ids)<3:return None
    design=np.column_stack((z[ids]**2+.2**2/12,z[ids],np.ones(len(ids))))
    a,bb,c=np.linalg.lstsq(design,b.mean(axis=0)[ids],rcond=None)[0]
    if a>=0:return None
    vertex=-bb/(2*a)
    if not lo*.2<=vertex<=(hi+1)*.2:return None
    return float(vertex)

def bootstrap_peak_difference(a,b,window,seed):
    rng=np.random.default_rng(seed);out=[];invalid=0
    for j in range(1000):
        va=peak(a[rng.integers(0,len(a),len(a))],window)
        vb=peak(b[rng.integers(0,len(b),len(b))],window)
        if va is None or vb is None:invalid+=1
        else:out.append(va-vb)
    # Do not silently discard invalid fits to claim uncertainty adequacy.
    return dict(SEM_cm=None if invalid else float(np.std(out,ddof=1)),invalid_fits=invalid,replicates=1000)

def pilot_window(coarse,fine):
    d,se=curve_stats(coarse);df,sf=curve_stats(fine)
    smooth=np.convolve(d,np.ones(5)/5,mode='same');center=int(np.argmax(smooth));ref=float(smooth[center])
    eligible=d>=.97*ref
    if not eligible[center]:return dict(status='未評価',reason='pilot maximum-bin raw dose is outside 97% contiguous window',Dref=ref)
    lo=hi=center
    while lo>0 and eligible[lo-1]:lo-=1
    while hi<len(d)-1 and eligible[hi+1]:hi+=1
    deep=np.where((np.arange(len(d))>hi)&(d>=.2*ref))[0]
    near=np.arange(lo,hi+1)
    build=np.where((np.arange(len(df))<lo*4)&(df>=.2*ref))[0]
    return dict(status='合格',Dref=ref,window=[lo,hi],deep=deep.tolist(),near=near.tolist(),build=build.tolist(),
                excluded_low_dose_fine=np.where(df<.2*ref)[0].tolist(),excluded_low_dose_coarse=np.where(d<.2*ref)[0].tolist(),
                window_rule='raw pilot dose >=97% Dref, contiguous around five-bin-smoothed maximum; low-dose <20% excluded')

def select(b,fine,lock):
    d,se=curve_stats(b);df,sf=curve_stats(fine)
    coarse=np.array(lock['deep']+lock['near'],dtype=int);build=np.array(lock['build'],dtype=int)
    return np.r_[d[coarse],df[build]],np.r_[se[coarse],sf[build]],np.r_[np.full(len(coarse),.02),np.full(len(build),.03)]

def required_n(coarse,fine,lock,n0):
    _,sem,_=select(coarse,fine,lock)
    return int(np.ceil(n0*float(np.max(sem/(.005*lock['Dref'])))**2/100)*100)

def compare(proto,reference,sensitivity,lock,seed,prerequisites,q7):
    if not prerequisites:return dict(status='未評価',reason='Q0–Q4 prerequisites not met',engineering='未評価',statistical='未評価')
    if not q7:return dict(status='未評価',reason='Q7 not met',engineering='未評価',statistical='未評価')
    if lock['status']!='合格':return dict(status='未評価',reason=lock.get('reason','pilot window invalid'),engineering='未評価',statistical='未評価')
    ref=lock['Dref'];p,ps,tol=select(proto['coarse'],proto['fine'],lock);r,rs,_=select(reference['coarse'],reference['fine'],lock);s,ss,_=select(sensitivity['coarse'],sensitivity['fine'],lock)
    delta=p-r;width=abs(s-r)/ref;combined=np.hypot(ps,rs)
    z=np.divide(delta,combined,out=np.zeros_like(delta),where=combined>0)
    z[(combined==0)&(delta!=0)]=np.inf
    bonf=float(norm.ppf(1-.05/(2*len(z))));stat='統計的にも一致' if np.all(abs(z)<=bonf) else '統計的一致を棄却'
    vp=peak(proto['coarse'],lock['window']);vr=peak(reference['coarse'],lock['window'])
    bs=bootstrap_peak_difference(proto['coarse'],reference['coarse'],lock['window'],seed)
    engineering='工学的基準に合格'
    if vp is None or vr is None:engineering='未評価'
    elif bs['SEM_cm'] is None:engineering='未評価'
    elif max(np.max(ps),np.max(rs))>.005*ref or bs['SEM_cm']>2/30:engineering='統計不足'
    elif np.any(abs(delta)>tol*ref) or abs(vp-vr)>.2:engineering='不合格'
    status='合格' if engineering=='工学的基準に合格' else engineering
    if np.any(width>.01):status='判定保留'
    return dict(status=status,engineering=engineering,statistical=stat,Dref=ref,max_difference_Dref=float(np.max(abs(delta))/ref),
                max_reference_width_Dref=float(max(width)),max_prototype_SEM_Dref=float(max(ps)/ref),max_reference_SEM_Dref=float(max(rs)/ref),
                prototype_vertex_cm=vp,reference_vertex_cm=vr,vertex_difference_cm=None if vp is None or vr is None else vp-vr,bootstrap=bs,
                bonferroni_z=bonf,significant_bins=int(np.sum(abs(z)>bonf)),evaluated_bins=len(z),
                differences_Gy_cm2=delta.tolist(),reference_width_Gy_cm2=(s-r).tolist(),
                low_dose_excluded=True,conclusion_scope='no overall-pass label')

def step_check(base,half,lock):
    """Q7 uses the fixed EGS pilot bins/Dref and P1 K8 dose/SEM rules."""
    if lock['status']!='合格':return dict(status='未評価',reason='pilot window invalid')
    a,sa,_=select(base['coarse'],base['fine'],lock);b,sb,_=select(half['coarse'],half['fine'],lock)
    delta=(b-a)/lock['Dref'];sem=np.hypot(sa,sb)/lock['Dref']
    status='統計不足' if max(sem)>.01/3 else '合格' if max(abs(delta))<=.01 else '不合格'
    return dict(status=status,max_difference_Dref=float(max(abs(delta))),max_difference_SEM_Dref=float(max(sem)),
                differences_Dref=delta.tolist(),difference_SEM_Dref=sem.tolist(),Dref=lock['Dref'])

def run_requirement(proto,reference,halves,lock,n0,seed):
    """N from pilot SEM, peak difference bootstrap, Q7 combined SEM; no dose residual tuning."""
    if lock['status']!='合格':return dict(required_N=n0,peak_status='未評価',reason=lock.get('reason'))
    dose=max(required_n(proto['coarse'],proto['fine'],lock,n0),required_n(reference['coarse'],reference['fine'],lock,n0))
    vp=peak(proto['coarse'],lock['window']);vr=peak(reference['coarse'],lock['window'])
    bs=bootstrap_peak_difference(proto['coarse'],reference['coarse'],lock['window'],seed)
    peak_n=n0;ps='合格'
    if vp is None or vr is None or bs['SEM_cm'] is None:ps='未評価'
    else:peak_n=max(100,int(np.ceil(n0*(bs['SEM_cm']/(2/30))**2/100)*100))
    q7_n=n0
    for half in halves:
        _,s,_=select(proto['coarse'],proto['fine'],lock);_,sh,_=select(half['coarse'],half['fine'],lock)
        q7_n=max(q7_n,int(np.ceil(n0*(np.max(np.hypot(s,sh))/(lock['Dref']*.01/3))**2/100)*100))
    return dict(required_N=max(n0,dose,peak_n,q7_n),dose_N=dose,peak_N=peak_n,Q7_N=q7_n,
                peak_status=ps,peak_bootstrap=bs,pilot_prototype_vertex_cm=vp,pilot_reference_vertex_cm=vr)
