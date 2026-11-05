"""A4u pre-registered CPU judgement functions. Never starts a simulation.

H3b/H5/H6 remain pending until independently sealed inputs and the separately
reviewed registration are supplied. No missing reference is a passing gate.
"""
import hashlib
import json
from statistics import NormalDist
import numpy as np

DIVERGENT=tuple('s2:category:3:'+roi for roi in ('all','roi0','roi1','roi2'))
DIVERGENCE_SHA256=hashlib.sha256(json.dumps(DIVERGENT,separators=(',',':')).encode()).hexdigest()
L_BANDS=((10,11),(12,13),(14,16),(9,10),(11,12),(9,16))
IDENTITY=('reference_seal_sha256','histories','seed','physics_set','bundle_sha256','edges','cum','divergence_sha256')


def identity(actual,expected):
    if set(expected)!=set(IDENTITY) or set(actual)!=set(IDENTITY):raise ValueError('incomplete H7 identity')
    if expected['divergence_sha256']!=DIVERGENCE_SHA256:raise ValueError('D13 hash mismatch')
    if actual!=expected:raise ValueError('H7 sealed reference/source/physics identity mismatch')
    if actual['histories']<=0 or not isinstance(actual['physics_set'],dict) or 'specialization' not in actual['physics_set']:raise ValueError('invalid H7 identity')
    return True


def branches(old,new):
    if set(old)!=set(new):raise ValueError('H2 quantity names differ')
    rows={};unexpected=[]
    for name in sorted(old):
        x,y=np.asarray(old[name],float),np.asarray(new[name],float)
        if x.ndim!=1 or x.shape!=y.shape or len(x)<2 or not np.isfinite([x,y]).all():raise ValueError('H2 outer-batch arrays')
        delta=float(y.mean()-x.mean());detectable=float(3*np.sqrt((x.var(ddof=1)+y.var(ddof=1))/len(x)))
        rows[name]=dict(K_only=float(x.mean()),all_shell=float(y.mean()),difference=delta,detectable_difference=detectable,anticipated=name in DIVERGENT)
        if name not in DIVERGENT and abs(delta)>detectable:unexpected.append(name)
    return dict(status='stop' if unexpected else 'pass',unexpected=unexpected,quantities=rows,divergence_sha256=DIVERGENCE_SHA256,
        interpretation='model change; detector ROI quantities are not independently validated by H3b')


def l_bands(reference,twin,*,reference_off=None,twin_off=None):
    """Outer-batch photon counts per source history, at integer-keV edges."""
    r,t=np.asarray(reference,float),np.asarray(twin,float)
    if r.ndim!=2 or t.ndim!=2 or r.shape[1]<87 or t.shape[1]<87 or min(len(r),len(t))<2:raise ValueError('H3b batch spectra required')
    bands=L_BANDS;rows=[]
    for lo,hi in bands:
        x,y=r[:,lo:hi].sum(axis=1),t[:,lo:hi].sum(axis=1)
        mu=float(x.mean());delta=float(y.mean()-mu);sem=float(np.sqrt(x.var(ddof=1)/len(x)+y.var(ddof=1)/len(y)))
        rows.append(dict(band=[lo,hi],reference=mu,twin=float(y.mean()),difference=delta,SEM=sem,pass_=abs(delta)<=max(2*sem,.05*mu)))
    k=None
    if reference_off is not None and twin_off is not None:
        k=[]
        for on,off in ((r,np.asarray(reference_off,float)),(t,np.asarray(twin_off,float))):
            on_fraction=on[:,72:86].sum()/on.sum();off_fraction=off[:,72:86].sum()/off.sum()
            k.append(dict(on_fraction=float(on_fraction),off_fraction=float(off_fraction),pass_=bool(on_fraction>=2*off_fraction)))
    status='pending' if k is None else 'pass' if all(row['pass_'] for row in rows) and all(row['pass_'] for row in k) else 'fail'
    return dict(status=status,bands=rows,K_band_qualitative=k)


def low_energy(reference,twin,reference_sem,twin_sem,registration):
    """D7. Registration is frozen before H4/H5 data, never inferred here."""
    required={'K','mask','mandatory_mask','minimum_evaluable','registration_sha256'}
    if set(registration)!=required:raise ValueError('H4 registration missing')
    raw={k:v for k,v in registration.items() if k!='registration_sha256'}
    seal=hashlib.sha256(json.dumps(raw,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if seal!=registration['registration_sha256']:raise ValueError('H4 registration seal mismatch')
    r,t,sr,st=map(lambda a:np.asarray(a,float),(reference,twin,reference_sem,twin_sem))
    if len({a.shape for a in (r,t,sr,st)})!=1 or not all(np.isfinite(a).all() for a in (r,t,sr,st)):raise ValueError('invalid low-energy arrays')
    mask=np.asarray(registration['mask'],bool);mandatory=np.asarray(registration['mandatory_mask'],bool)
    K=registration['K']
    if mask.shape!=r.shape or mandatory.shape!=r.shape or int(mask.sum())!=K or K<1:raise ValueError('pre-registered bin count')
    sem=np.hypot(sr,st);enough=sem<=.01*r;delta=np.abs(t-r);critical=NormalDist().inv_cdf(1-.05/(2*K))
    passed=(delta<=critical*sem)|(delta<=.02*r);historical=(delta<=2*sem)|(delta<=.02*r)
    evaluable=mask&enough;failed=evaluable&~passed
    sufficient=int(evaluable.sum())>=registration['minimum_evaluable'] and bool(enough[mandatory].all())
    return dict(status='insufficient' if not sufficient else 'fail' if failed.any() else 'pass',K=K,z_critical=critical,
        pass_count=int((evaluable&passed).sum()),fail_count=int(failed.sum()),insufficient_count=int((mask&~enough).sum()),historical_2sigma_fail=int((evaluable&~historical).sum()),failed_bins=np.flatnonzero(failed).tolist())


def pending():
    return dict(H2='pending CPU comparison',H3b='pending independent sealed EGS5 reference analysis',H4='pending pilot/registration',H5='pending EGS5 and entry runs',H6='pending single T4 session',H7='pending review and reference seal checks',divergence_sha256=DIVERGENCE_SHA256)

if __name__=='__main__':print(json.dumps(pending(),indent=2))


def sealed_reference(path,digest,actual,expected):
    from pathlib import Path
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest or actual.get('reference_seal_sha256')!=digest:raise ValueError('reference seal SHA256 mismatch')
    return identity(actual,expected)
