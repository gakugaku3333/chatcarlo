"""Q2 independent numerical quadrature of KN and the EGS5 pair rule."""
import json
import numpy as np
from scipy.integrate import quad
from scipy.stats import chisquare,chi2
from .physics import M,RE,NE
from .photon_data_p2 import kn_total
from .photon_sampling_p2 import samples,annihilation_samples,photoelectric,pair_parameters,photoelectric_check
from .p2_common import OUT,ENERGIES,SEED,write,save,progress

def chi(values,edges,cdf):
    observed=np.histogram(values,edges)[0];expected=np.diff(cdf)*len(values)
    merged_o=[];merged_e=[];o=0.;e=0.
    for oi,ei in zip(observed,expected):
        o+=oi;e+=ei
        if e>=100:merged_o.append(o);merged_e.append(e);o=e=0.
    if e or o:
        if merged_e:merged_o[-1]+=o;merged_e[-1]+=e
        else:return dict(status='統計不足')
    expected=np.array(merged_e);expected*=sum(merged_o)/expected.sum()
    stat,p=chisquare(merged_o,expected)
    return dict(chi2=float(stat),p=float(p),merged_bins=len(expected),min_expected=float(expected.min()),status='合格' if p>.001 else '不合格')

def kn_density(mu,E):
    q=1/(1+E/M*(1-mu))
    return np.pi*RE**2*NE*q*q*(q+1/q-(1-mu*mu))

def pair_density(t,E,dl,bpar,delcm,delpos):
    # Analytic mixture proposal on br in [0,.5]:
    # bpar * 2 + (1-bpar) * 24(br-.5)^2, multiplied by rejection factor.
    br=(t+M)/E
    if br<M/E or br>.5:return 0.
    d=1/(E*br*(1-br))
    if d>=delpos:return 0.
    delta=delcm*d
    def rejection(j):
        row=dl[j]
        return row[0]+delta*(row[1]+delta*row[2]) if delta<1 else row[3]+row[4]*np.log(delta+row[5])
    return (bpar*2*min(1.,rejection(2))+(1-bpar)*24*(br-.5)**2*min(1.,rejection(0)))/E

def check():
    if (OUT/'q2.json').exists():return json.loads((OUT/'q2.json').read_text())
    progress('T2','running',samples_per_case=1000000)
    dl,bpar,delcm,delpos=pair_parameters();rows=[];n=1000000
    for j,E in enumerate(ENERGIES):
        c,p=samples(E,n,SEED+200+j,dl,bpar,delcm,delpos)
        means=quad(lambda mu:E/(1+E/M*(1-mu))*kn_density(mu,E),-1,1,epsabs=1e-30,epsrel=1e-11)[0]/kn_total(E)
        sem=float(c[:,0].std(ddof=1)/np.sqrt(n));z=float((c[:,0].mean()-means)/sem)
        edges=np.linspace(-1,1,101);cdf=np.array([quad(lambda u:kn_density(u,E),-1,e,epsabs=1e-30,epsrel=1e-10)[0]/kn_total(E) for e in edges]);cdf[0]=0.;cdf[-1]=1.
        angle=chi(c[:,3],edges,cdf)
        momentum=c[:,0,None]*c[:,1:4]+np.sqrt(c[:,4]*(c[:,4]+2*M))[:,None]*c[:,5:8]-np.array([0,0,E])
        momentum_error=float(np.max(np.linalg.norm(momentum,axis=1))/E);energy_error=float(np.max(abs(c[:,0]+c[:,4]-E))/E)
        lo=0.;hi=E/2-M;pe=np.linspace(lo,hi,101)
        if E<=2.1:pcdf=(pe-lo)/(hi-lo)
        else:
            f=lambda t:pair_density(t,E,dl,bpar,delcm,delpos)
            # Split at DELTA=1 and DEL=DELPOS to respect non-smooth support.
            special=[]
            for d in (1/delcm,delpos):
                disc=1-4/(E*d)
                if disc>=0:
                    t=(1-np.sqrt(disc))*E/2-M
                    if 0<t<hi:special.append(t)
            norm=quad(f,lo,hi,points=special,epsabs=1e-10,epsrel=1e-10)[0]
            pcdf=np.array([quad(f,lo,x,points=[v for v in special if v<x],epsabs=1e-10,epsrel=1e-10)[0]/norm if x>lo else 0 for x in pe]);pcdf[-1]=1
        split=chi(p[:,5],pe,pcdf);phis=np.mod(np.arctan2(p[:,2],p[:,1]),2*np.pi)
        az=chi(phis,np.linspace(0,2*np.pi,101),np.linspace(0,1,101))
        charge_z=float((np.mean(p[:,4]>0)-.5)/np.sqrt(.25/n))
        charge_p=float(chi2.sf(charge_z**2,1))
        pair_energy=float(np.max(abs(p[:,0]+p[:,5]-(E-2*M)))/E)
        polar_error=float(np.max(abs(p[:,3]-np.cos(M/E))))
        opposite_transverse=float(np.max(np.linalg.norm(p[:,1:3]+p[:,6:8],axis=1)))
        tests=[abs(z)<=3,angle['status']=='合格',momentum_error<=1e-12,energy_error<=1e-12,split['status']=='合格',az['status']=='合格',charge_p>.001,pair_energy<=1e-12,polar_error<=1e-12,opposite_transverse<=1e-12]
        rows.append(dict(E=E,N=n,seed=SEED+200+j,compton_mean=float(c[:,0].mean()),compton_analytic_mean=means,mean_SEM=sem,mean_z=z,
                         compton_angle=angle,momentum_max_relative=momentum_error,energy_max_relative=energy_error,
                         pair_split=split,pair_azimuth=az,pair_charge_z=charge_z,pair_charge_chi2=charge_z**2,pair_charge_p=charge_p,pair_energy_max_relative=pair_energy,
                         pair_polar_max_error=polar_error,pair_transverse_antiparallel_max_error=opposite_transverse,
                         status='合格' if all(tests) else '不合格'))
        save(OUT/'sampling'/f'case_{j}.npz',compton_angle_hist=np.histogram(c[:,3],edges)[0],pair_split_hist=np.histogram(p[:,5],pe)[0])
        write('q2_checkpoint.json',dict(completed=rows))
    a=annihilation_samples(n,SEED+210);pe=np.linspace(-1,1,101);u=chi(a[:,2],pe,(pe+1)/2)
    phi=chi(np.mod(np.arctan2(a[:,1],a[:,0]),2*np.pi),np.linspace(0,2*np.pi,101),np.linspace(0,1,101))
    unit=float(max(abs(np.sum(a*a,axis=1)-1)))
    photo_rows=[]
    for j,E in enumerate(ENERGIES):
        en,di=photoelectric_check(E,n,SEED+200+j)
        photo_rows.append(dict(E=E,N=n,seed=SEED+200+j,max_energy_error=en,max_direction_error=di))
    photo=all(r['max_energy_error']==r['max_direction_error']==0 for r in photo_rows)
    try:
        samples(2*M-1e-8,1,SEED,dl,bpar,delcm,delpos);threshold=False
    except ValueError:threshold=True
    rest=dict(N=n,seed=SEED+210,polar=u,azimuth=phi,direction_norm_max_error=unit,energy_sum=2*M,momentum_sum_max=0.,status='合格' if u['status']==phi['status']=='合格' and unit<=1e-12 else '不合格')
    result=dict(status='合格' if all(r['status']=='合格' for r in rows) and rest['status']=='合格' and photo and threshold else '不合格',rows=rows,photoelectric=photo_rows,photoelectric_exact=photo,pair_below_threshold_rejected=threshold,annihilation=rest)
    write('q2.json',result);progress('T2','complete',result=result['status']);return result
