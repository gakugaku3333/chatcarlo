"""P3r scalar coupled history transport, separate from the unchanged P1 entry.
19-field electron hinge state + charge + pair ancestry. All descendants form one history.
No history ledger is retained. Rest budget is +2m per pair, -2m per annihilation;
escape/unprocessed are kinetic or photon energies; delta_rest holds surviving mass.
"""
import numpy as np
from numba import njit
from .physics import *
from .transport import initial as electron_initial,boundary,score_segment,local,CUT
from .sampling import gs_mu_cached
from .backend_sampling import table_rates,table_single
from .photon_data_p2 import coefficients,transfer_coefficient,log_interp
from .photon_sampling_p2 import compton,pair,annihilation
from .flight_annihilation_p2c import annihilation_rate, photons
from .bremsstrahlung_p3r import (quantities, restricted_stopping, sample_k, direction,
                                new_remaining, advance_remaining, event_order)
from . import photon_transport_p3s as previous
FIELDS=23
@njit
def initial(T,z,ux,uy,uz,region,primary,fe,k,brems_enabled=True):
    a=np.zeros(FIELDS);a[:19]=electron_initial(T,z,ux,uy,uz,region,primary,fe,k);a[19]=-1
    a[22]=new_remaining(brems_enabled)
    return a
@njit
def energy_init(a,fe):
    a[7]=a[6];a[11]=min(fe*a[7],max(0.,a[7]-CUT))
    a[10]=np.random.random()*a[11];a[8]=a[10];a[9]=a[11]-a[10];a[18]=0.
@njit
def photon_state(E,z,ux,uy,uz,ancestry):
    a=np.zeros(FIELDS);a[2]=z;a[3]=ux;a[4]=uy;a[5]=uz;a[6]=E;a[20]=ancestry
    return a
@njit
def accumulate_primary(batches,h,n,history):
    b=min(99,h*100//n)
    batches[b]+=history
    return b
@njit
def rule_value(T, k, kind, energies, values):
    """kind 0: constant; 1: upper-inclusive steps; 2: linear E–K."""
    if kind == 0:
        return k
    if kind == 1:
        return values[np.searchsorted(energies, T, side='left')]
    return np.interp(T, energies, values)

@njit
def hinge_transition(a, knew, u):
    old_k = a[14]
    a[12] = a[13] + u*knew
    a[13] = (1-u)*knew
    a[14] = knew
    return old_k

@njit
def scattering_hinge(a, ev, knew, ts, ks, us, ys, total, G1):
    # Sample the OLD step before creating the new step. Do not rescale its tail.
    mu = gs_mu_cached(ev, a[14], ts, ks, us, ys, total, G1)
    a[3], a[4], a[5] = rotate(a[3], a[4], a[5], mu, 2*np.pi*np.random.random())
    hinge_transition(a, knew, np.random.random())

@njit
def tracked_local(score, z, loss, dz, touched, j):
    local(score, z, loss, dz)
    b=max(0,min(len(score)-1,int(z/dz)))
    touched[j,0]=min(touched[j,0],b);touched[j,1]=max(touched[j,1],b)

@njit
def tracked_segment(score,z0,z1,loss,dz,touched,j):
    score_segment(score,z0,z1,loss,dz)
    if loss==0:return
    b0=max(0,min(len(score)-1,int(min(z0,z1)/dz)))
    b1=max(0,min(len(score)-1,int(max(z0,z1)/dz)))
    touched[j,0]=min(touched[j,0],b0);touched[j,1]=max(touched[j,1],b1)

@njit
def touched_sum(score,touched,j):
    result=0.
    for i in range(touched[j,0],touched[j,1]+1):result+=score[i]
    return result

@njit
def add_clear(batch, score, touched, j, b):
    for i in range(touched[j,0],touched[j,1]+1):
        batch[b,i]+=score[i]
        score[i]=0.
    touched[j,0]=len(score);touched[j,1]=-1

@njit
def _run_coupled(E,n,seed,fe,k,thickness,ts,ks,us,ys,backend_rates,backend_single,
                pes,pwater,pthresholds,dl,bpar,delcm,delpos,nist_E,nist_muen,
                maxstack=4096,maxsteps=10000000,source_state=None,annihilate_in_flight=False,
                positron_source_T=None,positron_source_z=15.,annihilation_scale=1.,
                spectrum_edges=None,spectrum_cum=None,
                rule_kind=0,rule_energies=None,rule_values=None,
                score_kerma=True,score_pair=True,score_lateral=False,
                charged_source_T=None,charged_source_charge=-1,charged_source_z=0.,
                brems_mode=2,brems_energies=None,brems_values=None,
                disable_brems=False,infinite_medium=False,record_first=False):
    if rule_kind not in (0,1,2):raise ValueError('unknown step rule')
    if rule_kind:
        if rule_energies is None or rule_values is None:raise ValueError('rule arrays required')
        if len(rule_values)!=len(rule_energies)+(1 if rule_kind==1 else 0):raise ValueError('rule dimensions')
        if not np.all(np.isfinite(rule_energies)) or not np.all(np.isfinite(rule_values)) or np.any(np.diff(rule_energies)<=0) or np.any(rule_values<=0):raise ValueError('invalid rule')
    else:
        rule_energies=np.array([0.,10.]);rule_values=np.array([k,k])
    if charged_source_T is not None:
        if charged_source_charge not in (-1,1) or charged_source_T<=0 or not 0<=charged_source_z<thickness:raise ValueError('invalid charged source')
        if source_state is not None or positron_source_T is not None:raise ValueError('conflicting sources')
    if (spectrum_edges is None) != (spectrum_cum is None):
        raise ValueError('both spectrum edges and cumulative probabilities required')
    nspec=np.zeros((0,10),np.int64)
    if spectrum_edges is not None and spectrum_cum is not None:
        if source_state is not None or positron_source_T is not None or charged_source_T is not None:
            raise ValueError('spectrum requires photon source without source_state')
        if len(spectrum_cum)<1 or len(spectrum_cum)>64 or len(spectrum_edges)!=len(spectrum_cum)+1:
            raise ValueError('invalid spectrum dimensions')
        if not np.all(np.isfinite(spectrum_edges)) or not np.all(np.isfinite(spectrum_cum)):
            raise ValueError('non-finite spectrum')
        if spectrum_edges[0]<0 or spectrum_edges[-1]>10 or np.any(np.diff(spectrum_edges)<=0):
            raise ValueError('invalid spectrum energy edges')
        if spectrum_cum[0]<=0 or np.any(np.diff(spectrum_cum)<=0) or spectrum_cum[-1]!=1.:
            raise ValueError('invalid cumulative probabilities')
        nspec=np.zeros((len(spectrum_cum),10),np.int64)
    source_energy=E
    np.random.seed(seed);nbatch=100;dz=.2;faces=np.array([0.,thickness])
    nb=int(round(thickness/dz));nf=int(round(thickness/.05))
    batches=np.zeros((nbatch,nb));fine_batches=np.zeros((nbatch,nf));pair_batches=np.zeros((nbatch,nf))
    kerma_batches=np.zeros((nbatch,nf));collision_batches=np.zeros((nbatch,nf))
    counts=np.zeros(16,np.int64);totals=np.zeros(6);maxima=np.zeros(3);violations=np.zeros(3,np.int64)
    stack=np.empty((maxstack,FIELDS));maxdepth=0;scoring=not infinite_medium;hard_on=True;mode=0;ignore_boundaries=infinite_medium
    # count, sum k, envelope excess, k>T; dN/sq, dE/sq, radiation/sq.
    brems=np.zeros(10); first_record=np.zeros(8)
    hs=np.zeros(nb);fine=np.zeros(nf);pair_score=np.zeros(nf);kerma=np.zeros(nf);collision_kerma=np.zeros(nf)
    touched=np.array([[nb,-1],[nf,-1],[nf,-1],[nf,-1],[nf,-1]])
    lateral_batches=np.zeros(nbatch);step_batches=np.zeros(nbatch,np.int64)
    for h in range(n):
        lateral=0.;hn=0.;expected_n=0.;de=0.;radiation=0.
        if spectrum_edges is not None and spectrum_cum is not None:
            u1=np.random.random();u2=np.random.random();j=0
            while spectrum_cum[j]<u1:
                j+=1
            width=spectrum_edges[j+1]-spectrum_edges[j]
            source_energy=spectrum_edges[j]+u2*width
            cell=min(9,int((source_energy-spectrum_edges[j])/width*10))
            nspec[j,cell]+=1
        stack[0]=photon_state(source_energy,0.,0.,0.,1.,0) if source_state is None else source_state
        if positron_source_T is not None:
            stack[0]=initial(positron_source_T,positron_source_z,0.,0.,1.,0,1,fe,rule_value(positron_source_T,k,rule_kind,rule_energies,rule_values),not disable_brems)
            stack[0,19]=1
        if charged_source_T is not None:
            stack[0]=initial(charged_source_T,charged_source_z,0.,0.,1.,0,1,fe,
                             rule_value(charged_source_T,k,rule_kind,rule_energies,rule_values),not disable_brems)
            stack[0,19]=charged_source_charge
        if source_state is not None and stack[0,19]!=0:
            stack[0,22]=new_remaining(not disable_brems)
        if annihilate_in_flight and stack[0,19]==1:
            stack[0,21]=-np.log(np.random.random())
        top=0;dep=0.;escaped=0.;rest=0.;steps=0;failed=False;positron_escape_rest=0.
        # A supplied positron source includes its two-rest-mass annihilation budget.
        incoming=source_energy if positron_source_T is None else positron_source_T
        if charged_source_T is not None:incoming=charged_source_T
        if stack[0,19]==1:rest=2*M;incoming+=2*M
        seen=np.zeros(2,np.int64)
        while top>=0:
            a=stack[top].copy();top-=1
            if a[19]==0:
                steps+=1
                if steps>maxsteps:
                    failed=True;counts[5]+=1;break
                E=a[6]
                if E<CUT:
                    if scoring:
                        tracked_local(hs,a[2],E,dz,touched,0);tracked_local(fine,a[2],E,.05,touched,1)
                        if score_pair and a[20]:tracked_local(pair_score,a[2],E,.05,touched,2)
                    dep+=E;continue
                c=coefficients(E,pes,pwater,pthresholds);mu=np.sum(c)
                if infinite_medium:bb=np.inf;nr=0
                else:bb,nr=boundary(a[2],a[5],int(a[16]),faces)
                flight=-np.log(np.random.random())/mu;length=min(flight,bb)
                z0=a[2];a[2]+=length*a[5]
                if score_kerma and scoring:
                    kt=E*transfer_coefficient(E,pes,pwater,pthresholds)*length
                    kc=E*log_interp(E,nist_E,nist_muen)*length
                    tracked_segment(kerma,z0,a[2],kt,.05,touched,3);tracked_segment(collision_kerma,z0,a[2],kc,.05,touched,4)
                if record_first and steps==1:
                    first_record[5]+=length;first_record[6]+=length*length
                if bb<=flight:
                    if record_first and steps==1:first_record[0]+=1
                    escaped+=E;counts[7]+=1;continue
                r=np.random.random()*mu;process=0
                while process<3 and r>=c[process]:r-=c[process];process+=1
                counts[8+process]+=1
                if record_first and steps==1:first_record[1+process]+=1
                if top+3>=maxstack:
                    failed=True;counts[6]+=1;break
                if process==0:
                    ep,gx,gy,gz,T,ex,ey,ez=compton(E,a[3],a[4],a[5])
                    photon=photon_state(ep,a[2],gx,gy,gz,int(a[20]));top+=1;stack[top]=photon
                    child=initial(T,a[2],ex,ey,ez,0,0,fe,rule_value(T,k,rule_kind,rule_energies,rule_values),not disable_brems);child[20]=a[20]
                    top+=1;stack[top]=child
                elif process==1:
                    child=initial(E,a[2],a[3],a[4],a[5],0,0,fe,rule_value(E,k,rule_kind,rule_energies,rule_values),not disable_brems);child[20]=a[20]
                    top+=1;stack[top]=child
                else:
                    p=pair(E,a[3],a[4],a[5],dl,bpar,delcm,delpos)
                    rest+=2*M
                    first=initial(p[0],a[2],p[1],p[2],p[3],0,0,fe,rule_value(p[0],k,rule_kind,rule_energies,rule_values),not disable_brems);first[19]=p[4];first[20]=1
                    second=initial(p[5],a[2],p[6],p[7],p[8],0,0,fe,rule_value(p[5],k,rule_kind,rule_energies,rule_values),not disable_brems);second[19]=p[9];second[20]=1
                    if annihilate_in_flight:
                        if first[19]==1:first[21]=-np.log(np.random.random())
                        if second[19]==1:second[21]=-np.log(np.random.random())
                    top+=1;stack[top]=first;top+=1;stack[top]=second
                maxdepth=max(maxdepth,top+1);continue
            cached_ev=-1.;annihilated=False;suspended=False;arate=0.
            eta=np.zeros(2);totalj=np.zeros(2)
            while True:
                steps+=1
                if steps>maxsteps:
                    failed=True;counts[5]+=1;break
                region=int(a[16])
                if not infinite_medium and (region<0 or region>=len(faces)-1):
                    escaped+=a[6]
                    if a[19]==1:counts[13]+=1;positron_escape_rest+=2*M
                    break
                if a[6]<=CUT+1e-14 and a[18]!=1:
                    if scoring:
                        tracked_local(hs,a[2],a[6],dz,touched,0);tracked_local(fine,a[2],a[6],.05,touched,1)
                        if score_pair and a[20]:tracked_local(pair_score,a[2],a[6],.05,touched,2)
                    if score_lateral:lateral+=a[6]*(a[0]**2+a[1]**2)
                    dep+=a[6];a[6]=0.;break
                ev=max(CUT,a[7])
                if ev!=cached_ev:
                    S=restricted_stopping(ev,brems_energies,brems_values,disable_brems,hard_on)
                    brate,bhard,bsoft,bmax=quantities(ev,brems_energies,brems_values,disable_brems)
                    rate=hard(ev)[0] if hard_on else 0.
                    if backend_rates is None:
                        total,G1=elastic_rates(ev)
                    else:
                        total,G1=table_rates(ev,ts,backend_rates)
                    if mode==1:
                        if backend_rates is None:eta,B,totalj,tmp=elastic(ev)
                        G1=total
                    if mode==2:G1=0.
                    if annihilate_in_flight and a[19]==1:arate=annihilation_scale*annihilation_rate(ev)
                    cached_ev=ev
                # Recompute all event distances after every direction/energy update.
                eb=a[8]/S
                mb=a[12]/G1 if G1>0 else np.inf
                hb=a[15]/rate if rate>0 else np.inf
                if a[18]==1:mb=np.inf;hb=np.inf
                if ignore_boundaries:bb=np.inf;nr=region
                else:
                    bb,nr=boundary(a[2],a[5],region,faces);counts[4]+=1
                rb=a[22]/brate if brate>0 and a[18]!=1 else np.inf
                step=min(eb,mb,hb,bb,rb,a[6]/S)
                ab=np.inf
                if annihilate_in_flight and a[19]==1 and a[18]!=1:
                    ab=a[21]/arate if arate>0 else np.inf
                    step=min(step,ab)
                if not np.isfinite(step) or step<0:
                    failed=True;counts[6]+=1;break
                x0=a[0];y0=a[1];z0=a[2];a[0]+=step*a[3];a[1]+=step*a[4];a[2]+=step*a[5]
                loss=step*S;a[6]-=loss;dep+=loss
                radiation+=step*bsoft
                if a[18]!=1:
                    expected_n+=step*brate
                    a[22]=advance_remaining(a[22],step,brate)
                if score_lateral:lateral+=loss*((x0+step*a[3]/2)**2+(y0+step*a[4]/2)**2)
                if scoring:
                    tracked_segment(hs,z0,a[2],loss,dz,touched,0);tracked_segment(fine,z0,a[2],loss,.05,touched,1)
                    if score_pair and a[20]:tracked_segment(pair_score,z0,a[2],loss,.05,touched,2)
                a[8]=max(0.,a[8]-loss);a[12]=max(0.,a[12]-step*G1);a[15]=max(0.,a[15]-step*rate)
                if annihilate_in_flight and a[19]==1 and a[18]!=1:
                    a[21]=max(0.,a[21]-step*arate)
                counts[0]+=1
                # Deterministic priority: boundary, energy, scattering, hard, brems, annihilation.
                priority=event_order(bb,eb,mb,hb,rb,ab)
                if bb<=step+1e-13:
                    a[2]=faces[region+1] if a[5]>0 else faces[region];a[16]=nr;counts[3]+=1
                    continue
                if eb<=step+1e-13:
                    if a[18]==1:
                        if scoring:
                            tracked_local(hs,a[2],a[6],dz,touched,0);tracked_local(fine,a[2],a[6],.05,touched,1)
                            if score_pair and a[20]:tracked_local(pair_score,a[2],a[6],.05,touched,2)
                        if score_lateral:lateral+=a[6]*(a[0]**2+a[1]**2)
                        dep+=a[6];a[6]=0.;break
                    a[7]=max(CUT,a[7]-a[11])
                    if a[7]<=CUT+1e-14:
                        a[8]=a[9];a[9]=0.;a[10]=0.;a[18]=1.;continue
                    prev=a[9];a[11]=min(fe*a[7],a[7]-CUT);a[10]=np.random.random()*a[11]
                    a[9]=a[11]-a[10];a[8]=prev+a[10];continue
                if mb<=step+1e-13:
                    scattering_hinge(a,ev,rule_value(ev,k,rule_kind,rule_energies,rule_values),ts,ks,us,ys,total,G1)
                    counts[1]+=1
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
                    child=initial(W,a[2],ds[0],ds[1],ds[2],int(a[16]),0,fe,rule_value(W,k,rule_kind,rule_energies,rule_values),not disable_brems);child[0]=a[0];child[1]=a[1];child[20]=a[20]
                    if mode==1:child[12]=-np.log(np.random.random())
                    stack[top]=child;maxdepth=max(maxdepth,top+1);break
                if rb<=step+1e-13:
                    energy_init(a,fe)
                    T=a[6]
                    a[22]=new_remaining(not disable_brems)
                    if T<=CUT:continue
                    photon_k,excess=sample_k(T,brems_energies,brems_values)
                    brems[2]+=excess
                    if photon_k>T:brems[3]+=1
                    if excess or photon_k>T:
                        failed=True;counts[6]+=1;break
                    q=quantities(T,brems_energies,brems_values)
                    hn+=1.;de+=photon_k-q[1]/q[0];radiation+=photon_k
                    brems[0]+=1.;brems[1]+=photon_k
                    gx,gy,gz=direction(T,a[3],a[4],a[5],2*np.pi*np.random.random())
                    # Overflow leaves the pre-emission kinetic budget unresolved.
                    if brems_mode==2 and top+2>=maxstack:
                        failed=True;counts[6]+=1;break
                    a[6]-=photon_k;energy_init(a,fe)
                    if brems_mode==1:
                        dep+=photon_k
                        if scoring:
                            tracked_local(hs,a[2],photon_k,dz,touched,0)
                            tracked_local(fine,a[2],photon_k,.05,touched,1)
                            if score_pair and a[20]:tracked_local(pair_score,a[2],photon_k,.05,touched,2)
                        if score_lateral:lateral+=photon_k*(a[0]**2+a[1]**2)
                        continue
                    top+=1;stack[top]=a
                    top+=1;stack[top]=photon_state(photon_k,a[2],gx,gy,gz,int(a[20]))
                    maxdepth=max(maxdepth,top+1);suspended=True;break
                if annihilate_in_flight and ab<=step+1e-13:
                    T=a[6]
                    if T<=CUT:
                        a[21]=-np.log(np.random.random());continue
                    if top+3>=maxstack:
                        failed=True;counts[6]+=1;break
                    g=photons(T,a[3],a[4],a[5])
                    top+=1;stack[top]=photon_state(g[0],a[2],g[1],g[2],g[3],int(a[20]))
                    top+=1;stack[top]=photon_state(g[4],a[2],g[5],g[6],g[7],int(a[20]))
                    rest-=2*M;a[6]=0.;annihilated=True;counts[14]+=1
                    if a[17]==1:counts[15]+=1
                    maxdepth=max(maxdepth,top+1);break
                # Actual-energy endpoint: deposit only the remaining ledger energy.
                if scoring:
                    tracked_local(hs,a[2],a[6],dz,touched,0);tracked_local(fine,a[2],a[6],.05,touched,1)
                    if score_pair and a[20]:tracked_local(pair_score,a[2],a[6],.05,touched,2)
                if score_lateral:lateral+=a[6]*(a[0]**2+a[1]**2)
                dep+=a[6];a[6]=0.;break
            if failed:break
            if not suspended and not annihilated and a[19]==1 and a[6]==0 and (infinite_medium or 0<=a[16]<len(faces)-1):
                # Kinetic cutoff was scored by the electron loop BEFORE releasing rest energy.
                if top+3>=maxstack:
                    # Keep the unresolved positron rest budget rather than depositing it.
                    a[6]=0.;failed=True;counts[6]+=1;break
                ux,uy,uz=annihilation();rest-=2*M;counts[12]+=1
                top+=1;stack[top]=photon_state(M,a[2],ux,uy,uz,int(a[20]))
                top+=1;stack[top]=photon_state(M,a[2],-ux,-uy,-uz,int(a[20]))
                maxdepth=max(maxdepth,top+1)
        unresolved=0.
        if failed:
            unresolved=a[6]
            for j in range(top+1):unresolved+=stack[j,6]
        residual=incoming-dep-escaped-unresolved-rest
        coarse_error=abs(touched_sum(hs,touched,0)-dep) if scoring else 0.
        fine_error=abs(touched_sum(fine,touched,1)-dep) if scoring else 0.
        dn=hn-expected_n
        brems[4]+=dn;brems[5]+=dn*dn;brems[6]+=de;brems[7]+=de*de
        brems[8]+=radiation;brems[9]+=radiation*radiation
        maxima[0]=max(maxima[0],abs(residual)/incoming)
        maxima[1]=max(maxima[1],coarse_error/incoming);maxima[2]=max(maxima[2],fine_error/incoming)
        if abs(residual)>incoming*1e-9:violations[0]+=1
        if coarse_error>incoming*1e-9:violations[1]+=1
        if fine_error>incoming*1e-9:violations[2]+=1
        totals[0]+=incoming;totals[1]+=dep;totals[2]+=escaped;totals[3]+=unresolved;totals[4]+=rest;totals[5]+=positron_escape_rest
        b=min(99,h*100//n)
        add_clear(batches,hs,touched,0,b);add_clear(fine_batches,fine,touched,1,b)
        add_clear(pair_batches,pair_score,touched,2,b);add_clear(kerma_batches,kerma,touched,3,b)
        add_clear(collision_batches,collision_kerma,touched,4,b)
        lateral_batches[b]+=lateral;step_batches[b]+=steps
    return batches,fine_batches,pair_batches,kerma_batches,collision_batches,totals,maxima,violations,counts,maxdepth,nspec,lateral_batches,step_batches,brems,first_record

def run_coupled(*args, brems_mode=2, disable_brems=False, infinite_medium=False,
                record_first=False, **kwargs):
    """Mode 0 returns precisely P3s's tuple, without building brems tables."""
    if brems_mode not in (0,1,2):raise ValueError('brems_mode must be 0, 1 or 2')
    if brems_mode==0:
        if disable_brems or infinite_medium or record_first:
            raise ValueError('P3r check hooks require mode 1 or 2')
        return previous.run_coupled(*args,**kwargs)
    from .bremsstrahlung_p3r import tables, MAX_T
    if float(args[0])>MAX_T:raise ValueError('source above 20 MeV')
    # The inherited scoring arrays hold round(thickness/bin) bins: other thicknesses lose deposits from the arrays.
    if not infinite_medium and abs(float(args[5])/.2-round(float(args[5])/.2))>1e-9:
        raise ValueError('slab thickness must be a multiple of the 0.2 cm scoring bin')
    for key in ('charged_source_T','positron_source_T'):
        if kwargs.get(key) is not None and kwargs[key]>MAX_T:
            raise ValueError('charged source above 20 MeV')
    if kwargs.get('source_state') is not None:
        a=np.zeros(FIELDS); supplied=kwargs['source_state']
        a[:len(supplied)]=supplied
        # Supplied charged states initialize inside the seeded kernel.
        kwargs['source_state']=a
    energies,values=tables()
    return _run_coupled(*args,**kwargs,brems_mode=brems_mode,
                        brems_energies=energies,brems_values=values,
                        disable_brems=disable_brems,infinite_medium=infinite_medium,
                        record_first=record_first)
