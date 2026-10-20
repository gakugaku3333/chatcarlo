"""P8e material transport: a copy of P4, with boundary hinge truncation.

Material arrays are built/verified in Python, never generated inside transport.
The P4 outputs are retained for regression; P8e additionally scores geometric
bins, absolute score/ledger residuals, and boundary-scattering diagnostics.
"""
import numpy as np
from numba import njit
from .physics import *
from .transport import initial as electron_initial,boundary,score_segment,local,CUT
from .sampling import gs_mu_cached
from .backend_sampling import table_rates,table_single
from .photon_data_p2 import coefficients,transfer_coefficient,log_interp,pair_interp
from .photon_sampling_p2 import compton,pair,annihilation
from .flight_annihilation_p2c import annihilation_rate, photons
from .bremsstrahlung_p3r import quantities, sample_k, direction, new_remaining, advance_remaining
from .scoring_p3b import score_box_bins, score_box_point
OUTPUTS = ('coarse','fine','pair','kerma','collision_kerma','totals',
    'maxima','violations','counts','maxdepth','nspec','lateral','steps','brems',
    'first_record','axis_coarse','axis_fine','slab_coarse','profile',
    'profile_fine','region_ledger','deposits','deposit_overflow','sources',
    'source_overflow','mass_batches','history_mass','mass_ledger','path_total')
FIELDS=23
BOUNDARY_STRENGTH_EDGES=np.array([0.,1e-12,1e-10,1e-8,1e-6,1e-5,1e-4,.0005,.001,.003125,.00625,.0125,.025,.05,.1,.2,1.])


def mass_geometry(thickness, layer_faces=None, layer_rho=None, mass_bin=None):
    if (layer_faces is None) != (layer_rho is None):
        raise ValueError('both layer faces and densities required')
    faces = np.array([0., thickness]) if layer_faces is None else np.asarray(layer_faces, float)
    rho = np.ones(1) if layer_rho is None else np.asarray(layer_rho, float)
    if (faces.ndim != 1 or len(faces) < 2 or not np.all(np.isfinite(faces))
        or faces[0] != 0 or faces[-1] != thickness or np.any(np.diff(faces) <= 0)):
        raise ValueError('invalid layer faces')
    if rho.shape != (len(faces)-1,) or not np.all(np.isfinite(rho)) or np.any(rho <= 0):
        raise ValueError('invalid layer densities')
    mf = np.r_[0., np.cumsum(rho*np.diff(faces))]
    if mass_bin is not None:
        if not np.isfinite(mass_bin) or mass_bin <= 0:
            raise ValueError('mass_bin must be finite and positive')
        ratio = mf/mass_bin
        if np.any(np.abs(ratio-np.rint(ratio)) > 1e-9*np.abs(ratio)):
            raise ValueError('mass faces must be multiples of mass_bin')
    return faces, rho, mf


@njit
def mass_coordinate(z, region, faces, rho, mf):
    return mf[region]+rho[region]*(z-faces[region])


@njit
def mass_local(score, z, region, loss, width, faces, rho, mf):
    local(score, mass_coordinate(z, region, faces, rho, mf), loss, width)


@njit
def mass_segment(score, z0, z1, region, loss, width, faces, rho, mf):
    score_segment(score, mass_coordinate(z0, region, faces, rho, mf),
                  mass_coordinate(z1, region, faces, rho, mf), loss, width)

@njit
def initial(T,z,ux,uy,uz,region,primary,fe,k,brems_enabled=True,x=0.,y=0.):
    a=np.zeros(FIELDS);a[:19]=electron_initial(T,z,ux,uy,uz,region,primary,fe,k);a[19]=-1
    a[22]=new_remaining(brems_enabled)
    a[0]=x;a[1]=y
    return a
@njit
def energy_init(a,fe):
    a[7]=a[6];a[11]=min(fe*a[7],max(0.,a[7]-CUT))
    a[10]=np.random.random()*a[11];a[8]=a[10];a[9]=a[11]-a[10];a[18]=0.
@njit
def photon_state(E,z,ux,uy,uz,ancestry,x=0.,y=0.):
    a=np.zeros(FIELDS);a[2]=z;a[3]=ux;a[4]=uy;a[5]=uz;a[6]=E;a[20]=ancestry
    a[0]=x;a[1]=y
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
def sample_field(field_half,ssd):
    x=(2*np.random.random()-1)*field_half;y=(2*np.random.random()-1)*field_half
    norm=np.sqrt(x*x+y*y+ssd*ssd)
    return x,y,x/norm,y/norm,ssd/norm

@njit
def deposit_segment(h,b,p0,p1,loss,scores,lo,hi,axes,origins,widths,sizes,sums,records,used):
    if len(records):
        if used[0]<len(records):
            i=used[0];records[i,0]=h;records[i,1]=0
            records[i,2:5]=p0;records[i,5:8]=p1;records[i,8]=loss
        else:used[1]+=1
        used[0]+=1
    for j in range(len(sizes)):
        # Both ends outside the same face: score_box_bins would add exactly zero.
        if ((p0[0]<lo[j,0] and p1[0]<lo[j,0]) or (p0[0]>=hi[j,0] and p1[0]>=hi[j,0]) or
            (p0[1]<lo[j,1] and p1[1]<lo[j,1]) or (p0[1]>=hi[j,1] and p1[1]>=hi[j,1]) or
            (p0[2]<lo[j,2] and p1[2]<lo[j,2]) or (p0[2]>=hi[j,2] and p1[2]>=hi[j,2])):continue
        sums[j]+=score_box_bins(scores[b,j,:sizes[j]],p0,p1,loss,axes[j],origins[j],widths[j],lo[j],hi[j])

@njit
def deposit_point(h,b,p,loss,scores,lo,hi,axes,origins,widths,sizes,sums,records,used):
    if len(records):
        if used[0]<len(records):
            i=used[0];records[i,0]=h;records[i,1]=1
            records[i,2:5]=p;records[i,5:8]=p;records[i,8]=loss
        else:used[1]+=1
        used[0]+=1
    for j in range(len(sizes)):
        if (p[0]<lo[j,0] or p[0]>=hi[j,0] or p[1]<lo[j,1] or p[1]>=hi[j,1] or
            p[2]<lo[j,2] or p[2]>=hi[j,2]):continue
        sums[j]+=score_box_point(scores[b,j,:sizes[j]],p,loss,axes[j],origins[j],widths[j],lo[j],hi[j])

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
                disable_brems=False,infinite_medium=False,record_first=False,
                field_half=None,ssd=None,source_offset=(0.,0.),
                region_lo=None,region_hi=None,region_axes=None,region_origins=None,
                region_widths=None,region_sizes=None,region_dz=.2,region_dz_fine=.05,
                nlayer=0,record_deposits=0,record_sources=0,
                faces=None,rho=None,m_faces=None,layered=False,mass_bin=None,
                fano_source_T=None,fano_source_charge=-1,history_rng=False,
                record_histories=False,fault=0,materials=None,layer_material=None,
                bin_edges=None,bin_owner=None,binding_enabled=True,
                fluorescence_enabled=True,record_boundaries=0,record_fluorescence=0,record_events=0):
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
    fluorescence_counts=np.zeros(8,np.int64)
    fluorescence_records=np.zeros((record_fluorescence,8))
    fluorescence_used=0
    event_records=np.zeros((record_events,10))
    event_used=0
    material = materials[0]
    history_bins=np.zeros(len(bin_edges)-1)
    p8_batches=np.zeros((100,len(history_bins)))
    boundary_counts=np.zeros(5,np.int64)
    boundary_hist=np.zeros(len(BOUNDARY_STRENGTH_EDGES)+1,np.int64)
    boundary_omitted_hist=np.zeros(len(BOUNDARY_STRENGTH_EDGES)+1,np.int64)
    boundary_records=np.zeros((record_boundaries,12))
    boundary_used=0
    p8_max=np.zeros(2)
    source_energy=E
    np.random.seed(seed);nbatch=100;dz=.2
    nmass=0 if mass_bin is None else int(round(m_faces[-1]/mass_bin))
    mass_batches=np.zeros((100,nmass));mass_history=np.zeros(nmass)
    history_mass=np.zeros((n if record_histories else 0,nmass))
    mass_ledger=0;path_total=0.
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
    if region_sizes is None:
        region_lo=np.empty((0,3));region_hi=np.empty((0,3));region_axes=np.empty(0,np.int64)
        region_origins=np.empty(0);region_widths=np.empty(0);region_sizes=np.empty(0,np.int64)
    size=1
    for j in range(len(region_sizes)):size=max(size,region_sizes[j])
    region_scores=np.zeros((100,len(region_sizes),size));region_ledger=np.zeros(len(region_sizes),np.int64)
    records=np.zeros((record_deposits,9));used=np.zeros(2,np.int64)
    sources=np.zeros((min(n,record_sources),7))
    start=np.empty(3)  # reused start point of a charged step (no per-step allocation)
    for h in range(n):
        if history_rng:np.random.seed((seed+h) % 2**32)
        lateral=0.;hn=0.;expected_n=0.;de=0.;radiation=0.
        b=min(99,h*100//n);region_sums=np.zeros(len(region_sizes))
        if spectrum_edges is not None and spectrum_cum is not None:
            u1=np.random.random();u2=np.random.random();j=0
            while spectrum_cum[j]<u1:
                j+=1
            width=spectrum_edges[j+1]-spectrum_edges[j]
            source_energy=spectrum_edges[j]+u2*width
            cell=min(9,int((source_energy-spectrum_edges[j])/width*10))
            nspec[j,cell]+=1
        sx=0.;sy=0.;ux=0.;uy=0.;uz=1.
        if field_half is not None and ssd is not None:sx,sy,ux,uy,uz=sample_field(field_half,ssd)
        sx+=source_offset[0];sy+=source_offset[1]
        stack[0]=photon_state(source_energy,0.,ux,uy,uz,0,sx,sy) if source_state is None else source_state
        if positron_source_T is not None:
            stack[0]=initial(positron_source_T,positron_source_z,0.,0.,1.,0,1,fe,rule_value(positron_source_T,k,rule_kind,rule_energies,rule_values),not disable_brems,sx,sy)
            stack[0,19]=1
        if charged_source_T is not None:
            stack[0]=initial(charged_source_T,charged_source_z,0.,0.,1.,0,1,fe,
                             rule_value(charged_source_T,k,rule_kind,rule_energies,rule_values),not disable_brems,sx,sy)
            stack[0,19]=charged_source_charge
        if fano_source_T is not None:
            m=np.random.random()*m_faces[-1]
            region=min(len(rho)-1,np.searchsorted(m_faces,m,side='right')-1)
            z=faces[region]+(m-m_faces[region])/rho[region]
            uz=2*np.random.random()-1;phi=2*np.pi*np.random.random()
            transverse=np.sqrt(max(0.,1-uz*uz))
            stack[0]=initial(fano_source_T,z,transverse*np.cos(phi),transverse*np.sin(phi),uz,
                             region,1,fe,rule_value(fano_source_T,k,rule_kind,rule_energies,rule_values),not disable_brems)
            stack[0,19]=fano_source_charge
        if source_state is not None and stack[0,19]!=0:
            stack[0,22]=new_remaining(not disable_brems)
        if annihilate_in_flight and stack[0,19]==1:
            stack[0,21]=-np.log(np.random.random())
        if h<len(sources):sources[h]=stack[0,:7]
        top=0;dep=0.;escaped=0.;rest=0.;steps=0;failed=False;positron_escape_rest=0.
        # A supplied positron source includes its two-rest-mass annihilation budget.
        incoming=source_energy if positron_source_T is None else positron_source_T
        if charged_source_T is not None:incoming=charged_source_T
        if fano_source_T is not None:incoming=fano_source_T
        if stack[0,19]==1:rest=2*M;incoming+=2*M
        seen=np.zeros(2,np.int64)
        while top>=0:
            a=stack[top].copy();top-=1
            if a[19]==0:
                steps+=1
                if steps>maxsteps:
                    failed=True;counts[5]+=1;break
                E=a[6]
                material=materials[layer_material[int(a[16])]]
                if E<CUT:
                    if a[17]>=2:fluorescence_counts[3]+=1
                    if a[17]==3:fluorescence_counts[7]+=1
                    if nmass:mass_local(mass_history,a[2],int(a[16]),E,mass_bin,faces,rho,m_faces)
                    score_bins(history_bins,a[2],a[2],int(a[16]),E,bin_edges,bin_owner)
                    if scoring:
                        deposit_point(h,b,a[:3],E,region_scores,region_lo,region_hi,region_axes,region_origins,region_widths,region_sizes,region_sums,records,used)
                        tracked_local(hs,a[2],E,dz,touched,0);tracked_local(fine,a[2],E,.05,touched,1)
                        if score_pair and a[20]:tracked_local(pair_score,a[2],E,.05,touched,2)
                    dep+=E;continue
                c=material_coefficients(E,material)*rho[int(a[16])];mu=np.sum(c)
                if infinite_medium:bb=np.inf;nr=0
                else:bb,nr=boundary(a[2],a[5],int(a[16]),faces)
                tau=-np.log(np.random.random())
                flight=tau/mu;length=min(flight,bb)
                escaped_photon=False;flight_length=0.
                while True:
                    z0=a[2];a[0]+=length*a[3];a[1]+=length*a[4];a[2]+=length*a[5]
                    flight_length+=length
                    if score_kerma and scoring:
                        kt=E*transfer_coefficient(E,pes,pwater,pthresholds)*length
                        kc=E*log_interp(E,nist_E,nist_muen)*length
                        tracked_segment(kerma,z0,a[2],kt,.05,touched,3);tracked_segment(collision_kerma,z0,a[2],kc,.05,touched,4)
                    if bb>flight:break
                    if nr<0 or nr>=len(rho):
                        escaped_photon=True;break
                    tau=max(0.,tau-length*mu)
                    region=int(a[16]);a[2]=faces[region+1] if a[5]>0 else faces[region]
                    a[16]=nr
                    material=materials[layer_material[nr]]
                    c=material_coefficients(E,material)*rho[nr];mu=np.sum(c)
                    bb,nr=boundary(a[2],a[5],int(a[16]),faces)
                    flight=tau/mu;length=min(flight,bb)
                if record_first and steps==1:
                    first_record[5]+=flight_length;first_record[6]+=flight_length*flight_length
                if escaped_photon:
                    if record_first and steps==1:first_record[0]+=1
                    if a[17]>=2:fluorescence_counts[1]+=1
                    if a[17]==3:fluorescence_counts[5]+=1
                    escaped+=E;counts[7]+=1;continue
                r=np.random.random()*mu;process=0
                while process<3 and r>=c[process]:r-=c[process];process+=1
                counts[8+process]+=1
                if record_first and steps==1:first_record[1+process]+=1
                if top+3>=maxstack:
                    failed=True;counts[6]+=1;break
                if process==0:
                    ep,gx,gy,gz,T,ex,ey,ez=compton(E,a[3],a[4],a[5])
                    photon=photon_state(ep,a[2],gx,gy,gz,int(a[20]),a[0],a[1]);photon[16]=a[16];photon[17]=a[17];top+=1;stack[top]=photon
                    child=initial(T,a[2],ex,ey,ez,int(a[16]),0,fe,rule_value(T,k,rule_kind,rule_energies,rule_values),not disable_brems,a[0],a[1]);child[20]=a[20]
                    top+=1;stack[top]=child
                elif process==1:
                    if a[17]>=2:fluorescence_counts[2]+=1
                    if a[17]==3:fluorescence_counts[6]+=1
                    T=E
                    if binding_enabled:
                        element,shell=select_shell(E,material)
                        if element<0:
                            binding=E
                            emitted=np.empty((0,4))
                            local_energy=E
                        else:
                            binding=material.bindings[element,shell]
                            emitted,local_energy,trace=relax_vacancy(element,shell,material,fluorescence_enabled)
                        T=E-binding
                        if T<0:raise ValueError('negative photoelectron energy')
                        if top+len(emitted)+2>=maxstack:
                            failed=True;counts[6]+=1;break
                        dep+=local_energy
                        score_bins(history_bins,a[2],a[2],int(a[16]),local_energy,bin_edges,bin_owner)
                        if scoring:
                            tracked_local(hs,a[2],local_energy,dz,touched,0)
                            tracked_local(fine,a[2],local_energy,.05,touched,1)
                            if score_pair and a[20]:tracked_local(pair_score,a[2],local_energy,.05,touched,2)
                        for j in range(len(emitted)):
                            top+=1
                            stack[top]=photon_state(emitted[j,0],a[2],emitted[j,1],emitted[j,2],emitted[j,3],int(a[20]),a[0],a[1])
                            stack[top,16]=a[16];stack[top,17]=2
                            for line_index in range(material.line_counts[element,0]):
                                if emitted[j,0]==material.lines[element,0,line_index,0]:stack[top,17]=3
                            fluorescence_counts[0]+=1
                            if stack[top,17]==3:fluorescence_counts[4]+=1
                            if fluorescence_used<len(fluorescence_records):
                                fluorescence_records[fluorescence_used]=np.array([emitted[j,0],a[0],a[1],a[2],emitted[j,1],emitted[j,2],emitted[j,3],a[16]])
                            fluorescence_used+=1
                    child=initial(T,a[2],a[3],a[4],a[5],int(a[16]),0,fe,rule_value(T,k,rule_kind,rule_energies,rule_values),not disable_brems,a[0],a[1]);child[20]=a[20]
                    top+=1;stack[top]=child
                else:
                    p=pair(E,a[3],a[4],a[5],material.dl,material.bpar,material.delcm,material.delpos)
                    rest+=2*M
                    first=initial(p[0],a[2],p[1],p[2],p[3],int(a[16]),0,fe,rule_value(p[0],k,rule_kind,rule_energies,rule_values),not disable_brems,a[0],a[1]);first[19]=p[4];first[20]=1
                    second=initial(p[5],a[2],p[6],p[7],p[8],int(a[16]),0,fe,rule_value(p[5],k,rule_kind,rule_energies,rule_values),not disable_brems,a[0],a[1]);second[19]=p[9];second[20]=1
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
                    if nmass:mass_local(mass_history,a[2],int(a[16]),a[6],mass_bin,faces,rho,m_faces)
                    score_bins(history_bins,a[2],a[2],int(a[16]),a[6],bin_edges,bin_owner)
                    if scoring:
                        deposit_point(h,b,a[:3],a[6],region_scores,region_lo,region_hi,region_axes,region_origins,region_widths,region_sizes,region_sums,records,used)
                        tracked_local(hs,a[2],a[6],dz,touched,0);tracked_local(fine,a[2],a[6],.05,touched,1)
                        if score_pair and a[20]:tracked_local(pair_score,a[2],a[6],.05,touched,2)
                    if score_lateral:lateral+=a[6]*(a[0]**2+a[1]**2)
                    dep+=a[6];a[6]=0.;break
                material=materials[layer_material[region]]
                ev=max(CUT,a[7])
                if ev!=cached_ev:
                    brate,bhard,bsoft,bmax=material_quantities(ev,material,disable_brems)
                    # Same operation order as restricted_stopping (and P3s's stopping when disabled).
                    S=material_collision(ev,material)-(material_hard(ev,material)[1] if hard_on else 0.)+bsoft
                    rate=material_hard(ev,material)[0] if hard_on else 0.
                    if backend_rates is None:
                        total,G1=elastic_rates(ev)
                    else:
                        total,G1=table_rates(ev,material.gs_T,material.rates)
                    if mode==1:
                        if backend_rates is None:eta,B,totalj,tmp=elastic(ev)
                        G1=total
                    if mode==2:G1=0.
                    if annihilate_in_flight and a[19]==1:arate=annihilation_scale*material_annihilation_rate(ev,material.ne)
                    cached_ev=ev
                density=rho[region]
                Sq=S*(1. if fault==2 else density)
                rateq=rate*density;G1q=G1*(1. if fault==1 else density)
                brateq=brate*density;bsoftq=bsoft*density;arateq=arate*density
                # Recompute all event distances after every direction/energy update.
                eb=a[8]/Sq
                mb=a[12]/G1q if G1q>0 else np.inf
                hb=a[15]/rateq if rateq>0 else np.inf
                if a[18]==1:mb=np.inf;hb=np.inf
                if ignore_boundaries:bb=np.inf;nr=region
                else:
                    bb,nr=boundary(a[2],a[5],region,faces);counts[4]+=1
                rb=a[22]/brateq if brateq>0 and a[18]!=1 else np.inf
                step=min(eb,mb,hb,bb,rb,a[6]/Sq)
                ab=np.inf
                if annihilate_in_flight and a[19]==1 and a[18]!=1:
                    ab=a[21]/arateq if arateq>0 else np.inf
                    step=min(step,ab)
                if not np.isfinite(step) or step<0:
                    failed=True;counts[6]+=1;break
                x0=a[0];y0=a[1];z0=a[2];a[0]+=step*a[3];a[1]+=step*a[4];a[2]+=step*a[5]
                path_total+=step*rho[region]
                loss=step*Sq;a[6]-=loss;dep+=loss
                if nmass:mass_segment(mass_history,z0,a[2],region,loss,mass_bin,faces,rho,m_faces)
                score_bins(history_bins,z0,a[2],region,loss,bin_edges,bin_owner)
                radiation+=step*bsoftq
                if a[18]!=1:
                    expected_n+=step*brateq
                    a[22]=advance_remaining(a[22],step,brateq)
                if score_lateral:lateral+=loss*((x0+step*a[3]/2)**2+(y0+step*a[4]/2)**2)
                if scoring:
                    start[0]=x0;start[1]=y0;start[2]=z0;deposit_segment(h,b,start,a[:3],loss,region_scores,region_lo,region_hi,region_axes,region_origins,region_widths,region_sizes,region_sums,records,used)
                    tracked_segment(hs,z0,a[2],loss,dz,touched,0);tracked_segment(fine,z0,a[2],loss,.05,touched,1)
                    if score_pair and a[20]:tracked_segment(pair_score,z0,a[2],loss,.05,touched,2)
                a[8]=max(0.,a[8]-loss);a[12]=max(0.,a[12]-step*G1q);a[15]=max(0.,a[15]-step*rateq)
                if annihilate_in_flight and a[19]==1 and a[18]!=1:
                    a[21]=max(0.,a[21]-step*arateq)
                counts[0]+=1
                # Deterministic priority: boundary, energy, scattering, hard, brems, annihilation.
                if event_used<len(event_records):
                    event=6
                    if bb<=step+1e-13:event=0
                    elif eb<=step+1e-13:event=1
                    elif mb<=step+1e-13:event=2
                    elif hb<=step+1e-13:event=3
                    elif rb<=step+1e-13:event=4
                    elif ab<=step+1e-13:event=5
                    event_records[event_used]=np.array([event,region,a[2],step,bb if np.isfinite(bb) else -1.,eb if np.isfinite(eb) else -1.,mb if np.isfinite(mb) else -1.,hb if np.isfinite(hb) else -1.,rb if np.isfinite(rb) else -1.,ab if np.isfinite(ab) else -1.])
                event_used+=1
                if bb<=step+1e-13:
                    a[2]=faces[region+1] if a[5]>0 else faces[region]
                    counts[3]+=1
                    if 0<=nr<len(rho) and layer_material[region]!=layer_material[nr]:
                        info=material_boundary(a,region,nr,material,total,G1,
                            rule_value(ev,k,rule_kind,rule_energies,rule_values))
                        boundary_counts[0]+=1
                        boundary_counts[1]+=int(info[2])
                        boundary_counts[2]+=int(info[0]==0.)
                        boundary_counts[3]+=int(info[3])
                        boundary_counts[4]+=int(int(a[16])==region)
                        cell=np.searchsorted(BOUNDARY_STRENGTH_EDGES,info[0],side='right')
                        boundary_hist[cell]+=1
                        if info[2]:boundary_omitted_hist[cell]+=1
                        if boundary_used<len(boundary_records):
                            boundary_records[boundary_used]=np.array([a[2],region,nr,a[16],info[0],info[1],a[8],a[12],a[13],a[14],a[15],a[22]])
                        boundary_used+=1
                        cached_ev=-1.
                    else:
                        a[16]=nr
                    continue
                if eb<=step+1e-13:
                    if a[18]==1:
                        if nmass:mass_local(mass_history,a[2],int(a[16]),a[6],mass_bin,faces,rho,m_faces)
                        score_bins(history_bins,a[2],a[2],int(a[16]),a[6],bin_edges,bin_owner)
                        if scoring:
                            deposit_point(h,b,a[:3],a[6],region_scores,region_lo,region_hi,region_axes,region_origins,region_widths,region_sizes,region_sums,records,used)
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
                    scattering_hinge(a,ev,rule_value(ev,k,rule_kind,rule_energies,rule_values),material.gs_T,material.gs_K,material.gs_u,material.gs_y,total,G1)
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
                    child=initial(W,a[2],ds[0],ds[1],ds[2],int(a[16]),0,fe,rule_value(W,k,rule_kind,rule_energies,rule_values),not disable_brems,a[0],a[1]);child[20]=a[20]
                    if mode==1:child[12]=-np.log(np.random.random())
                    stack[top]=child;maxdepth=max(maxdepth,top+1);break
                if rb<=step+1e-13:
                    energy_init(a,fe)
                    T=a[6]
                    a[22]=new_remaining(not disable_brems)
                    if T<=CUT:continue
                    photon_k,excess=material_sample_k(T,material)
                    brems[2]+=excess
                    if photon_k>T:brems[3]+=1
                    if excess or photon_k>T:
                        failed=True;counts[6]+=1;break
                    q=material_quantities(T,material)
                    hn+=1.;de+=photon_k-q[1]/q[0];radiation+=photon_k
                    brems[0]+=1.;brems[1]+=photon_k
                    gx,gy,gz=direction(T,a[3],a[4],a[5],2*np.pi*np.random.random())
                    # Overflow leaves the pre-emission kinetic budget unresolved.
                    if brems_mode==2 and top+2>=maxstack:
                        failed=True;counts[6]+=1;break
                    a[6]-=photon_k;energy_init(a,fe)
                    if brems_mode==1:
                        dep+=photon_k
                        if nmass:mass_local(mass_history,a[2],int(a[16]),photon_k,mass_bin,faces,rho,m_faces)
                        score_bins(history_bins,a[2],a[2],int(a[16]),photon_k,bin_edges,bin_owner)
                        if scoring:
                            deposit_point(h,b,a[:3],photon_k,region_scores,region_lo,region_hi,region_axes,region_origins,region_widths,region_sizes,region_sums,records,used)
                            tracked_local(hs,a[2],photon_k,dz,touched,0)
                            tracked_local(fine,a[2],photon_k,.05,touched,1)
                            if score_pair and a[20]:tracked_local(pair_score,a[2],photon_k,.05,touched,2)
                        if score_lateral:lateral+=photon_k*(a[0]**2+a[1]**2)
                        continue
                    top+=1;stack[top]=a
                    top+=1;stack[top]=photon_state(photon_k,a[2],gx,gy,gz,int(a[20]),a[0],a[1])
                    stack[top,16]=a[16]
                    maxdepth=max(maxdepth,top+1);suspended=True;break
                if annihilate_in_flight and ab<=step+1e-13:
                    T=a[6]
                    if T<=CUT:
                        a[21]=-np.log(np.random.random());continue
                    if top+3>=maxstack:
                        failed=True;counts[6]+=1;break
                    g=photons(T,a[3],a[4],a[5])
                    top+=1;stack[top]=photon_state(g[0],a[2],g[1],g[2],g[3],int(a[20]),a[0],a[1])
                    stack[top,16]=a[16]
                    top+=1;stack[top]=photon_state(g[4],a[2],g[5],g[6],g[7],int(a[20]),a[0],a[1])
                    stack[top,16]=a[16]
                    rest-=2*M;a[6]=0.;annihilated=True;counts[14]+=1
                    if a[17]==1:counts[15]+=1
                    maxdepth=max(maxdepth,top+1);break
                # Actual-energy endpoint: deposit only the remaining ledger energy.
                if nmass:mass_local(mass_history,a[2],int(a[16]),a[6],mass_bin,faces,rho,m_faces)
                score_bins(history_bins,a[2],a[2],int(a[16]),a[6],bin_edges,bin_owner)
                if scoring:
                    deposit_point(h,b,a[:3],a[6],region_scores,region_lo,region_hi,region_axes,region_origins,region_widths,region_sizes,region_sums,records,used)
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
                top+=1;stack[top]=photon_state(M,a[2],ux,uy,uz,int(a[20]),a[0],a[1])
                stack[top,16]=a[16]
                top+=1;stack[top]=photon_state(M,a[2],-ux,-uy,-uz,int(a[20]),a[0],a[1])
                stack[top,16]=a[16]
                maxdepth=max(maxdepth,top+1)
        unresolved=0.
        if failed:
            unresolved=a[6]
            for j in range(top+1):unresolved+=stack[j,6]
        for j in range(len(region_sizes)):
            error=abs(region_sums[j]-dep) if j==2 else max(0.,region_sums[j]-dep)
            if error>1e-9*incoming:region_ledger[j]+=1
        if nmass:
            if abs(np.sum(mass_history)-dep)>1e-9*incoming:mass_ledger+=1
            mass_batches[b]+=mass_history
            if record_histories:history_mass[h]=mass_history
            mass_history[:]=0.
        p8_error=abs(np.sum(history_bins)-dep)
        p8_max[0]=max(p8_max[0],p8_error)
        p8_batches[b]+=history_bins
        history_bins[:]=0.
        residual=incoming-dep-escaped-unresolved-rest
        coarse_error=abs(touched_sum(hs,touched,0)-dep) if scoring else 0.
        fine_error=abs(touched_sum(fine,touched,1)-dep) if scoring else 0.
        dn=hn-expected_n
        brems[4]+=dn;brems[5]+=dn*dn;brems[6]+=de;brems[7]+=de*de
        brems[8]+=radiation;brems[9]+=radiation*radiation
        maxima[0]=max(maxima[0],abs(residual)/incoming)
        maxima[1]=max(maxima[1],coarse_error/incoming);maxima[2]=max(maxima[2],fine_error/incoming)
        if abs(residual)>incoming*1e-12:violations[0]+=1
        if coarse_error>incoming*1e-9:violations[1]+=1
        if fine_error>incoming*1e-9:violations[2]+=1
        totals[0]+=incoming;totals[1]+=dep;totals[2]+=escaped;totals[3]+=unresolved;totals[4]+=rest;totals[5]+=positron_escape_rest
        b=min(99,h*100//n)
        add_clear(batches,hs,touched,0,b);add_clear(fine_batches,fine,touched,1,b)
        add_clear(pair_batches,pair_score,touched,2,b);add_clear(kerma_batches,kerma,touched,3,b)
        add_clear(collision_batches,collision_kerma,touched,4,b)
        lateral_batches[b]+=lateral;step_batches[b]+=steps
    nc=int(round(thickness/region_dz));nfine=int(round(thickness/region_dz_fine))
    axis_coarse=np.zeros((100,nc));axis_fine=np.zeros((100,nfine));slab_coarse=np.zeros((100,nc))
    npcoarse=0;npfine=0
    if len(region_sizes):
        axis_coarse=region_scores[:,0,:nc].copy();axis_fine=region_scores[:,1,:nfine].copy()
        slab_coarse=region_scores[:,2,:nc].copy()
        npcoarse=region_sizes[3];npfine=region_sizes[3+2*nlayer]
    profile=np.zeros((100,2,nlayer,npcoarse));profile_fine=np.zeros((100,2,nlayer,npfine))
    for axis in range(2):
        for d in range(nlayer):
            profile[:,axis,d,:]=region_scores[:,3+axis*nlayer+d,:npcoarse]
            profile_fine[:,axis,d,:]=region_scores[:,3+2*nlayer+axis*nlayer+d,:npfine]
    return (batches,fine_batches,pair_batches,kerma_batches,collision_batches,totals,maxima,violations,counts,maxdepth,nspec,lateral_batches,step_batches,brems,first_record,axis_coarse,axis_fine,slab_coarse,profile,profile_fine,region_ledger,records[:min(used[0],len(records))],used[1],sources,(max(0,n-record_sources) if record_sources else 0),mass_batches,history_mass,mass_ledger,path_total),(p8_batches,p8_max,boundary_counts,boundary_hist,boundary_records[:min(boundary_used,len(boundary_records))],fluorescence_counts,fluorescence_records[:min(fluorescence_used,len(fluorescence_records))],event_records[:min(event_used,len(event_records))],boundary_omitted_hist)

def expand_regions(regions,thickness,dz,dz_fine):
    """Expand boxes once at the Python entrance; no per-history bin arrays."""
    for width in (dz,dz_fine,.2):
        if not np.isfinite(width) or width<=0 or abs(thickness/width-round(thickness/width))>1e-9:
            raise ValueError('slab thickness must be a multiple of scoring bins')
    if regions is None:return {}
    r=np.asarray(regions,dtype=float)
    if r.ndim!=1 or not 9<=len(r)<=11 or not np.all(np.isfinite(r)):
        raise ValueError('invalid regions length or non-finite regions')
    cx,cy,half,orth,extent,coarse,fine,layer=r[:8];depths=r[8:]
    if np.any(r[2:8]<=0):raise ValueError('region widths must be positive')
    for width in (coarse,fine):
        if abs(extent/width-round(extent/width))>1e-9:raise ValueError('scan extent must be a multiple of bin width')
    if np.any(depths-layer/2<0) or np.any(depths+layer/2>thickness):raise ValueError('depth layer outside slab')
    lo=[];hi=[];axes=[];origins=[];widths=[];sizes=[]
    def box(low,high,axis,origin,width,number):
        lo.append(low);hi.append(high);axes.append(axis);origins.append(origin);widths.append(width);sizes.append(number)
    for width in (dz,dz_fine):box([cx-half,cy-half,0.],[cx+half,cy+half,thickness],2,0.,width,int(round(thickness/width)))
    box([-np.inf,-np.inf,0.],[np.inf,np.inf,thickness],2,0.,dz,int(round(thickness/dz)))
    for width in (coarse,fine):
        for axis in (0,1):
            for depth in depths:
                low=[cx-orth,cy-orth,depth-layer/2];high=[cx+orth,cy+orth,depth+layer/2]
                center=cx if axis==0 else cy
                low[axis]=center-extent;high[axis]=center+extent
                box(low,high,axis,center-extent,width,int(round(2*extent/width)))
    return dict(region_lo=np.array(lo),region_hi=np.array(hi),region_axes=np.array(axes,np.int64),
        region_origins=np.array(origins),region_widths=np.array(widths),region_sizes=np.array(sizes,np.int64),nlayer=len(depths))


# All rates/stopping powers in this bundle are per gram (cm2/g, MeV cm2/g).
# Only the transport loop multiplies by mass density; GS total/G1 are divided
# by the generation density when P8a's length-based tables are loaded.
from collections import namedtuple
from functools import lru_cache
from pathlib import Path
import json
import hashlib

Material = namedtuple('Material', 'ne excitation c z nj stop_T stop_delta stop_collision stop_rad '
    'gs_T gs_K gs_u gs_y rates photo_E photo pair_E pair_n pair_e thresholds '
    'dl bpar delcm delpos bindings pm shell_nj yields lines line_counts brems_E brems_V log_stop_T')


@njit
def material_collision(T, m):
    if len(m.stop_collision):
        return log_interp(T,m.stop_T,m.stop_collision)
    # The legacy bundle represents P4's analytic stopping model, including its
    # exact constants and operation order. Production bundles carry ESTAR data.
    tau=T/M;g=1+tau;b2=1-1/(g*g);d=tau/2
    F=-1-b2+np.log((tau-d)*d)+tau/(tau-d)+(d*d/2+(2*tau+1)*np.log1p(-d/tau))/(g*g)
    density=np.interp(np.log(T),m.log_stop_T,m.stop_delta)
    return m.c/b2*(np.log(2*(tau+2)/(m.excitation/M)**2)+F-density)


@njit
def material_hard(T, m):
    if T<=2*DELTA:return 0.,0.
    g=1+T/M;b2=1-1/g**2
    a=(g-1)**2/g**2;b=(2*g-1)/g**2;c=m.c/(b2*T);x=DELTA/T
    if .5-x<1e-7:
        h=(T/2-DELTA)/T;mid=.5-h/2
        v=a+1/mid**2+1/(1-mid)**2-b/(mid*(1-mid))
        return c*h*v,c*T*h*mid*v
    return c*(primitive0(.5,a,b)-primitive0(x,a,b)),c*T*(primitive1(.5,a,b)-primitive1(x,a,b))


@njit
def material_annihilation_rate(T, ne):
    if T<=0:raise ValueError('annihilation requires positive kinetic energy')
    g=1+T/M;p=np.sqrt(g*g-1)
    cross=np.pi*RE**2/(g+1)*((g*g+4*g+1)/(g*g-1)*np.log(g+p)-(g+3)/p)
    return ne*cross


@njit
def material_coefficients(E, m):
    if E<m.pair_E[0] or E>m.pair_E[-1]:raise ValueError('photon outside table')
    x=E/M;L=np.log1p(2*x)
    kn=2*np.pi*RE**2*m.ne*((1+x)/x**2*(2*(1+x)/(1+2*x)-L/x)+L/(2*x)-(1+3*x)/(1+2*x)**2)
    # np.interp selects the upper side at duplicate absorption-edge nodes.
    pe=log_interp(E,m.photo_E,m.photo)
    i=np.searchsorted(m.photo_E,E,side='right')-1
    if 0<=i<len(m.photo_E)-1:
        near_edge=(i>0 and m.photo_E[i]==m.photo_E[i-1]) or (i+2<len(m.photo_E) and m.photo_E[i+1]==m.photo_E[i+2])
        if near_edge:
            # Select the physical side BEFORE taking logarithms: neighbouring
            # floating-point energies can have identical rounded logarithms.
            u=np.log(E/m.photo_E[i])/np.log(m.photo_E[i+1]/m.photo_E[i])
            pe=np.exp(np.log(m.photo[i])+u*np.log(m.photo[i+1]/m.photo[i]))
    return np.array([kn,pe,pair_interp(E,m.pair_E,m.pair_n,m.thresholds[0]),
                     pair_interp(E,m.pair_E,m.pair_e,m.thresholds[1])])


@njit
def shell_distribution(E,m):
    l=np.log(E*1000);xs=np.zeros(m.bindings.shape)
    for i in range(len(m.bindings)):
        for s in range(5):
            if E>m.bindings[i,s] and (s==0 or m.pm[i,s,0]!=0):
                q=m.pm[i,s]
                xs[i,s]=np.exp(q[0]+q[1]*l+q[2]*l*l+q[3]*l*l*l)
        if E<=m.bindings[i,4]:xs[i,:]=0.
        xs[i,:]*=m.shell_nj[i]
        if E<=m.bindings[i,3]:
            total=np.sum(xs[i]);xs[i,:]=0.;xs[i,4]=total
    total=np.sum(xs)
    return xs/total if total>0 else xs


@njit
def select_shell(E,m):
    probs=shell_distribution(E,m)
    if np.sum(probs)==0:return -1,-1
    r=np.random.random();last_i=-1;last_s=-1
    for i in range(len(probs)):
        for s in range(5):
            if probs[i,s]>0:
                last_i=i;last_s=s
                if r<probs[i,s]:return i,s
                r-=probs[i,s]
    return last_i,last_s


@njit
def relax_vacancy(element,shell,m,enabled=True):
    """Return transported photons (E,ux,uy,uz), local budget, transition trace.

    A small explicit vacancy stack is independent of the charged-particle loop.
    Trace rows: origin shell, selected line index (-1 nonradiative), child shell.
    """
    initial_binding=m.bindings[element,shell]
    queue=np.empty(8,np.int64);queue[0]=shell;top=0
    out=np.zeros((8,4));trace=np.full((8,3),-1,np.int64);used=0;events=0;emitted=0.
    while top>=0:
        s=queue[top];top-=1
        if s>=4 or not enabled:continue
        if np.random.random()>=m.yields[element,s]:
            trace[events,0]=s;events+=1;continue
        r=np.random.random();j=0
        while j<m.line_counts[element,s]-1 and r>=m.lines[element,s,j,1]:
            r-=m.lines[element,s,j,1];j+=1
        energy=m.lines[element,s,j,0];child=int(m.lines[element,s,j,2])
        trace[events]=np.array([s,j,child]);events+=1
        if energy>=CUT:
            ux,uy,uz=annihilation()
            out[used]=np.array([energy,ux,uy,uz]);used+=1;emitted+=energy
        if child<4:
            top+=1;queue[top]=child
    local_energy=initial_binding-emitted
    if local_energy<0:raise ValueError('negative relaxation deposition')
    return out[:used],local_energy,trace[:events]


@njit
def score_bins(score,z0,z1,region,loss,edges,owner):
    if loss==0:return
    lo=min(z0,z1);hi=max(z0,z1)
    if hi-lo<=1e-15:
        j=min(len(score)-1,max(0,np.searchsorted(edges,z0,side='right')-1))
        if owner[j]!=region and j>0 and z0==edges[j] and owner[j-1]==region:j-=1
        if owner[j]!=region:raise ValueError('point outside scoring layer')
        score[j]+=loss;return
    first=max(0,np.searchsorted(edges,lo,side='right')-1)
    last=min(len(score)-1,np.searchsorted(edges,hi,side='left'))
    for j in range(first,last+1):
        if owner[j]==region:
            overlap=max(0.,min(hi,edges[j+1])-max(lo,edges[j]))
            score[j]+=loss*overlap/(hi-lo)


@njit
def material_boundary(a,old,nr,m,total,G1,knew,mu_override=2.):
    """Truncate the old scattering hinge; retain loss and Poisson clocks.

    mu_override is solely a deterministic test hook; normal transport samples
    the old material's GS distribution. Return s,mu,omitted,terminal,cache=-1.
    """
    s=max(0.,(a[14]-a[13])-a[12]);mu=1.;terminal=a[18]==1
    omitted=False
    if not terminal:
        if s>=m.gs_K[0]:
            mu=gs_mu_cached(max(CUT,a[7]),s,m.gs_T,m.gs_K,m.gs_u,m.gs_y,total,G1) if mu_override>1 else mu_override
            a[3],a[4],a[5]=rotate(a[3],a[4],a[5],mu,2*np.pi*np.random.random())
        elif s>0:omitted=True
    forward=(nr>old and a[5]>0) or (nr<old and a[5]<0)
    a[16]=nr if forward else old
    if not terminal:
        u=np.random.random();a[12]=u*knew;a[13]=(1-u)*knew;a[14]=knew
    return np.array([s,mu,(1. if omitted else 0.),(1. if terminal else 0.),-1.])

from scipy.integrate import quad
from .bremsstrahlung_p3r import (MAX_T, ALRAD_H, ALRADP_H, LOG_NUCLEAR,
    LOG_ELECTRON, SCREEN_SCALE, SCREEN_OFFSET)

@njit
def material_shape(T, k, Z, NJ):
    """Unnormalized k*dSigma/dk, summed over the supplied composition, cm2/g."""
    if T > MAX_T or T <= 0.:raise ValueError('brems electron energy outside (0,20] MeV')
    if k < 0. or k > T: return 0.
    E = T+M; rem = E-k; y = rem/E; result = 0.
    for j in range(len(Z)):
        z = Z[j]; az2 = (ALPHA*z)**2
        fc = az2*(1/(1+az2)+.20206+az2*(-.0369+az2*(.0083-.002*az2)))
        xi = ALRADP_H/(ALRAD_H-fc) if z==1 else np.log(LOG_ELECTRON*z**(-2/3))/(np.log(LOG_NUCLEAR*z**(-1/3))-fc)
        xl = 4/3*np.log(z); d = SCREEN_SCALE*z**(-1/3)*M/E*k/rem
        if d >= np.exp((21.12-xl)/4.184)-SCREEN_OFFSET:continue
        if d <= 1:
            s1 = 20.867+d*(-3.242+.625*d)-xl
            s2 = 20.209+d*(-1.930-.086*d)-xl
        else:
            s1 = 21.12-4.184*np.log(d+SCREEN_OFFSET)-xl; s2 = s1
        result += NJ[j]*RE**2*ALPHA*z*(z+xi)*((1+y*y)*s1-2/3*y*s2)
    return result


def material_integral(T, lo, hi, Z, NJ, logarithmic=False):
    # Split at analytic branch switches and the zero endpoint for each element.
    E = T+M; points = []
    for z in Z:
        c = SCREEN_SCALE*z**(-1/3)*M/E
        for d in (1.,np.exp((21.12-4/3*np.log(z))/4.184)-SCREEN_OFFSET):
            k = d*E/(c+d)
            if lo < k < hi:points.append(np.log(k) if logarithmic else k)
    if logarithmic:
        return quad(lambda u:material_shape(T,np.exp(u),Z,NJ),np.log(lo),np.log(hi),
                    points=sorted(points),epsabs=1e-12,epsrel=2e-10,limit=100)[0]
    return quad(lambda k:material_shape(T,k,Z,NJ),lo,hi,points=sorted(points),
                epsabs=1e-12,epsrel=2e-10,limit=100)[0]


def brems_tables(z,nj,stop,energies=None):
    """Parameterised P3r B normalization; no EGS correction table."""
    if energies is None:
        dense=np.geomspace(CUT,MAX_T,int(np.ceil(np.log(MAX_T/CUT)/.003))+1)
        near=CUT+np.geomspace(1e-8,1e-3,401)
        energies=np.unique(np.r_[stop['T'][(stop['T']>=CUT)&(stop['T']<=MAX_T)],dense,near,CUT,MAX_T])
    energies=np.asarray(energies,dtype=float)
    values=np.zeros((len(energies),5))
    for i,T in enumerate(energies):
        rad=np.interp(np.log(T),np.log(stop['T']),stop['radiative'])
        norm=rad/material_integral(T,0.,T,z,nj)
        soft=norm*material_integral(T,0.,min(T,CUT),z,nj)
        rate=norm*material_integral(T,CUT,T,z,nj,True) if T>CUT else 0.
        envelope=max(material_shape(T,x,z,nj) for x in np.geomspace(CUT,T,1025))*1.01
        values[i]=rate,rad-soft if T>CUT else 0.,soft,envelope,norm
    return energies,values


@njit
def material_quantities(T,m,disabled=False):
    if T>MAX_T:raise ValueError('brems energy above 20 MeV')
    rad=np.interp(np.log(max(T,CUT)),m.log_stop_T,m.stop_rad)
    if disabled or T<=CUT:return 0.,0.,rad,0.
    energies=m.brems_E;v=m.brems_V
    i=min(len(energies)-2,max(0,np.searchsorted(energies,T)-1))
    # Preserve P3r's exact logarithm subtraction/order in the legacy data model.
    u=(np.log(T)-np.log(energies[i]))/(np.log(energies[i+1])-np.log(energies[i]))
    if i==0:
        rate=u*v[1,0];discrete=u*v[1,1]
        return rate,discrete,rad-discrete,u*v[1,3]+(1-u)*v[0,3]
    rate=np.exp((1-u)*np.log(v[i,0])+u*np.log(v[i+1,0]))
    soft=np.exp((1-u)*np.log(v[i,2])+u*np.log(v[i+1,2]))
    discrete=np.exp((1-u)*np.log(v[i,1])+u*np.log(v[i+1,1]))
    if len(m.stop_collision):soft=rad-discrete
    return rate,discrete,soft,(1-u)*v[i,3]+u*v[i+1,3]


@njit
def material_sample_k(T,m):
    envelope=material_quantities(T,m)[3]
    lo=np.log(CUT);width=np.log(T/CUT)
    while True:
        k=min(T,max(CUT,np.exp(lo+width*np.random.random())))
        f=material_shape(T,k,m.z,m.nj)
        if f>envelope:return k,1
        if np.random.random()*envelope<=f:return k,0


def make_material(c,stop,photon,gs,pair_data,shells,fluor,brems,*,analytic=False,density=1.):
    """Build one verified shape/units-compatible numeric bundle for Numba."""
    rates=np.asarray(gs['rates'],dtype=float).copy()
    rates[:,1 if rates.shape[1]==3 else 0:]/=density
    legacy='water' in photon
    photo_E=photon['E'] if legacy else photon['photo_E']
    photo=photon['water'][:,2] if legacy else photon['photo']
    pn=photon['water'][:,3] if legacy else photon['pair_nuclear']
    pe=photon['water'][:,4] if legacy else photon['pair_electron']
    bpar=float(np.ravel(pair_data['bpar'])[0]);delpos=float(np.ravel(pair_data['delpos'])[0])
    arrays=[np.asarray(v,dtype=float) for v in (c['Z'],c['NJ'],stop['T'],stop['delta'],
        np.empty(0) if analytic else stop['collision'],stop['radiative'],gs['T'],gs['K'],gs['u'],gs['y'],rates,
        photo_E,photo,photon['E'],pn,pe,photon['thresholds'],pair_data['dl'])]
    for a in arrays:
        if not np.all(np.isfinite(a)):raise ValueError('nonfinite material data')
    return Material(float(c['NE']),float(c['I_MeV']),float(2*np.pi*RE**2*M*c['NE']),*arrays[:17],
        arrays[17],bpar,float(pair_data['delcm']),delpos,
        np.asarray(shells['binding_MeV'],float),np.asarray(shells['pm'],float),np.asarray(shells['NJ'],float),
        np.asarray(fluor['yields'],float),np.asarray(fluor['lines'],float),np.asarray(fluor['line_counts'],np.int64),
        np.asarray(brems[0],float),np.asarray(brems[1],float),np.log(np.asarray(stop['T'],float)))


def empty_relaxation(z,nj):
    return (dict(binding_MeV=np.zeros((len(z),5)),pm=np.zeros((len(z),5,4)),NJ=np.asarray(nj)),
            dict(yields=np.zeros((len(z),4)),lines=np.zeros((len(z),4,1,4)),line_counts=np.zeros((len(z),4),np.int64)))


def legacy_bundle():
    """Actual P8a legacy_water data, never dispatching to a different transport."""
    from .materials_p8a import legacy_water
    from .bremsstrahlung_p3r import tables
    data=legacy_water()
    # Read the existing pair artifact; pair_parameters() would create cache data.
    from .p2_common import CACHE
    with np.load(CACHE/'pair_parameters.npz') as f:pair_data={k:f[k].copy() for k in f.files}
    z=np.array([int(k.split('_')[0][1:]) for k in data['elastic'] if k.endswith('q-1')],float)
    z=np.sort(z)
    c=dict(Z=z,NJ=data['NJ'],NE=data['NE'],I_MeV=data['I'])
    shells,fluor=empty_relaxation(z,data['NJ'])
    return make_material(c,data['stopping'],data['photon'],data['gs'],pair_data,shells,fluor,tables(),analytic=True)


def bundle_checks(m):
    """Independent quadrature at nodes and held-out midpoints; reject bad tables."""
    errors=[];excess=0
    stride=max(1,len(m.brems_E)//32)
    mids=np.sqrt(m.brems_E[:-1]*m.brems_E[1:])[::stride]
    for T in np.r_[m.brems_E[::stride],mids,10.]:
        v=material_quantities(float(T),m)
        # Reconstruct normalization independently at each validation energy.
        rad=np.interp(np.log(T),m.log_stop_T,m.stop_rad)
        integral=material_integral(float(T),0.,float(T),m.z,m.nj)
        norm=rad/integral
        hard=norm*material_integral(float(T),CUT,float(T),m.z,m.nj) if T>CUT else 0.
        soft=norm*material_integral(float(T),0.,min(CUT,float(T)),m.z,m.nj)
        errors.append(abs(v[1]+v[2]-rad)/rad)
        errors.append(abs(v[1]-hard)/rad)
        errors.append(abs(v[2]-soft)/rad)
        if T>CUT:
            excess+=sum(material_shape(float(T),x,m.z,m.nj)>v[3] for x in np.geomspace(CUT,T,2049))
    if max(errors)>1e-3 or excess:
        raise ValueError('brems integral / soft-hard split / envelope failed')
    from .physics import hard, NE
    from .flight_annihilation_p2c import annihilation_rate
    for T in (.03,.1,1.,10.):
        if not np.allclose(material_hard(T,m),np.asarray(hard(T))*m.ne/NE,rtol=1e-12,atol=0):
            raise ValueError('Moller NE scaling')
        if not np.isclose(material_annihilation_rate(T,m.ne),annihilation_rate(T)*m.ne/NE,rtol=1e-12):
            raise ValueError('annihilation NE scaling')
    return dict(status='pass',max_relative_integral_error=max(errors),envelope_excess=excess,
                units='cm2/g; MeV cm2/g; density multiplied only in transport')


def prepare_bundles(cache):
    from .materials_p8a import load_manifest,load_reference,OUT,sha,write,MATERIALS
    cache=Path(cache);manifest=load_manifest();reports={}
    for name in MATERIALS:
        row=manifest['materials'][name]
        data={k:load_reference(row[k],name) for k in ('definition','constants','stopping','photon','gs','pair','shells')}
        # Exercise composition-aware identity construction without regenerating DCS.
        from .materials_p8a import MixtureElsepa
        refs=load_reference(row['elastic'],name)
        atomic={int(load_reference(ref,'element')['Z']):load_reference(ref,'element')
                for key,ref in refs.items() if key.endswith('q-1.npz')}
        provider=MixtureElsepa(data['definition'],atomic)
        if provider.energy_max<10.:raise ValueError('ELSEPA range')
        path=cache/(name+'_fluorescence.npz')
        receipt=json.loads(path.with_suffix('.json').read_text())
        if (sha(path)!=receipt['sha256'] or receipt['shells_sha256']!=row['shells']['sha256']
            or receipt['generator_sha256']!=sha(Path(__file__).with_name('generate_fluorescence_p8e.py'))):
            raise ValueError('fluorescence identity changed')
        with np.load(path) as f:fluor={k:f[k].copy() for k in f.files}
        c=data['constants'];brems=brems_tables(np.array(c['Z'],float),np.array(c['NJ'],float),data['stopping'])
        m=make_material(c,data['stopping'],data['photon'],data['gs'],data['pair'],data['shells'],fluor,brems,
                        density=c['density_g_cm3'])
        reports[name]=bundle_checks(m)
        dest=cache/(name+'_bundle.npz')
        np.savez(dest,**m._asdict())
    identity=dict(manifest_sha256=sha(OUT/'manifest.json'),
                  files={f.name:sha(f) for f in cache.glob('*.npz')},
                  source_hashes=part2_sources(),units='mass',checks=reports,
                  densities=[load_reference(manifest['materials'][n]['constants'],n)['density_g_cm3'] for n in MATERIALS])
    write(cache/'bundles.json',identity)
    return identity


def part2_sources():
    from .materials_p8a import sha,ROOT
    from .physics import DATA
    names=('photon_transport_p8e.py','generate_fluorescence_p8e.py','run_p8e.py',
           'materials_p8a.py','pegs_p8a.py','physics.py','sampling.py','backend_sampling.py',
           'photon_sampling_p2.py','flight_annihilation_p2c.py','transport.py','bremsstrahlung_p3r.py',
           'photon_data_p2.py','scoring_p3b.py','batch_runner_p3u.py','batch_kernels_p3u.py')
    hashes={name:sha(Path(__file__).with_name(name)) for name in names}
    hashes[str((DATA/'metadata.json').relative_to(ROOT))]=sha(DATA/'metadata.json')
    return hashes


def load_bundles(cache):
    from .materials_p8a import sha,OUT,load_manifest,load_reference,MATERIALS
    cache=Path(cache);identity=json.loads((cache/'bundles.json').read_text())
    if identity['manifest_sha256']!=sha(OUT/'manifest.json') or identity['source_hashes']!=part2_sources():
        raise ValueError('P8a or P8e bundle identity changed')
    manifest=load_manifest()  # Verify every referenced table and composition again.
    if (identity['units']!='mass' or any(identity['checks'][n]['status']!='pass' for n in MATERIALS)):
        raise ValueError('material bundle checks/units not passed')
    expected_density=[load_reference(manifest['materials'][n]['constants'],n)['density_g_cm3'] for n in MATERIALS]
    if identity['densities']!=expected_density:raise ValueError('material bundle densities changed')
    for name,expected in identity['files'].items():
        if sha(cache/name)!=expected:raise ValueError('material bundle hash changed')
    result=[]
    for name in MATERIALS:
        with np.load(cache/(name+'_bundle.npz')) as f:
            values=[]
            for field in Material._fields:
                v=f[field].copy()
                values.append(v.item() if v.ndim==0 else v)
            result.append(Material(*values))
    return result,identity


def run_coupled(materials,layers,n,seed,*,densities=None,E=1.,
                binding_enabled=True,fluorescence_enabled=True,**options):
    """One P4-derived path for legacy and P8a bundles; return raw diagnostics.

    Adjacent identical materials are merged for transport, independently of the
    scoring table. This makes artificial subdivisions transparent to all clocks
    and arithmetic. Equal-width scoring subdivisions are canonicalised too.
    """
    from numba.typed import List
    from .run_p8e import layout
    if not isinstance(n,(int,np.integer)) or n<=0:raise ValueError('positive n required')
    if not 0<=int(seed)<2**32:raise ValueError('seed outside uint32')
    if not materials:raise ValueError('material bundle required')
    densities=np.ones(4) if densities is None else np.asarray(densities,float)
    layout(layers,densities)  # Validate original layers before transparent merging.
    canonical=[]
    for mid,thick,width in layers:
        if mid>len(materials):raise ValueError('layer material missing')
        if canonical and canonical[-1][0]==mid and canonical[-1][2]==width:
            a,b,c=canonical.pop();canonical.append((a,b+thick,c))
        else:canonical.append((mid,thick,width))
    grid=layout(canonical,densities)
    faces=[0.];ids=[];rho=[]
    for mid,thick,width in canonical:
        if ids and ids[-1]==mid-1:
            faces[-1]+=thick
        else:
            faces.append(faces[-1]+thick);ids.append(mid-1);rho.append(densities[mid-1])
    faces=np.asarray(faces);ids=np.asarray(ids,np.int64);rho=np.asarray(rho)
    centers=(grid['edges_cm'][1:]+grid['edges_cm'][:-1])/2
    owner=np.searchsorted(faces,centers,side='right').astype(np.int64)-1
    typed=List()
    for m in materials:typed.append(m)
    m=materials[0]
    dummy=np.zeros((len(m.pair_E),7));dummy[:,2]=np.interp(m.pair_E,m.photo_E,m.photo)
    dummy[:,3]=m.pair_n;dummy[:,4]=m.pair_e
    defaults=dict(annihilate_in_flight=True,score_kerma=False,score_pair=False,
        rule_kind=1,rule_energies=np.array([.02,1.]),rule_values=np.array([.1,.05,.025]))
    defaults.update(options)
    legacy_values,extra_values=_run_coupled(float(E),int(n),int(seed),.05,.025,float(faces[-1]),
        m.gs_T,m.gs_K,m.gs_u,m.gs_y,m.rates,np.empty((0,0)),
        m.pair_E,dummy,m.thresholds,m.dl,m.bpar,m.delcm,m.delpos,
        np.array([.001,20.]),np.ones(2),faces=faces,rho=rho,m_faces=np.r_[0.,np.cumsum(np.diff(faces)*rho)],
        materials=typed,layer_material=ids,bin_edges=grid['edges_cm'],bin_owner=owner,
        binding_enabled=binding_enabled,fluorescence_enabled=fluorescence_enabled,**defaults)
    values=legacy_values+extra_values
    result=dict(zip(OUTPUTS+('p8_batches','p8_max','boundary_counts','boundary_hist','boundary_records','fluorescence_counts','fluorescence_records','event_records','boundary_omitted_hist'),values))
    result['dose']=result['p8_batches'].sum(axis=0)/n/grid['mass_g_cm2']
    return result


KEYS=('dose','totals','maxima','violations','counts','brems','p8_max',
      'boundary_counts','boundary_hist','boundary_omitted_hist','fluorescence_counts','steps')


def healthy(batch):
    try:
        return (all(np.all(np.isfinite(v)) for v in batch.values())
            and np.all(np.asarray(batch['dose'])>=0)
            and batch['totals'][3]==0
            and batch['maxima'][0]<=1e-12 and not np.any(batch['violations'])
            and not np.any(batch['counts'][5:7]) and not np.any(batch['brems'][2:4])
            and batch['p8_max'][0]<=1e-12)
    except (KeyError,IndexError,TypeError):return False


def check_params(params,n):
    if set(params)!={'bundle','layers','densities','spectrum'}:raise ValueError('P8e params')
    from .run_p8e import layout
    layout(params['layers'],params['densities'])
    if params['spectrum']!='mohan':raise ValueError('P8e source must be Mohan 10 MV')
    if n<=0:raise ValueError('empty batch')


def fingerprint(params):
    from .materials_p8a import sha
    from .run_p3a import SPECTRUM
    from .batch_kernels_p3u import versions
    _,identity=load_bundles(params['bundle'])
    return dict(source_hashes=part2_sources(),bundle_sha256=sha(Path(params['bundle'])/'bundles.json'),
                inputs=identity['files'],spectrum_sha256=sha(SPECTRUM),**versions())


@lru_cache(maxsize=2)
def _worker_materials(folder):
    return load_bundles(folder)[0]


def batch_run(n,seed,params):
    from .run_p3a import load_spectrum
    edges,_,cum=load_spectrum()
    values=run_coupled(_worker_materials(params['bundle']),params['layers'],n,seed,
        densities=params['densities'],spectrum_edges=edges,spectrum_cum=cum)
    return {k:np.asarray(values[k]) for k in KEYS}
