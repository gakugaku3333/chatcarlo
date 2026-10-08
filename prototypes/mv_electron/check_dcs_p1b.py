"""M2/M3 raw input acceptance, without cross-code matching adjustments."""
import json
import numpy as np
from .dcs import get_backend,moments,Z
POINTS=[.02,.1,.256,1.,2.,10.]

def inputs(backend):
    b=get_backend(backend);rows=[]
    for z in [1,8]:
        for t in POINTS:
            G,total,back=moments(b,t,z=z);G2,total2,_=moments(b,t,order=32,z=z)
            y,w=b.quadrature(t,16,z);r={'Z':z,'T':t,'total_cm2':total,'G1_cm2':G[1],'G2_cm2':G[2],'nonnegative':bool(np.all(w>=0)),'doubling_max_relative':float(max(abs(np.r_[G2[1:]/G[1:]-1,total2/total-1])))}
            if backend=='dcslib':
                e,d,h=b.atomic[z];j=np.argmin(abs(e-t));grid=bool(abs(e[j]/t-1)<1e-12)
                r.update(native_energy=grid,nearest_grid_MeV=float(e[j]),header=h[j].tolist() if grid else None)
                if grid:
                    diff=np.r_[total,G[1:]]/h[j]-1;r['header_relative']=diff.tolist()
                expected={1:1.13312333e-20,8:2.86423422e-19}
                r['independent_2MeV_relative']=float(total/expected[z]-1) if t==2 else None
                r['pass']=bool(r['nonnegative'] and r['doubling_max_relative']<=1e-6 and (not grid or max(abs(diff[1:]))<=.01) and (t!=2 or abs(r['independent_2MeV_relative'])<=1e-4))
            else:
                mu,p,yc,eta,A,large,ref=b.parameters(t,z);yc_mu=1-mu[-1]
                norm=float(np.trapezoid(p,mu));front=float(A*(1/eta-1/(eta+yc)))
                fy,fw=b.quadrature(t,32,z);frontmask=fy<yc
                r.update(angular_integral=norm,sigma525_cm2=large,sigma526_cm2=ref,total_relative=total/ref-1,large_integral_cm2=large*norm,forward_integral_cm2=front,mu_c=float(mu[-1]),eta=eta,seam_large_cm2=large*p[-1],seam_forward_cm2=A/(eta+yc)**2,forward_G1_cm2=float(np.dot(fw[frontmask],fy[frontmask])),forward_G2_cm2=float(np.dot(fw[frontmask],3*fy[frontmask]-1.5*fy[frontmask]**2)))
                # ENDF tables have rounded finite-precision normalization, no renormalization.
                r['pass']=bool(r['nonnegative'] and abs(norm-1)<=1e-3 and abs(total/ref-1)<=1e-3 and (abs(front/(ref-large)-1)<=1e-3 if ref!=large else front==0))
            rows.append(r)
    return {'status':'pass' if all(r['pass'] for r in rows) else 'fail','rows':rows}

def scattering():
    rows=[]
    for name in ['rutherford','dcslib','eedl']:
        b=get_backend(name)
        for t in [.02,.1,1.,2.,10.]:
            G,total,back=moments(b,t)
            # Split at back-hemisphere boundary exactly, rather than node selection.
            from scipy.integrate import quad
            edges=b.edges(t) if name!='rutherford' else np.r_[0,np.geomspace(1e-10,2,606)]
            edges=np.unique(np.r_[edges,1])
            back=sum(quad(lambda y:float(b.density(t,np.asarray(y))),a,c,epsabs=1e-12,epsrel=1e-8)[0] for a,c in zip(edges[:-1],edges[1:]) if a>=1)
            rows.append({'backend':name,'T':t,'total_cm-1':total,'G1_cm-1':G[1],'G2_cm-1':G[2],'back_cm-1':float(back)})
    return {'status':'record-only','rows':rows}
