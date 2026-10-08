"""Read-only cross-check using NumPy's fixed-width reader and scalar interpolation.
Does not use the production ENDF reader or its angular interpolation to form references.
Not an independent transport code validation; no alternative physics is selected.
"""
import bisect,io,json,hashlib,math
from pathlib import Path
import numpy as np
from .physics import ROOT
P=ROOT/'docs/validation/mv/p1b'

def raw_section(path,mf,mt):
    lines=[l[:66] for l in path.read_text().splitlines() if l[70:72].strip()==str(mf) and l[72:75].strip()==str(mt)]
    return np.genfromtxt(io.StringIO('\n'.join(lines)),delimiter=11,filling_values=0.)

def pairs(rows,index,number):
    return rows[index:index+math.ceil(2*number/6)].ravel()[:2*number].reshape(number,2)

def scalar_linear(x,xs,ys):
    if x<=xs[0]:return float(ys[0])
    if x>=xs[-1]:return float(ys[-1])
    j=bisect.bisect_right(xs,x)-1
    return float(ys[j]+(ys[j+1]-ys[j])*(x-xs[j])/(xs[j+1]-xs[j]))

def independent_atom(path):
    cross={};rules={}
    for mt in [525,526]:
        rows=raw_section(path,23,mt);nr,n=int(rows[1,4]),int(rows[1,5])
        rules[str(mt)]=rows[2:2+math.ceil(2*nr/6)].ravel()[:2*nr].reshape(nr,2).astype(int).tolist()
        cross[mt]=pairs(rows,2+math.ceil(2*nr/6),n)
    rows=raw_section(path,26,525);nr,np_=int(rows[1,4]),int(rows[1,5])
    pos=2+math.ceil(2*nr/6)+math.ceil(2*np_/6)
    nr,ne=int(rows[pos,4]),int(rows[pos,5]);pos+=1
    regions=rows[pos:pos+math.ceil(2*nr/6)].ravel()[:2*nr].reshape(nr,2).astype(int).tolist();pos+=math.ceil(2*nr/6)
    angular=[];languages=[]
    for i in range(ne):
        energy,lang,nw,np_=rows[pos,1],int(rows[pos,2]),int(rows[pos,4]),int(rows[pos,5]);pos+=1
        assert nw==2*np_
        angular.append((energy,pairs(rows,pos,np_)));languages.append(lang);pos+=math.ceil(nw/6)
    return cross,rules,angular,regions,languages

def main():
    from .eedl import Eedl
    from .dcs import moments
    meta=json.loads((ROOT/'docs/validation/mv/p1/data/metadata.json').read_text())
    m,alpha=meta['electron_mass_MeV'],meta['alpha'];production=Eedl();result={'method':'independent fixed-width NumPy reader; scalar two-stage linear interpolation; exact polynomial two-point Gauss integration on each linear angular segment plus analytic forward moments','limits':'Checks file reading, specified interpolation, units and elemental moments at listed energies; does not prove all code paths or reference/model correctness, and is not a second independent transport implementation.','atoms':{}}
    for z in [1,8]:
        path=P/'data'/f'ZA{z:03}000';xs,rules,angular,regions,languages=independent_atom(path);atom=production.atomic[z]
        all_raw=all(np.array_equal(xs[mt],atom.xs[mt]) for mt in [525,526]) and all(np.array_equal(v,p[1]) and e*1e-6==p[0] for (e,v),p in zip(angular,atom.angular)) and len(angular)==len(atom.angular)
        assert all_raw and all(a[1]==2 for a in regions) and set(languages)=={12}
        energies=np.array([e for e,v in angular]);checks=[]
        for T in [.02,.1,.256,1.,2.,10.]:
            ev=T*1e6;j=max(0,min(bisect.bisect_right(energies,ev)-1,len(energies)-2));a=(ev-energies[j])/(energies[j+1]-energies[j]);v0,v1=angular[j][1],angular[j+1][1]
            mu=np.unique(np.r_[v0[:,0],v1[:,0]])
            pdf=np.array([(1-a)*scalar_linear(x,v0[:,0],v0[:,1])+a*scalar_linear(x,v1[:,0],v1[:,1]) for x in mu])
            prod_mu,prod_pdf=atom.angular_at(T)
            reference_at_prod=np.array([(1-a)*scalar_linear(x,v0[:,0],v0[:,1])+a*scalar_linear(x,v1[:,0],v1[:,1]) for x in prod_mu])
            pdf_rel=float(max(abs(reference_at_prod/prod_pdf-1)))
            total=scalar_linear(ev,xs[526][:,0],xs[526][:,1])*1e-24;large=scalar_linear(ev,xs[525][:,0],xs[525][:,1])*1e-24
            tau=T/m;eta=alpha**2*z**(2/3)/(2*.885**2*tau*(tau+2))*(1.13+3.76*alpha**2*z*z*(tau+1)**2/(tau*(tau+2))*math.sqrt(tau/(tau+1)))
            yc=1-mu[-1];A=(total-large)/(1/eta-1/(eta+yc))
            x,w=np.polynomial.legendre.leggauss(2);quad=(mu[:-1,None]+mu[1:,None])/2+(mu[1:,None]-mu[:-1,None])*x/2
            pp=pdf[:-1,None]+(pdf[1:,None]-pdf[:-1,None])*(quad-mu[:-1,None])/(mu[1:,None]-mu[:-1,None])
            weights=pp*w*(mu[1:,None]-mu[:-1,None])/2
            front1=A*(math.log1p(yc/eta)+eta/(eta+yc)-1)
            front2=3*front1-1.5*A*(yc-2*eta*math.log1p(yc/eta)+eta*yc/(eta+yc))
            ref=np.array([large*weights.sum()+total-large,large*np.sum(weights*(1-quad))+front1,large*np.sum(weights*1.5*(1-quad*quad))+front2])
            G,got,_=moments(production,T,z=z);actual=np.r_[got,G[1:3]];relative=actual/ref-1
            checks.append({'T_MeV':T,'bracket_MeV':[float(energies[j]*1e-6),float(energies[j+1]*1e-6)],'upper_energy_weight':float(a),'PDF_max_relative_difference':pdf_rel,'independent_atomic_total_G1_G2_cm2':ref.tolist(),'production_relative_difference':relative.tolist(),'pass':bool(pdf_rel<1e-12 and max(abs(relative))<1e-7)})
        assert all(c['pass'] for c in checks)
        result['atoms'][str(z)]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'all_MF23_MF26_numeric_records_match':bool(all_raw),'MF23_regions':rules,'MF26_regions':regions,'LANG_values':sorted(set(languages)),'angular_energy_grid_MeV':(energies*1e-6).tolist(),'checks':checks}
    (P/'eedl_independent_verification.json').write_text(json.dumps(result,indent=2)+'\n');print('independent ENDF read/interpolation/moment checks passed')
if __name__=='__main__':main()
