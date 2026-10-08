"""Read-only diagnostic of reference partial-wave transport coefficients.
These data NEVER enter prototype tables or transport.
"""
import json,hashlib
import numpy as np
from .physics import ROOT,Z,NJ,elastic
from .gs import coefficients
from .component_checks import OUT
def main():
    rows=[];elements=[];hashes={}
    for z in Z:
        p=ROOT/f'docs/egs5_crosscheck/egs5/data/dcslib/eeldx{int(z):03d}.tab'
        hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        hdr=[]
        for line in p.read_text().splitlines():
            parts=line.split()
            if len(parts)==6 and parts[0]=='-1' and int(parts[1])==int(z):
                hdr.append([float(parts[2])*1e-6,*map(float,parts[3:])])
        a=np.array(hdr);assert len(a)>10;elements.append(a)
    for t in [.02,.1,1,2,10]:
        # Use actual tabulated energies (no spline approximation in diagnostic).
        pw=np.zeros(3)
        for j,a in enumerate(elements):
            idx=np.argmin(abs(a[:,0]-t));assert abs(a[idx,0]-t)<1e-10
            pw+=NJ[j]*(Z[j]+1)/Z[j]*a[idx,1:]
        G,total=coefficients(t,2);proto=np.array([total,G[1],G[2]])
        rows.append({'T':t,'EGS5_partial_wave_total_G1_G2_cm_inverse':pw.tolist(),'prototype_Rutherford_total_G1_G2_cm_inverse':proto.tolist(),'relative':(proto/pw-1).tolist()})
    result={'rows':rows,'data_sha256':hashes,'reference_source':'egs5/pegs/elinit.f header csin,csin1,csin2; Z+1 soft scattering correction','interpretation':'Coefficient/model difference established. Attribution of full depth-dose residual to this difference remains a hypothesis; no substitution or tuning performed.'}
    (OUT/'model_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
