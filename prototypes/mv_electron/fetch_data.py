"""Retrieve NIST ESTAR/aqueous composition, retaining responses and SHA256.
No PEGS stopping powers are consumed by the prototype.
"""
from pathlib import Path
import subprocess,re,json,hashlib,datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/validation/mv/p1/data'
def request(url,args=()):
    p=subprocess.run(['curl','--fail','--silent','--show-error','--location','--retry','2','--max-time','60',*args,url],capture_output=True)
    if p.returncode: raise RuntimeError(p.stderr.decode())
    return p.stdout

def main():
    OUT.mkdir(exist_ok=True)
    # NIST's default grid augmented by a logarithmic grid; all returned points tested.
    energies=np.unique(np.r_[np.geomspace(.01,20,161),.02,.05,.1,1,2,10])
    payload='\n'.join(format(e,'.12g') for e in energies)+'\n'
    url='https://physics.nist.gov/cgi-bin/Star/e_table-t.pl'
    raw=request(url,['--data-urlencode','matno=276','--data-urlencode','Energies='+payload])
    compurl='https://physics.nist.gov/cgi-bin/Star/compos.pl?matno=276'
    comp=request(compurl)
    (OUT/'estar_water.html').write_bytes(raw);(OUT/'composition.html').write_bytes(comp)
    rows=re.findall(r'([\d.]+E[+\-]\d+)\s+([\d.]+E[+\-]\d+)\s+([\d.]+E[+\-]\d+)\s+([\d.]+E[+\-]\d+)\s+([\d.]+E[+\-]\d+)',raw.decode())
    a=np.array([[float(x) for x in row] for row in rows]); assert len(a)>100
    a=a[(a[:,0]>=.01)&(a[:,0]<=20)];a=a[np.argsort(a[:,0])];a=np.unique(a,axis=0)
    np.savez(OUT/'estar.npz',T=a[:,0],collision=a[:,1],radiative=a[:,2],total=a[:,3],delta=a[:,4])
    c=comp.decode();I=float(re.search(r'Mean Excitation Energy.*?</td><td[^>]*>([\d.]+)',c).group(1))*1e-6
    weights={z:float(w) for z,w in re.findall(r'<td>\s*(\d+)\s*</td>\s*<td>\s*([\d.]+)\s*</td>',c)}
    # Atomic weights retrieved from xraylib's installed published element data, not typed.
    import xraylib
    from scipy.constants import physical_constants,Avogadro,alpha
    data={'I_MeV':I,'weights':weights,'atomic_weights':{z:xraylib.AtomicWeight(int(z)) for z in weights},'electron_mass_MeV':physical_constants['electron mass energy equivalent in MeV'][0], 'classical_radius_cm':physical_constants['classical electron radius'][0]*100,'avogadro':Avogadro,'alpha':alpha,'sources':[url,compurl,'scipy.constants CODATA','xraylib.AtomicWeight'],'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'energies_requested':energies.tolist(),'hashes':{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['estar_water.html','composition.html','estar.npz']}}
    (OUT/'metadata.json').write_text(json.dumps(data,indent=2)+'\n')
    print('ESTAR points:',len(a),'range:',a[0,0],a[-1,0])
if __name__=='__main__':main()
