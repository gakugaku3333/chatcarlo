from pathlib import Path
import re,json,hashlib
import numpy as np
BASE=Path(__file__).resolve().parent
def main():
 p=BASE/'data/mohan10.spectrum'
 lines=p.read_text().splitlines(); n,lower,mode=map(float,lines[1].split(','));a=np.array([[float(v) for v in s.split(',')] for s in lines[2:]])
 assert len(a)==n and mode==1 and np.all(np.diff(a[:,0])>0) and np.all(a[:,1]>=0)
 edges=np.r_[lower,a[:,0]];prob=a[:,1]*np.diff(edges);prob/=prob.sum()
 definition=(BASE/'data/egs_spectra.cpp').read_text()
 assert 'en_array[0] = dum' in definition and 'counts per MeV' in definition
 result={'J7':'pass','title':lines[0].strip(),'bins':int(n),'mode':int(mode),'definition':'lower edge from header; rows are upper edge MeV and counts/MeV; uniform within each bin','min_MeV':float(edges[0]),'max_MeV':float(edges[-1]),'bin_width_MeV':np.diff(edges).tolist(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'fluence_weighted_mean_MeV':float(np.dot(prob,(edges[1:]+edges[:-1])/2)),'source':'NRC EGSnrc public distribution; spectrum attributed there to Mohan et al','caution':'0-0.5 MeV first bin includes below-cutoff photons; P1 must explicitly handle low energy cutoff identically in both codes'}
 (BASE/'spectrum_check.json').write_text(json.dumps(result,indent=2)+'\n')
 np.savetxt(BASE/'data/mohan10_bins.csv',np.c_[edges[:-1],edges[1:],a[:,1],prob],delimiter=',',header='lower_MeV,upper_MeV,counts_per_MeV,probability',comments='')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
