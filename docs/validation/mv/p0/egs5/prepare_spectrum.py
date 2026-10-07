"""Use fetched Mohan bin probabilities, piecewise uniform sampling, no hand values."""
from pathlib import Path
import argparse,json,numpy as np
from prepare_inputs import prepare,BASE

def main():
 p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--histories',type=int,default=100000);p.add_argument('--no-timer',action='store_true');p.add_argument('--no-pdd',action='store_true');args=p.parse_args()
 prepare(args.name,args.histories,False,not args.no_timer)
 data=np.loadtxt(BASE.parent/'data/mohan10_bins.csv',delimiter=',',skiprows=1)
 cdf=np.cumsum(data[:,3]);cdf[-1]=1.
 path=BASE/args.name/f'{args.name}.f';s=path.read_text()
 s=s.replace('      real*8 rn1,rn2','''      real*8 rn1,rn2,rne,rnu,sumin
      real*8 specupper(20),speccdf(20),loweredge
      integer ibin''')
 decl=''
 for array,values in [('specupper',data[:,1]),('speccdf',cdf)]:
  decl+=f'      data {array}/\n'
  for i,v in enumerate(values):decl+=f'     * {v:.16e}'.replace('e','d')+('/\n' if i==len(values)-1 else ',\n')
 s=s.replace('      open(UNIT= 6',decl+'      open(UNIT= 6',1)
 s=s.replace('      sumesc=0.d0','      sumin=0.d0\n      sumesc=0.d0',1)
 s=s.replace('        yin=(2.d0*rn2-1.d0)*fieldhw','''        yin=(2.d0*rn2-1.d0)*fieldhw
        call randomset(rne)
        call randomset(rnu)
        ibin=1
        do while(rne.gt.speccdf(ibin).and.ibin.lt.20)
          ibin=ibin+1
        end do
        loweredge=0.d0
        if(ibin.gt.1) loweredge=specupper(ibin-1)
        ein=loweredge+rnu*(specupper(ibin)-loweredge)
        sumin=sumin+ein''')
 s=s.replace("'LEDGER',ncase*ein","'LEDGER',sumin")
 if args.no_pdd:
  s=s.replace('            edeph(iz)=edeph(iz)+edep','! PDD scoring disabled for matched CPU cost measurement').replace('          sumx(j)  = sumx(j)  + edeph(j)','! per-bin moments disabled').replace('          sumx2(j) = sumx2(j) + edeph(j)*edeph(j)','! per-bin moments disabled')
 path.write_text(s)
 (BASE/args.name/'spectrum_sampling.json').write_text(json.dumps({'source':'data/mohan10.spectrum','mode':'uniform within bin, photon counts proportional to counts/MeV times bin width','cutoff_handling':'sample complete published histogram; photons below PCUT deposit residual energy in source water cell per EGS5 cutoff rule; retained, not rejected','cdf':cdf.tolist(),'upper_edges_MeV':data[:,1].tolist()},indent=2)+'\n')
if __name__=='__main__':main()
