"""Derive MV user code from the existing proven PDD code, never edit EGS5."""
from pathlib import Path
import re,hashlib,json
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[4]
ORIGINAL=ROOT/'docs/egs5_crosscheck/revalidation_2026_10_07/egs5/pdd_phantom/pdd_phantom.f'
def prepare(name,ncase,kerma=False,scoring=True):
 d=BASE/name;d.mkdir(exist_ok=False)
 s=ORIGINAL.read_text();s=s[s.index('      implicit none'):]
 # Remove old per-bin labels/legacy result code, retaining initialization recipe.
 start=s.index('!     ------------------------------------------------------------\n!     Bin labels')
 end=s.index('!     ----------\n!     Open files',start)
 s=s[:start]+s[end:]
 s=s.replace('(47)','(150)').replace('1,47','1,150')
 s=s.replace('      real*8 sumx(150),sumx2(150),meanb(150),varb(150),semb(150),relb(150)', '      real*8 sumx(150),sumx2(150),meanb(150),varb(150),\n     * semb(150),relb(150)')
 s=s.replace('      nreg=3\n\n      med(1)=0\n      med(3)=0\n      med(2)=1', '''      nreg=1352
      med(1)=0
      med(nreg)=0
      do j=2,nreg-1
        med(j)=1
        ecut(j)=ECUTVALUE
        pcut(j)=0.001d0
        iraylr(j)=1
        incohr(j)=1
      end do'''.replace('ECUTVALUE','21.0d0' if kerma else '0.521d0'))
 s=s.replace('      ecut(2)=1.5','      ecut(2)='+('21.0d0' if kerma else '0.521d0'))
 s=s.replace('      inseed=1','      inseed=20261007').replace('      ein=0.060','      ein=10.0d0').replace('      zin=-0.0001d0','      zin=0.d0').replace('      zback=20.0d0','      zback=30.0d0').replace('      chard(1) = 0.5d0','      chard(1) = 0.2d0')
 s=s.replace('      integer i,j,ncase','''      integer i,j,ncase,ix0,iy0
      integer nhsteps,nphsteps,npair,nann,nposesc
      integer nleft,nabnormal
      real*8 escapeh,resth,sumesc,sumrest,unprocessed
      real*8 sumunproc,maxres,resid,tstart,tend,scortime
      real*8 pairenergy,annenergy,posescrest
      real*8 tscore0,tscore1
      common/ledger/escapeh,nhsteps,nphsteps,npair,nann,
     * nposesc,scortime
      integer histstep(100000)
      integer istepbin''')
 s=s.replace('      call hatch\n','''      call hatch
      iausfl(13)=1
      iausfl(15)=1
      iausfl(16)=1
''')
 a=s.index('      write(6,*) "REVALIDATION PHYSICS: WATER REGIONS"');b=s.index('      close(UNIT=KMPI)',a)
 s=s[:a]+'''      write(6,*) 'PHYSICS',rhom(1),ae(1),ap(1),ue(1),up(1),
     * ecut(2),pcut(2),iraylr(2),incohr(2)
'''+s[b:]
 s=s.replace('      ncase=100000000',f'''      ncase={ncase}
      sumesc=0.d0
      sumrest=0.d0
      sumunproc=0.d0
      pairenergy=0.d0
      annenergy=0.d0
      posescrest=0.d0
      maxres=0.d0
      nleft=0
      nabnormal=0
      nphsteps=0
      scortime=0.d0
      histstep=0
      call cpu_time(tstart)''')
 s=s.replace('        edtot=0.d0','''        edtot=0.d0
        escapeh=0.d0
        nhsteps=0
        npair=0
        nann=0
        nposesc=0
        ix0=1
        iy0=1
        if(xin.ge.-1.d0) ix0=2
        if(xin.ge.1.d0) ix0=3
        if(yin.ge.-1.d0) iy0=2
        if(yin.ge.1.d0) iy0=3
        irin=2+(iy0-1)*3+ix0-1''')
 s=s.replace('        do j=1,150\n          sumx', '''        unprocessed=0.d0
        if(np.gt.0) then
          nleft=nleft+1
          do j=1,np
            if(iq(j).eq.0) then
              unprocessed=unprocessed+e(j)
            else
              unprocessed=unprocessed+e(j)-RM
            end if
          end do
        end if
        resth=2.d0*RM*(npair-nann)
        sumesc=sumesc+escapeh
        sumrest=sumrest+resth
        sumunproc=sumunproc+unprocessed
        pairenergy=pairenergy+2.d0*RM*npair
        annenergy=annenergy+2.d0*RM*nann
        posescrest=posescrest+2.d0*RM*nposesc
        resid=ein-edtot-escapeh-resth-unprocessed
        if(resid.ne.resid) nabnormal=nabnormal+1
        maxres=max(maxres,abs(resid))
        istepbin=min(100000,nhsteps+1)
        histstep(istepbin)=histstep(istepbin)+1
        do j=1,150
          sumx''')
 a=s.index('      write(6,160) ncase');b=s.index('      stop\n      end',a)
 s=s[:a]+'''      call cpu_time(tend)
      write(6,*) 'NCASE',ncase
      write(6,*) 'TIME',tend-tstart,scortime
      write(6,*) 'LEDGER',ncase*ein,sumtot,sumesc,sumunproc,
     * sumrest,maxres,nleft,nabnormal
      write(6,*) 'REST_COMPONENTS',pairenergy,annenergy,posescrest
      write(6,*) 'PHOTON_STEPS',nphsteps
      do j=1,150
        write(6,*) 'PDD',j,(j-0.5d0)*0.2d0,meanb(j),semb(j)
      end do
      do j=1,100000
        if(histstep(j).gt.0) write(6,*) 'STEPS',j-1,histstep(j)
      end do
      call counters_out(1)
'''+s[b:]
 a=s.index('      subroutine ausgab(iarg)');b=s.index('!--------------------------last line of ausgab.f',a)
 s=s[:a]+'''      subroutine ausgab(iarg)
      implicit none
      include 'include/egs5_h.f'
      include 'include/egs5_epcont.f'
      include 'include/egs5_stack.f'
      include 'include/egs5_useful.f'
      common/score/edeph,edtot
      real*8 edeph(150),edtot
      integer nhsteps,nphsteps,npair,nann,nposesc
      real*8 escapeh,scortime,t0,t1
      common/ledger/escapeh,nhsteps,nphsteps,npair,nann,
     * nposesc,scortime
      integer iarg,k,ix,iy,iz
      call cpu_time(t0)
      if(iarg.eq.0) then
        if(iq(np).ne.0) then
          nhsteps=nhsteps+1
        else
          nphsteps=nphsteps+1
        end if
      end if
      if(iarg.eq.15) npair=npair+1
      if(iarg.eq.12.or.iarg.eq.14) nann=nann+1
      if(iarg.eq.3) then
        if(iq(np).eq.0) then
          escapeh=escapeh+e(np)
        else
          escapeh=escapeh+e(np)-RM
          if(iq(np).eq.1) nposesc=nposesc+1
        end if
      else if(iarg.le.4) then
        k=ir(np)-2
        if(k.ge.0.and.k.lt.1350) then
          edtot=edtot+edep
          iz=k/9+1
          iy=mod(k,9)/3+1
          ix=mod(k,3)+1
          if(ix.eq.2.and.iy.eq.2) then
            edeph(iz)=edeph(iz)+edep
          end if
        end if
      end if
      call cpu_time(t1)
      scortime=scortime+t1-t0
      return
      end
'''+s[b:]
 a=s.index('      subroutine howfar')
 s=s[:a]+'''      subroutine howfar
      implicit none
      include 'include/egs5_h.f'
      include 'include/egs5_epcont.f'
      include 'include/egs5_stack.f'
      real*8 edges(4),tx,ty,tz,tmin,loz,hiz
      integer k,ix,iy,iz,nextx,nexty,nextz
      data edges/-15.d0,-1.d0,1.d0,15.d0/
      if(ir(np).eq.1.or.ir(np).eq.1352) then
        idisc=1
        return
      end if
      k=ir(np)-2
      iz=k/9+1
      iy=mod(k,9)/3+1
      ix=mod(k,3)+1
      tx=1.d30
      ty=1.d30
      tz=1.d30
      nextx=ix
      nexty=iy
      nextz=iz
      if(u(np).gt.0.d0) then
        tx=(edges(ix+1)-x(np))/u(np)
        nextx=ix+1
      else if(u(np).lt.0.d0) then
        tx=(edges(ix)-x(np))/u(np)
        nextx=ix-1
      end if
      if(v(np).gt.0.d0) then
        ty=(edges(iy+1)-y(np))/v(np)
        nexty=iy+1
      else if(v(np).lt.0.d0) then
        ty=(edges(iy)-y(np))/v(np)
        nexty=iy-1
      end if
      loz=(iz-1)*0.2d0
      hiz=iz*0.2d0
      if(w(np).gt.0.d0) then
        tz=(hiz-z(np))/w(np)
        nextz=iz+1
      else if(w(np).lt.0.d0) then
        tz=(loz-z(np))/w(np)
        nextz=iz-1
      end if
      tmin=min(tx,ty,tz)
      if(tmin.gt.ustep) return
      ustep=max(0.d0,tmin)
      if(tmin.eq.tx) then
        ix=nextx
      else if(tmin.eq.ty) then
        iy=nexty
      else
        iz=nextz
      end if
      if(ix.lt.1.or.ix.gt.3.or.iy.lt.1.or.iy.gt.3.or.
     * iz.lt.1.or.iz.gt.150) then
        irnew=1352
      else
        irnew=2+(iz-1)*9+(iy-1)*3+ix-1
      end if
      return
      end
'''
 if not scoring:
  s=s.replace('      call cpu_time(t0)','! timing disabled').replace('      call cpu_time(t1)','! timing disabled').replace('      scortime=scortime+t1-t0','! timing disabled')
 (d/f'{name}.f').write_text('! P0 MV derived from revalidation pdd_phantom; 1350 water cells.\n'+s)
 inp='''COMP
 &INP NE=2,RHO=1.000,PZ=2,1,
      IRAYL=1,IBOUND=1,INCOH=1,ICPROF=0,IMPACT=0 &END
H2O                           H2O
H  O
ENER
 &INP AE=0.521,AP=0.001,UE=20.511,UP=20.0 &END
PWLF
 &INP &END
DECK
 &INP &END
'''
 # Use ESTAR's water mass fractions and tabulated density effect.
 html=(BASE.parent/'data/water_composition.html').read_text()
 weights={int(z):float(w) for z,w in re.findall(r'<td>\s*(\d+)\s*</td>\s*<td>\s*([\d.]+)\s*</td>',html)}
 inp=inp.replace('COMP\n','MIXT\n').replace('PZ=2,1,',f'RHOZ={weights[1]},{weights[8]},EPSTFL=1,')
 (d/'epstar.dat').write_bytes((ROOT/'docs/egs5_crosscheck/egs5/data/density_corrections/compounds/water_liquid.density').read_bytes())
 (d/f'{name}.inp').write_text(inp)
 (d/'derivation.json').write_text(json.dumps({'original':str(ORIGINAL.relative_to(ROOT)),'original_sha256':hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),'ncase':ncase,'seed':20261007,'kerma':kerma,'scoring_timer':scoring},indent=2)+'\n')
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--histories',type=int,default=10000);p.add_argument('--kerma',action='store_true');p.add_argument('--no-timer',action='store_true');a=p.parse_args();prepare(a.name,a.histories,a.kerma,not a.no_timer)
