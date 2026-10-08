"""Electron slab user code derived from validated PDD initialization/AUSGAB conventions.
EGS5 is read-only. Fresh inputs, builds and outputs remain in P1 or scratch.
"""
from pathlib import Path
import argparse,json,hashlib,re,shutil
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[4]
EGS=ROOT/'docs/egs5_crosscheck/egs5'
ORIGINAL=ROOT/'docs/egs5_crosscheck/revalidation_2026_10_07/egs5/pdd_phantom/pdd_phantom.f'
def prepare(name,T,n,gs=1,ms=1.,energy=1.,fit=False,ap=None,iunrst=0,nale=150):
 d=BASE/name;d.mkdir(exist_ok=False)
 orig=ORIGINAL.read_text()
 includes=orig[orig.index("      include 'include/egs5_h.f'"):orig.index('      common/geom')]
 init=orig[orig.index('      call block_set'):orig.index('      nreg=3')]
 init=re.sub(r'      write\(6,100\).*?100   FORMAT.*?\n','',init,flags=re.S)
 init=init.replace("chard(1) = 0.5d0",f"chard(1) = {0.5*ms}d0")
 source=includes+"""
      real*8 thick,dz,kin,score(120),edtot,back,forward
      real*8 sumscore(120),sum2(120),batch(120,100)
      real*8 sumdep,sumb,sumf,maxres,resid,unproc
      real*8 xin,yin,zin,uin,vin,win,wtin,ein
      real*8 hsteps(20000),stepbins(160),r2(2),r22(2)
      real*8 cross(2),cross2(2),cpu0,cpu1,lo,hi
      integer nx(2),seen(2),seenstack(2,MXSTACK)
      integer ncase,i,j,b,iqin,irin,nleft,photons,steps
      character*24 medarr(1)
      common/p1geom/thick,dz
      common/p1score/score,edtot,back,forward,steps,photons,
     * stepbins,r2,r22,nx,seen,seenstack
      open(6,file='egs5job.out',status='unknown')
      call counters_out(0)
"""+init+"""
      nreg=3
      med(1)=0
      med(2)=1
      med(3)=0
      ecut(2)=0.521d0
      pcut(2)=0.001d0
      k1hscl(2)=0.d0
      k1lscl(2)=0.d0
      estepr(2)=ESCALE
      usegsd(1)=GSFLAG
      luxlev=1
      inseed=20261007
      call rluxinit
      iqin=-1
      xin=0.d0
      yin=0.d0
      zin=0.d0
      uin=0.d0
      vin=0.d0
      win=1.d0
      irin=2
      wtin=1.d0
      latchi=0
      kin=KINETIC
      ein=kin+RM
      emaxe=kin
      open(KMPI,file='pgs5job.pegs5dat',status='old')
      open(KMPO,file='egs5job.dummy',status='unknown')
      call hatch
      iausfl(6)=1
      iausfl(9)=1
      close(KMPI)
      close(KMPO)
      thick=THICKNESS
      dz=BINWIDTH
      ncase=NCOUNT
      sumscore=0.d0
      sum2=0.d0
      batch=0.d0
      hsteps=0.d0
      stepbins=0.d0
      sumdep=0.d0
      sumb=0.d0
      sumf=0.d0
      maxres=0.d0
      nleft=0
      photons=0
      r2=0.d0
      r22=0.d0
      nx=0
      call cpu_time(cpu0)
      do i=1,ncase
        score=0.d0
        edtot=0.d0
        back=0.d0
        forward=0.d0
        steps=0
        seen=0
        seenstack=0
        call shower(iqin,ein,xin,yin,zin,uin,vin,win,
     *              irin,wtin)
        unproc=0.d0
        if(np.gt.0) then
          nleft=nleft+np
          do j=1,np
            if(iq(j).eq.0) then
              unproc=unproc+e(j)
            else
              unproc=unproc+e(j)-RM
            end if
          end do
        end if
        resid=kin-edtot-back-forward-unproc
        maxres=max(maxres,abs(resid))
        sumdep=sumdep+edtot
        sumb=sumb+back
        sumf=sumf+forward
        hsteps(min(20000,steps+1))=
     *     hsteps(min(20000,steps+1))+1
        b=min(100,1+(i-1)*100/ncase)
        do j=1,120
          sumscore(j)=sumscore(j)+score(j)
          sum2(j)=sum2(j)+score(j)**2
          batch(j,b)=batch(j,b)+score(j)
        end do
      end do
      call cpu_time(cpu1)
      write(6,*) 'CONFIG',kin,ncase,rhom(1),usegsd(1),
     * k1hscl(2),estepr(2),ae(1),ap(1),ue(1),up(1)
      write(6,*) 'LEDGER',ncase*kin,sumdep,sumb,sumf,
     * maxres,nleft,photons
      write(6,*) 'CPU',cpu1-cpu0
      do j=1,120
        write(6,*) 'PDD',j,(j-.5d0)*dz,
     * sumscore(j)/(ncase*dz),
     * sqrt(max(0.d0,(sum2(j)/ncase-
     * (sumscore(j)/ncase)**2)/(ncase-1)))/dz
        do b=1,100
          write(6,*) 'BATCH',b,j,batch(j,b)/(ncase/100*dz)
        end do
      end do
      do j=1,20000
        if(hsteps(j).gt.0) write(6,*) 'STEPS',j-1,hsteps(j)
      end do
      do j=1,160
        if(stepbins(j).gt.0)
     * write(6,*) 'STEPDIST',j,stepbins(j)
      end do
      do j=1,2
        write(6,*) 'CROSS',j,r2(j),r22(j),nx(j)
      end do
      call counters_out(1)
      stop
      end

      subroutine ausgab(iarg)
      implicit none
      include 'include/egs5_h.f'
      include 'include/egs5_epcont.f'
      include 'include/egs5_stack.f'
      include 'include/egs5_useful.f'
      real*8 thick,dz,score(120),edtot,back,forward
      real*8 stepbins(160),r2(2),r22(2),z0,z1,lo,hi
      real*8 ov,loss,plane,frac,xx,yy,val
      integer nx(2),seen(2),seenstack(2,MXSTACK)
      integer steps,photons,iarg,j,k
      common/p1geom/thick,dz
      common/p1score/score,edtot,back,forward,steps,photons,
     * stepbins,r2,r22,nx,seen,seenstack
      if(iq(np).eq.0.and.iarg.le.4) photons=photons+1
      if(iarg.eq.8) then
!       Moller before/after track identifiers: EGS5 puts higher energy parent at npold.
!       Source lineage handled by latch, set in HOWFAR/initial stack below.
        return
      end if
      if(iarg.eq.0.and.iq(np).ne.0.and.ir(np).eq.2) then
        steps=steps+1
        k=max(1,min(160,int((log10(max(ustep,1.d-12))
     *    +12.d0)*10.d0)+1))
        stepbins(k)=stepbins(k)+1.d0
        z0=z(np)
        z1=z0+w(np)*ustep
        if(np.eq.1.and.w(np).gt.0.d0) then
          do j=1,2
            plane=.2d0
            if(j.eq.2) plane=.5d0
            if(seen(j).eq.0.and.z0.lt.plane.and.z1.ge.plane) then
              frac=(plane-z0)/(z1-z0)
              xx=x(np)+frac*ustep*u(np)
              yy=y(np)+frac*ustep*v(np)
              val=xx*xx+yy*yy
              r2(j)=r2(j)+val
              r22(j)=r22(j)+val*val
              nx(j)=nx(j)+1
              seen(j)=1
            end if
          end do
        end if
        lo=min(z0,z1)
        hi=max(z0,z1)
        edtot=edtot+edep
        if(abs(z1-z0).gt.1.d-15) then
          do j=max(1,int(lo/dz)+1),min(120,int(hi/dz)+1)
            ov=max(0.d0,min(hi,j*dz)-max(lo,(j-1)*dz))
            score(j)=score(j)+edep*ov/(hi-lo)
          end do
        else
          j=max(1,min(120,int(z0/dz)+1))
          score(j)=score(j)+edep
        end if
      else if(iarg.eq.3.and.
     *       (iq(np).ne.0.or.ir(np).ne.2)) then
        if(iq(np).eq.0) then
          loss=e(np)
        else
          loss=e(np)-RM
        end if
        if(z(np).le.0.d0) then
          back=back+loss
        else
          forward=forward+loss
        end if
      else if(iarg.le.4.and.ir(np).eq.2) then
        edtot=edtot+edep
        j=max(1,min(120,int(z(np)/dz)+1))
        score(j)=score(j)+edep
      end if
      return
      end
      subroutine howfar
      implicit none
      include 'include/egs5_h.f'
      include 'include/egs5_epcont.f'
      include 'include/egs5_stack.f'
      real*8 thick,dz,dist
      common/p1geom/thick,dz
      if(iq(np).eq.0) then
!       Only sensitivity run can create photons; local discard deposition.
        idisc=1
        return
      end if
      if(ir(np).ne.2) then
        idisc=1
        return
      end if
      if(w(np).gt.0.d0) then
        dist=(thick-z(np))/w(np)
        if(dist.le.ustep) then
          ustep=max(0.d0,dist)
          irnew=3
        end if
      else if(w(np).lt.0.d0) then
        dist=-z(np)/w(np)
        if(dist.le.ustep) then
          ustep=max(0.d0,dist)
          irnew=1
        end if
      end if
      return
      end
"""
 for a,b in {'MSSCALE':str(ms)+'d0','ESCALE':str(energy)+'d0','GSFLAG':str(gs),'KINETIC':str(T)+'d0','THICKNESS':str(1.2 if T==2 else 6.)+'d0','BINWIDTH':str(.01 if T==2 else .05)+'d0','NCOUNT':str(n)}.items():source=source.replace(a,b)
 (d/(name+'.f')).write_text('! Derived from validated pdd_phantom.f; P1 slab scoring.\n'+source)
 meta=json.loads((BASE.parent/'data/metadata.json').read_text());w=list(meta['weights'].values())
 ap=T if ap is None else ap
 inp=f"""MIXT
 &INP NE=2,RHO=1.000,RHOZ={w[0]},{w[1]},EPSTFL=1 &END
H2O                           H2O
H  O
ENER
 &INP AE=0.521,AP={ap},UE=20.511,UP=21.0,
 IUNRST={iunrst} &END
PWLF
 &INP EPE={.005 if fit else .01},NIPE={30 if fit else 20},NALE={nale} &END
"""
 # Actual PEGS radiative stopping power, RLC normalization from listing.
 for e in [2,10]:
  inp+=f"CALL\n &INP XP(1)={e+meta['electron_mass_MeV']},XP(2)={e+meta['electron_mass_MeV']} &END\nBRMSTM\n"
 inp+="DECK\n &INP &END\n"
 (d/(name+'.inp')).write_text(inp)
 shutil.copy2(EGS/'data/density_corrections/compounds/water_liquid.density',d/'epstar.dat')
 (d/'derivation.json').write_text(json.dumps({'source':str(ORIGINAL.relative_to(ROOT)),'sha256':hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),'T':T,'N':n,'USEGSD':gs,'ms_scale':ms,'energy_scale':energy,'fit':fit,'AP':ap,'IUNRST':iunrst,'NALE':nale},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--energy',type=float,default=2);p.add_argument('-n',type=int,default=1000000);p.add_argument('--gs',type=int,default=1);p.add_argument('--ms',type=float,default=1);p.add_argument('--es',type=float,default=1);p.add_argument('--fit',action='store_true');p.add_argument('--ap',type=float);p.add_argument('--iunrst',type=int,default=0);p.add_argument('--nale',type=int,default=150)
 a=p.parse_args();prepare(a.name,a.energy,a.n,a.gs,a.ms,a.es,a.fit,a.ap,a.iunrst,a.nale)
