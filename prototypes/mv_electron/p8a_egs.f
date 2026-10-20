! P8a fluorescence sensitivity only: vacuum/water/lead/vacuum.
! No lateral boundaries; transmitted brems and annihilation photons transported.
      implicit none
      include 'include/egs5_h.f'
      include 'include/egs5_bounds.f'
      include 'include/egs5_epcont.f'
      include 'include/egs5_media.f'
      include 'include/egs5_elecin.f'
      include 'include/egs5_misc.f'
      include 'include/egs5_stack.f'
      include 'include/egs5_thresh.f'
      include 'include/egs5_useful.f'
      include 'include/egs5_usersc.f'
      include 'include/egs5_brempr.f'
      include 'include/egs5_edge.f'
      include 'include/randomm.f'
      real*8 p8score(55),p8dep,p8escape
      common/p8score/p8score,p8dep,p8escape
      real*8 batches(55,100),elo(64),ehi(64),cum(64)
      real*8 ein,depall,escall,ehist,u1,u2,maxres,res,width,rho
      real*8 cpu0,cpu1,maxscore
      integer ncase,seed,fluor,i,j,b,nb,ispec,nleft
      character*24 medname(2)
      call block_set
      open(6,file='egs5job.out',status='replace')
      open(41,file='config.dat',status='old')
      read(41,*) ncase,seed,fluor
      close(41)
      if(ncase.lt.100.or.mod(ncase,100).ne.0) stop 21
      if(fluor.ne.0.and.fluor.ne.1) stop 22
      open(41,file='spectrum.dat',status='old')
      read(41,*) nb
      if(nb.lt.1.or.nb.gt.64) stop 23
      do j=1,nb
        read(41,*) elo(j),ehi(j),cum(j)
        if(elo(j).lt.0.d0.or.ehi(j).le.elo(j)) stop 24
        if(cum(j).le.0.d0.or.cum(j).gt.1.d0) stop 25
        if(j.gt.1) then
          if(elo(j).ne.ehi(j-1)) stop 26
          if(cum(j).le.cum(j-1)) stop 27
        endif
      enddo
      if(cum(nb).ne.1.d0) stop 28
      close(41)
      nmed=2
      medname(1)='P8A_WATER'
      medname(2)='P8A_LEAD'
      do j=1,2
        do i=1,24
          media(i,j)=medname(j)(i:i)
        enddo
        chard(j)=.5d0
        usegsd(j)=1
      enddo
      call pegs5
      nreg=4
      med(1)=0
      med(2)=1
      med(3)=2
      med(4)=0
      iraylr=0
      incohr=0
      iprofr=0
      iedgfl=0
      iauger=0
      ibrspl=0
      do i=2,3
        ecut(i)=.521d0
        pcut(i)=.01d0
        k1hscl(i)=0.d0
        k1lscl(i)=0.d0
        estepr(i)=1.d0
        iedgfl(i)=fluor
        iauger(i)=fluor
      enddo
      iprdst=0
      luxlev=1
      inseed=seed
      call rluxinit
      latchi=0
      emaxe=ehi(nb)
      open(KMPI,file='pgs5job.pegs5dat',status='old')
      open(KMPO,file='egs5job.dummy',status='unknown')
      call hatch
      close(KMPI)
      close(KMPO)
      iausfl=0
      do i=1,5
        iausfl(i)=1
      enddo
      do i=1,4
        write(6,*) 'REGION',i,med(i),incohr(i),iprofr(i),
     *    iraylr(i),iedgfl(i),iauger(i),ibrspl(i),
     *    ecut(i),pcut(i)
      enddo
      do i=1,2
        write(6,*) 'MEDIUM',i,rhom(i),ae(i),ap(i),
     *    incohm(i),iprofm(i),iraylm(i),usegsd(i),iunrst(i)
      enddo
      batches=0.d0
      ein=0.d0
      depall=0.d0
      escall=0.d0
      maxres=0.d0
      maxscore=0.d0
      nleft=0
      call cpu_time(cpu0)
      do i=1,ncase
        p8score=0.d0
        p8dep=0.d0
        p8escape=0.d0
        call randomset(u1)
        call randomset(u2)
        ispec=1
        do while(cum(ispec).lt.u1)
          ispec=ispec+1
        enddo
        ehist=elo(ispec)+u2*(ehi(ispec)-elo(ispec))
        ein=ein+ehist
        call shower(0,ehist,0.d0,0.d0,0.d0,0.d0,
     *    0.d0,1.d0,2,1.d0)
        nleft=nleft+np
        res=abs(ehist-p8dep-p8escape)/max(ehist,1.d-30)
        maxres=max(maxres,res)
        maxscore=max(maxscore,abs(sum(p8score)-p8dep))
        depall=depall+p8dep
        escall=escall+p8escape
        b=1+(i-1)/(ncase/100)
        batches(:,b)=batches(:,b)+p8score
      enddo
      call cpu_time(cpu1)
      write(6,*) 'CONFIG',ncase,seed,fluor
      write(6,*) 'LEDGER',ein,depall,escall,maxres,
     *    nleft,maxscore
      write(6,*) 'CPU',cpu1-cpu0
      do b=1,100
        do j=1,55
          if(j.le.50) then
            width=.2d0
            rho=rhom(1)
          else
            width=.1d0
            rho=rhom(2)
          endif
! Dose times unit conversion: MeV cm2/g/history (unit transverse area).
          write(6,*) 'DOSE',b,j,
     *      batches(j,b)/(ncase/100*width*rho)
        enddo
      enddo
      write(6,*) 'DONE'
      close(6)
      stop
      end

      subroutine ausgab(iarg)
      implicit none
      include 'include/egs5_h.f'
      include 'include/egs5_epcont.f'
      include 'include/egs5_stack.f'
      real*8 p8score(55),p8dep,p8escape
      common/p8score/p8score,p8dep,p8escape
      integer iarg,j,k,reg
      real*8 loss,z0,z1,lo,hi,left,right,ov
      if(iarg.gt.4) return
      loss=edep*wt(np)
      if(iarg.eq.3) then
        p8escape=p8escape+loss
        return
      endif
      reg=ir(np)
      if(reg.ne.2.and.reg.ne.3) return
      p8dep=p8dep+loss
      z0=z(np)
      if(iarg.eq.0.and.iq(np).ne.0) then
        z1=z0+w(np)*ustep
      else
        z1=z0
      endif
      if(reg.eq.2) then
        j=1
        k=50
      else
        j=51
        k=55
      endif
      lo=min(z0,z1)
      hi=max(z0,z1)
      do while(j.le.k)
        if(j.le.50) then
          left=(j-1)*.2d0
          right=j*.2d0
        else
          left=10.d0+(j-51)*.1d0
          right=10.d0+(j-50)*.1d0
        endif
        if(hi-lo.gt.1.d-15) then
          ov=max(0.d0,min(hi,right)-max(lo,left))/(hi-lo)
        else
! Tie at a material interface is owned by the current EGS region.
          ov=0.d0
          if(z0.ge.left.and.z0.lt.right) ov=1.d0
          if(j.eq.k.and.abs(z0-right).lt.1.d-12) ov=1.d0
        endif
        p8score(j)=p8score(j)+loss*ov
        j=j+1
      enddo
      return
      end

      subroutine howfar
      implicit none
      include 'include/egs5_h.f'
      include 'include/egs5_epcont.f'
      include 'include/egs5_stack.f'
      real*8 dist,left,right
      if(ir(np).eq.1.or.ir(np).eq.4) then
        idisc=1
        return
      endif
      if(ir(np).eq.2) then
        left=0.d0
        right=10.d0
      else if(ir(np).eq.3) then
        left=10.d0
        right=10.5d0
      else
        stop 31
      endif
      if(w(np).gt.0.d0) then
        dist=(right-z(np))/w(np)
        if(dist.le.ustep) then
          ustep=max(0.d0,dist)
          irnew=ir(np)+1
        endif
      else if(w(np).lt.0.d0) then
        dist=(left-z(np))/w(np)
        if(dist.le.ustep) then
          ustep=max(0.d0,dist)
          irnew=ir(np)-1
        endif
      endif
      return
      end
