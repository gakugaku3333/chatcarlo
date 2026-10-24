! P9a EGS5: configurable laterally infinite stack.
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
      real*8 p8score(1000),p8dep,p8escape
      common/p8score/p8score,p8dep,p8escape
      integer nl,nt,lmed(16),first(16),last(16)
      real*8 bound(0:16),bw(16),bleft(1000),bright(1000)
      common/p8geom/bound,bw,bleft,bright,nl,nt,lmed,first,last
      real*8 fscore(1000)
      common/p9score/fscore
      integer nf,ffirst(16),flast(16)
      real*8 fleft(1000),fright(1000)
      common/p9geom/fleft,fright,nf,ffirst,flast
      real*8 fbatches(1000,100)
      real*8 batches(1000,100),elo(64),ehi(64),cum(64)
      real*8 ein,depall,escall,ehist,u1,u2,maxres,res,width,rho
      real*8 cpu0,cpu1,maxscore
      integer ncase,seed,fluor,i,j,b,nb,ispec,nleft
      character*24 medname(5)
      call block_set
      open(6,file='egs5job.out',status='replace')
      call p8config(ncase,seed,fluor,nmed)
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
      medname(1)='P8E_WATER'
      medname(2)='P8E_LUNG'
      medname(3)='P8E_BONE'
      medname(4)='P8E_LEAD'
      medname(5)='P9A_SEED'
      do j=1,nmed
        do i=1,24
          media(i,j)=medname(j)(i:i)
        enddo
        chard(j)=.5d0
        usegsd(j)=1
      enddo
      call pegs5
      nreg=nl+2
      med(1)=0
      med(nreg)=0
      do i=1,nl
        med(i+1)=lmed(i)
      enddo
      rhor=0.d0
      iraylr=0
      incohr=0
      iprofr=0
      iedgfl=0
      iauger=0
      ibrspl=0
      do i=1,nreg
        ecut(i)=.521d0
        pcut(i)=.01d0
        k1hscl(i)=0.d0
        k1lscl(i)=0.d0
        estepr(i)=1.d0
        if(med(i).ne.0) then
          iedgfl(i)=fluor
          iauger(i)=fluor
        endif
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
      do i=1,nreg
        write(6,*) 'REGION',i,med(i),rhor(i),ecut(i),pcut(i),
     *    iedgfl(i),iauger(i),iraylr(i),incohr(i),iprofr(i),
     *    ibrspl(i),estepr(i),k1hscl(i),k1lscl(i)
      enddo
      do i=1,nmed
        write(6,*) 'MEDIUM',i,trim(medname(i)),rhom(i),
     *    ae(i),ap(i),usegsd(i),chard(i),incohm(i),iprofm(i),
     *    iraylm(i),iunrst(i)
      enddo
      write(6,*) 'SETTING',iprdst,luxlev
      do i=1,nl
        write(6,*) 'LAYER',i,lmed(i),bound(i)-bound(i-1),
     *    bw(i),last(i)-first(i)+1
      enddo
      fbatches=0.d0
      batches=0.d0
      ein=0.d0
      depall=0.d0
      escall=0.d0
      maxres=0.d0
      maxscore=0.d0
      nleft=0
      call cpu_time(cpu0)
      do i=1,ncase
        fscore=0.d0
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
        fbatches(:,b)=fbatches(:,b)+fscore
      enddo
      call cpu_time(cpu1)
      write(6,*) 'CONFIG',ncase,seed,fluor
      write(6,*) 'LEDGER',ein,depall,escall,maxres,
     *    nleft,maxscore
      write(6,*) 'CPU',cpu1-cpu0
      do b=1,100
        do j=1,nt
          do i=1,nl
            if(j.ge.first(i).and.j.le.last(i)) then
              width=bw(i)
              rho=rhom(lmed(i))
            endif
          enddo
! Dose times unit conversion: MeV cm2/g/history (unit transverse area).
          write(6,*) 'DOSE',b,j,
     *      batches(j,b)/(ncase/100*width*rho)
        enddo
      enddo
      do b=1,100
        do j=1,nf
          do i=1,nl
            if(j.ge.ffirst(i).and.j.le.flast(i)) then
              rho=rhom(lmed(i))
            endif
          enddo
          write(6,*) 'FINE',b,j,
     *      fbatches(j,b)/(ncase/100*.05d0*rho)
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
      real*8 p8score(1000),p8dep,p8escape
      common/p8score/p8score,p8dep,p8escape
      integer nl,nt,lmed(16),first(16),last(16)
      real*8 bound(0:16),bw(16),bleft(1000),bright(1000)
      common/p8geom/bound,bw,bleft,bright,nl,nt,lmed,first,last
      real*8 fscore(1000)
      common/p9score/fscore
      integer nf,ffirst(16),flast(16)
      real*8 fleft(1000),fright(1000)
      common/p9geom/fleft,fright,nf,ffirst,flast
      integer iarg,reg
      real*8 loss,z0,z1
      if(iarg.gt.4) return
      loss=edep*wt(np)
      if(iarg.eq.3) then
        p8escape=p8escape+loss
        return
      endif
      reg=ir(np)
      if(reg.lt.2.or.reg.gt.nl+1) return
      p8dep=p8dep+loss
      z0=z(np)
      if(iarg.eq.0.and.iq(np).ne.0) then
        z1=z0+w(np)*ustep
      else
        z1=z0
      endif
      call p8deposit(reg,z0,z1,loss,p8score)
      call p9deposit(reg,z0,z1,loss,fscore)
      return
      end

      subroutine howfar
      implicit none
      include 'include/egs5_h.f'
      include 'include/egs5_epcont.f'
      include 'include/egs5_stack.f'
      integer nl,nt,lmed(16),first(16),last(16)
      real*8 bound(0:16),bw(16),bleft(1000),bright(1000)
      common/p8geom/bound,bw,bleft,bright,nl,nt,lmed,first,last
      real*8 dist,left,right
      if(ir(np).eq.1.or.ir(np).eq.nl+2) then
        idisc=1
        return
      endif
      if(ir(np).lt.2.or.ir(np).gt.nl+1) stop 49
      left=bound(ir(np)-2)
      right=bound(ir(np)-1)
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

! Standalone routines: also compiled by test_p8e without EGS5.
      subroutine p8config(ncase,seed,fluor,media_set)
      use, intrinsic :: ieee_arithmetic
      implicit none
      integer nl,nt,lmed(16),first(16),last(16)
      real*8 bound(0:16),bw(16),bleft(1000),bright(1000)
      common/p8geom/bound,bw,bleft,bright,nl,nt,lmed,first,last
      real*8 fscore(1000)
      common/p9score/fscore
      integer nf,ffirst(16),flast(16)
      real*8 fleft(1000),fright(1000)
      common/p9geom/fleft,fright,nf,ffirst,flast
      integer ncase,seed,fluor,media_set,i,j,k,nb,ios
      character*256 header
      real*8 thick,ratio
      open(41,file='config.dat',status='old')
      read(41,'(A)',iostat=ios) header
      if(ios.ne.0) stop 50
      read(header,*,iostat=ios) ncase,seed,fluor,nl,media_set
      if(ios.ne.0) stop 50
      if(media_set.ne.4.and.media_set.ne.5) stop 51
      if(ncase.lt.100.or.mod(ncase,100).ne.0) stop 41
      if(fluor.ne.0.and.fluor.ne.1) stop 42
      if(nl.lt.1.or.nl.gt.16) stop 43
      nf=0
      nt=0
      bound(0)=0.d0
      do i=1,nl
        read(41,*) lmed(i),thick,bw(i)
        if(lmed(i).lt.1.or.lmed(i).gt.media_set) stop 44
        if(.not.ieee_is_finite(thick).or.
     *     .not.ieee_is_finite(bw(i))) stop 45
        if(thick.le.0.d0.or.bw(i).le.0.d0) stop 46
! Check quotient BEFORE integer conversion (including overflow).
        ratio=thick/bw(i)
        if(.not.ieee_is_finite(ratio)) stop 48
        if(ratio.gt.1001.d0) stop 48
        if(abs(ratio-dnint(ratio)).gt.1.d-9*ratio.or.
     *     dnint(ratio).lt.1.d0) stop 47
        nb=nint(ratio)
        if(nt+nb.gt.1000) stop 48
        if(.not.ieee_is_finite(bound(i-1)+thick)) stop 45
        bound(i)=bound(i-1)+thick
! Fine boundaries must lie on the .05 cm lattice. No overlap masses.
        ratio=bound(i)/.05d0
        if(.not.ieee_is_finite(ratio)) stop 52
        if(ratio.gt.1000.d0+1.d-8) stop 52
        if(abs(ratio-dnint(ratio)).gt.1.d-8) stop 52
        ffirst(i)=nf+1
        flast(i)=nint(ratio)
        if(flast(i).lt.ffirst(i)) stop 52
        do j=ffirst(i),flast(i)
          fleft(j)=bound(i-1)+(j-ffirst(i))*.05d0
          fright(j)=bound(i-1)+(j-ffirst(i)+1)*.05d0
        enddo
        fright(flast(i))=bound(i)
        nf=flast(i)
        first(i)=nt+1
        last(i)=nt+nb
        do k=1,nb
          j=nt+k
          bleft(j)=bound(i-1)+(k-1)*bw(i)
          bright(j)=bound(i-1)+k*bw(i)
        enddo
        nt=nt+nb
      enddo
      close(41)
      return
      end

      subroutine p8deposit(reg,z0,z1,loss,score)
      implicit none
      integer nl,nt,lmed(16),first(16),last(16)
      real*8 bound(0:16),bw(16),bleft(1000),bright(1000)
      common/p8geom/bound,bw,bleft,bright,nl,nt,lmed,first,last
      integer reg,j,k
      real*8 z0,z1,loss,score(1000),lo,hi,ov,left,right
      if(reg.lt.2.or.reg.gt.nl+1) return
      j=first(reg-1)
      k=last(reg-1)
      lo=min(z0,z1)
      hi=max(z0,z1)
      do while(j.le.k)
        left=bleft(j)
        right=bright(j)
        if(hi-lo.gt.1.d-15) then
          ov=max(0.d0,min(hi,right)-max(lo,left))/(hi-lo)
        else
! Exact interface points belong to the current EGS layer.
          ov=0.d0
          if(z0.ge.left.and.z0.lt.right) ov=1.d0
          if(j.eq.k.and.abs(z0-right).lt.1.d-12) ov=1.d0
        endif
        score(j)=score(j)+loss*ov
        j=j+1
      enddo
      return
      end

      subroutine p9deposit(reg,z0,z1,loss,score)
      implicit none
      integer nl,nt,lmed(16),first(16),last(16)
      real*8 bound(0:16),bw(16),bleft(1000),bright(1000)
      common/p8geom/bound,bw,bleft,bright,nl,nt,lmed,first,last
      integer nf,ffirst(16),flast(16)
      real*8 fleft(1000),fright(1000)
      common/p9geom/fleft,fright,nf,ffirst,flast
      integer reg,j,k
      real*8 z0,z1,loss,score(1000),lo,hi,ov,left,right
      if(reg.lt.2.or.reg.gt.nl+1) return
      j=ffirst(reg-1)
      k=flast(reg-1)
      lo=min(z0,z1)
      hi=max(z0,z1)
      do while(j.le.k)
        left=fleft(j)
        right=fright(j)
        if(hi-lo.gt.1.d-15) then
          ov=max(0.d0,min(hi,right)-max(lo,left))/(hi-lo)
        else
! Exact interface points belong to the current EGS layer.
          ov=0.d0
          if(z0.ge.left.and.z0.lt.right) ov=1.d0
          if(j.eq.k.and.abs(z0-right).lt.1.d-12) ov=1.d0
        endif
        score(j)=score(j)+loss*ov
        j=j+1
      enddo
      return
      end
