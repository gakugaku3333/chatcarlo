! P2b user code: p2_egs.f plus optional rejection of in-flight
! annihilation (noflight.dat = 1). Without that file it is p2_egs.f.
! External build only. All generated brems photons tagged, not removed
! by stack position. No energy scored at IARG=7.
      implicit none
      include 'include/egs5_h.f'
      include 'include/egs5_bounds.f'
      include 'include/egs5_epcont.f'
      include 'include/egs5_media.f'
      include 'include/egs5_misc.f'
      include 'include/egs5_stack.f'
      include 'include/egs5_thresh.f'
      include 'include/egs5_useful.f'
      include 'include/egs5_usersc.f'
      include 'include/egs5_brempr.f'
      include 'include/egs5_edge.f'
      include 'include/randomm.f'
      include 'p2b_egs_common.inc'
      real*8 batches(150,100),fbatches(600,100),Esource
      real*8 sumdep,sumescape,maxres,res,unproc,cpu0,cpu1
      real*8 elapsed,priorwall,budget
      integer done,clock0,clocknow,clockrate,rstat,kout,mkout
      logical exists
      integer ncase,seed,mode,i,j,b,nleft,violations
      character*24 medarr
      call system_clock(clock0,clockrate)
      priorwall=0.d0
      budget=86400.d0
      inquire(file='budget.dat',exist=exists)
      if(exists) then
        open(43,file='budget.dat',status='old')
        read(43,*) budget
        close(43)
      endif
      open(6,file='egs5job.out',status='replace')
      open(41,file='config.dat',status='old')
      read(41,*) Esource,ncase,seed,mode
      close(41)
      noflight=0
      inquire(file='noflight.dat',exist=exists)
      if(exists) then
        open(45,file='noflight.dat',status='old')
        read(45,*) noflight
        close(45)
      endif
      call counters_out(0)
      call block_set
      nmed=1
      medarr='H2O                     '
      do i=1,24
        media(i,1)=medarr(i:i)
      end do
      chard(1)=.5d0
      call pegs5
      nreg=3
      med(1)=0
      med(2)=1
      med(3)=0
      ecut(2)=.521d0
      pcut(2)=.01d0
      k1hscl(2)=0.d0
      k1lscl(2)=0.d0
      estepr(2)=1.d0
      usegsd(1)=1
      iraylr=0
      incohr=0
      iprofr=0
      iedgfl=0
      iauger=0
      ibrspl=0
      iprdst=0
      luxlev=1
      inseed=seed
      call rluxinit
      latchi=0
      emaxe=Esource
      open(KMPI,file='pgs5job.pegs5dat',status='old')
      open(KMPO,file='egs5job.dummy',status='unknown')
      call hatch
      close(KMPI)
      close(KMPO)
      iausfl=0
      do i=1,5
        iausfl(i)=1
      end do
      iausfl(8)=1
      if(noflight.eq.1) iausfl(13)=1
      iausfl(14)=1
      iausfl(15)=1
      iausfl(17)=1
      write(6,*) 'FLAGS',incohm(1),incohr(2),iprofm(1),
     * iprofr(2),iraylm(1),iraylr(2),iedgfl(2),iauger(2),
     * iprdst,ibrspl(2),iausfl(8),usegsd(1)
      write(6,*) 'CUTOFFS',ae(1),ap(1),ecut(2),pcut(2),
     * ue(1),up(1),rhom(1)
      write(6,*) 'KINETIC_CUTOFF',ecut(2)-RM,RM
      thick=30.d0
      batches=0.d0
      fbatches=0.d0
      sumdep=0.d0
      sumescape=0.d0
      maxres=0.d0
      maxscore=0.d0
      maxfine=0.d0
      nleft=0
      violations=0
      brcreated=0.d0
      brdiscard=0.d0
      brenergy=0.d0
      flightann=0.d0
      restann=0.d0
      paircount=0.d0
      brwrong=0
      done=0
      inquire(file='checkpoint.bin',exist=exists)
      if(exists) then
        open(42,file='checkpoint.bin',form='unformatted',
     *       status='old')
        read(42) done,batches,fbatches,sumdep,sumescape,maxres,
     *    maxscore,maxfine,nleft,violations,brcreated,brdiscard,
     *    brenergy,flightann,restann,paircount,brwrong,isdext,
     *    kout,mkout,priorwall
        close(42)
        call rluxin
        kount=kout
        mkount=mkout
      endif
      call cpu_time(cpu0)
      do i=done+1,ncase
        if(mod(i-1,1000).eq.0) then
          call system_clock(clocknow)
          elapsed=priorwall+dble(clocknow-clock0)/clockrate
          if(elapsed.ge.budget) then
            write(6,*) 'TIME_LIMIT',elapsed,done
            close(6)
            stop
          endif
        endif
        score=0.d0
        fine=0.d0
        dep=0.d0
        escape=0.d0
        call shower(0,Esource,0.d0,0.d0,0.d0,0.d0,
     *              0.d0,1.d0,2,1.d0)
        unproc=0.d0
        if(np.gt.0) then
          nleft=nleft+np
          do j=1,np
            if(iq(j).eq.0) then
              unproc=unproc+e(j)
            else if(iq(j).eq.1) then
              unproc=unproc+e(j)+RM
            else
              unproc=unproc+e(j)-RM
            end if
          end do
        endif
        res=Esource-dep-escape-unproc
        maxres=max(maxres,abs(res))
        if(abs(res).gt.Esource*1.d-9) violations=violations+1
        maxscore=max(maxscore,abs(sum(score)-dep))
        maxfine=max(maxfine,abs(sum(fine)-dep))
        sumdep=sumdep+dep
        sumescape=sumescape+escape
        b=min(100,1+(i-1)/(ncase/100))
        batches(:,b)=batches(:,b)+score
        fbatches(:,b)=fbatches(:,b)+fine
        if(mod(i,ncase/100).eq.0) then
          done=i
          call rluxout
          call system_clock(clocknow)
          elapsed=priorwall+dble(clocknow-clock0)/clockrate
          open(42,file='checkpoint.tmp',form='unformatted',
     *         status='replace')
          write(42) done,batches,fbatches,sumdep,sumescape,maxres,
     *      maxscore,maxfine,nleft,violations,brcreated,brdiscard,
     *      brenergy,flightann,restann,paircount,brwrong,isdext,
     *      kount,mkount,elapsed
          close(42)
          call rename('checkpoint.tmp','checkpoint.bin',rstat)
          if(rstat.ne.0) stop 9
          open(44,file='batch_progress.tmp',status='replace')
          write(44,*) done,b,elapsed
          close(44)
          call rename('batch_progress.tmp','batch_progress.txt',
     *                rstat)
          if(rstat.ne.0) stop 10
        endif
      end do
      call cpu_time(cpu1)
      write(6,*) 'CONFIG',Esource,ncase,seed,mode
      write(6,*) 'NOFLIGHT',noflight
      write(6,*) 'LEDGER',ncase*Esource,sumdep,sumescape,
     * maxres,nleft,violations,maxscore,maxfine
      write(6,*) 'CALLBACKS',brcreated,brdiscard,brenergy,
     * flightann,restann,paircount,brwrong
      write(6,*) 'CPU',cpu1-cpu0
      call system_clock(clocknow)
      write(6,*) 'WALL',priorwall+
     * dble(clocknow-clock0)/clockrate
      do b=1,100
        do j=1,150
          write(6,*) 'BATCH',b,j,
     *      batches(j,b)/(ncase/100*.2d0)
        end do
        do j=1,600
          write(6,*) 'FINE',b,j,
     *      fbatches(j,b)/(ncase/100*.05d0)
        end do
      end do
      call counters_out(1)
      stop
      end

      subroutine ausgab(iarg)
      implicit none
      include 'include/egs5_h.f'
      include 'include/egs5_epcont.f'
      include 'include/egs5_stack.f'
      include 'p2b_egs_common.inc'
      real*8 z0,z1,loss
      integer iarg,j,marked
      if(iarg.eq.7) then
        marked=0
        do j=max(1,np-1),np
          if(iq(j).eq.0) then
            latch(j)=9001
            dnear(j)=0.d0
            marked=marked+1
            brcreated=brcreated+1.d0
          endif
        end do
        if(marked.ne.1) brwrong=brwrong+1
        return
      endif
      if(iarg.eq.12.and.noflight.eq.1) then
        sve=e(np)
        svu=u(np)
        svv=v(np)
        svw=w(np)
        return
      endif
      if(iarg.eq.13) then
        flightann=flightann+1.d0
        if(noflight.eq.1) then
! Reject: drop both photons, restore the positron as it was at IARG=12.
          np=np-1
          iq(np)=1
          e(np)=sve
          u(np)=svu
          v(np)=svv
          w(np)=svw
        endif
        return
      endif
      if(iarg.eq.14) restann=restann+1.d0
      if(iarg.eq.16) paircount=paircount+1.d0
      if(iarg.eq.3) then
        loss=edep*wt(np)
        if(latch(np).eq.9001) then
          if(iq(np).ne.0) brwrong=brwrong+1
          dep=dep+loss
          call p2point(z(np),loss)
          brdiscard=brdiscard+1.d0
          brenergy=brenergy+loss
        else
          escape=escape+loss
        endif
      else if(iarg.eq.0.and.iq(np).ne.0.and.ir(np).eq.2) then
        z0=z(np)
        z1=z0+w(np)*ustep
        loss=edep*wt(np)
        dep=dep+loss
        call p2segment(z0,z1,loss)
      else if(iarg.le.4.and.ir(np).eq.2) then
        loss=edep*wt(np)
        dep=dep+loss
        call p2point(z(np),loss)
      endif
      return
      end

      subroutine p2point(zz,loss)
      implicit none
      include 'p2b_egs_common.inc'
      real*8 zz,loss
      integer j
      j=max(1,min(150,int(zz/.2d0)+1))
      score(j)=score(j)+loss
      j=max(1,min(600,int(zz/.05d0)+1))
      fine(j)=fine(j)+loss
      return
      end

      subroutine p2segment(z0,z1,loss)
      implicit none
      include 'p2b_egs_common.inc'
      real*8 z0,z1,loss,lo,hi,ov
      integer j
      if(abs(z1-z0).le.1.d-15) then
        call p2point(z0,loss)
        return
      endif
      lo=min(z0,z1)
      hi=max(z0,z1)
      do j=max(1,int(lo/.2d0)+1),min(150,int(hi/.2d0)+1)
        ov=max(0.d0,min(hi,j*.2d0)-max(lo,(j-1)*.2d0))
        score(j)=score(j)+loss*ov/(hi-lo)
      end do
      do j=max(1,int(lo/.05d0)+1),min(600,int(hi/.05d0)+1)
        ov=max(0.d0,min(hi,j*.05d0)-max(lo,(j-1)*.05d0))
        fine(j)=fine(j)+loss*ov/(hi-lo)
      end do
      return
      end

      subroutine howfar
      implicit none
      include 'include/egs5_h.f'
      include 'include/egs5_epcont.f'
      include 'include/egs5_stack.f'
      include 'p2b_egs_common.inc'
      real*8 dist
      if(latch(np).eq.9001.or.ir(np).ne.2) then
        idisc=1
        return
      endif
      if(w(np).gt.0.d0) then
        dist=(thick-z(np))/w(np)
        if(dist.le.ustep) then
          ustep=max(0.d0,dist)
          irnew=3
        endif
      else if(w(np).lt.0.d0) then
        dist=-z(np)/w(np)
        if(dist.le.ustep) then
          ustep=max(0.d0,dist)
          irnew=1
        endif
      endif
      return
      end
