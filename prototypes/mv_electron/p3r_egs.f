! P3r: transported brems photons, count only at IARG=7.
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
      include 'p3r_egs_common.inc'
      real*8 batches(150,100),fbatches(600,100),Esource
      real*8 sumdep,sumescape,maxres,res,unproc,cpu0,cpu1
      real*8 elapsed,priorwall,budget
      integer done,clock0,clocknow,clockrate,rstat,kout,mkout
      real*8 elo(64),ehi(64),cum(64),ehist,u1,u2,sumin
      real*8 sinb(100),pairb(100),restb(100),flightb(100)
      real*8 prevpair,prevrest,prevflight,maxrel
      real*8 brnb(100),breb(100),prevbrn,prevbre
      integer stopbatch
      integer nb,ispec,cell,nspec(64,10)
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
      stopbatch=0
      inquire(file='stop_batches.dat',exist=exists)
      if(exists) then
        open(47,file='stop_batches.dat',status='old')
        read(47,*) stopbatch
        close(47)
      endif
      nb=0
      inquire(file='spectrum.dat',exist=exists)
      if(exists) then
        open(46,file='spectrum.dat',status='old')
        read(46,*) nb
        if(nb.lt.1.or.nb.gt.64) stop 11
        do j=1,nb
          read(46,*) elo(j),ehi(j),cum(j)
          if(elo(j).lt.0.d0.or.ehi(j).le.elo(j)) stop 12
          if(cum(j).le.0.d0.or.cum(j).gt.1.d0) stop 13
          if(j.gt.1) then
            if(elo(j).ne.ehi(j-1)) stop 14
            if(cum(j).le.cum(j-1)) stop 15
          endif
        enddo
        if(cum(nb).ne.1.d0) stop 16
        close(46)
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
      if(nb.gt.0) emaxe=ehi(nb)
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
      nescape=0
      sumin=0.d0
      sinb=0.d0
      nspec=0
      pairb=0.d0
      restb=0.d0
      flightb=0.d0
      maxrel=0.d0
      brnb=0.d0
      breb=0.d0
      done=0
      inquire(file='checkpoint.bin',exist=exists)
      if(exists) then
        open(42,file='checkpoint.bin',form='unformatted',
     *       status='old')
        read(42) done,batches,fbatches,sumdep,sumescape,maxres,
     *    maxscore,maxfine,nleft,violations,brcreated,brdiscard,
     *    brenergy,flightann,restann,paircount,brwrong,isdext,
     *    kout,mkout,priorwall,sumin,sinb,nspec,pairb,restb,
     *    flightb,maxrel,nescape,brnb,breb
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
        ehist=Esource
        b=min(100,1+(i-1)/(ncase/100))
        if(nb.gt.0) then
          call randomset(u1)
          call randomset(u2)
          ispec=1
          do while(cum(ispec).lt.u1)
            ispec=ispec+1
          enddo
          ehist=elo(ispec)+u2*(ehi(ispec)-elo(ispec))
          cell=min(10,1+int(10.d0*(ehist-elo(ispec))/
     *                     (ehi(ispec)-elo(ispec))))
          nspec(ispec,cell)=nspec(ispec,cell)+1
        endif
        sumin=sumin+ehist
        sinb(b)=sinb(b)+ehist
        prevbrn=brcreated
        prevbre=brenergy
        prevpair=paircount
        prevrest=restann
        prevflight=flightann
        call shower(0,ehist,0.d0,0.d0,0.d0,0.d0,
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
        pairb(b)=pairb(b)+paircount-prevpair
        restb(b)=restb(b)+restann-prevrest
        flightb(b)=flightb(b)+flightann-prevflight
        brnb(b)=brnb(b)+brcreated-prevbrn
        breb(b)=breb(b)+brenergy-prevbre
        res=ehist-dep-escape-unproc
        if(ehist.gt.0.d0) maxrel=max(maxrel,abs(res)/ehist)
        maxres=max(maxres,abs(res))
        if(abs(res).gt.ehist*1.d-9) violations=violations+1
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
     *      kount,mkount,elapsed,sumin,sinb,nspec,pairb,restb,
     *      flightb,maxrel,nescape,brnb,breb
          close(42)
          call rename('checkpoint.tmp','checkpoint.bin',rstat)
          if(rstat.ne.0) stop 9
          open(44,file='batch_progress.tmp',status='replace')
          write(44,*) done,b,elapsed
          close(44)
          call rename('batch_progress.tmp','batch_progress.txt',
     *                rstat)
          if(rstat.ne.0) stop 10
          if(b.eq.stopbatch) then
            write(6,*) 'P3R_PAUSE',done,b
            close(6)
            stop
          endif
        endif
      end do
      call cpu_time(cpu1)
      write(6,*) 'CONFIG',Esource,ncase,seed,mode
      write(6,*) 'NOFLIGHT',noflight
      write(6,*) 'LEDGER',sumin,sumdep,sumescape,
     * maxres,nleft,violations,maxscore,maxfine
      write(6,*) 'CALLBACKS',brcreated,brdiscard,brenergy,
     * flightann,restann,paircount,brwrong
      write(6,*) 'ESCAPES',nescape
      write(6,*) 'MAXREL',maxrel
      write(6,*) 'SPECTRUM',nb
      do j=1,nb
        write(6,*) 'SPEC',j,(nspec(j,cell),cell=1,10)
      enddo
      do b=1,100
        write(6,*) 'BATCHIN',b,sinb(b)
        write(6,*) 'BATCHBR',b,brnb(b),breb(b)
        write(6,*) 'BATCHCB',b,pairb(b),restb(b),flightb(b)
      enddo
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
      include 'p3r_egs_common.inc'
      real*8 z0,z1,loss
      integer iarg,j,marked
      if(iarg.eq.7) then
        marked=0
        do j=max(1,np-1),np
          if(iq(j).eq.0) then
            marked=marked+1
            brcreated=brcreated+1.d0
            brenergy=brenergy+e(j)*wt(j)
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
        escape=escape+loss
        nescape=nescape+1
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
      include 'p3r_egs_common.inc'
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
      include 'p3r_egs_common.inc'
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
      include 'p3r_egs_common.inc'
      real*8 dist
      if(ir(np).ne.2) then
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
