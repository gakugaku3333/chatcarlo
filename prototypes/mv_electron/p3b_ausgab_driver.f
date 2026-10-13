      program ausgab_driver
      implicit none
      include 'include/egs5_h.f'
      include 'include/egs5_epcont.f'
      include 'include/egs5_stack.f'
      include 'p3b_egs_common.inc'
      integer iarg,cutstep,j,k
      np=1
      thick=30.d0
      score=0.d0
      fine=0.d0
      dep=0.d0
      escape=0.d0
      nescape=0
      noflight=0
      brcreated=0.d0
      brdiscard=0.d0
      brenergy=0.d0
      flightann=0.d0
      restann=0.d0
      paircount=0.d0
      brwrong=0
      rb=1
      rbatch=0.d0
      rsum=0.d0
      call rread(rn,rlo,rhi,raxis,rorigin,rwidth,rsize)
      read(*,*) iarg,iq(1),ir(1),x(1),y(1),z(1),
     * u(1),v(1),w(1),ustep,edep,wt(1),cutstep
      if(cutstep.eq.1) call howfar
      call ausgab(iarg)
      write(*,'(A,3ES26.17E3,I5)') 'TOTAL ',dep,escape,ustep,
     * nescape
      do j=1,rn
        do k=1,rsize(j)
          write(*,'(2I5,1X,ES26.17E3)') j,k,rbatch(k,j,1)
        enddo
      enddo
      end
      include 'p3b_egs_ausgab.inc'
      subroutine p2point(zz,loss)
      implicit none
      include 'p3b_egs_common.inc'
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
      include 'p3b_egs_common.inc'
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

      include 'p3b_egs_score.inc'
