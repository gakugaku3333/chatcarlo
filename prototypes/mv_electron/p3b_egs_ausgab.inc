      subroutine ausgab(iarg)
      implicit none
      include 'include/egs5_h.f'
      include 'include/egs5_epcont.f'
      include 'include/egs5_stack.f'
      include 'p3b_egs_common.inc'
      real*8 z0,z1,loss,p0(3),p1(3),amount
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
        p0=(/x(np),y(np),z0/)
        p1=(/x(np)+u(np)*ustep,y(np)+v(np)*ustep,z1/)
        do j=1,rn
          call rsegment(rbatch(:,j,rb),p0,p1,loss,raxis(j),
     *      rorigin(j),rwidth(j),rlo(:,j),rhi(:,j),rsize(j),
     *      amount)
          rsum(j)=rsum(j)+amount
        enddo
      else if(iarg.le.4.and.ir(np).eq.2) then
        loss=edep*wt(np)
        dep=dep+loss
        call p2point(z(np),loss)
        p0=(/x(np),y(np),z(np)/)
        do j=1,rn
          call rpoint(rbatch(:,j,rb),p0,loss,raxis(j),
     *      rorigin(j),rwidth(j),rlo(:,j),rhi(:,j),rsize(j),
     *      amount)
          rsum(j)=rsum(j)+amount
        enddo
      endif
      return
      end

      subroutine howfar
      implicit none
      include 'include/egs5_h.f'
      include 'include/egs5_epcont.f'
      include 'include/egs5_stack.f'
      include 'p3b_egs_common.inc'
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
