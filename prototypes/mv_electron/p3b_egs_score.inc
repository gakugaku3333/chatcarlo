! Argument-only half-open scoring, shared by EGS5 and both drivers.
      subroutine rpoint(s,p,loss,axis,origin,width,lo,hi,n,total)
      implicit none
      integer axis,n,j,k
      real*8 s(n),p(3),loss,origin,width,lo(3),hi(3),total
      total=0.d0
      do j=1,3
        if(p(j).lt.lo(j).or.p(j).ge.hi(j)) return
      enddo
      k=floor((p(axis)-origin)/width)+1
      if(k.ge.1.and.k.le.n) then
        s(k)=s(k)+loss
        total=loss
      endif
      end

      subroutine rsegment(s,p,q,loss,axis,origin,width,
     *                    lo,hi,n,total)
      implicit none
      integer axis,n,j,k,first,last
      real*8 s(n),p(3),q(3),loss,origin,width,lo(3),hi(3)
      real*8 total,t0,t1,d,a,b,q0,q1,amount
      if(all(p.eq.q)) then
        call rpoint(s,p,loss,axis,origin,width,lo,hi,n,total)
        return
      endif
      t0=0.d0
      t1=1.d0
      total=0.d0
      do j=1,3
        if(p(j).lt.lo(j).and.q(j).lt.lo(j)) return
        if(p(j).ge.hi(j).and.q(j).ge.hi(j)) return
      enddo
      do j=1,3
        d=q(j)-p(j)
        if(d.eq.0.d0) then
          if(p(j).lt.lo(j).or.p(j).ge.hi(j)) return
        else
          a=(lo(j)-p(j))/d
          b=(hi(j)-p(j))/d
          t0=max(t0,min(a,b))
          t1=min(t1,max(a,b))
        endif
      enddo
      if(t0.ge.t1) return
      d=q(axis)-p(axis)
      if(d.eq.0.d0) then
        k=floor((p(axis)-origin)/width)+1
        if(k.ge.1.and.k.le.n) then
          total=loss*(t1-t0)
          s(k)=s(k)+total
        endif
        return
      endif
      q0=p(axis)+t0*d
      q1=p(axis)+t1*d
      first=max(1,floor((min(q0,q1)-origin)/width)+1)
      last=min(n,floor((max(q0,q1)-origin)/width)+1)
      do k=first,last
        a=(origin+(k-1)*width-p(axis))/d
        b=(origin+k*width-p(axis))/d
        amount=loss*max(0.d0,min(t1,max(a,b))-max(t0,min(a,b)))
        s(k)=s(k)+amount
        total=total+amount
      enddo
      end

      subroutine rread(n,lo,hi,axis,origin,width,size)
      use, intrinsic :: ieee_arithmetic
      implicit none
      integer n,axis(16),size(16),j,ios
      real*8 lo(3,16),hi(3,16),origin(16),width(16)
      logical exists
      n=0
      lo=0.d0
      hi=0.d0
      axis=0
      origin=0.d0
      width=0.d0
      size=0
      inquire(file='regions.dat',exist=exists)
      if(.not.exists) return
      open(48,file='regions.dat',status='old')
      read(48,*,iostat=ios) n
      if(ios.ne.0.or.n.lt.1.or.n.gt.16) stop 31
      do j=1,n
        read(48,*,iostat=ios) lo(:,j),hi(:,j),axis(j),
     *                         origin(j),width(j),size(j)
        if(ios.ne.0) stop 32
        if(any(ieee_is_nan(lo(:,j))).or.
     *     any(ieee_is_nan(hi(:,j)))) stop 33
        if(any(lo(:,j).ge.hi(:,j))) stop 33
        if(.not.ieee_is_finite(origin(j)).or.
     *     .not.ieee_is_finite(width(j))) stop 34
        if(width(j).le.0.d0.or.axis(j).lt.1.or.
     *     axis(j).gt.3.or.size(j).lt.1.or.size(j).gt.320)
     *     stop 35
      enddo
      close(48)
      end
