      program score_driver
      implicit none
      integer n,axis(16),size(16),j,k,ios
      real*8 lo(3,16),hi(3,16),origin(16),width(16)
      real*8 scores(320,16),p(3),q(3),loss,total
      call rread(n,lo,hi,axis,origin,width,size)
      scores=0.d0
      open(51,file='segments.dat',status='old')
      do
        read(51,*,iostat=ios) p,q,loss
        if(ios.lt.0) exit
        if(ios.ne.0) stop 40
        do j=1,n
          call rsegment(scores(:,j),p,q,loss,axis(j),origin(j),
     *      width(j),lo(:,j),hi(:,j),size(j),total)
        enddo
      enddo
      close(51)
      do j=1,n
        do k=1,size(j)
          write(*,'(2I5,1X,ES26.17E3)') j,k,scores(k,j)
        enddo
      enddo
      end
      include 'p3b_egs_score.inc'
