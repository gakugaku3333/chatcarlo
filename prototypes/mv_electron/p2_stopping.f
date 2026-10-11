! Diagnostic only: no showers. Direct PEGS functions and actual PWLF.
      implicit none
      call block_set
      call pegs5
      call p2diagnostic
      stop
      end
      subroutine p2diagnostic
      implicit none
      include 'include/egs5_h.f'
      include 'pegscommons/dercon.f'
      include 'pegscommons/thres2.f'
      include 'pegscommons/molvar.f'
      include 'pegscommons/rslts.f'
      real*8 t,en,x,v(15),fit(15),spione,brmstm
      integer i,j,h,ix
      rho=1.d0
      open(42,file='stopping.txt',status='replace')
      write(42,*) 'META',rm,ae,ap,rlc,rho,axe,bxe,nel
! Fixed 401-node diagnostic grid over .01--10 MeV.
      do i=0,400
        t=.01d0*exp(log(1000.d0)*i/400.d0)
        en=t+rm
        x=log(t)
        ix=max(1,min(nel,int(axe*x+bxe)))
        call efuns(en,v)
        fit=afe(ix,:)*x+bfe(ix,:)
        write(42,'(A,1X,36ES25.16)') 'GRID',t,
     *   spione(en,ae)/rlc,brmstm(en,en)/rlc,
     *   brmstm(en,ap)/rlc,v,fit
      end do
! Same interior nodes as final QFIT (NIP=3 for 148 intervals).
! Read-only: do not refit or alter tables.
      do j=2,nel-1
        do h=1,3
          x=(j-bxe+h/4.d0)/axe
          t=exp(x)
          call efuns(t+rm,v)
          fit=afe(j,:)*x+bfe(j,:)
          write(42,'(A,1X,31ES25.16)') 'AUDIT',t,v,fit
        end do
      end do
      write(42,*) 'DONE'
      close(42)
      return
      end
! No transport in this diagnostic.
      subroutine ausgab(iarg)
      integer iarg
      return
      end
      subroutine howfar
      return
      end
