! PEGS-only, derived from pdd_phantom MAIN initialization.
      implicit none

!     ------------
!     EGS5 COMMONs
!     ------------
      include 'include/egs5_h.f'                ! Main EGS "header" file

      include 'include/egs5_bounds.f'
      include 'include/egs5_epcont.f'
      include 'include/egs5_media.f'
      include 'include/egs5_misc.f'
      include 'include/egs5_stack.f'
      include 'include/egs5_thresh.f'
      include 'include/egs5_useful.f'
      include 'include/egs5_usersc.f'
      include 'include/randomm.f'

      common/geom/zback,xyhw
      real*8 zback,xyhw
!     zback = total phantom depth (20 cm)
!     xyhw  = phantom lateral half-width (15 cm, for a 30x30 cm face)

      common/score/edeph,edtot
      real*8 edeph(47),edtot

      real*8 ein,xin,yin,zin,             ! Arguments
     *       uin,vin,win,wtin
      integer iqin,irin

      real*8 fieldhw                          ! Local variables
      real*8 sumx(47),sumx2(47),meanb(47),varb(47),semb(47),relb(47)
      real*8 sumtot,sumtot2,meantot,vartot,semtot,reltot
      real*8 rn1,rn2
      integer i,j,ncase
      character*24 medarr(1)
      character*24 label(47)

!     ----------
!     Open files
!     ----------
      open(UNIT= 6,FILE='egs5job.out',STATUS='unknown')

!     ====================
      call counters_out(0)
!     ====================

!-----------------------------------------------------------------------
! Step 2: pegs5-call
!-----------------------------------------------------------------------
!     ==============
      call block_set
!     ==============

      nmed=1
      medarr(1)='H2O                     '

      do j=1,nmed
        do i=1,24
          media(i,j)=medarr(j)(i:i)
        end do
      end do

      chard(1) = 0.5d0

      write(6,100)
100   FORMAT(' PEGS5-call comes next'/)

!     ==========
      call pegs5
      stop
      end
      subroutine ausgab(iarg)
      integer iarg
      return
      end
      subroutine howfar
      return
      end
