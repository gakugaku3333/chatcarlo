! P0 MV derived from revalidation pdd_phantom; 1350 water cells.
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
      real*8 edeph(150),edtot

      real*8 ein,xin,yin,zin,             ! Arguments
     *       uin,vin,win,wtin
      integer iqin,irin

      real*8 fieldhw                          ! Local variables
      real*8 sumx(150),sumx2(150),meanb(150),varb(150),semb(150),relb(150)
      real*8 sumtot,sumtot2,meantot,vartot,semtot,reltot
      real*8 rn1,rn2
      integer i,j,ncase,ix0,iy0
      integer nhsteps,nphsteps,npair,nann,nposesc
      integer nleft,nabnormal
      real*8 escapeh,resth,sumesc,sumrest,unprocessed
      real*8 sumunproc,maxres,resid,tstart,tend,scortime
      real*8 pairenergy,annenergy,posescrest
      real*8 tscore0,tscore1
      common/ledger/escapeh,nhsteps,nphsteps,npair,nann,
     * nposesc,scortime
      integer histstep(100000)
      integer istepbin
      character*24 medarr(1)
      character*24 label(150)

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

      chard(1) = 0.2d0

      write(6,100)
100   FORMAT(' PEGS5-call comes next'/)

!     ==========
      call pegs5
!     ==========

!-----------------------------------------------------------------------
! Step 3: Pre-hatch-call-initialization
!-----------------------------------------------------------------------
      nreg=1352
      med(1)=0
      med(nreg)=0
      do j=2,nreg-1
        med(j)=1
        ecut(j)=0.521d0
        pcut(j)=0.001d0
        iraylr(j)=1
        incohr(j)=1
      end do
!     Region 2 is water (whole phantom bulk); 1,3 are vacuum
      ecut(2)=0.521d0
      pcut(2)=0.001
      iraylr(2)=1
      incohr(2)=1
!     Turn on S(q)-weighted (Waller-Hartree) incoherent scattering for
!     the phantom region, matching the INCOH=1 PEGS5 data generated
!     for this run (Step 1 of plan_residual_check.md). Region 2 is the
!     water phantom bulk (see region map above); regions 1/3 are
!     vacuum and never score, so incohr for them is left at the
!     egs5_block_set.f default of 0.

      luxlev=1
      inseed=20261007
      write(6,120) inseed
120   FORMAT(/,' inseed=',I12,5X,
     *         ' (seed for generating unique sequences of Ranlux)')

!     =============
      call rluxinit
!     =============

!-----------------------------------------------------------------------
! Step 4:  Determination-of-incident-particle-parameters
!-----------------------------------------------------------------------
      iqin=0
      ein=10.0d0
      zin=0.d0
      uin=0.0
      vin=0.0
      win=1.0
      irin=1
      wtin=1.0
      latchi=0

      fieldhw=5.0d0
!     Half-width of the 10x10 cm^2 field (non-divergent approximation
!     of the point-source beam at SSD=100 cm -- see pdd60_NOTES.md)

!-----------------------------------------------------------------------
! Step 5:   hatch-call
!-----------------------------------------------------------------------
      emaxe = ein + RM

      write(6,130)
130   format(/' Start pdd60_phantom_incoh1'/
     *        ' Call hatch to get cross-section data')

      open(UNIT=KMPI,FILE='pgs5job.pegs5dat',STATUS='old')
      open(UNIT=KMPO,FILE='egs5job.dummy',STATUS='unknown')

      write(6,140)
140   format(/,' HATCH-call comes next',/)

!     ==========
      call hatch
      iausfl(13)=1
      iausfl(15)=1
      iausfl(16)=1
      write(6,*) 'PHYSICS',rhom(1),ae(1),ap(1),ue(1),up(1),
     * ecut(2),pcut(2),iraylr(2),incohr(2)
      close(UNIT=KMPI)
      close(UNIT=KMPO)

      write(6,145) incohr(2),iprofr(2)
145   format(/' incohr(2)=',i2,' iprofr(2)=',i2,
     *        '  (confirmation, cf. tutor7.f line 298-299)')

      write(6,150) ae(1)-RM, ap(1)
150   format(/' Knock-on electrons can be created and any electron ',
     *'followed down to' /T40,F8.3,' MeV kinetic energy'/
     *' Brem photons can be created and any photon followed down to',
     */T40,F8.3,' MeV')

!-----------------------------------------------------------------------
! Step 6:  Initialization-for-howfar
!-----------------------------------------------------------------------
      zback=30.0d0
      xyhw=15.0d0
!     30x30x20 cm water phantom, front face at z=0, single region

!-----------------------------------------------------------------------
! Step 7:  Initialization-for-ausgab
!-----------------------------------------------------------------------
      do i=1,150
        sumx(i)=0.d0
        sumx2(i)=0.d0
      end do
      sumtot=0.d0
      sumtot2=0.d0

!-----------------------------------------------------------------------
! Step 8:  Shower-call
!-----------------------------------------------------------------------
      ncase=10000
      sumesc=0.d0
      sumrest=0.d0
      sumunproc=0.d0
      pairenergy=0.d0
      annenergy=0.d0
      posescrest=0.d0
      maxres=0.d0
      nleft=0
      nabnormal=0
      nphsteps=0
      scortime=0.d0
      histstep=0
      call cpu_time(tstart)
      do i=1,ncase
        call randomset(rn1)
        call randomset(rn2)
        xin=(2.d0*rn1-1.d0)*fieldhw
        yin=(2.d0*rn2-1.d0)*fieldhw

        do j=1,150
          edeph(j)=0.d0
        end do
        edtot=0.d0
        escapeh=0.d0
        nhsteps=0
        npair=0
        nann=0
        nposesc=0
        ix0=1
        iy0=1
        if(xin.ge.-1.d0) ix0=2
        if(xin.ge.1.d0) ix0=3
        if(yin.ge.-1.d0) iy0=2
        if(yin.ge.1.d0) iy0=3
        irin=2+(iy0-1)*3+ix0-1

        call shower(iqin,ein,xin,yin,zin,uin,vin,win,irin,wtin)

        unprocessed=0.d0
        if(np.gt.0) then
          nleft=nleft+1
          do j=1,np
            if(iq(j).eq.0) then
              unprocessed=unprocessed+e(j)
            else
              unprocessed=unprocessed+e(j)-RM
            end if
          end do
        end if
        resth=2.d0*RM*(npair-nann)
        sumesc=sumesc+escapeh
        sumrest=sumrest+resth
        sumunproc=sumunproc+unprocessed
        pairenergy=pairenergy+2.d0*RM*npair
        annenergy=annenergy+2.d0*RM*nann
        posescrest=posescrest+2.d0*RM*nposesc
        resid=ein-edtot-escapeh-resth-unprocessed
        if(resid.ne.resid) nabnormal=nabnormal+1
        maxres=max(maxres,abs(resid))
        istepbin=min(100000,nhsteps+1)
        histstep(istepbin)=histstep(istepbin)+1
        do j=1,150
          sumx(j)  = sumx(j)  + edeph(j)
          sumx2(j) = sumx2(j) + edeph(j)*edeph(j)
        end do
        sumtot  = sumtot  + edtot
        sumtot2 = sumtot2 + edtot*edtot
      end do

!-----------------------------------------------------------------------
! Step 9:  Output-of-results
!-----------------------------------------------------------------------
      do j=1,150
        meanb(j) = sumx(j)/dfloat(ncase)
        varb(j)  = sumx2(j)/dfloat(ncase) - meanb(j)*meanb(j)
        if (varb(j).lt.0.d0) varb(j)=0.d0
        varb(j)  = varb(j)*dfloat(ncase)/dfloat(ncase-1)
        semb(j)  = dsqrt(varb(j)/dfloat(ncase))
        if (meanb(j).gt.0.d0) then
          relb(j) = 100.d0*semb(j)/meanb(j)
        else
          relb(j) = -1.d0
        end if
      end do

      meantot = sumtot/dfloat(ncase)
      vartot  = sumtot2/dfloat(ncase) - meantot*meantot
      if (vartot.lt.0.d0) vartot=0.d0
      vartot  = vartot*dfloat(ncase)/dfloat(ncase-1)
      semtot  = dsqrt(vartot/dfloat(ncase))
      if (meantot.gt.0.d0) then
        reltot = 100.d0*semtot/meantot
      else
        reltot = -1.d0
      end if

      call cpu_time(tend)
      write(6,*) 'NCASE',ncase
      write(6,*) 'TIME',tend-tstart,scortime
      write(6,*) 'LEDGER',ncase*ein,sumtot,sumesc,sumunproc,
     * sumrest,maxres,nleft,nabnormal
      write(6,*) 'REST_COMPONENTS',pairenergy,annenergy,posescrest
      write(6,*) 'PHOTON_STEPS',nphsteps
      do j=1,150
        write(6,*) 'PDD',j,(j-0.5d0)*0.2d0,meanb(j),semb(j)
      end do
      do j=1,100000
        if(histstep(j).gt.0) write(6,*) 'STEPS',j-1,histstep(j)
      end do
      call counters_out(1)
      stop
      end
!-------------------------last line of main code------------------------

!-------------------------------ausgab.f--------------------------------
!-----------------------------------------------------------------------
      subroutine ausgab(iarg)
      implicit none
      include 'include/egs5_h.f'
      include 'include/egs5_epcont.f'
      include 'include/egs5_stack.f'
      include 'include/egs5_useful.f'
      common/score/edeph,edtot
      real*8 edeph(150),edtot
      integer nhsteps,nphsteps,npair,nann,nposesc
      real*8 escapeh,scortime,t0,t1
      common/ledger/escapeh,nhsteps,nphsteps,npair,nann,
     * nposesc,scortime
      integer iarg,k,ix,iy,iz
      call cpu_time(t0)
      if(iarg.eq.0) then
        if(iq(np).ne.0) then
          nhsteps=nhsteps+1
        else
          nphsteps=nphsteps+1
        end if
      end if
      if(iarg.eq.15) npair=npair+1
      if(iarg.eq.12.or.iarg.eq.14) nann=nann+1
      if(iarg.eq.3) then
        if(iq(np).eq.0) then
          escapeh=escapeh+e(np)
        else
          escapeh=escapeh+e(np)-RM
          if(iq(np).eq.1) nposesc=nposesc+1
        end if
      else if(iarg.le.4) then
        k=ir(np)-2
        if(k.ge.0.and.k.lt.1350) then
          edtot=edtot+edep
          iz=k/9+1
          iy=mod(k,9)/3+1
          ix=mod(k,3)+1
          if(ix.eq.2.and.iy.eq.2) then
            edeph(iz)=edeph(iz)+edep
          end if
        end if
      end if
      call cpu_time(t1)
      scortime=scortime+t1-t0
      return
      end
!--------------------------last line of ausgab.f------------------------

!-------------------------------howfar.f--------------------------------
!-----------------------------------------------------------------------
!  True 3-D rectangular box geometry (RPP-style distance-to-surface),
!  reused unchanged in structure from bsf60_phantom.f's howfar, but
!  simplified to a single water region (no front-layer/bulk split,
!  since scoring is now analytic-coordinate based in ausgab).
!-----------------------------------------------------------------------
      subroutine howfar
      implicit none
      include 'include/egs5_h.f'
      include 'include/egs5_epcont.f'
      include 'include/egs5_stack.f'
      real*8 edges(4),tx,ty,tz,tmin,loz,hiz
      integer k,ix,iy,iz,nextx,nexty,nextz
      data edges/-15.d0,-1.d0,1.d0,15.d0/
      if(ir(np).eq.1.or.ir(np).eq.1352) then
        idisc=1
        return
      end if
      k=ir(np)-2
      iz=k/9+1
      iy=mod(k,9)/3+1
      ix=mod(k,3)+1
      tx=1.d30
      ty=1.d30
      tz=1.d30
      nextx=ix
      nexty=iy
      nextz=iz
      if(u(np).gt.0.d0) then
        tx=(edges(ix+1)-x(np))/u(np)
        nextx=ix+1
      else if(u(np).lt.0.d0) then
        tx=(edges(ix)-x(np))/u(np)
        nextx=ix-1
      end if
      if(v(np).gt.0.d0) then
        ty=(edges(iy+1)-y(np))/v(np)
        nexty=iy+1
      else if(v(np).lt.0.d0) then
        ty=(edges(iy)-y(np))/v(np)
        nexty=iy-1
      end if
      loz=(iz-1)*0.2d0
      hiz=iz*0.2d0
      if(w(np).gt.0.d0) then
        tz=(hiz-z(np))/w(np)
        nextz=iz+1
      else if(w(np).lt.0.d0) then
        tz=(loz-z(np))/w(np)
        nextz=iz-1
      end if
      tmin=min(tx,ty,tz)
      if(tmin.gt.ustep) return
      ustep=max(0.d0,tmin)
      if(tmin.eq.tx) then
        ix=nextx
      else if(tmin.eq.ty) then
        iy=nexty
      else
        iz=nextz
      end if
      if(ix.lt.1.or.ix.gt.3.or.iy.lt.1.or.iy.gt.3.or.
     * iz.lt.1.or.iz.gt.150) then
        irnew=1352
      else
        irnew=2+(iz-1)*9+(iy-1)*3+ix-1
      end if
      return
      end
