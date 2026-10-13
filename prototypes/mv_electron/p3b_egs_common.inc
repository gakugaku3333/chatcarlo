      real*8 score(150),fine(600),dep,escape,thick
      real*8 brcreated,brdiscard,brenergy,flightann,restann
      real*8 paircount,maxscore,maxfine
      integer brwrong
      common/p2score/score,fine,dep,escape,thick,
     * brcreated,brdiscard,brenergy,flightann,restann,
     * paircount,maxscore,maxfine,brwrong
      real*8 sve,svu,svv,svw
      integer noflight
      common/p2bflt/sve,svu,svv,svw,noflight
      integer nescape
      common/p3escape/nescape
      integer rn,raxis(16),rsize(16),rledger(16),rb
      integer srcgrid(10,10),srccount
      real*8 rlo(3,16),rhi(3,16),rorigin(16),rwidth(16)
      real*8 rbatch(320,16,100),rsum(16),srcrec(5,1000)
      common/p3boxes/rlo,rhi,rorigin,rwidth,rbatch,rsum,
     * srcrec,rn,raxis,rsize,rledger,rb,srcgrid,srccount
