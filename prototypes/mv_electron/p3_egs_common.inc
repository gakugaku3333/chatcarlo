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
