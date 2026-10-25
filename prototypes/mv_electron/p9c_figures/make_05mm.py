import os
import numpy as np,os,matplotlib,sys
matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from prototypes.mv_electron.gpu import p9c_core as core
from combined import load
plt.rcParams['font.family']='Hiragino Sans'
G=core.FINE;x,y,z=G.centers();m=load('fine',(240,200,300));r=m.result();mask=core.water_mask(G)
def bin2(a,op=np.sum):return op(op(op(a.reshape(120,2,100,2,150,2),axis=5),axis=3),axis=1)
W=bin2(np.where(mask,r['mean_with'],0.));O=bin2(np.where(mask,r['mean_without'],0.))
ok=bin2(mask.astype(int))==8
# ratio SEM of the 8-cell sum: per cell, sem of ratio times cell weight; independent (lower) and fully correlated (upper)
cellsem=np.where(mask,r['ratio_sem']*r['mean_without'],0.)
lo=np.sqrt(bin2(cellsem**2))/O;hi=bin2(cellsem)/O
rat=np.where(ok,W/O,np.nan);lo=np.where(ok,lo,np.nan);hi=np.where(ok,hi,np.nan)
X=x.reshape(120,2).mean(1);Y=y.reshape(100,2).mean(1);Z=z.reshape(150,2).mean(1);V=8*1.5625e-5
pros=((X[:,None,None]/2.25)**2+(Y[None,:,None]/1.75)**2+(Z[None,None,:]/1.5)**2<=1)&ok
ref=(O[pros]/V).mean()
iy=int(round((-0.2+2.5)/.05));ix=int(round(3/.05));print('layer y',Y[iy],'x',X[ix])
S=np.array(core.seeds())
fig=plt.figure(figsize=(22,13));gs=fig.add_gridspec(3,4)
c1=plt.get_cmap('jet').copy();c1.set_bad('0.6');c2=plt.get_cmap('RdBu_r').copy();c2.set_bad('0.6');c3=plt.get_cmap('viridis').copy();c3.set_bad('0.6')
for row,(axial,sl,H,xl,lab) in enumerate(((True,(slice(None),iy,slice(None)),X,'x 左右 (cm)',f'横断面 y = {Y[iy]:+.3f} cm（0.5 mm 層、シードの中心）'),(False,(ix,slice(None),slice(None)),Y,'y 頭尾 (cm)',f'矢状断面 x = {X[ix]:+.3f} cm（0.5 mm 層、中央の針）'))):
    E=[H[0]-.025,H[-1]+.025,Z[-1]+.025,Z[0]-.025]
    for col,(arr,cm,v0,v1,t) in enumerate(((np.where(ok,W/V,np.nan)[sl]/ref,c1,.8,1.15,'シードあり（前立腺の平均で規格化）'),((bin2(r['mean_without'])/V)[sl]/ref,c1,.8,1.15,'シードなし'),(rat[sl],c2,.95,1.05,'比（あり／なし）±5%'),(hi[sl],c3,0,.04,'比の SEM の上限（相関 1）'))):
        a=fig.add_subplot(gs[row,col]);im=a.imshow(arr.T,extent=E,cmap=cm,vmin=v0,vmax=v1,aspect='equal',interpolation='nearest')
        a.add_patch(Ellipse((0,0),4.5 if axial else 3.5,3.0,fill=False,ec='k',lw=1));a.set_xlim(-2.5,2.5) if axial else a.set_xlim(-2,2);a.set_ylim(1.75,-1.75)
        a.set_title(lab+'\n'+t,fontsize=9);a.set_xlabel(xl);a.set_ylabel('z 前→後 (cm)');plt.colorbar(im,ax=a,fraction=.046)
# profiles at 0.5 mm through seed centres along z, and across the z=0 row along x
a=fig.add_subplot(gs[2,0:2])
for (sx,sy),c in (((0.,-0.2),'C0'),((0.5,0.5),'C1'),((-1.,-0.9),'C2')):
    i=int(round((sx+3)/.05));j=int(round((sy+2.5)/.05))
    a.plot(Z,rat[i,j],'.-',color=c,lw=.8,ms=3,label=f'x={X[i]:+.3f}, y={Y[j]:+.3f} cm')
    a.fill_between(Z,rat[i,j]-hi[i,j],rat[i,j]+hi[i,j],color=c,alpha=.12);a.fill_between(Z,rat[i,j]-lo[i,j],rat[i,j]+lo[i,j],color=c,alpha=.25)
a.axhline(1,color='k',lw=.6);a.set_xlim(-1.75,1.75);a.set_ylim(.9,1.1);a.set_xlabel('z 前→後 (cm)');a.set_ylabel('比（あり／なし）')
a.set_title('シードの中心を通る z 方向の線（0.5 mm セル）。濃い帯 = SEM（独立とみなす）、薄い帯 = SEM の上限（相関 1）。切れ目 = シードを含むセル',fontsize=9);a.legend(fontsize=8)
a=fig.add_subplot(gs[2,2:4]);k=int(round(2.5/.05))
a.plot(X,rat[:,iy,k],'.-',color='C3',lw=.8,ms=3,label=f'y={Y[iy]:+.3f}, z={Z[k]:+.3f} cm')
a.fill_between(X,rat[:,iy,k]-hi[:,iy,k],rat[:,iy,k]+hi[:,iy,k],color='C3',alpha=.12);a.fill_between(X,rat[:,iy,k]-lo[:,iy,k],rat[:,iy,k]+lo[:,iy,k],color='C3',alpha=.25)
for v in (-2,-1,0,1,2):a.axvline(v,color='0.7',lw=.6)
a.axhline(1,color='k',lw=.6);a.set_xlim(-2.5,2.5);a.set_ylim(.9,1.1);a.set_xlabel('x 左右 (cm)');a.set_ylabel('比（あり／なし）')
a.set_title('シードの列を横切る x 方向の線（0.5 mm セル、灰色の縦線 = シードの位置）',fontsize=9);a.legend(fontsize=8)
fig.suptitle('P9c 前立腺の形の照射野・7 門・10 MV、シード 58 本のあり／なし、0.5 mm セルに合算（0.25 mm × 8、'+os.environ.get('P9C_LABEL','')+'、灰色 = シードを含むセル）\n参考: P9b は不合格のまま記録、シードの近くの線量は EGS5 と照合していない',fontsize=12)
fig.tight_layout();fig.savefig(os.environ.get('P9C_OUT','runs/prostate_seed_7field')+'/p9c_05mm'+os.environ.get('P9C_TAG','')+'.png',dpi=105)
print('typical SEM in prostate: independent',np.nanmedian(lo[pros]),'upper',np.nanmedian(hi[pros]))
