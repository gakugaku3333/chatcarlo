import os
import numpy as np,os,matplotlib
matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Circle,Rectangle
from prototypes.mv_electron.gpu import p9c_core as core
from prototypes.mv_electron.gpu.p9c_stats import Moments
plt.rcParams['font.family']='Hiragino Sans'
G=core.FINE;x,y,z=G.centers()
import sys;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)));from combined import load
m=load('fine',(240,200,300))
r=m.result();mask=core.water_mask(G);V=1.5625e-5
dw=np.where(mask,r['mean_with']/V,np.nan);do=r['mean_without']/V  # no seeds in this geometry: show every cell
rat=np.where(mask,r['ratio'],np.nan);rs=np.where(mask,r['ratio_sem'],np.nan)
pros=((x[:,None,None]/2.25)**2+(y[None,:,None]/1.75)**2+(z[None,None,:]/1.5)**2<=1)&mask
ref=np.nanmean(do[pros])
iy=int(round((-0.2+2.5)/.025));ix=int(round((0+3)/.025));print('y',y[iy],'x',x[ix])
cm=plt.get_cmap('jet').copy();cm.set_bad('0.6');cr=plt.get_cmap('RdBu_r').copy();cr.set_bad('0.6');cs=plt.get_cmap('viridis').copy();cs.set_bad('0.6')
fig,ax=plt.subplots(2,4,figsize=(22,9))
def outline(a,axial):
    if axial:a.add_patch(Ellipse((0,0),4.5,3.0,fill=False,ec='w',lw=1.2));a.set_xlim(-2.5,2.5)
    else:a.add_patch(Ellipse((0,0),3.5,3.0,fill=False,ec='w',lw=1.2));a.set_xlim(-2.0,2.0)
    a.set_ylim(1.75,-1.75)
for row,(axial,sl,ext,xl,lab) in enumerate(((True,(slice(None),iy,slice(None)),[x[0]-.0125,x[-1]+.0125],'x 左右 (cm)',f'横断面 y = {y[iy]:+.4f} cm（シードの中心の層）'),(False,(ix,slice(None),slice(None)),[y[0]-.0125,y[-1]+.0125],'y 頭尾 (cm)',f'矢状断面 x = {x[ix]:+.4f} cm（中央の針の列）'))):
    E=[ext[0],ext[1],z[-1]+.0125,z[0]-.0125]
    for col,(arr,cmap,vmin,vmax,title) in enumerate(((dw[sl]/ref,cm,0,1.3,'シードあり（前立腺の平均で規格化）'),(do[sl]/ref,cm,0,1.3,'シードなし（同じ規格化）'),(rat[sl],cr,0.8,1.2,'比（あり／なし）'),(rs[sl],cs,0,0.1,'比の SEM'))):
        a=ax[row,col];im=a.imshow(arr.T,extent=E,cmap=cmap,vmin=vmin,vmax=vmax,aspect='equal',interpolation='nearest')
        outline(a,axial);a.set_xlabel(xl);a.set_ylabel('z 前→後 (cm)');a.set_title(lab+'\n'+title,fontsize=10);plt.colorbar(im,ax=a,fraction=.046)
fig.suptitle('P9c 前立腺の形の照射野・7 門・10 MV、シード 58 本のあり／なし（'+os.environ.get('P9C_LABEL','')+'、0.25 mm 格子、灰色 = シードと交わるセル）\n参考: P9b は不合格のまま記録、シードの近くの線量は EGS5 と照合していない',fontsize=12)
fig.tight_layout();fig.savefig(os.environ.get('P9C_OUT','runs/prostate_seed_7field')+'/p9c_dose_slices_prostate'+os.environ.get('P9C_TAG','')+'.png',dpi=110)
