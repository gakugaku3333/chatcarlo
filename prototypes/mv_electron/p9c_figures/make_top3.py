import os
import numpy as np,os,json,matplotlib
matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Rectangle
from prototypes.mv_electron.gpu import p9c_core as core
from prototypes.mv_electron.gpu.p9c_stats import Moments
plt.rcParams['font.family']='Hiragino Sans'
R=os.path.expanduser('~/.cache/chatcarlo/p9c2/prod_prostate/')
G=core.FINE;x,y,z=G.centers();V=1.5625e-5
import sys;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)));from combined import load
m=load('fine',(240,200,300))
L=load('lines',(3,300))
r=m.result();mask=core.water_mask(G);lr=L.result();s=json.load(open(R+'prostate_summary.json'))
pros=((x[:,None,None]/2.25)**2+(y[None,:,None]/1.75)**2+(z[None,None,:]/1.5)**2<=1)&mask
ref=np.nanmean(np.where(mask,r['mean_without'],np.nan)[pros])
# slab over the seed length of the y=-0.2 layer: y in [-0.45,0.05)
ys=(y>=-0.45)&(y<0.05)
def slab(k,axis_sel):
    a=np.where(mask,r[k],np.nan);return np.nanmean(a[:,ys,:],axis=1)
w=slab('mean_with',0);o=slab('mean_without',0);rat=w/o
o_all=r['mean_without'][:,ys,:].mean(axis=1)  # display only: no seeds in this geometry
# conservative (fully correlated) SEM bound of the slab-averaged ratio: mean of per-cell ratio SEM
bound=np.nanmean(np.where(mask,r['ratio_sem'],np.nan)[:,ys,:],axis=1)
low=np.nanmean(np.where(mask,r['ratio_sem'],np.nan)[:,ys,:]**2,axis=1)**.5/np.sqrt(ys.sum())
S=np.array(core.seeds())
fig=plt.figure(figsize=(20,6.2));gs=fig.add_gridspec(1,3)
E=[x[0]-.0125,x[-1]+.0125,z[-1]+.0125,z[0]-.0125]
c1=plt.get_cmap('jet').copy();c1.set_bad('0.6');c2=plt.get_cmap('RdBu_r').copy();c2.set_bad('0.6')
for col,(arr,cmap,vmin,vmax,title) in enumerate(((w/ref,c1,0.8,1.15,'シードあり（前立腺の平均で規格化、0.80〜1.15）'),(o_all/ref,c1,0.8,1.15,'シードなし（同じ規格化）'),(rat,c2,0.95,1.05,'比（あり／なし）、±5%'))):
    a=fig.add_subplot(gs[0,col]);im=a.imshow(arr.T,extent=E,cmap=cmap,vmin=vmin,vmax=vmax,aspect='equal',interpolation='nearest')
    a.add_patch(Ellipse((0,0),4.5,3.0,fill=False,ec='k',lw=1));a.set_xlim(-2.5,2.5);a.set_ylim(1.75,-1.75)
    for sx,sy,sz,*_ in S:
        if col!=1 and abs(sy+0.2)<1e-9:a.add_patch(Ellipse((sx,sz),.1,.1,fill=False,ec='k',lw=.6))
    a.set_title('横断面: シードの長さ y −0.45〜+0.05 cm で平均（20 セル）\n'+title,fontsize=10);a.set_xlabel('x 左右 (cm)');a.set_ylabel('z 前→後 (cm)');plt.colorbar(im,ax=a,fraction=.046)
fig.suptitle('P9c 前立腺の形の照射野・7 門・10 MV、シード 58 本のあり／なし（'+os.environ.get('P9C_LABEL','')+'）\n参考: P9b は不合格のまま記録、シードの近くの線量は EGS5 と照合していない',fontsize=12)
fig.tight_layout();fig.savefig(os.environ.get('P9C_OUT','runs/prostate_seed_7field')+'/p9c_seed_slab_top3'+os.environ.get('P9C_TAG','')+'.png',dpi=110)
