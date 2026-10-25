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
fig=plt.figure(figsize=(20,12));gs=fig.add_gridspec(2,3)
E=[x[0]-.0125,x[-1]+.0125,z[-1]+.0125,z[0]-.0125]
c1=plt.get_cmap('jet').copy();c1.set_bad('0.6');c2=plt.get_cmap('RdBu_r').copy();c2.set_bad('0.6')
for col,(arr,cmap,vmin,vmax,title) in enumerate(((w/ref,c1,0.8,1.15,'シードあり（前立腺の平均で規格化、0.80〜1.15）'),(o_all/ref,c1,0.8,1.15,'シードなし（同じ規格化）'),(rat,c2,0.95,1.05,'比（あり／なし）、±5%'))):
    a=fig.add_subplot(gs[0,col]);im=a.imshow(arr.T,extent=E,cmap=cmap,vmin=vmin,vmax=vmax,aspect='equal',interpolation='nearest')
    a.add_patch(Ellipse((0,0),4.5,3.0,fill=False,ec='k',lw=1));a.set_xlim(-2.5,2.5);a.set_ylim(1.75,-1.75)
    for sx,sy,sz,*_ in S:
        if col!=1 and abs(sy+0.2)<1e-9:a.add_patch(Ellipse((sx,sz),.1,.1,fill=False,ec='k',lw=.6))
    a.axhline(0.0125,color='m',lw=.8,ls=':');[a.axvline(v,color='m',lw=.8,ls=':') for v in (0.0125,-0.9875)]
    a.set_title('横断面: シードの長さ y −0.45〜+0.05 cm で平均（20 セル）\n'+title,fontsize=10);a.set_xlabel('x 左右 (cm)');a.set_ylabel('z 前→後 (cm)');plt.colorbar(im,ax=a,fraction=.046)
# z profiles through seeds (recorded lines, exact paired SEM)
a=fig.add_subplot(gs[1,0]);zz=np.array(s['line_z_cm'])
for j,(cc,col) in enumerate(zip(s['line_cell_centers'],('C0','C1','C2'))):
    rr=lr['ratio'][j];ee=lr['ratio_sem'][j];v=np.array(s['line_valid'][j],bool);rr=np.where(v,rr,np.nan)
    a.plot(zz,rr,color=col,lw=.8,label=f'x={cc[0]:+.2f}, y={cc[1]:+.2f} cm（シード {s["line_seed_ids"][j]}）');a.fill_between(zz,rr-ee,rr+ee,color=col,alpha=.2)
a.axhline(1,color='k',lw=.6);a.set_xlim(-1.75,1.75);a.set_ylim(0.8,1.2);a.set_xlabel('z 前→後 (cm)');a.set_ylabel('比（あり／なし）')
a.set_title('シードの中心を通る z 方向の線（0.25 mm のセル 1 列、帯 = ±1 SEM）\n切れ目 = シードの中（水だけのセルではない）',fontsize=10);a.legend(fontsize=8)
# z profile slab-averaged through x=0 needle
a=fig.add_subplot(gs[1,1]);ix=int(round(3/.025));
for xi,col,lab in ((int(round(3/.025)),'C0','x = +0.01 cm（中央の針、シード z = −1, 0, +1 付近）'),(int(round(2/.025)),'C2','x = −0.99 cm（シード z = −1, 0, +1 付近）')):
    a.plot(z,rat[xi],color=col,lw=.9,label=lab);a.fill_between(z,rat[xi]-bound[xi],rat[xi]+bound[xi],color=col,alpha=.15)
a.axhline(1,color='k',lw=.6);a.set_xlim(-1.75,1.75);a.set_ylim(0.9,1.1);a.set_xlabel('z 前→後 (cm)');a.set_ylabel('比（あり／なし）')
a.set_title('z 方向の線（シードの長さで平均）\n帯 = SEM の上限（セルの相関を 1 と仮定）',fontsize=10);a.legend(fontsize=8)
# x profile across seed row z=0
a=fig.add_subplot(gs[1,2]);iz=int(round(2.5/.025))
for arr,col,lab in ((w[:,iz]/ref,'C3','シードあり'),(o_all[:,iz]/ref,'k','シードなし')):a.plot(x,arr,color=col,lw=.9,label=lab)
a2=a.twinx();a2.plot(x,rat[:,iz],color='C0',lw=.8,label='比');a2.fill_between(x,rat[:,iz]-bound[:,iz],rat[:,iz]+bound[:,iz],color='C0',alpha=.15);a2.set_ylim(0.9,1.1);a2.set_ylabel('比（あり／なし）',color='C0');a2.axhline(1,color='C0',lw=.5,ls='--')
for v in (-2,-1,0,1,2):a.axvline(v,color='0.7',lw=.6)
a.set_xlim(-2.5,2.5);a.set_ylim(0,1.2);a.set_xlabel('x 左右 (cm)');a.set_ylabel('線量（前立腺の平均で規格化）')
a.set_title('シードの列（z = +0.01 cm）を横切る x 方向の線（シードの長さで平均）\n灰色の縦線 = シードの位置 x = −2, −1, 0, 1, 2',fontsize=10);a.legend(loc='lower left',fontsize=8);a2.legend(loc='lower right',fontsize=8)
fig.suptitle('P9c 前立腺の形の照射野・7 門・10 MV、シード 58 本のあり／なし（'+os.environ.get('P9C_LABEL','')+'）\n参考: P9b は不合格のまま記録、シードの近くの線量は EGS5 と照合していない',fontsize=12)
fig.tight_layout();fig.savefig(os.environ.get('P9C_OUT','runs/prostate_seed_7field')+'/p9c_seed_profiles'+os.environ.get('P9C_TAG','')+'.png',dpi=110)
iz0=iz;print('row ratio near seeds:',[(v,round(float(np.nanmean(rat[(abs(x-v)<0.15)&(abs(x-v)>0.05),iz0])),4)) for v in (-2,-1,0,1,2)])
print('bound typical',float(np.nanmedian(bound[abs(x)<2,iz0])),'low',float(np.nanmedian(low[abs(x)<2,iz0])))
