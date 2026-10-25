import os
import numpy as np,os,matplotlib
matplotlib.use('Agg');import matplotlib.pyplot as plt
from prototypes.mv_electron.gpu import p9c_core as core
from prototypes.mv_electron.gpu.p9c_stats import Moments
plt.rcParams['font.family']='Hiragino Sans'
R=os.path.expanduser('~/.cache/chatcarlo/p9c2/prod_prostate/')
G=core.FINE;x,y,z=G.centers()
import sys;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)));from combined import load
m=load('fine',(240,200,300))
r=m.result();mask=core.water_mask(G);S=np.array(core.seeds())
X,Y,Z=np.meshgrid(x,y,z,indexing='ij')
pros=((X/2.25)**2+(Y/1.75)**2+(Z/1.5)**2<=1)&mask
# distance to nearest seed axis segment (y-aligned), within the seed's length; and angle relative to the 0° beam
d=np.full(X.shape,np.inf);zz=np.zeros(X.shape)
for sx,sy,sz,rad,h,_ in S:
    inlen=np.abs(Y-sy)<=h-0.025
    dd=np.where(inlen,np.hypot(X-sx,Z-sz),np.inf);sel=dd<d;d[sel]=dd[sel];zz[sel]=(Z-sz)[sel]
ok=pros&np.isfinite(d)
edges=np.arange(0.05,0.36,0.025);cent=(edges[:-1]+edges[1:])/2
rat=r['ratio'];w=r['mean_with'];o=r['mean_without']
out=[]
for lo,hi in zip(edges[:-1],edges[1:]):
    sel=ok&(d>=lo)&(d<hi);n=sel.sum()
    ratio=w[sel].sum()/o[sel].sum()
    sem_ind=np.sqrt(np.sum(r['ratio_sem'][sel]**2))/n   # cells treated independent
    out.append((ratio,sem_ind,n))
out=np.array(out)
far=ok&(d>=0.30);far_ratio=w[far].sum()/o[far].sum()
fig,ax=plt.subplots(1,1,figsize=(8,5.5))
ax.errorbar(cent*10,(out[:,0]-1)*100,yerr=out[:,1]*100,fmt='o-',capsize=3,label='58 本のシードのまわりの平均（シードの長さの範囲）')
ax.axhline(0,color='k',lw=.6);ax.axvline(0.5,color='0.6',ls='--',lw=.8);ax.text(0.52,ax.get_ylim()[1]*0.9 if False else 0,'シードの表面 (0.5 mm)',fontsize=8,color='0.4',rotation=90,va='bottom')
ax.set_xlabel('シードの軸からの距離 (mm)');ax.set_ylabel('線量の変化（あり／なし − 1、%）')
ax.set_title('前立腺の形の照射野・7 門・10 MV：シードのまわりの線量の変化\n誤差棒 = ±1 SEM（セルを独立とみなした値）',fontsize=11);ax.legend(fontsize=9)
fig.tight_layout();fig.savefig(os.environ.get('P9C_OUT','runs/prostate_seed_7field')+'/p9c_seed_radial'+os.environ.get('P9C_TAG','')+'.png',dpi=120)
for c,(ra,se,n) in zip(cent,out):print(f'{c*10:.2f} mm  {100*(ra-1):+.2f} ± {100*se:.2f} %  cells {n}')
print('far (>=3 mm, in prostate)',100*(far_ratio-1))
