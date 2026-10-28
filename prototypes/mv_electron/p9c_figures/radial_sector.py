"""P9c: dose ratio around the seeds by distance from the seed axis and by direction (2026-10-28).
7-field total only: prostate.npz stores no per-field tally.
Run from the repository root: PYTHONPATH=. .venv/bin/python prototypes/mv_electron/p9c_figures/radial_sector.py
Part 1: the distance bins of make_radial.py, and where the voxel centres in each bin lie relative to the seed surface.
Part 2: 8 directions around the seed axis (0 = anterior side of the seed, 180 = posterior side; +z is anterior to posterior)."""
import sys,numpy as np
import os
# all six production runs (600 batches); combined.load() reads only the first two unless P9C_RUNS is set
os.environ.setdefault('P9C_RUNS','prod_prostate,prod_prostate2,prod_prostate3,prod_prostate4,prod_prostate5,prod_prostate6')
sys.path.insert(0,'prototypes/mv_electron/p9c_figures')
from prototypes.mv_electron.gpu import p9c_core as core
from combined import load
G=core.FINE;x,y,z=G.centers()
m=load('fine',(240,200,300));r=m.result();mask=core.water_mask(G);S=np.array(core.seeds())
X,Y,Z=np.meshgrid(x,y,z,indexing='ij')
pros=((X/2.25)**2+(Y/1.75)**2+(Z/1.5)**2<=1)&mask
d=np.full(X.shape,np.inf);dx=np.zeros(X.shape);dz=np.zeros(X.shape)
for sx,sy,sz,rad,h,_ in S:
    inlen=np.abs(Y-sy)<=h-0.025
    dd=np.where(inlen,np.hypot(X-sx,Z-sz),np.inf);sel=dd<d;d[sel]=dd[sel];dx[sel]=(X-sx)[sel];dz[sel]=(Z-sz)[sel]
ok=pros&np.isfinite(d)
w=r['mean_with'];o=r['mean_without'];rs=r['ratio_sem']
# z: anterior -> posterior (+z = posterior). phi: 0 = anterior side of seed, 180 = posterior side, 90 = +x side
phi=np.degrees(np.arctan2(dx,-dz))%360

print("== part 1: distance bins (as make_radial.py) ==")
for lo,hi in zip(np.arange(0.05,0.36,0.025)[:-1],np.arange(0.05,0.36,0.025)[1:]):
    sel=ok&(d>=lo)&(d<hi);dc=d[sel]
    print(f"{lo*10:.2f}-{hi*10:.2f} mm from axis | cells {sel.sum():6d} | {100*(w[sel].sum()/o[sel].sum()-1):+.2f}% | voxel centre from surface mean {10*(dc.mean()-0.05):.3f} (min {10*(dc.min()-0.05):.3f}, max {10*(dc.max()-0.05):.3f}) mm")
print("== part 2: by direction ==")
rb=[(0.050,0.075),(0.075,0.100),(0.100,0.125),(0.125,0.150),(0.175,0.200),(0.250,0.275)]
print("sector centre: 0=anterior side, 180=posterior side (no 180 beam), 90=+x side")
hdr="dist-from-axis | "+" | ".join(f"{c:>3d}°" for c in range(0,360,45))
print(hdr)
for lo,hi in rb:
    row=[]
    for c in range(0,360,45):
        a=(phi-c+180)%360-180
        sel=ok&(d>=lo)&(d<hi)&(np.abs(a)<22.5);n=sel.sum()
        ra=w[sel].sum()/o[sel].sum();se=np.sqrt(np.sum(rs[sel]**2))/n
        row.append(f"{100*(ra-1):+5.2f}±{100*se:.2f}({n})")
    print(f"{lo*10:.2f}-{hi*10:.2f} mm | "+" | ".join(row))
