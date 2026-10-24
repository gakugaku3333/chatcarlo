"""60 keV water (GPU, runs/diag_water_60kev_gpu/dose.npz) vs EGS5 (revalidation_2026_10_07/comparison.csv):
reproduce the abstract's v1 numbers, then the in-field OCR (bin centres <= 0.8 x50, centre bin excluded)."""
import csv, numpy as np
rows = {r['bin']: r for r in csv.DictReader(open('docs/egs5_crosscheck/revalidation_2026_10_07/comparison.csv'))}
d = np.load('runs/diag_water_60kev_gpu/dose.npz'); D = d['dose_per_history_Gy']
ix = lambda x: int(round(x + 65)); iz = lambda z: int(round(z + 50))   # origin -65,-65,-50, 1 cm voxels
yy = slice(ix(-1), ix(1))
gp = np.array([D[yy, yy, iz(z)].mean() for z in range(15)])
ep = np.array([float(rows[f'pdd_z{z}-{z+1}']['egs5_Gy_history']) for z in range(15)])
im = int(ep.argmax()); dp = (gp/gp[im])/(ep/ep[im]) - 1
print(f'PDD n={len(dp)} mean {100*abs(dp).mean():.2f} max {100*abs(dp).max():.2f}')
allv, alli = [], []
for name, z in (('shallow', 0), ('10cm', 9)):
    g = np.array([(D[ix(x), yy, iz(z)].mean() + D[ix(-x-1), yy, iz(z)].mean())/2 for x in range(8)])
    e = np.array([(float(rows[f'lat_{name}_x{x}-{x+1}']['egs5_Gy_history']) + float(rows[f'lat_{name}_x{-x-1}-{-x}'.replace('x-0', 'x0')]['egs5_Gy_history']))/2 for x in range(8)])
    g, e = g/g[0], e/e[0]; dd = g/e - 1; xc = np.arange(8) + .5
    i = int(np.argmax(e < .5)); x50 = xc[i-1] + (e[i-1]-.5)/(e[i-1]-e[i])*(xc[i]-xc[i-1])
    v1 = e >= .2; inner = (xc <= .8*x50) & (xc > .5)
    allv += list(dd[v1]); alli += list(dd[inner])
    print(f'{name}: x50 {x50:.2f} | v1(>=20%) n={v1.sum()} | inner n={inner.sum()} {np.round(100*dd[inner],2)}')
allv, alli = np.array(allv), np.array(alli)
print(f'OCR v1 n={len(allv)} mean {100*abs(allv).mean():.2f} max {100*abs(allv).max():.2f}')
print(f'OCR in-field n={len(alli)} mean {100*abs(alli).mean():.2f} max {100*abs(alli).max():.2f}')
