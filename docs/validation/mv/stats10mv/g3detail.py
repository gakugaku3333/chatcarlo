import sys, numpy as np
sys.argv += ['pooled', 'gpu_pooled']; exec(open(sys.argv[1]).read().split('def table')[0])
a, b = gruns
z = (b['m']-a['m'])/np.sqrt(a['s']**2+b['s']**2)
lab = [f'PDD {(i+.5)*.25:.3f}cm' for i in range(120)] + [f'OCR d{d} x{(i+.5)*.25:.3f}cm' for d in (5, 10, 20) for i in range(36)]
o = np.argsort(-z**2)
print('top10 z:', [(lab[i], round(float(z[i]), 2)) for i in o[:10]])
for nm, sl in (('PDD', slice(0, 120)), ('OCR5', slice(120, 156)), ('OCR10', slice(156, 192)), ('OCR20', slice(192, 228))):
    zz = z[sl]; print(nm, 'chi2/dof %.2f' % (np.mean(zz**2)), 'mean z %.2f' % zz.mean(), '|z|>3:', int((abs(zz) > 3).sum()))
r = b['m']/a['m']-1; print('PDD mean ratio (54M/46M -1) %.3f%%' % (100*r[:120].mean()))
