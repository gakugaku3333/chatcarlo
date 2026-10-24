"""Venselaar-style region split of the abstract comparisons (0.5 cm bins). usage: regions.py 10|4"""
import sys, numpy as np
which = sys.argv[1]
def region(gm, gs, em, es, nb, no, depths):
    gp, ep = gm[:nb], em[:nb]; im = int(ep.argmax())
    d = (gp/gp[im])/(ep/ep[im])-1; s = np.sqrt((gs[:nb]/gp)**2+(es[:nb]/ep)**2+(gs[im]/gp[im])**2+(es[im]/ep[im])**2)
    z = (np.arange(nb)+.5)*30/nb; dm = z[im]
    deep = z > dm; bu = z < dm
    f = lambda a: f'mean {100*abs(a).mean():.2f} max {100*abs(a).max():.2f} n {len(a)}'
    print(f'PDD all: {f(d)} sem {100*s.mean():.2f}'); print(f'dmax {dm} cm | PDD z>dmax: {f(d[deep])}  sem {100*s[deep].mean():.2f} | buildup z<dmax: {f(d[bu])} (z of max {z[bu][np.argmax(abs(d[bu]))]}) | max overall at z={z[np.argmax(abs(d))]}')
    # buildup distance-to-agreement: depth shift giving the GPU value on the EGS5 curve
    allin, alls = [], []
    for j in range(len(depths)):
        sl = slice(nb+no*j, nb+no*(j+1)); g, e = gm[sl]/gm[sl][0], em[sl]/em[sl][0]
        x = (np.arange(no)+.5)*0.5
        def x50(p): i = int(np.argmax(p < .5)); return x[i-1]+(p[i-1]-.5)/(p[i-1]-p[i])*(x[i]-x[i-1])
        xe, xg = x50(e), x50(g); inner = (x <= .8*xe) & (np.arange(no) > 0)
        dd = g/e-1; allin += list(dd[inner])
        ss = np.sqrt((gs[sl]/gm[sl])**2+(es[sl]/em[sl])**2+(gs[sl][0]/gm[sl][0])**2+(es[sl][0]/em[sl][0])**2); alls += list(ss[inner])
        pen = (e >= .2) & ~inner
        print(f'  depth {depths[j]}: x50 EGS5 {xe:.3f} GPU {xg:.3f} shift {10*(xg-xe):+.2f} mm | inner(<=0.8 x50) {f(dd[inner])} | penumbra(>=20%,outside inner) {f(dd[pen])}')
    allin = np.array(allin); print('OCR inner all depths:', f(allin), f'sem mean {100*np.mean(alls):.2f}')
if which == '10':
    sys.argv[1:] = ['docs/validation/mv/stats10mv/pool.py', 'pooled', 'gpu_pooled']
    src = open(sys.argv[1]).read().split('def table')[0]; exec(src)
    pass
    k = 2
    def merged(quant):
        out = []
        for q, N in quant:
            q2 = np.concatenate([q[:, :120].reshape(100, -1, k).mean(-1), q[:, 120:].reshape(100, 3, -1, k).mean(-1).reshape(100, -1)], 1)
            out.append((st.mean_sem(q2), N))
        T = sum(N for _, N in out)
        return sum(N/T*m for (m, _), N in out), np.sqrt(sum((N/T*s)**2 for (_, s), N in out))
    eqs = [(quantities(p3.validate(dict(np.load(Path(p)/'result.npz')), 'egs', N, sd, json.load(open(Path(p)/'summary.json')))), N) for p, N, sd in RUNS]
    gqs = []
    for e in GPUS:
        c = bundle.read_json(e/'bundle/condition.json'); mt = bundle.read_json(e/'bundle/materials.json')
        gqs.append((quantities(compare.stacked(bundle.collect(e/'results', c, mt), c)), c['histories']))
    em, es = merged(eqs); gm, gs = merged(gqs)
    region(gm, gs, em, es, 60, 18, (5, 10, 20))
else:
    sys.argv[1:] = ['docs/validation/mv/stats4mv/pool.py']
    src = open(sys.argv[1]).read().split("print('EGS5'")[0]; exec(src)
    em, es, _ = merge(egs, 2); gm, gs, _ = merge(gpu, 2)
    region(gm, gs, em, es, 60, 18, ('a', 'b', 'c'))
