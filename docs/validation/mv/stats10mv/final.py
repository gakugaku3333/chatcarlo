import sys, numpy as np
sys.argv += ['pooled', 'gpu_pooled']; src = open(sys.argv[1]).read().split('def table')[0]; exec(src)
print('GPU runs', [(r['name'], r['N'], r['seed']) for r in gruns], 'EGS5 N', Ntot)
for g in g3: print('G3', g['pair'], 'chi2 %.1f p %.3g' % (g['chi2'], g['p']))
def stats(d, s): return '差の平均%.2f 最大%.2f | 誤差平均%.2f 最大%.2f | 2σ超%d/%d' % (100*abs(d).mean(), 100*abs(d).max(), 100*s.mean(), 100*s.max(), (abs(d) > 2*s).sum(), len(d))
for k in (1, 2):
    def rb(m, s):  # merge k bins: means average; SEM combine (bins independent within same batch? use batch-level instead)
        return m
    # batch-level merging for exact SEM
    def merged(runs_, quant):
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
    em, es = merged(None, eqs); gm, gs = merged(None, gqs); nb = 120//k; no = 36//k
    gp, ep = gm[:nb], em[:nb]; im = int(ep.argmax())
    d = (gp/gp[im])/(ep/ep[im])-1; s = np.sqrt((gs[:nb]/gp)**2+(es[:nb]/ep)**2+(gs[im]/gp[im])**2+(es[im]/ep[im])**2)
    print(f'{0.25*k} cm PDD (基準 {(im+.5)*0.25*k} cm):', stats(d, s))
    D, Sg = [], []
    for j in range(3):
        sl = slice(nb+no*j, nb+no*(j+1)); g, e = gm[sl], em[sl]
        dd = (g/g[0])/(e/e[0])-1; ss = np.sqrt((gs[sl]/g)**2+(es[sl]/e)**2+(gs[sl][0]/g[0])**2+(es[sl][0]/e[0])**2); m = e/e[0] >= .2
        D += list(dd[m]); Sg += list(ss[m])
    print(f'{0.25*k} cm OCR:', stats(np.array(D), np.array(Sg)))
