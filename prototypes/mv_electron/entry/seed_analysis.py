"""Single-checkpoint analysis; radial/slab formulas from p9c_figures, no cache paths."""
from pathlib import Path
import warnings
import numpy as np
from ..gpu import p9c_core as core
from ..gpu.p9c_stats import Moments
from ..gpu.p5b_core import write_json

NOTES = [
    'Distance SEM treats cells as independent and may be underestimated.',
    'At low photon counts a 3% increase cannot be resolved.',
    'No near-seed EGS5 cross-check; P9b remains failed.',
]


def fields(values, batches):
    # Bound transient memory to one x slice while keeping the original Moments formula.
    shape = values.shape[1:]
    result = {k:np.empty(shape) for k in ('mean_with','mean_without','ratio_sem')}
    for i in range(shape[0]):
        m = Moments(shape[1:]); m.n = batches; m.values = values[:,i]
        r = m.result()
        for k in result:
            result[k][i] = r[k]
    return result


def statistic(w, o, sem, selected):
    count = int(selected.sum())
    reason = 'no valid cells' if not count else 'zero denominator' if o[selected].sum() == 0 else None
    ratio = None if reason else float(w[selected].sum()/o[selected].sum())
    error = None if reason else float(np.sqrt(np.sum(sem[selected]**2))/count)
    if error is not None and not np.isfinite(error):
        error = None
    return dict(ratio=ratio, sem=error, cells=count, reason=reason,
                sem_reason=reason or ('undefined cell SEM' if error is None else None))


def tables(r, grid):
    x,y,z = grid.centers(); X=x[:,None,None]; Y=y[None,:,None]; Z=z[None,None,:]
    mask = core.water_mask(grid)
    pros = ((X/2.25)**2+(Y/1.75)**2+(Z/1.5)**2<=1)&mask
    rectum = (X**2+(Z-3.)**2<=1.5**2)&(abs(Y)<=1.75)&mask
    w,o,se = (r[k] for k in ('mean_with','mean_without','ratio_sem'))
    d = np.full(grid.shape,np.inf)
    for sx,sy,sz,rad,h,_ in core.seeds():
        dd = np.where(abs(Y-sy)<=h-.025,np.hypot(X-sx,Z-sz),np.inf)
        np.minimum(d,dd,out=d)
    ok = pros&np.isfinite(d)
    radial = []
    edges = np.arange(.05,.36,.025)
    for lo,hi in zip(edges[:-1],edges[1:]):
        radial.append(dict(interval_cm=[float(lo),float(hi)], interval='[lo, hi)',
                           center_mm=float((lo+hi)*5),
                           **statistic(w,o,se,ok&(d>=lo)&(d<hi))))
    return dict(prostate=statistic(w,o,se,pros),rectum=statistic(w,o,se,rectum),radial=radial,
                far=dict(interval_cm=[.3,None],interval='[lo, infinity)',**statistic(w,o,se,ok&(d>=.3)))), mask, pros


def slab_arrays(r, mask, prostate, grid):
    x,y,z = grid.centers(); ys = (y>=-.45)&(y<.05)
    def mean(a, axis=None):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            return np.nanmean(a,axis=axis)
    ref = mean(np.where(mask,r['mean_without'],np.nan)[prostate])
    w = mean(np.where(mask,r['mean_with'],np.nan)[:,ys,:],1)
    o = mean(np.where(mask,r['mean_without'],np.nan)[:,ys,:],1)
    o_all = mean(r['mean_without'][:,ys,:],1)
    with np.errstate(divide='ignore',invalid='ignore'):
        arrays = [w/ref,o_all/ref,w/o]
    return [np.where(np.isfinite(a),a,np.nan) for a in arrays]


def analyse(checkpoint, output, grid=None):
    grid = core.FINE if grid is None else grid
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    with np.load(checkpoint,allow_pickle=False) as f:
        r = fields(f['fine'],len(f['completed']))
        aggregate = Moments((2,)); aggregate.values=f['aggregate']; aggregate.n=len(f['completed'])
    report,mask,pros = tables(r,grid)
    # Region mean ratios and paired SEM come from complete seven-field batch means.
    ar = aggregate.result()
    for j,key in enumerate(('prostate','rectum')):
        if report[key]['reason'] is None:
            for dest,src in (('ratio','ratio'),('sem','ratio_sem')):
                v = float(ar[src][j]); report[key][dest] = v if np.isfinite(v) else None
            report[key]['sem_reason'] = None if report[key]['sem'] is not None else 'undefined paired SEM'
        report[key]['sem_method'] = 'paired batch region means'
    report['notes'] = NOTES
    report['checkpoint'] = str(Path(checkpoint).resolve())
    write_json(output/'seed_analysis.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    x,y,z = grid.centers(); arrays = slab_arrays(r,mask,pros,grid)
    fig,axes = plt.subplots(1,3,figsize=(16,6))
    extent = [x[0]-grid.step[0]/2,x[-1]+grid.step[0]/2,z[-1]+grid.step[2]/2,z[0]-grid.step[2]/2]
    for j,(ax,a,title) in enumerate(zip(axes,arrays,('With seeds / reference','Without seeds / reference','With / without'))):
        cm = plt.get_cmap('RdBu_r' if j==2 else 'jet').with_extremes(bad='0.6')
        im=ax.imshow(a.T,extent=extent,cmap=cm,vmin=.95 if j==2 else .8,vmax=1.05 if j==2 else 1.15,
                     aspect='equal',interpolation='nearest')
        ax.add_patch(Ellipse((0,0),4.5,3.,fill=False,ec='k',lw=1))
        if j!=1:
            for sx,sy,sz,*_ in core.seeds():
                if abs(sy+.2)<1e-9:ax.add_patch(Ellipse((sx,sz),.1,.1,fill=False,ec='k',lw=.6))
        ax.set(xlim=(-2.5,2.5),ylim=(1.75,-1.75),xlabel='x (cm)',ylabel='z anterior -> posterior (cm)',title=title)
        fig.colorbar(im,ax=ax,fraction=.046)
    fig.suptitle('Mohan 10 MV, 7 prostate fields, 58 seeds; y in [-0.45, 0.05) cm\n'+'\n'.join(NOTES),fontsize=10)
    fig.tight_layout(rect=(0,0,1,.82)); fig.savefig(output/'seed_transverse.png',dpi=110); plt.close(fig)
    return report
