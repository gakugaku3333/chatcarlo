"""Scientific plots from saved raw curves; no renormalization."""
import os
from pathlib import Path
CACHE=Path(__file__).resolve().parents[2]/"docs/validation/mv/p1/.cache"
os.environ["MPLCONFIGDIR"]=str(CACHE/"matplotlib")
os.environ["XDG_CACHE_HOME"]=str(CACHE)
CACHE.mkdir(exist_ok=True)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .component_checks import OUT
from .check_egs5 import parse
from .analysis import dref
def main():
    for T in [2,10]:
        ref=parse(f'base_{T}');pro=np.load(OUT/f'runs/base_{T}.npz')
        q=pro['batches'].mean(0);sem=pro['batches'].std(0,ddof=1)/10;z=ref['z']*10;D=dref(ref['q'])
        fig,(ax,rx)=plt.subplots(2,1,figsize=(8,6),sharex=True,gridspec_kw={'height_ratios':[3,1]},layout='constrained')
        ax.plot(z,ref['q'],label='EGS5 GS, radiation local')
        ax.plot(z,q,label='Rutherford GS / dual random hinge')
        ax.fill_between(z,q-sem,q+sem,alpha=.25)
        ax.set_ylabel('q [MeV cm²/g per primary]');ax.set_title(f'{T} MeV electron, water slab');ax.legend();ax.grid(alpha=.2)
        rx.plot(z,100*(q-ref['q'])/D,color='tab:red')
        rx.axhline(0,color='black',lw=.7)
        for v in [-3,3]:rx.axhline(v,color='gray',ls='--',lw=.7)
        rx.set_ylabel('Difference / Dref [%]');rx.set_xlabel('Depth [mm]');rx.grid(alpha=.2)
        fig.savefig(OUT/f'depth_{T}.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
