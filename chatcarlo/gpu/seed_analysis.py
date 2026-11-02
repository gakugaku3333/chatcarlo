"""Common-mask energy density and paired batch statistics (MeV/cm³/photon)."""
import json
from pathlib import Path
import numpy as np
from . import seed_geometry as core
from .therapy_bundle import sha
from .seed_stats import Moments

MASK_PATH=Path(__file__).resolve().parents[2]/'docs/validation/gpu/g5/a2b_masks.npz'

def coarsen(values, *, mean=False):
    a=np.asarray(values)
    if a.shape!=core.OLD_FINE.shape:raise ValueError('old fine grid shape')
    b=a.reshape(120,2,100,2,150,2)
    return b.mean((1,3,5)) if mean else b.sum((1,3,5))

def make_masks():
    """Freeze ALL eight OLD water/organ cells, never reapply the new water mask."""
    g=core.OLD_FINE;x,y,z=g.centers();xx=x[:,None,None];yy=y[None,:,None];zz=z[None,None,:]
    water=core.water_mask(g)
    prostate=((xx/2.25)**2+(yy/1.75)**2+(zz/1.5)**2<=1)&water
    rectum=(xx**2+(zz-3.)**2<=1.5**2)&(abs(yy)<=1.75)&water
    masks={k:coarsen(v.astype(np.uint8))==8 for k,v in dict(water=water,prostate=prostate,rectum=rectum).items()}
    x,y,z=core.FINE.centers();near=np.zeros(core.FINE.shape,bool)
    for a,b,c,r,h,m in core.seeds():near|=((x[:,None,None]-a)**2+(z[None,None,:]-c)**2<=.2**2)&(abs(y[None,:,None]-b)<=h)
    masks['near']=near&masks['water']
    return masks

def load_masks(path=MASK_PATH):
    path=Path(path);record=json.loads(path.with_suffix('.json').read_text())
    if sha(path)!=record['sha256']:raise ValueError('mask SHA256 mismatch')
    with np.load(path,allow_pickle=False) as f:m={k:f[k] for k in f.files}
    if set(m)!={'water','prostate','rectum','near'} or any(v.shape!=core.FINE.shape or v.dtype!=bool for v in m.values()):raise ValueError('mask schema')
    if int(m['near'].sum())!=25400:raise ValueError('near mask count')
    if {k:int(v.sum()) for k,v in m.items()}!=record['counts']:raise ValueError('mask count seal')
    return m

def densities(values,photons,masks):
    a=np.asarray(values)
    if a.shape!=core.FINE.shape or not np.isfinite(a).all() or np.any(a<0) or photons<=0:raise ValueError('energy/count')
    return np.array([a[masks[k]].sum()/photons/(masks[k].sum()*core.FINE.volume) for k in ('prostate','rectum','near')])

def old_densities(mean_cell_energy,masks):
    # P9c stores energy per accepted photon per OLD cell; coarsening SUM conserves energy.
    return densities(coarsen(mean_cell_energy),1,masks)

def analyse(rows):
    a=np.asarray(rows,float)
    if a.ndim!=3 or a.shape[1:]!=(2,3) or len(a)<2 or not np.isfinite(a).all() or np.any(a<=0):raise ValueError('paired density rows')
    m=Moments((3,))
    for x,y in a:m.add(x,y)
    return {k:np.where(np.isfinite(v),v,None).tolist() for k,v in m.result().items()}
