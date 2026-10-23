"""NumPy-only source copy for VM execution (source.py:28-43,67-212).

Only the approved rect/cone/parallel branches without heel or rotation.
The host supplies the exact spectrum table, including its normalization.
"""
import math
import numpy as np

def beam_basis(d):
    if abs(d[2])<0.999:u=np.array([-d[1],d[0],0.0])
    else:u=np.array([1.0,0.0,0.0])
    u=u/np.linalg.norm(u)
    return u,np.cross(d,u)

def uniform(rng,low,high,n):
    """low + (high-low)*U: written out so the last bit does not depend on the NumPy version.

    Generator.uniform rounds differently in NumPy 2.4 (Mac) and the NumPy on Colab; the host replay and
    the VM then disagree on the origins by one ulp and the byte-exact source check stops the run (2026-10-23)."""
    return low+(high-low)*rng.random(n)

def sample_source_photons(src,n,rng,spectrum=None):
    for key in ('mas','ctdi_vol_mGy','rotation'):
        if src.get(key) is not None:raise ValueError('G1 source does not support '+key)
    if src.get('heel_effect'):raise ValueError('G1 source does not support heel_effect')
    pos=np.asarray(src['position'],dtype=float);d=np.asarray(src['direction'],dtype=float)
    fld=src['field'];sid=fld.get('sid_cm');u,v=beam_basis(d)
    origins=np.tile(pos,(n,1));shape=fld.get('shape','rect')
    if shape=='cone':
        cos_half=np.cos(math.atan((fld['diameter_cm']/2.0)/fld['sid_cm']))
        cos_t=uniform(rng,cos_half,1.0,n);phi=uniform(rng,0.0,2.0*np.pi,n)
        sin_t=np.sqrt(1.0-cos_t**2)
        dirs=(cos_t[:,None]*d+(sin_t*np.cos(phi))[:,None]*u+(sin_t*np.sin(phi))[:,None]*v)
    elif shape=='parallel':
        w,h=fld['size_cm'];su=uniform(rng,-w/2,w/2,n);sv=uniform(rng,-h/2,h/2,n)
        origins=origins+su[:,None]*u+sv[:,None]*v
        dirs=np.broadcast_to(d,(n,3)).copy()
    elif shape=='rect':
        w,h=fld['size_cm'];su=uniform(rng,-w/2,w/2,n);sv=uniform(rng,-h/2,h/2,n)
        target=pos+d*sid+su[:,None]*u+sv[:,None]*v
        dirs=target-origins;dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
    else:raise ValueError('G1 field must be rect, cone or parallel')
    if spectrum is None:
        spec=src['spectrum'];e=np.array([s['energy_keV'] for s in spec],dtype=float);w=np.array([s['weight'] for s in spec],dtype=float);w=w/w.sum()
    else:e,w=spectrum
    energies=e[rng.choice(len(e),size=n,p=w)]
    return origins,dirs,energies
