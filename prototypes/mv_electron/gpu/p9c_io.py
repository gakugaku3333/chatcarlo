"""Read-only verified P9a inputs, P9c export and Colab handoff."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from . import p9c_core as core

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def export(work):
    from .. import p9a_data
    from ..run_p3a import load_spectrum
    from ..photon_transport_p8e import Material
    from .p9c import CPU
    from .p9c_checks import local,SMALL
    work=Path(work);dest=work/'export'
    if dest.exists() and any(dest.iterdir()):raise ValueError('refuse export overwrite')
    materials,identity=p9a_data.load_bundles();edges,_,cum=load_spectrum()
    gates=local(work/'local',materials,identity['densities'])
    record=dict(sources=core.fingerprint(),bundles=identity,densities=identity['densities'],edges=edges.tolist(),cum=cum.tolist(),seed=940100,local=gates)
    refs={}
    for name,angle in (('rectangle',0),('prostate',45)):
        e=CPU(work/'build',materials,densities=identity['densities'],edges=edges,cum=cum,beam=name,angle=angle,grids=(SMALL,core.Grid((-20.,)*3,(10.,)*3,(4,)*3)))
        r=e.run(2000,record['seed'],field=core.ANGLES.index(angle))
        refs.update({name+'_'+k:v for k,v in r['histories'].items()})
    dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'cpu_reference.npz',**refs)
    for j,m in enumerate(materials):np.savez(dest/f'material_{j}.npz',**dict(zip(Material._fields,m)))
    (dest/'input.json').write_text(json.dumps(record,indent=2)+'\n')
    root=core.HERE.parents[2]
    with zipfile.ZipFile(dest/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        names=set(core.fingerprint())|{p.name for p in core.HERE.glob('*.py')}
        for name in sorted(names):z.write(core.HERE/name,'prototypes/mv_electron/gpu/'+name)
        for p in core.HERE.parent.glob('*.py'):z.write(p,'prototypes/mv_electron/'+p.name)
        for name in ('metadata.json','estar_water.html','estar.npz'):
            path=root/'docs/validation/mv/p1/data'/name;z.write(path,str(path.relative_to(root)))
        z.writestr('prototypes/__init__.py','')
    (dest/'manifest.json').write_text(json.dumps({p.name:sha(p) for p in dest.iterdir() if p.is_file()},indent=2)+'\n')
    return dict(status='exported',folder=str(dest),local=gates)

def load(bundle):
    from ..photon_transport_p8e import Material
    bundle=Path(bundle);manifest=json.loads((bundle/'manifest.json').read_text())
    required={'input.json','cpu_reference.npz','source.zip',*(f'material_{j}.npz' for j in range(5))}
    if set(manifest)!=required or any(Path(k).name!=k or sha(bundle/k)!=v for k,v in manifest.items()):raise ValueError('export manifest mismatch')
    record=json.loads((bundle/'input.json').read_text())
    if record['sources']!=core.fingerprint():raise ValueError('P9c source mismatch')
    for gate in ('V0','V1a','V2a','V3','V5','V6','V6b'):
        if record['local'][gate]['status']!='pass':raise ValueError('local gate required: '+gate)
    if not record['local']['V5']['cpu_pair'].startswith('exact'):raise ValueError('CPU paired transport gate required')
    materials=[]
    for j in range(5):
        with np.load(bundle/f'material_{j}.npz',allow_pickle=False) as f:materials.append(Material(*(f[k].item() if f[k].ndim==0 else f[k].copy() for k in Material._fields)))
    return record,materials

def sections(checkpoint,grid,density=1.):
    """Requested sections with water-only mask and actual half-open cell centers.

    Returns per-photon MeV/g. Convert to Gy externally with a declared conversion.
    """
    from .p9c_stats import Checkpoint
    cp=checkpoint;mom=cp.moments['fine'];x,y,z=grid.centers();mask=core.water_mask(grid)
    result={};mass=grid.volume*density
    for value in (-.9,-.2,.5,1.2):
        index=grid.index((0.,value,0.))[1];m=type(mom)((grid.shape[0],grid.shape[2]));m.n=mom.n;m.values=mom.values[:,:,index,:];r={k:np.where(mask[:,index,:],v,np.nan) for k,v in m.result().items()}
        result['xz_y_'+str(value)]=dict(y_cm=float(y[index]),x_cm=x,z_cm=z,valid=mask[:,index,:],**r,dose_with=r['mean_with']/mass,dose_without=r['mean_without']/mass,dose_sem_with=r['sem_with']/mass,dose_sem_without=r['sem_without']/mass)
    index=grid.index((0.,0.,0.))[0];m=type(mom)((grid.shape[1],grid.shape[2]));m.n=mom.n;m.values=mom.values[:,index,:,:];r={k:np.where(mask[index,:,:],v,np.nan) for k,v in m.result().items()}
    result['yz_x_0']=dict(x_cm=float(x[index]),y_cm=y,z_cm=z,valid=mask[index,:,:],**r,dose_with=r['mean_with']/mass,dose_without=r['mean_without']/mass,dose_sem_with=r['sem_with']/mass,dose_sem_without=r['sem_without']/mass)
    return result
