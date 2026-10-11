"""Fresh prototype run; outputs per-primary ledger and independent equal-sized batches."""
import argparse,json,time,hashlib,shutil
import numpy as np
from pathlib import Path
from .physics import ROOT
from .transport import run,FIELDS
from .component_checks import OUT,SEED
def execute(name,T=2.,n=100000,fe=.05,k=.05,split=0,shift=0.,scoring=True,hard_on=True,mode=0,thickness_override=0,ignore_boundaries=False,output_dir=None,backend="rutherford",seed=None,gs_table=None,elsepa_directory=None,expected_ihef=None,expected_hashes=None):
    run_seed=SEED if seed is None else seed
    if backend=='elsepa':
        if output_dir is None or seed is None or gs_table is None or elsepa_directory is None or expected_ihef is None or expected_hashes is None:
            raise ValueError('elsepa requires explicit output directory, seed, GS table, table directory, IHEF and hashes')
        from .elsepa import Elsepa
        provider=Elsepa(Path(elsepa_directory),expected_ihef=expected_ihef,expected_hashes=expected_hashes)
        provider.validate_energy(T)
    if n%100:raise ValueError('N must be divisible by 100 (equal independent batches)')
    out=OUT if output_dir is None else Path(output_dir)
    if backend=='elsepa' and any(out.resolve().is_relative_to(ROOT/'docs/validation/mv'/name) for name in ('p0','p1','p1b','p1c','p1d','p1e')):
        raise ValueError('elsepa output must not overwrite historical artifacts')
    dest=out/'runs';dest.mkdir(exist_ok=True);path=dest/(name+'.npz')
    if path.exists():raise FileExistsError(path)
    thickness=(1.2 if T==2 else 6.) if not thickness_override else thickness_override;dz=.01 if T==2 else .05
    faces=np.array([0.,thickness]) if split==0 else np.unique(np.r_[0.,np.arange(shift if shift else split,thickness,split),thickness])
    table_path=out/'data/gs.npz' if gs_table is None else Path(gs_table)
    g=np.load(table_path)
    if backend=='elsepa':
        if int(g['IHEF'])!=expected_ihef or json.loads(str(g['elsepa_table_hashes']))!=expected_hashes:
            raise ValueError('elsepa GS source/IHEF mismatch')
    kwargs={} if backend=='rutherford' else {'backend_rates':g['rates'],'backend_single':g['single']}
    if backend!='rutherford' and str(g['backend'])!=backend:raise ValueError('selected backend/table mismatch')
    args=tuple(g[x] for x in ['T','K','u','y'])
    source_dir=dest/(name+'_source');source_dir.mkdir()
    hashes={}
    for file in Path(__file__).parent.glob('*.py'):
        shutil.copy2(file,source_dir/file.name);hashes[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
    (source_dir/'hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    t=time.perf_counter();run(T,1,run_seed,fe,k,faces,dz,*args,scoring,hard_on,mode,ignore_boundaries=ignore_boundaries,**kwargs);jit=time.perf_counter()-t
    t=time.perf_counter();b,l,c,x,x2,nx,depth=run(T,n,run_seed,fe,k,faces,dz,*args,scoring,hard_on,mode,ignore_boundaries=ignore_boundaries,**kwargs);wall=time.perf_counter()-t
    q=b/(n/100*dz)
    np.savez(path,T=T,N=n,fe=fe,K=k,faces=faces,dz=dz,batches=q,ledger=l,counts=c,cross_sum=x,cross_sum2=x2,cross_n=nx,maxdepth=depth,wall_seconds=wall,jit_seconds=jit,scoring=scoring,hard_on=hard_on,mode=mode,seed=run_seed,ignore_boundaries=ignore_boundaries,backend=backend)
    meta={'elsepa_table_hashes':expected_hashes if backend=='elsepa' else None,'IHEF':expected_ihef if backend=='elsepa' else None,'table_sha256':hashlib.sha256(table_path.read_bytes()).hexdigest(),'backend':backend,'name':name,'N':n,'T':T,'fe':fe,'K':k,'mode':mode,'seconds':wall,'us_per_primary':wall/n*1e6,'counts_per_primary':(c/n).tolist(),'ledger_global_relative':float(abs(np.sum(l[:,3]))/(n*T)),'ledger_max_relative':float(max(abs(l[:,3]))/T),'unprocessed':float(sum(l[:,4])),'unprocessed_energy':float(sum(l[:,5])),'maxdepth':int(depth),'stack_allocated_bytes':4096*FIELDS*8,'table_bytes':sum(g[x].nbytes for x in ['T','K','u','y','p0','G1','total']+([] if backend=='rutherford' else ['rates','single'])),'cross_mean':(x/np.maximum(nx,1)).tolist(),'cross_n':nx.tolist()}
    (dest/(name+'.json')).write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2),flush=True)
    return path
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--energy',type=float,default=2);p.add_argument('-n',type=int,default=100000);p.add_argument('--fe',type=float,default=.05);p.add_argument('--K',type=float,default=.05);p.add_argument('--split',type=float,default=0);p.add_argument('--shift',type=float,default=0);p.add_argument('--no-scoring',action='store_true');p.add_argument('--no-hard',action='store_true');p.add_argument('--mode',type=int,default=0);p.add_argument('--thickness',type=float,default=0);p.add_argument('--no-boundaries',action='store_true')
    p.add_argument('--backend',choices=['rutherford','dcslib','eedl','mf','elsepa'],default='rutherford');p.add_argument('--output-dir',type=Path)
    p.add_argument('--seed',type=int);p.add_argument('--gs-table',type=Path);p.add_argument('--elsepa-directory',type=Path);p.add_argument('--expected-ihef',type=int,choices=[0,1]);p.add_argument('--table-hashes',type=Path)
    a=p.parse_args();execute(a.name,a.energy,a.n,a.fe,a.K,a.split,a.shift,not a.no_scoring,not a.no_hard,a.mode,a.thickness,a.no_boundaries,a.output_dir,a.backend,a.seed,a.gs_table,a.elsepa_directory,a.expected_ihef,json.loads(a.table_hashes.read_text()) if a.table_hashes else None)
