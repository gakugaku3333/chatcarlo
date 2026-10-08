"""Fresh prototype run; outputs per-primary ledger and independent equal-sized batches."""
import argparse,json,time,hashlib,shutil
import numpy as np
from pathlib import Path
from .physics import ROOT
from .transport import run,FIELDS
from .component_checks import OUT,SEED
def execute(name,T=2.,n=100000,fe=.05,k=.05,split=0,shift=0.,scoring=True,hard_on=True,mode=0,thickness_override=0,ignore_boundaries=False):
    if n%100:raise ValueError('N must be divisible by 100 (equal independent batches)')
    dest=OUT/'runs';dest.mkdir(exist_ok=True);path=dest/(name+'.npz')
    if path.exists():raise FileExistsError(path)
    thickness=(1.2 if T==2 else 6.) if not thickness_override else thickness_override;dz=.01 if T==2 else .05
    faces=np.array([0.,thickness]) if split==0 else np.unique(np.r_[0.,np.arange(shift if shift else split,thickness,split),thickness])
    g=np.load(OUT/'data/gs.npz');args=tuple(g[x] for x in ['T','K','u','y'])
    source_dir=dest/(name+'_source');source_dir.mkdir()
    hashes={}
    for file in Path(__file__).parent.glob('*.py'):
        shutil.copy2(file,source_dir/file.name);hashes[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
    (source_dir/'hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    t=time.perf_counter();run(T,1,SEED,fe,k,faces,dz,*args,scoring,hard_on,mode,ignore_boundaries=ignore_boundaries);jit=time.perf_counter()-t
    t=time.perf_counter();b,l,c,x,x2,nx,depth=run(T,n,SEED,fe,k,faces,dz,*args,scoring,hard_on,mode,ignore_boundaries=ignore_boundaries);wall=time.perf_counter()-t
    q=b/(n/100*dz)
    np.savez(path,T=T,N=n,fe=fe,K=k,faces=faces,dz=dz,batches=q,ledger=l,counts=c,cross_sum=x,cross_sum2=x2,cross_n=nx,maxdepth=depth,wall_seconds=wall,jit_seconds=jit,scoring=scoring,hard_on=hard_on,mode=mode,seed=SEED,ignore_boundaries=ignore_boundaries)
    meta={'name':name,'N':n,'T':T,'fe':fe,'K':k,'mode':mode,'seconds':wall,'us_per_primary':wall/n*1e6,'counts_per_primary':(c/n).tolist(),'ledger_global_relative':float(abs(np.sum(l[:,3]))/(n*T)),'ledger_max_relative':float(max(abs(l[:,3]))/T),'unprocessed':float(sum(l[:,4])),'unprocessed_energy':float(sum(l[:,5])),'maxdepth':int(depth),'stack_allocated_bytes':4096*FIELDS*8,'table_bytes':sum(g[x].nbytes for x in ['T','K','u','y','p0','G1','total']),'cross_mean':(x/np.maximum(nx,1)).tolist(),'cross_n':nx.tolist()}
    (dest/(name+'.json')).write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2),flush=True)
    return path
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--energy',type=float,default=2);p.add_argument('-n',type=int,default=100000);p.add_argument('--fe',type=float,default=.05);p.add_argument('--K',type=float,default=.05);p.add_argument('--split',type=float,default=0);p.add_argument('--shift',type=float,default=0);p.add_argument('--no-scoring',action='store_true');p.add_argument('--no-hard',action='store_true');p.add_argument('--mode',type=int,default=0);p.add_argument('--thickness',type=float,default=0);p.add_argument('--no-boundaries',action='store_true')
    a=p.parse_args();execute(a.name,a.energy,a.n,a.fe,a.K,a.split,a.shift,not a.no_scoring,not a.no_hard,a.mode,a.thickness,a.no_boundaries)
