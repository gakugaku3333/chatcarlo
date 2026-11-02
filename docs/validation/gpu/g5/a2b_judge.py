"""Freeze six explicit old files/600 batches BEFORE production, then judge B3.

No default combined.load(): the two-run default is deliberately never used.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from chatcarlo.gpu import seed_analysis as analysis,seed_geometry as core,contracts
from chatcarlo.gpu.seed_stats import Moments
from chatcarlo.gpu.therapy_bundle import sha

DELTAS=dict(prostate_ratio=.001,rectum_ratio=.001,near_ratio=.003,prostate_with=.005,prostate_without=.005,rectum_with=.007,rectum_without=.007)


def freeze(paths,out):
    out=Path(out)
    if out.exists():raise ValueError('reference already frozen')
    if len(paths)!=6 or len({Path(p).resolve() for p in paths})!=6:raise ValueError('six explicit independent runs required')
    masks=analysis.load_masks();org=Moments((2,));rows=[];seeds=[]
    energy=np.zeros((2,3))
    for p in map(Path,paths):
        digest=sha(p)
        with np.load(p,allow_pickle=False) as f:
            batches=f['completed'].tolist();ident=json.loads(str(f['identity']))
            if batches!=list(range(100)):raise ValueError('100 explicit old batches per run')
            if ident.get('n')!=7142900 or ident.get('angles')!=list(core.ANGLES) or ident.get('beam')!='prostate' or ident.get('grids')!=[core.OLD_FINE.record(),core.COARSE.record()]:raise ValueError('old fixed P9c configuration')
            if rows and any(ident[k]!=rows[0]['identity'][k] for k in ident if k!='seed'):raise ValueError('old run identity mismatch')
            if ident['seed'] in seeds or ident['seed'] in (9502001,9502002):raise ValueError('old/new seed collision')
            seeds.append(ident['seed']);org.values+=f['aggregate'];org.n+=100
            fine=f['fine']
            if fine.shape!=(5,*core.OLD_FINE.shape):raise ValueError('old fine grid')
            for side in range(2):energy[side]+=analysis.old_densities(fine[side]/100,masks)/6
            # Near SEM: independent OLD cells, retaining within-cell with/without covariance.
            mask=np.repeat(np.repeat(np.repeat(masks['near'],2,0),2,1),2,2)
            # Sum of per-cell covariance numerators, not covariance of the cell sum.
            # Each run is independent; pooled second moments are needed below.
        rows.append(dict(path=str(p.resolve()),sha256=digest,batches=batches,seed=ident['seed'],identity=ident))
    # Compute independent-cell variance from pooled per-cell moments in a second
    # pass. This is old provenance only; new SEM comes from between-batch rows.
    pooled=np.zeros((5,int(mask.sum())))
    for p in map(Path,paths):
        with np.load(p,allow_pickle=False) as f:
            fine=f['fine']
            for j in range(5):pooled[j]+=fine[j][mask]
    n=600;sa,sb,saa,sbb,sab=pooled
    a=sa.sum()/n;b=sb.sum()/n;ratio=a/b
    va=np.maximum(0,(saa-sa*sa/n)/(n-1)/n).sum();vb=np.maximum(0,(sbb-sb*sb/n)/(n-1)/n).sum();cov=((sab-sa*sb/n)/(n-1)/n).sum()
    near_sem=float(np.sqrt(max(0,va+ratio*ratio*vb-2*ratio*cov))/b)
    r=org.result();values={};sem={}
    for j,name in enumerate(('prostate','rectum','near')):values[name+'_ratio']=float(energy[0,j]/energy[1,j])
    sem['near_ratio']=near_sem
    for j,name in enumerate(('prostate','rectum')):
        sem[name+'_ratio']=float(r['ratio_sem'][j]/r['ratio'][j]*values[name+'_ratio'])
        for side,label in enumerate(('with','without')):
            key=name+'_'+label;values[key]=float(energy[side,j]);sem[key]=float(r['sem_'+label][j]/r['mean_'+label][j]*values[key])
    record=dict(format='a2b-reference-v1',runs=rows,batches=600,histories_per_side=6*7*7142900,masks_sha256=sha(analysis.MASK_PATH),values=values,sem=sem,deltas=DELTAS,near_sem='independent old 0.25 mm cells',organ_sem='relative SEM from existing aggregate; means recomputed on common masks',review_required=True)
    contracts.write(out,record);return record


def judge(reference,new_summary,out):
    ref=json.loads(Path(reference).read_text());new=json.loads(Path(new_summary).read_text())
    if ref['deltas']!=DELTAS or ref['masks_sha256']!=sha(analysis.MASK_PATH) or ref['batches']!=600:raise ValueError('reference seal/criteria')
    if len(ref['runs'])!=6 or len({r['path'] for r in ref['runs']})!=6 or len({r['seed'] for r in ref['runs']})!=6:raise ValueError('six independent old runs required')
    for row in ref['runs']:
        if sha(row['path'])!=row['sha256'] or row['batches']!=list(range(100)):raise ValueError('old reference changed')
    if new.get('baseline_only'):raise ValueError('B5 legacy baseline cannot be used for B3')
    if new['batches']!=50:raise ValueError('50 new batches required')
    from chatcarlo.gpu import seed_condition,seed_run
    run_dir=Path(new_summary).parent.parent
    cond=seed_condition.verify(run_dir)['condition'];contracts.require(run_dir,cond,seed_condition.verify)
    if cond['seed'] not in (9502001,9502002):raise ValueError('new fixed B3 seed')
    if any(row['seed']==cond['seed'] for row in ref['runs']):raise ValueError('old/new seed collision')
    store=seed_run.preflight(Path(new_summary).parent,cond)
    if store.completed!=list(range(50)) or sha(store.path)!=new['checkpoint_sha256']:raise ValueError('new checkpoint SHA256/completion')
    if new['statistics']!=analysis.analyse(store.rows):raise ValueError('summary / paired batch table mismatch')
    r=new['statistics'];rows={}
    for key,delta in DELTAS.items():
        name,side=key.rsplit('_',1);j=('prostate','rectum','near').index(name)
        v=float(r['ratio' if side=='ratio' else 'mean_'+side][j]);s=float(r['ratio_sem' if side=='ratio' else 'sem_'+side][j])
        old=ref['values'][key];sigma=float(np.hypot(s,ref['sem'][key]));scale=1. if side=='ratio' else abs(old)
        diff=abs(v-old)/scale;three=3*sigma/scale
        status='insufficient-statistics' if not np.isfinite([diff,three]).all() or three>delta else 'pass' if diff<=delta else 'fail'
        rows[key]=dict(old=old,new=v,difference=diff,three_sigma=three,delta=delta,status=status)
    statuses=[r['status'] for r in rows.values()]
    record=dict(status='fail' if 'fail' in statuses else 'insufficient-statistics' if 'insufficient-statistics' in statuses else 'pass',rows=rows,reference_sha256=sha(reference),new_summary_sha256=sha(new_summary),notes='Old near SEM: cell independent. New near SEM: paired outer batches. No entry switch performed.')
    contracts.write(out,record);return record

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze');f.add_argument('out',type=Path);f.add_argument('runs',nargs=6,type=Path)
    j=sub.add_parser('judge');j.add_argument('reference',type=Path);j.add_argument('summary',type=Path);j.add_argument('out',type=Path)
    a=p.parse_args();print(json.dumps(freeze(a.runs,a.out) if a.command=='freeze' else judge(a.reference,a.summary,a.out),indent=2))
