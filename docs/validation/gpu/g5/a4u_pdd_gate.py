"""Read-only central-axis PDD judgement under A4U_PDD_PREREG.md.

Use --run DIR for saved CPU batches and --out DIR for the new report.
No transport, EGS5 execution, retries, or history/ROI/threshold overrides.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
import numpy as np
from . import a4u_h4_gate as h4, a4u_h5_gate as h5
from .a4u_h4_gate import (
    analyse, moments, PDD_FRACTION, MAX_JUDGED_DEPTH_CM, ALPHA, SEM_LIMIT,
    RELATIVE_LIMIT, CUTOFF_SIGMA, MIN_EVALUABLE_COUNT, MIN_EVALUABLE_FRACTION,
    MIN_MANDATORY_BINS,
)
from .a4u_h5_gate import CASES, BATCHES, CHUNK_SIZE, chunks
from . import a4u_pdd_cases

# Preregistered exception: 0.25 cm bins give only two centers below 0.5 cm.
# Use z < 1 cm to retain the requirement of four evaluable surface bins.
MANDATORY_PDD_END_CM=1.0
REGISTRATION_PROPOSAL={**h4.REGISTRATION_PROPOSAL,
    'status':'approved_by_user_2026-11-05: A4U_PDD_PREREG.md',
    'mandatory_pdd_end_cm':MANDATORY_PDD_END_CM}


def candidate_samples(axis,group=1):
    if group not in (1,4):raise ValueError('PDD group must be 1 or 4')
    axis=np.asarray(axis,float)
    if axis.shape!=(BATCHES,120) or not np.isfinite(axis).all() or np.any(axis<0):
        raise ValueError('PDD axis_coarse scores')
    if group==4:
        # Sum within each batch on both sides, retaining inter-bin covariance.
        axis=axis.reshape(BATCHES,30,4).sum(axis=2)
        meta=[dict(name=f'axis_1cm:{j}',kind='fine',index=j,z_cm=(j+.5)*1.0,
                   unit='Gy cm2/history') for j in range(30)]
        return axis,meta
    meta=[dict(name=f'axis_coarse:{j}',kind='fine',index=j,z_cm=(j+.5)*.25,
               unit='Gy cm2/history') for j in range(120)]
    return axis,meta


def registration(reference,group=1):
    if group not in (1,4):raise ValueError('PDD group must be 1 or 4')
    means=reference['samples'].mean(0);meta=reference['meta'];count=min(120,len(meta));peak=float(means[:count].max())
    if peak<=0:raise ValueError('empty reference depth curve')
    depth=np.array([meta[i]['z_cm'] for i in range(count)],float)
    selected=((means[:count]>=PDD_FRACTION*peak)&(depth<=MAX_JUDGED_DEPTH_CM)).tolist()+[False]*(len(meta)-count)
    ids=np.flatnonzero(selected);chosen=[meta[i] for i in ids]
    mandatory=[r['z_cm']<MANDATORY_PDD_END_CM for r in chosen]
    counts={kind:sum(flag and r['kind']==kind for flag,r in zip(mandatory,chosen)) for kind in MIN_MANDATORY_BINS}
    # Addendum 2 correction: cover the same surface 0–1 cm with 4 or 1 bins.
    valid=all(counts[k]>=n/group for k,n in MIN_MANDATORY_BINS.items())
    K=len(ids)
    return dict(proposal=REGISTRATION_PROPOSAL,indices=ids.tolist(),bins=chosen,mandatory=mandatory,
        mandatory_counts=counts,valid=valid,K=K,z_critical=NormalDist().inv_cdf(1-ALPHA/(2*K)),
        minimum_evaluable=max(MIN_EVALUABLE_COUNT,math.ceil(MIN_EVALUABLE_FRACTION*K)))


def check_reference_precision(reference,reg):
    mean,sem=moments(reference['samples'][:,reg['indices']])
    exceeded=int(np.count_nonzero(sem>(SEM_LIMIT/math.sqrt(2))*mean))
    if exceeded*5>reg['K']:
        raise ValueError(f"参照側だけで予算を使い切っている。bin 幅か参照の履歴数を見直す "
                         f"（相対 SEM > SEM_LIMIT/sqrt(2): {exceeded}/{reg['K']} bins）")


def read_reference(energy,cases=None):
    c=(CASES if cases is None else cases)[energy];path=h5.REFERENCE_ROOT/c['label']/'result.npz';pin=h4.sha(path)
    with np.load(path,allow_pickle=False) as z:
        # Check identity before reading any scoring array. axis_coarse is raw
        # MeV per batch, in a 2*2*.25 = 1 g water voxel.
        if int(z['N'])!=c['EGS5_N'] or int(z['seed'])!=c['EGS5_seed']:
            raise ValueError('PDD EGS5 result identity mismatch')
        axis=z['axis_coarse']*h4.GY_PER_MEV_G/(c['EGS5_N']/BATCHES)
    if h4.sha(path)!=pin:raise ValueError('PDD EGS5 reference changed while reading')
    samples,meta=candidate_samples(axis)
    return dict(samples=samples,meta=meta,N=c['EGS5_N'],seed=c['EGS5_seed'],
                files={'result.npz':pin})


def read_cpu(out,energy,cases=None):
    out=Path(out);c=(CASES if cases is None else cases)[energy]
    seal=json.loads((out/'condition.json').read_text());cond=seal['condition']
    if seal['sha256']!=h4.digest(cond):raise ValueError('PDD CPU condition SHA256 mismatch')
    if (cond['N']!=c['CPU_N'] or cond['cpu_seed']!=c['CPU_seed'] or
        cond['attempt']!='initial' or cond['chunk_size']!=CHUNK_SIZE or cond['batches']!=BATCHES):
        raise ValueError('PDD CPU fixed condition mismatch')
    fine=[];profile=[];fields=[];n=cond['N']//BATCHES
    for b in range(BATCHES):
        path=out/f'batch_{b:03d}.npz';record=json.loads(path.with_suffix('.json').read_text())
        identity=dict(condition_sha256=seal['sha256'],batch=b,start=b*n,count=n,seed=cond['cpu_seed'])
        spans=[(r['start'],r['count']) for r in record['chunks']]
        if (record['identity']!=identity or record['sha256']!=h4.sha(path) or
            record['physics_set']!=cond['physics_set'] or spans!=chunks(b*n,n,CHUNK_SIZE)):
            raise ValueError('PDD CPU batch receipt/identity/SHA256 mismatch')
        with np.load(path,allow_pickle=False) as z:
            fine.append(z['fine'].copy());profile.append(z['profile_fine'].copy());fields.append(z['raw_field'].copy())
    fine=np.asarray(fine);profile=np.asarray(profile);fields=np.asarray(fields)
    h4.candidate_samples(fine,profile)  # same saved fine/profile validation as H5
    if fields.shape!=(BATCHES,540):raise ValueError('PDD raw_field shape')
    axis=fields[:,:120]*h4.GY_PER_MEV_G/n
    samples,_=candidate_samples(axis)
    return dict(samples=samples,fine=fine,profile_fine=profile,condition=cond)


def check_fine(energy,fine):
    saved=json.loads(h5.RESULT.read_text())['cases'][str(energy)]['measured']['fine']['CPU_mean']
    if not np.array_equal(moments(np.asarray(fine))[0],np.asarray(saved)):
        raise ValueError('PDD fine does not exactly match sealed H5 CPU_mean')


def evaluate_case(energy,reference,cpu,cases='h5',group=1):
    if cases!='pdd':check_fine(energy,cpu['fine'])  # mandatory for the same H5 N/seed
    reg=registration(reference,group)
    check_reference_precision(reference,reg)
    shared=analyse(reference,cpu['samples'],reg)
    # H4's pilot projections are not permission to change histories. Retain its
    # D7 judgement and cutoff only, with no retry or projected-history output.
    status='stop_cutoff' if shared['cutoff_bins'] else shared['pilot_D7']['status']
    rows=[{k:v for k,v in row.items() if k!='required_histories'} for row in shared['bins']]
    rm,rs=moments(reference['samples']);cm,cs=moments(cpu['samples'])
    return dict(status=status,D7=shared['pilot_D7'],K=shared['K'],z_critical=shared['z_critical'],
        bins=rows,cutoff_bins=shared['cutoff_bins'],known_failure_bins=shared['known_failure_bins'],
        registration=reg,fine_matches_H5=cases!='pdd',OCR_judged=False,
        **({'fine_matches_H5_reason':'別の N・seed なので H5 との一致検査は適用外'} if cases=='pdd' else {}),
        measured=dict(axis_coarse=dict(EGS5_mean=rm.tolist(),EGS5_SEM=rs.tolist(),
            CPU_mean=cm.tolist(),CPU_SEM=cs.tolist())),
        condition=cpu['condition'],reference_files=reference['files'])


def gate(run,out,cases='h5',group=1,energy=None):
    if group not in (1,4):raise ValueError('PDD group must be 1 or 4')
    if energy is not None and energy not in (20,60):raise ValueError('PDD energy must be 20 or 60')
    if group==4 and energy==20:raise ValueError('--energy 20 cannot be combined with --group 4')
    selected=a4u_pdd_cases.CASES if cases=='pdd' else CASES
    if energy is not None:selected={energy:selected[energy]}
    if group==4:
        # Addendum 2 concerns 60 keV only; 20 keV retains its 0.25 cm gate.
        selected={60:selected[60]}
    refs={e:read_reference(e,selected) for e in selected}
    cpus={e:read_cpu(Path(run)/str(e),e,selected) for e in selected}
    if group==4:
        for energy in selected:
            refs[energy]['samples'],refs[energy]['meta']=candidate_samples(refs[energy]['samples'],group)
            cpus[energy]['samples'],_=candidate_samples(cpus[energy]['samples'],group)
    # Verify the selected energies before judging any central-axis curve.
    if cases!='pdd':
        for energy in selected:check_fine(energy,cpus[energy]['fine'])
    results={}
    for energy in selected:
        results[str(energy)]=evaluate_case(energy,refs[energy],cpus[energy],cases,group)
        if results[str(energy)]['status']!='pass':break
    status=next((r['status'] for r in results.values() if r['status']!='pass'),'pass')
    report=dict(status=status,cases=results,pending_energies=[e for e in selected if str(e) not in results])
    if group==4:
        report.update(group=4,bin_width_cm=1.0,post_hoc=True,
            post_hoc_reason='0.25 cm では統計不足だったのを見てから粗くした後出しの判定。A4U_PDD_PREREG.md の追補 2')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'a4u_pdd.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cases',choices=('h5','pdd'),default='h5')
    parser.add_argument('--group',type=int,choices=(1,4),default=1)
    parser.add_argument('--energy',type=int,choices=(20,60))
    args=parser.parse_args();report=gate(args.run,args.out,args.cases,args.group,args.energy)
    print(json.dumps(dict(status=report['status'],pending_energies=report['pending_energies']),indent=2))
    return 0 if report['status']=='pass' else 1


if __name__=='__main__':raise SystemExit(main())
