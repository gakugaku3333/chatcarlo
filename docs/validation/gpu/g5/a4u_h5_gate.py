"""Explicit H5 CPU main-run judgement. Import/--help never build or transport.

No EGS5/CUDA execution and no automatic retry. D7 rules are imported from H4.
"""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from . import a4u_h4_gate as h4
from .a4u_h4_gate import registration, analyse, moments, candidate_samples

HERE=Path(__file__).resolve().parent
RESULT=HERE/'a4u_h5.json'
REFERENCE_ROOT=HERE/'a4_run/egs5'
APPROVAL='approved_by_user_2026-11-04: PDD only, judged depth <= 15 cm'
CASES={20:dict(label='h5_main_20kev',EGS5_N=18438400,EGS5_seed=20101,CPU_N=20127000,CPU_seed=20111,retry_CPU_seed=20121),
       60:dict(label='h5_main_60kev',EGS5_N=23216500,EGS5_seed=60101,CPU_N=23164000,CPU_seed=60111,retry_CPU_seed=60121)}
BATCHES=100
# H4 uses 10k histories/call. Keep that memory bound: a main outer batch is
# >200k histories, exceeding CPU.histories' API limit and large history tensor.
CHUNK_SIZE=10000


def read_reference(energy):
    from .a4_egs import WANT_ON
    c=CASES[energy];base=REFERENCE_ROOT/c['label'];pilot=REFERENCE_ROOT/h4.CASES[energy]['label']
    paths={f'main/{n}':base/n for n in ('config.json','summary.json','result.npz')}
    paths.update({f'pilot/{n}':pilot/n for n in ('config.json','summary.json')})
    if not all(p.is_file() for p in paths.values()):raise ValueError('H5 requires completed saved EGS5 inputs; never launches EGS5')
    pins={n:h4.sha(p) for n,p in paths.items()}
    cfg=json.loads(paths['main/config.json'].read_text());summary=json.loads(paths['main/summary.json'].read_text())
    edges,cum=h4.mono(energy)
    expected=dict(N=c['EGS5_N'],seed=c['EGS5_seed'],batches=100,field=[h4.HALF,h4.SSD],regions=h4.REGIONS,thickness=30.,nb=1)
    if any(cfg.get(k)!=v for k,v in expected.items()) or cfg.get('spectrum_arrays_sha256')!=h4.array_hash(edges,cum):raise ValueError('H5 EGS5 condition/spectrum mismatch')
    if summary.get('status')!='合格' or len(summary.get('tests',{}))!=19 or not all(summary['tests'].values()) or summary.get('effective_flags')!=WANT_ON:raise ValueError('H5 EGS5 health/physics ON mismatch')
    with np.load(paths['main/result.npz'],allow_pickle=False) as z:
        if int(z['N'])!=c['EGS5_N'] or int(z['seed'])!=c['EGS5_seed'] or str(z['unit'])!='Gy cm2':raise ValueError('H5 EGS5 result identity/unit')
        fine=z['fine'].copy();profile=z['profile_fine']*h4.GY_PER_MEV_G/(c['EGS5_N']/100*.25)
        wall=float(z['seconds'])
    if not math.isclose(wall,float(summary['wall_seconds']),rel_tol=1e-12):raise ValueError('H5 EGS5 timing mismatch')
    # Two already-completed runs separate fixed startup from marginal cost.
    # Never label main WALL/N as a marginal history cost.
    pc=json.loads(paths['pilot/config.json'].read_text());ps=json.loads(paths['pilot/summary.json'].read_text())
    if pc['N']!=h4.CASES[energy]['N'] or pc['seed']!=h4.CASES[energy]['egs_seed'] or pc['spectrum_arrays_sha256']!=h4.array_hash(edges,cum):raise ValueError('H5 timing calibration identity')
    slope=(wall-float(ps['wall_seconds']))/(c['EGS5_N']-pc['N']);fixed=wall-slope*c['EGS5_N']
    if not all(math.isfinite(v) and v>0 for v in (wall,slope,fixed,float(summary['CPU_seconds']))):raise ValueError('H5 timing calibration invalid')
    if any(h4.sha(p)!=pins[n] for n,p in paths.items()):raise ValueError('EGS5 reference changed while reading')
    samples,meta=candidate_samples(fine,profile)
    return dict(samples=samples,meta=meta,fine=fine,profile_fine=profile,N=c['EGS5_N'],seed=c['EGS5_seed'],files=pins,
        edges=edges.tolist(),cum=cum.tolist(),timing=dict(total_wall_seconds=wall,CPU_seconds_total=float(summary['CPU_seconds']),
        fixed_startup_seconds_estimate=fixed,marginal_seconds_per_history_estimate=slope,
        marginal_wall_seconds_estimate=slope*c['EGS5_N'],
        scope='two-point fit from saved pilot/main WALL times; marginal term includes per-history and peripheral costs; fixed term is estimated, no separate I/O measurement'))


def make_condition(energy,ref,reg,materials,arrays,fg):
    c=CASES[energy]
    cond=h4.condition(energy,ref,reg,materials,arrays,fg)
    cond.update(format='a4u-h5-main-v1',N=c['CPU_N'],cpu_seed=c['CPU_seed'],retry_cpu_seed=c['retry_CPU_seed'],
        EGS5_N=c['EGS5_N'],EGS5_seed=c['EGS5_seed'],attempt='initial',chunk_size=CHUNK_SIZE,
        retry_policy='one human-decided fresh CPU seed; same sealed EGS5 reference; never automatic')
    return cond


def chunks(start,count,size):
    return [(s,min(size,start+count-s)) for s in range(start,start+count,size)]


def run_cpu(out,cond,materials,arrays,fg):
    from chatcarlo.gpu import therapy,physics_sets,therapy_health,contracts
    from chatcarlo.gpu.therapy_observation import descriptor
    # This launcher executes only the initial run. A retry seed is a plan only.
    if cond['attempt']!='initial':raise ValueError('automatic retry execution is not supported')
    out=Path(out);seal=h4.check_condition(out,cond,write=True);driver=None;fine=[];profile=[];seconds=0.
    if cond['N']%BATCHES:raise ValueError('H5 requires 100 equal outer batches')
    n=cond['N']//BATCHES
    for b in range(BATCHES):
        identity=dict(condition_sha256=seal['sha256'],batch=b,start=b*n,count=n,seed=cond['cpu_seed'])
        spans=chunks(b*n,n,cond['chunk_size'])
        expected=[dict(start=s,count=k,observation_sha256=h4.digest(descriptor(arrays,fg,(h4.HALF,h4.HALF),h4.SSD,cond['cpu_seed'],s,k,cond['physics_set']))) for s,k in spans]
        path=out/f'batch_{b:03d}.npz';rp=path.with_suffix('.json')
        if path.exists() or rp.exists():
            record=json.loads(rp.read_text())
            if record['identity']!=identity or record['sha256']!=h4.sha(path) or record['chunks']!=expected or record['physics_set']!=cond['physics_set']:raise ValueError('H5 batch receipt/identity/SHA256 mismatch')
            with np.load(path,allow_pickle=False) as z:f=z['fine'].copy();p=z['profile_fine'].copy()
            elapsed=record['seconds']
        else:
            if driver is None:
                driver=therapy.CPU(out/'build',materials,arrays,field_geometry=fg,half=(h4.HALF,h4.HALF),ssd=h4.SSD,physics_declaration=cond['physics_set'])
                physics_sets.check(driver.observed,cond['physics_set'])
            raw_fine=np.zeros(600);raw_field=np.zeros(3420);elapsed=0.;observations=[]
            for s,k in spans:
                begin=time.perf_counter();r=driver.histories(k,cond['cpu_seed'],start=s,slots=h4.SLOTS);elapsed+=time.perf_counter()-begin
                want=descriptor(arrays,fg,(h4.HALF,h4.HALF),h4.SSD,cond['cpu_seed'],s,k,cond['physics_set'])
                contracts.check_observation(driver.last_observation,want,'H5 native CPU')
                therapy_health.check(r,arrays,fg)
                observations.append(dict(start=s,count=k,observation_sha256=h4.digest(driver.last_observation)))
                raw_fine+=r['groups'][:,450:1050].sum(0);raw_field+=r['field'];del r
            if observations!=expected:raise ValueError('H5 native chunk observations')
            f=raw_fine*h4.GY_PER_MEV_G/(n*.05)
            p=raw_field[1500:3420].reshape(2,3,320)*h4.GY_PER_MEV_G/(n*.25)
            np.savez(path,fine=f,profile_fine=p)
            rp.write_text(json.dumps(dict(identity=identity,sha256=h4.sha(path),seconds=elapsed,
                physics_set=cond['physics_set'],chunks=observations),indent=2)+'\n')
        if f.shape!=(600,) or p.shape!=(2,3,320) or not math.isfinite(elapsed) or elapsed<=0:raise ValueError('H5 cached batch/timing')
        fine.append(f);profile.append(p);seconds+=elapsed
    f=np.asarray(fine);p=np.asarray(profile);samples,meta=candidate_samples(f,p)
    return dict(samples=samples,meta=meta,fine=f,profile_fine=p,N=cond['N'],
        timing=dict(native_call_wall_seconds=seconds,native_call_seconds_per_history=seconds/cond['N'],
        scope='CPU.histories including allocations and observation; excludes build, table generation, validation and I/O'))


def evaluate_case(reference,cpu,reg,cond):
    # D7 lives in H4. N can differ between the sides; means and SEM still use
    # their own 100 correctly-normalised outer batches. Pilot N projections
    # and pilot cutoff are not H5 criteria and are not copied into this report.
    shared=analyse(reference,cpu['samples'],reg);decision=shared['pilot_D7']
    rows=[{k:v for k,v in row.items() if k not in ('required_histories','cutoff','lower_abs_difference')} for row in shared['bins']]
    measured={}
    for key in ('fine','profile_fine'):
        rm,rs=moments(reference[key]);cm,cs=moments(cpu[key])
        measured[key]=dict(EGS5_mean=rm.tolist(),EGS5_SEM=rs.tolist(),CPU_mean=cm.tolist(),CPU_SEM=cs.tolist())
    status='retry_required' if decision['status']=='fail' else decision['status']
    return dict(status=status,D7=decision,K=shared['K'],z_critical=shared['z_critical'],bins=rows,
        known_failure_bins=shared['known_failure_bins'],condition=cond,condition_sha256=h4.digest(cond),
        measured=measured,judged_candidate_indices=reg['indices'],OCR_judged=False,
        retry=dict(required=status=='retry_required',automatic=False,CPU_seed=cond['retry_cpu_seed'],CPU_N=cond['N'],
            EGS5_seed=cond['EGS5_seed'],EGS5_N=cond['EGS5_N'],EGS5_reference_reused=True),
        EGS5_timing=reference.get('timing'),CPU_timing=cpu.get('timing'))


def judge_retry(first,repeated):
    """Read-only decision for a human-supplied second case report; no transport.

    A manual evaluator can use evaluate_case with an explicitly produced CPU
    sample and a condition whose attempt='retry', cpu_seed=retry_cpu_seed.
    """
    c=first['condition'];expected={**c,'attempt':'retry','cpu_seed':c['retry_cpu_seed']}
    if first['status']!='retry_required' or c['attempt']!='initial' or c['cpu_seed']==c['retry_cpu_seed']:raise ValueError('H5 initial failure/retry seed required')
    if repeated['condition']!=expected or first['condition_sha256']!=h4.digest(c) or repeated['condition_sha256']!=h4.digest(expected):raise ValueError('H5 retry condition/reference/registration mismatch')
    common=sorted(set(first['known_failure_bins'])&set(repeated['known_failure_bins']))
    status='fail' if common else 'insufficient' if repeated['D7']['status']=='insufficient' else 'review_required' if repeated['known_failure_bins'] else 'pass'
    return dict(status=status,repeated_failure_bins=common,initial_failure_bins=first['known_failure_bins'],
        second_failure_bins=repeated['known_failure_bins'],further_retry_allowed=False,
        note='same sealed EGS5 reference; different new failures stay unresolved, never an automatic third run')


def gate(out):
    if h4.REGISTRATION_PROPOSAL['status']!=APPROVAL:raise ValueError('H5 requires the user-approved H4 registration')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    refs={e:read_reference(e) for e in CASES}
    materials,base_arrays,fg=h4.prepare_cpu(20);prepared={}
    for energy,ref in refs.items():
        edges,cum=h4.mono(energy);arrays={**base_arrays,'edges':edges,'cum':cum}
        reg=registration(ref);cond=make_condition(energy,ref,reg,materials,arrays,fg)
        h4.check_condition(out/str(energy),cond)
        prepared[energy]=(reg,cond,arrays)
    cases={}
    for energy in CASES:
        reg,cond,arrays=prepared[energy]
        cpu=run_cpu(out/str(energy),cond,materials,arrays,fg)
        cases[str(energy)]=evaluate_case(refs[energy],cpu,reg,cond)
        if cases[str(energy)]['status']!='pass':break
    statuses=[r['status'] for r in cases.values()]
    status=next((s for s in statuses if s!='pass'),'pass')
    report=dict(status=status,cases=cases,pending_energies=[e for e in CASES if str(e) not in cases],
        registration_status=h4.REGISTRATION_PROPOSAL['status'],retry_executed=False,
        scope='H5 CPU-twin statistical gate only; entry approval/runs and H6 are separate; H3b remains closed as failure with unexplained residual')
    RESULT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    r=gate(parser.parse_args().out);print(json.dumps(dict(status=r['status'],pending_energies=r['pending_energies']),indent=2))
    return 0 if r['status']=='pass' else 1


if __name__=='__main__':raise SystemExit(main())
