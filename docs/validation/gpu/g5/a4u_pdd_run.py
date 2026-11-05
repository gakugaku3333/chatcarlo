"""Rerun the fixed H5 CPU histories, additionally saving raw field[0:540].

Run with python -m docs.validation.gpu.g5.a4u_pdd_run --out DIR.
Import/--help never build or transport. No EGS5 execution or automatic retry.
"""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from . import a4u_h4_gate as h4, a4u_h5_gate as h5
from .a4u_h5_gate import CASES, BATCHES, CHUNK_SIZE, chunks
from .a4u_h4_gate import candidate_samples
from . import a4u_pdd_cases, a4u_pdd_gate as pdd


def run_cpu(out,cond,materials,arrays,fg):
    from chatcarlo.gpu import therapy,physics_sets,therapy_health,contracts
    from chatcarlo.gpu.therapy_observation import descriptor
    # This launcher executes only the initial run. A retry seed is a plan only.
    if cond['attempt']!='initial':raise ValueError('automatic retry execution is not supported')
    out=Path(out);seal=h4.check_condition(out,cond,write=True);driver=None;fine=[];profile=[];fields=[];seconds=0.
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
            with np.load(path,allow_pickle=False) as z:f=z['fine'].copy();p=z['profile_fine'].copy();field=z['raw_field'].copy()
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
            field=raw_field[:540].copy()
            np.savez(path,fine=f,profile_fine=p,raw_field=field)
            rp.write_text(json.dumps(dict(identity=identity,sha256=h4.sha(path),seconds=elapsed,
                physics_set=cond['physics_set'],chunks=observations),indent=2)+'\n')
        if field.shape!=(540,) or f.shape!=(600,) or p.shape!=(2,3,320) or not math.isfinite(elapsed) or elapsed<=0:raise ValueError('H5 cached batch/timing')
        fine.append(f);profile.append(p);fields.append(field);seconds+=elapsed
    f=np.asarray(fine);p=np.asarray(profile);samples,meta=candidate_samples(f,p)
    return dict(samples=samples,meta=meta,fine=f,profile_fine=p,raw_field=np.asarray(fields),N=cond['N'],
        timing=dict(native_call_wall_seconds=seconds,native_call_seconds_per_history=seconds/cond['N'],
        scope='CPU.histories including allocations and observation; excludes build, table generation, validation and I/O'))


def make_condition(energy,ref,reg,materials,arrays,fg,cases='h5'):
    cond=h5.make_condition(energy,ref,reg,materials,arrays,fg)
    if cases=='pdd':
        c=a4u_pdd_cases.CASES[energy]
        cond.update(N=c['CPU_N'],cpu_seed=c['CPU_seed'],EGS5_N=c['EGS5_N'],EGS5_seed=c['EGS5_seed'],
            retry_cpu_seed='not applicable: see A4U_PDD_PREREG.md stop condition',
            retry_policy='not applicable: see A4U_PDD_PREREG.md stop condition')
    return cond


def run(out,cases='h5',energy=None):
    out=Path(out)
    selected=a4u_pdd_cases.CASES if cases=='pdd' else CASES
    energies=selected if energy is None else [energy]
    refs={e:pdd.read_reference(e,selected) if cases=='pdd' else h5.read_reference(e) for e in energies}
    materials,base_arrays,fg=h4.prepare_cpu(20);prepared={}
    for energy,ref in refs.items():
        edges,cum=h4.mono(energy);arrays={**base_arrays,'edges':edges,'cum':cum}
        if cases=='pdd':ref={**ref,'edges':edges.tolist(),'cum':cum.tolist()}
        reg=pdd.registration(ref) if cases=='pdd' else h5.registration(ref)
        cond=make_condition(energy,ref,reg,materials,arrays,fg,cases)
        h4.check_condition(out/str(energy),cond)
        prepared[energy]=(cond,arrays)
    for energy in energies:
        cond,arrays=prepared[energy]
        run_cpu(out/str(energy),cond,materials,arrays,fg)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cases',choices=('h5','pdd'),default='h5')
    parser.add_argument('--energy',type=int,choices=(20,60))
    args=parser.parse_args();run(args.out,args.cases,args.energy)
    return 0


if __name__=='__main__':raise SystemExit(main())
