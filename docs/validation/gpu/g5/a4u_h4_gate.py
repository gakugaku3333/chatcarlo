"""Explicit H4 CPU pilot launcher. Import/--help do not build or transport.

Reads completed EGS5 pilots only; never starts EGS5, CUDA, H5, or a retry.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import time
from types import SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
RESULT=HERE/'a4u_h4.json'
REFERENCE_ROOT=HERE/'a4_run/egs5'
CASES={20:dict(label='h4_pilot_20kev',N=1000000,egs_seed=20001,cpu_seed=20011),
       60:dict(label='h4_pilot_60kev',N=1000000,egs_seed=60001,cpu_seed=60011)}
BATCHES=100
SLOTS=32
HALF=5.
SSD=100.
REGIONS=[0.,0.,1.,1.,20.,.25,.125,1.,5.,10.,20.]
GY_PER_MEV_G=1.602176634e-10
ALPHA=.05
RELATIVE_LIMIT=.02
SEM_LIMIT=.01
CUTOFF_SIGMA=2
N_MARGIN=1.5
# APPROVED BY THE USER 2026-11-04 ("PDD だけ"): the judged set is the depth
# curve only. The OCR bins stay at the one saved shallow depth, 5 cm, where a
# 20 keV beam leaves about 1/2600 of the entrance dose (13-16% relative SEM at
# 1e6 histories); making them decidable needs ~4.9e8 histories, 18 h of EGS5
# per point, against 1.8e7 and 27 min for the depth curve alone. P7 saves no
# shallower profile, so the plan's own remedy applies: narrow the ROI rather
# than buy histories. OCR is still measured and reported, never judged.
# Original PROPOSAL text follows. The entrance is where the
# added low-energy physics matters most. Require both OCR axes at the only
# shallow depth already saved by P7 (5 cm); do not request new EGS5 regions.
# A narrow central ROI avoids spending histories on the field edge/tails.
# All mandatory bins must be evaluable; 80% permits only nonmandatory tails
# to be insufficient. At least 4 entrance PDD and 2 bins/axis prevent an empty
# mandatory intersection from silently satisfying the rule.
PDD_FRACTION=.05
# APPROVED BY THE USER 2026-11-04 ("15 cm まで"). At 60 keV the 5%-of-peak rule
# alone reaches 26.9 cm (531 bins), which pushes the multiplicity threshold to
# 3.905 and demands decidable statistics in the deep tail: 1.28e8 histories,
# 3.37 h of EGS5. Capping the judged depth at 15 cm keeps 300 bins - still ample
# for the Bonferroni correction - at 2.32e7 histories and 0.61 h. 15 cm is about
# e^-3 of the entrance at 60 keV, and the added physics (Rayleigh, bound Compton,
# relaxation) acts well inside it. At 20 keV the 5% rule already stops shallower,
# so the cap changes nothing there. Bins beyond it are measured, never judged.
MAX_JUDGED_DEPTH_CM=15.
OCR_DEPTH_CM=5.
OCR_HALF_WIDTH_CM=.25
MANDATORY_PDD_END_CM=.5
MANDATORY_OCR_HALF_WIDTH_CM=.125
MIN_MANDATORY_BINS={'fine':4}
MIN_EVALUABLE_COUNT=8
MIN_EVALUABLE_FRACTION=.8
# Proposal using the plan's stated upper history estimate, not an instruction
# to run that many histories. Exceeding it stops for review, never expands ROI/N.
MAX_MAIN_HISTORIES=240000000
REGISTRATION_PROPOSAL=dict(status='approved_by_user_2026-11-04: PDD only, judged depth <= 15 cm',
    pdd_fraction=PDD_FRACTION,max_judged_depth_cm=MAX_JUDGED_DEPTH_CM,ocr_depth_cm=OCR_DEPTH_CM,ocr_half_width_cm=OCR_HALF_WIDTH_CM,
    mandatory_pdd_end_cm=MANDATORY_PDD_END_CM,mandatory_ocr_half_width_cm=MANDATORY_OCR_HALF_WIDTH_CM,
    minimum_mandatory_bins=MIN_MANDATORY_BINS,minimum_evaluable_count=MIN_EVALUABLE_COUNT,
    minimum_evaluable_fraction=MIN_EVALUABLE_FRACTION,max_main_histories=MAX_MAIN_HISTORIES,
    rule=dict(multiplicity_family='per_energy',alpha=ALPHA,relative_limit=RELATIVE_LIMIT,SEM_limit=SEM_LIMIT,cutoff_sigma=CUTOFF_SIGMA,N_margin=N_MARGIN,variance_allocation='equal sides'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def mono(energy):
    from .a4_egs import mono_edges
    return mono_edges(energy)  # no scope(), build_code(), or run() calls


def array_hash(*arrays):
    # Identical to run_p3a.array_hash used in the saved EGS5 config.
    h=hashlib.sha256()
    for value in arrays:
        a=np.ascontiguousarray(value);h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def candidate_samples(fine,profile):
    fine=np.asarray(fine,float);profile=np.asarray(profile,float)
    if fine.shape!=(100,600) or profile.shape!=(100,2,3,320):raise ValueError('H4 scoring shapes')
    if not np.isfinite(fine).all() or not np.isfinite(profile).all() or np.any(fine<0) or np.any(profile<0):raise ValueError('invalid H4 scores')
    centers=-20+(np.arange(320)+.5)*.125
    indices=np.flatnonzero(np.abs(centers)<OCR_HALF_WIDTH_CM)
    columns=[fine];meta=[dict(name=f'fine:{j}',kind='fine',index=j,z_cm=(j+.5)*.05,unit='Gy cm2/history') for j in range(600)]
    for axis in range(2):
        columns.append(profile[:,axis,0,indices])  # shallowest P7 OCR: 5 cm
        meta.extend(dict(name=f'ocr_{"xy"[axis]}:5:{j}',kind='ocr_'+'xy'[axis],index=int(j),z_cm=5.,position_cm=float(centers[j]),unit='Gy/history') for j in indices)
    return np.column_stack(columns),meta


def read_reference(energy,root=None):
    from .a4_egs import WANT_ON
    case=CASES[energy];path=Path(root or REFERENCE_ROOT)/case['label']
    files=[path/n for n in ('config.json','summary.json','result.npz')]
    if not all(p.is_file() for p in files):raise ValueError('EGS5 pilot incomplete; read only, no launch: '+str(path))
    pins={p.name:sha(p) for p in files};config=json.loads(files[0].read_text());summary=json.loads(files[1].read_text())
    edges,cum=mono(energy)
    expected=dict(N=case['N'],seed=case['egs_seed'],batches=100,field=[HALF,SSD],regions=REGIONS,thickness=30.,nb=1)
    if any(config.get(k)!=v for k,v in expected.items()) or config.get('spectrum_arrays_sha256')!=array_hash(edges,cum):raise ValueError('EGS5 pilot condition/spectrum mismatch')
    if summary.get('status')!='合格' or not summary.get('tests') or not all(summary['tests'].values()) or summary.get('effective_flags')!=WANT_ON:raise ValueError('EGS5 pilot health/physics ON mismatch')
    with np.load(files[2],allow_pickle=False) as z:
        if int(z['N'])!=case['N'] or int(z['seed'])!=case['egs_seed'] or str(z['unit'])!='Gy cm2':raise ValueError('EGS5 result identity/unit')
        # Legacy FINE is already normalised per primary and 0.05 cm depth.
        # REG/profile_fine are raw MeV per outer batch in 0.25 g water voxels.
        fine=z['fine'].copy();profile=z['profile_fine']*GY_PER_MEV_G/(case['N']/100*.25)
        seconds=float(z['seconds'])
    wall=float(summary['wall_seconds']);cpu=float(summary['CPU_seconds'])
    if not all(math.isfinite(x) and x>0 for x in (wall,cpu,seconds)) or not math.isclose(wall,seconds,rel_tol=1e-12):raise ValueError('EGS5 recorded timing missing/mismatch')
    if any(sha(p)!=pins[p.name] for p in files):raise ValueError('EGS5 pilot changed while reading')
    samples,meta=candidate_samples(fine,profile)
    return dict(samples=samples,meta=meta,files=pins,config_sha256=pins['config.json'],
        N=case['N'],seed=case['egs_seed'],edges=edges.tolist(),cum=cum.tolist(),
        timing=dict(wall_seconds=wall,CPU_seconds=cpu,wall_seconds_per_history=wall/case['N'],CPU_seconds_per_history=cpu/case['N'],scope='saved native EGS5 WALL/CPU timers; no startup subtraction'),
        note='config.E=10 is a legacy placeholder; the sampled energy is sealed by mono_edges/spectrum_arrays_sha256')


def registration(reference):
    means=reference['samples'].mean(0);meta=reference['meta'];peak=float(means[:600].max())
    if peak<=0:raise ValueError('empty reference depth curve')
    # OCR excluded from the judged set: see the registration note above.
    depth=np.array([meta[i]['z_cm'] for i in range(600)],float)
    selected=((means[:600]>=PDD_FRACTION*peak)&(depth<=MAX_JUDGED_DEPTH_CM)).tolist()+[False]*(len(meta)-600)
    ids=np.flatnonzero(selected);chosen=[meta[i] for i in ids]
    mandatory=[(r['z_cm']<MANDATORY_PDD_END_CM if r['kind']=='fine' else abs(r['position_cm'])<MANDATORY_OCR_HALF_WIDTH_CM) for r in chosen]
    counts={kind:sum(flag and r['kind']==kind for flag,r in zip(mandatory,chosen)) for kind in MIN_MANDATORY_BINS}
    valid=all(counts[k]>=n for k,n in MIN_MANDATORY_BINS.items())
    K=len(ids)
    return dict(proposal=REGISTRATION_PROPOSAL,indices=ids.tolist(),bins=chosen,mandatory=mandatory,
        mandatory_counts=counts,valid=valid,K=K,z_critical=NormalDist().inv_cdf(1-ALPHA/(2*K)),
        minimum_evaluable=max(MIN_EVALUABLE_COUNT,math.ceil(MIN_EVALUABLE_FRACTION*K)))


def moments(a):return a.mean(0),a.std(0,ddof=1)/np.sqrt(len(a))


def analyse(reference,twin,reg):
    ids=reg['indices'];x=reference['samples'][:,ids];y=np.asarray(twin,float)[:,ids]
    if y.shape!=x.shape or not np.isfinite(y).all() or np.any(y<0):raise ValueError('CPU pilot scores')
    r,sr=moments(x);t,st=moments(y);sem=np.hypot(sr,st);delta=t-r;tol=np.maximum(reg['z_critical']*sem,RELATIVE_LIMIT*r)
    cutoff=np.abs(delta)-CUTOFF_SIGMA*sem>tol  # D7 OR is a maximum, not two simultaneous limits.
    enough=(r>0)&(sem<=SEM_LIMIT*r);passed=np.abs(delta)<=tol;old=np.abs(delta)<=np.maximum(2*sem,RELATIVE_LIMIT*r)
    mandatory=np.asarray(reg['mandatory'],bool);coverage=reg['valid'] and int(enough.sum())>=reg['minimum_evaluable'] and bool(enough[mandatory].all())
    rows=[];side_max=[100,100];unestimable=[]
    for i,bin_info in enumerate(reg['bins']):
        needs=[]
        for s in (sr[i],st[i]):
            # Equal variance budget: each side <= (1% EGS5 value)/sqrt(2).
            # This supplies precision for a decision, not a guarantee of agreement.
            n=None if cutoff.any() or r[i]<=0 else max(100,math.ceil(reference['N']*(s/(SEM_LIMIT*r[i]/np.sqrt(2)))**2))
            needs.append(n)
        if None in needs:
            if not cutoff.any():unestimable.append(bin_info['name'])
        else:side_max=[max(side_max[j],needs[j]) for j in range(2)]
        rows.append(dict(**bin_info,EGS5=float(r[i]),CPU=float(t[i]),EGS5_SEM=float(sr[i]),CPU_SEM=float(st[i]),
            difference=float(delta[i]),SEM_difference=float(sem[i]),z=float(delta[i]/sem[i]) if sem[i]>0 else (0. if delta[i]==0 else None),
            D7_tolerance=float(tol[i]),lower_abs_difference=float(max(0,abs(delta[i])-CUTOFF_SIGMA*sem[i])),cutoff=bool(cutoff[i]),
            evaluable=bool(enough[i]),D7_pass=bool(passed[i]),historical_2sigma_pass=bool(old[i]),
            mandatory=bool(mandatory[i]),required_histories=dict(EGS5=needs[0],CPU=needs[1])))
    estimates=None if unestimable or cutoff.any() else {name:100*math.ceil(N_MARGIN*n/100) for name,n in zip(('EGS5','CPU'),side_max)}
    status='stop_cutoff' if cutoff.any() else 'stop_registration' if not reg['valid'] else 'stop_unestimable' if unestimable else 'stop_budget' if max(estimates.values())>MAX_MAIN_HISTORIES else 'pilot_complete'
    return dict(status=status,K=reg['K'],z_critical=reg['z_critical'],bins=rows,
        cutoff_bins=[row['name'] for row in rows if row['cutoff']],unestimable_bins=unestimable,
        known_failure_bins=[row['name'] for row in rows if row['evaluable'] and not row['D7_pass']],
        pilot_D7=dict(status='insufficient' if not coverage else 'fail' if np.any(enough&~passed) else 'pass',
            passed=int((enough&passed).sum()),failed=int((enough&~passed).sum()),insufficient=int((~enough).sum()),
            historical_2sigma_failed=int((enough&~old).sum()),minimum_evaluable=reg['minimum_evaluable'],mandatory_evaluable=bool(enough[mandatory].all())),
        main_histories=estimates,required_max_before_margin=None if estimates is None else dict(zip(('EGS5','CPU'),side_max)),
        interpretation='N estimates concern statistical eligibility. H5 is not run; registration is a proposal. Known failures are retained.')


def catalog_table_hashes(materials):
    """Use the same float64 catalog encoding as production therapy bundles.

    Raw validation exports can contain int64 line_counts. Production load()
    reconstructs float64 catalog blobs before therapy_inputs hashes attributes.
    Keep native table_hashes independent; derive expectations from the catalog.
    """
    from chatcarlo.gpu import catalog
    generators={path:catalog.generator(path,ROOT)
                for path in {d['generator_path'] for d in catalog.P8_DEFINITIONS.values()}}
    hashes={}
    for mid,m in enumerate(materials):
        cat,_=catalog.p8_material(m,str(mid),generators)
        hashes.update({field['name']:field['data_sha256'] for field in cat['fields']})
    return hashes


def prepare_cpu(energy):
    from . import a2_common
    from chatcarlo.gpu import catalog,therapy
    from chatcarlo.gpu.generate_photon_tables import ensure
    _,raw,original=a2_common.inputs();materials=[]
    for m in raw:
        # Discard any exported photon field: always regenerate using the CURRENT
        # generator, including both corrections found during H3b.
        values={k:getattr(m,k) for k in catalog.P8_DEFINITIONS if k!='photon'}
        view=SimpleNamespace(**values);materials.append(SimpleNamespace(**values,photon=ensure(view)))
    edges,cum=mono(energy);bins=np.arange(301,dtype=float)*.1
    arrays=dict(faces=np.array([0.,30.]),rho=np.array([1.]),bins=bins,strength=original['strength'],
        edges=edges,cum=cum,ids=np.array([0],np.int64),owner=np.zeros(300,np.int64),constants=original['constants'],
        sizes=np.array([1,300,150,600,1],np.int64),mass_g_cm2=np.diff(bins))
    arrays['_table_hashes']=catalog_table_hashes(materials)
    return materials,arrays,a2_common.field_geometry()


def condition(energy,ref,reg,materials,arrays,fg):
    from chatcarlo.gpu import catalog,therapy,physics_sets
    tables={str(i)+'.'+k:catalog.digest(getattr(m,k)) for i,m in enumerate(materials) for k in catalog.P8_DEFINITIONS}
    inputs={k:array_hash(v) for k,v in arrays.items() if not k.startswith('_')}
    paths=[p for p in therapy.HERE.glob('t2_*') if p.suffix in ('.h','.cpp')]+[therapy.HERE/n for n in ('photon_physics.h','g5_base.h','physics_set.h','therapy.py','physics_sets.py','generate_photon_tables.py')]
    return dict(format='a4u-h4-pilot-v1',energy_keV=energy,N=CASES[energy]['N'],cpu_seed=CASES[energy]['cpu_seed'],
        batches=100,slots=SLOTS,half=HALF,ssd=SSD,regions=REGIONS,edges=ref['edges'],cum=ref['cum'],
        physics_set=physics_sets.declaration(physics_sets.NAME,specialization=physics_sets.ELECTRON),
        legacy=False,table_hashes=tables,input_hashes=inputs,field_geometry_sha256=hashlib.sha256(bytes(fg)).hexdigest(),
        sources={p.name:sha(p) for p in paths},reference_files=ref['files'],registration=reg)


def check_condition(out,cond,*,write=False):
    out=Path(out);seal=dict(condition=cond,sha256=digest(cond));path=out/'condition.json'
    if path.exists():
        if json.loads(path.read_text())!=seal:raise ValueError('H4 resume condition mismatch')
    elif write:
        out.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(seal,indent=2,allow_nan=False)+'\n')
    return seal


def run_cpu(out,cond,materials,arrays,fg):
    from chatcarlo.gpu import therapy,physics_sets,therapy_health,contracts
    from chatcarlo.gpu.therapy_observation import descriptor
    out=Path(out);seal=check_condition(out,cond,write=True)
    driver=None;fine=[];profile=[];seconds=0.;n=cond['N']//100
    for b in range(100):
        identity=dict(condition_sha256=seal['sha256'],batch=b,start=b*n,count=n,seed=cond['cpu_seed'])
        path=out/f'batch_{b:03d}.npz';receipt=path.with_suffix('.json')
        if path.exists() or receipt.exists():
            record=json.loads(receipt.read_text())
            if record['identity']!=identity or record['sha256']!=sha(path):raise ValueError('H4 batch resume identity/SHA256 mismatch')
            with np.load(path,allow_pickle=False) as z:f=z['fine'].copy();p=z['profile_fine'].copy()
            elapsed=record['seconds']
        else:
            if driver is None:
                driver=therapy.CPU(out/'build',materials,arrays,field_geometry=fg,half=(HALF,HALF),ssd=SSD,physics_declaration=cond['physics_set'])
                physics_sets.check(driver.observed,cond['physics_set'])
            begin=time.perf_counter();r=driver.histories(n,cond['cpu_seed'],start=b*n,slots=SLOTS);elapsed=time.perf_counter()-begin
            contracts.check_observation(driver.last_observation,descriptor(arrays,fg,(HALF,HALF),SSD,cond['cpu_seed'],b*n,n,cond['physics_set']),'H4 native CPU')
            therapy_health.check(r,arrays,fg)
            f=r['groups'][:,450:1050].sum(0)*GY_PER_MEV_G/(n*.05)
            parts=np.stack([r['field'][1500+j*320:1500+(j+1)*320] for j in range(6)]).reshape(2,3,320)
            p=parts*GY_PER_MEV_G/(n*.25)
            np.savez(path,fine=f,profile_fine=p)
            receipt.write_text(json.dumps(dict(identity=identity,sha256=sha(path),seconds=elapsed),indent=2)+'\n')
        if f.shape!=(600,) or p.shape!=(2,3,320) or not math.isfinite(elapsed) or elapsed<=0:raise ValueError('H4 saved CPU batch/timing')
        fine.append(f);profile.append(p);seconds+=elapsed
    samples,_=candidate_samples(np.asarray(fine),np.asarray(profile))
    return samples,dict(wall_seconds=seconds,wall_seconds_per_history=seconds/cond['N'],scope='sum of timed CPU.histories calls; includes call allocation/observation, excludes build, table generation and file I/O')


def gate(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    # Both references must be complete/healthy before any CPU build or transport.
    refs={e:read_reference(e) for e in CASES};reports={};status='pending'
    materials,base_arrays,fg=prepare_cpu(next(iter(CASES)));prepared={}
    for energy,ref in refs.items():
        edges,cum=mono(energy);arrays={**base_arrays,'edges':edges,'cum':cum}
        reg=registration(ref);cond=condition(energy,ref,reg,materials,arrays,fg)
        check_condition(out/str(energy),cond)  # all cases, before any CPU build
        prepared[energy]=(reg,cond,arrays)
    for energy in CASES:
        ref=refs[energy];reg,cond,arrays=prepared[energy]
        samples,timing=run_cpu(out/str(energy),cond,materials,arrays,fg)
        report=analyse(ref,samples,reg);report.update(condition=cond,condition_sha256=digest(cond),
            EGS5_timing=ref['timing'],CPU_timing=timing)
        if report['main_histories'] is not None:
            report['estimated_main_wall_seconds']={k:report['main_histories'][k]*v['wall_seconds_per_history'] for k,v in (('EGS5',ref['timing']),('CPU',timing))}
        reports[str(energy)]=report;status=report['status']
        if status!='pilot_complete':break
    final=dict(status=status,cases=reports,pending_energies=[e for e in CASES if str(e) not in reports],
        registration_status=REGISTRATION_PROPOSAL['status'],H3b='closed as failure with unexplained residual; not reclassified',H5_executed=False)
    RESULT.write_text(json.dumps(final,indent=2,allow_nan=False)+'\n');return final


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    report=gate(parser.parse_args().out)
    print(json.dumps(dict(status=report['status'],pending_energies=report['pending_energies']),indent=2))
    return 0 if report['status']=='pilot_complete' else 1


if __name__=='__main__':raise SystemExit(main())
