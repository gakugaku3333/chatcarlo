"""Explicit CPU H3b launcher; import/--help never builds or transports.

Reads the saved EGS5 histogram only. No EGS5, CUDA or shell-injection runs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from chatcarlo.gpu import core

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
REFERENCE=ROOT/'docs/egs5_crosscheck/fluorescence/fluor_on/egs5job.out'
REFERENCE_SHA256='a23ffc257cf53c314d764f2f9581111abbcb253202831a16f26346991f6769b8'
RESULT=HERE/'a4u_h3b.json'
REFERENCE_N=1000000
BANDS=((9,10),(10,11),(11,12),(12,13),(14,16),(9,16),(72,86))
ARMS=dict(production=dict(fault=0,histories=4000000,batch_size=200000,seed=1),
          CK_off=dict(fault=1,histories=1000000,batch_size=50000,seed=1),
          f13_zero=dict(fault=2,histories=1000000,batch_size=50000,seed=1))
BEAM=dict(origin=[-5.025,0.,0.],direction=[1.,0.,0.],energy_keV=100.)
GEOMETRY=dict(center=[0.,0.,0.],size_cm=[.05,100.,100.],background='air',
              world=[0.,0.,0.,50.025,100.,100.],detector=None)
SCOPE=('Relaxation component comparison only; does not independently validate '
       'the four D13 detector ROI quantities. Shell injection passed in S2; '
       'its single CUDA check belongs to H6.')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read_reference(path):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=REFERENCE_SHA256:raise ValueError('EGS5 reference SHA256 mismatch')
    text=raw.decode();marker='Escaped-photon energy histogram (both directions), 1 keV bins:'
    if text.count(marker)!=1:raise ValueError('EGS5 histogram heading')
    lines=text.split(marker)[1].splitlines()[2:];rows=[]
    for line in lines:
        fields=line.split()
        if len(fields)!=2:break
        try:edge,count=map(float,fields)
        except ValueError:break
        rows.append((edge,count))
    values=np.asarray(rows,float)
    if values.shape!=(120,2) or not np.allclose(values[:,0],np.arange(1,121)/1000,rtol=0,atol=1e-12):raise ValueError('EGS5 requires 120 one-keV bins')
    counts=values[:,1]
    if not np.isfinite(counts).all() or np.any(counts<0) or np.any(counts!=np.floor(counts)):raise ValueError('EGS5 histogram counts')
    total=re.search(r'Total escaped-photon count=\s*([0-9.]+)',text)
    if total is None or counts.sum()!=float(total[1]):raise ValueError('EGS5 escaped count mismatch')
    return dict(counts=counts.tolist(),escaped=int(counts.sum()),histories=REFERENCE_N,
                sha256=REFERENCE_SHA256,path=str(REFERENCE.relative_to(ROOT)),
                CK='used in saved run; established in A4U_H0.md (8)',
                uncertainty='Poisson count SEM; K fraction uses binomial SEM. Saved EGS5 output has no per-history moments.')


def make_arrays():
    from chatcarlo.kernel import bake_scene_materials
    a=core.pack(bake_scene_materials(['air','lead']))
    # Vacuum outside the foil, exactly as tests/test_gpu_g5_a4u_s2.py does.
    # Air would attenuate the 10 keV L lines by ~26% over the 50 cm flight to
    # the world edge (EGS5's geometry has no medium outside the slab).
    a['data'][int(a['offsets'][19])]=0.
    a.update(centers=np.array([GEOMETRY['center']],float),halves=np.array([GEOMETRY['size_cm']],float)/2,
             materials=np.array([1],np.int64),background=np.array(0),
             world=np.array(GEOMETRY['world']),detector=np.array([]))
    return a


def conditions(arrays,reference):
    inputs={name:dict(sha256=core.array_sha(value),shape=list(value.shape),dtype=str(value.dtype)) for name,value in sorted(arrays.items())}
    sources={name:hashlib.sha256((core.HERE/name).read_bytes()).hexdigest() for name in (*core.SOURCES,'core.py','physics_sets.py')}
    return dict(format='a4u-h3b-v1',device='cpu',rng='philox',cpu_flags=list(core.CPU_FLAGS),
                physics_set=core.physics_sets.declaration(),material_names=['air','lead'],
                beam=BEAM,geometry=GEOMETRY,arms=ARMS,bands=[list(b) for b in BANDS],
                acceptance=dict(sigma=2,relative=.05,fault_self_2SEM_relative_max=.05),
                inputs=inputs,bundle_sha256=digest(inputs),sources=sources,
                reference=dict(sha256=reference['sha256'],histories=reference['histories']),
                fluorescence=True,legacy=False)


def valid_spectrum(values):
    v=np.asarray(values,float)
    if v.shape!=(10001,) or not np.isfinite(v).all() or np.any(v<0) or np.any(v!=np.floor(v)):
        raise ValueError('H3b escaped_spectrum must contain 10001 integer counts')
    return v


def run_arm(out,arrays,name,condition_sha):
    spec=ARMS[name];n=spec['histories'];size=spec['batch_size'];path=out/name;path.mkdir(exist_ok=True)
    driver=None;histograms=[]
    for start in range(0,n,size):
        count=min(size,n-start)
        identity=dict(condition_sha256=condition_sha,arm=name,fault=spec['fault'],seed=spec['seed'],start=start,histories=count)
        filename=path/f'batch_{start//size:04d}.json'
        if filename.exists():
            saved=json.loads(filename.read_text())
            if saved['identity']!=identity:raise ValueError('H3b batch resume identity mismatch')
            histogram=valid_spectrum(saved['escaped_spectrum'])
        else:
            if driver is None:driver=core.Core(arrays,'cpu',build_dir=path/'build',_test_photon_fault=spec['fault'])
            origins=np.broadcast_to(BEAM['origin'],(count,3)).copy()
            directions=np.broadcast_to(BEAM['direction'],(count,3)).copy()
            energies=np.full(count,BEAM['energy_keV'])
            result=driver.run(origins,directions,energies,seed=spec['seed'],start=start,fluorescence=True)
            histogram=valid_spectrum(result['escaped_spectrum'])
            if histogram.sum()!=result['endings'][1] or result['completed']!=count:raise ValueError('H3b escaped photon/history conservation')
            filename.write_text(json.dumps(dict(identity=identity,escaped_spectrum=histogram.tolist()),allow_nan=False)+'\n')
        histograms.append(histogram)
    return np.asarray(histograms)


def compare(histograms,arm,reference):
    spec=ARMS[arm];h=np.asarray(histograms,float)
    expected=spec['histories']//spec['batch_size']
    if h.shape!=(expected,10001) or expected<2:raise ValueError('H3b outer-batch shape')
    for row in h:valid_spectrum(row)
    ref=np.asarray(reference['counts'],float);escaped=h.sum(axis=1);rows={}
    if escaped.sum()<=0:raise ValueError('H3b no escaped photons')
    for lo,hi in BANDS:
        counts=h[:,lo:hi].sum(axis=1);rc=float(ref[lo:hi].sum())
        if (lo,hi)==(72,86):
            mean=float(counts.sum()/escaped.sum());rmean=rc/reference['escaped']
            influence=(counts-mean*escaped)/spec['batch_size']
            sem=float(influence.std(ddof=1)/np.sqrt(expected)/(escaped/spec['batch_size']).mean())
            rsem=float(np.sqrt(rmean*(1-rmean)/reference['escaped']))
            quantity='fraction of all escaped photons (observation B)'
        else:
            rates=counts/spec['batch_size'];mean=float(rates.mean());sem=float(rates.std(ddof=1)/np.sqrt(expected))
            rmean=rc/reference['histories'];rsem=float(np.sqrt(rc)/reference['histories'])
            quantity='escaped photons per source history'
        delta=mean-rmean;combined=float(np.hypot(sem,rsem))
        rows[f'{lo}-{hi}']=dict(band=[lo,hi],quantity=quantity,ChatCarlo=mean,ChatCarlo_SEM=sem,
            EGS5=rmean,EGS5_SEM=rsem,ChatCarlo_count=int(counts.sum()),EGS5_count=int(rc),
            difference=delta,relative_difference=delta/rmean,SEM=combined,
            z=delta/combined if combined else None,threshold=max(2*combined,.05*rmean),
            pass_=bool(abs(delta)<=max(2*combined,.05*rmean)))
    report=dict(bands=rows,spectrum_comparison_status='pass' if all(row['pass_'] for row in rows.values()) else 'fail',
                histograms_0_120_keV=h[:,:120].tolist(),histories=spec['histories'],seed=spec['seed'],fault=spec['fault'],
                escaped=int(escaped.sum()),precision_note='ChatCarlo SEM from outer batches, including secondary photon correlations')
    if arm=='production':report['status']=report['spectrum_comparison_status']
    else:
        alpha=rows['10-11'];total=rows['9-16']
        relative_2sem=2*alpha['ChatCarlo_SEM']/alpha['ChatCarlo'] if alpha['ChatCarlo']>0 else None
        enough=relative_2sem is not None and relative_2sem<.05
        report.update(self_2SEM_relative_10_11=relative_2sem,precision_sufficient=enough,
                      detected_10_11=not alpha['pass_'],L_total_pass=total['pass_'],
                      discrimination_basis='10-11 only; L total is an agreement/prediction check, not discrimination evidence')
        report['status']='insufficient' if not enough else 'pass' if not alpha['pass_'] and total['pass_'] else 'fail'
    return report


def gate(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    reference=read_reference(REFERENCE);arrays=make_arrays();condition=conditions(arrays,reference);sha=digest(condition)
    seal=dict(condition=condition,condition_sha256=sha);path=out/'condition.json'
    if path.exists():
        if json.loads(path.read_text())!=seal:raise ValueError('H3b resume condition mismatch')
    else:path.write_text(json.dumps(seal,indent=2,allow_nan=False)+'\n')
    reports={};status='pending'
    for arm in ARMS:
        reports[arm]=compare(run_arm(out,arrays,arm,sha),arm,reference)
        status=reports[arm]['status']
        if status!='pass':break  # no additional measurement after a failed prerequisite
    final=dict(status=status,condition=condition,condition_sha256=sha,reference=reference,arms=reports,
               pending_arms=[arm for arm in ARMS if arm not in reports],scope=SCOPE,
               fault_count_policy='fixed 1M each; no adaptive increase or threshold changes',
               output=str(RESULT))
    RESULT.write_text(json.dumps(final,indent=2,allow_nan=False)+'\n')
    return final


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True,help='condition and per-batch resume directory')
    args=parser.parse_args();report=gate(args.out)
    print(json.dumps({k:report[k] for k in ('status','condition_sha256','pending_arms')},indent=2))
    return 0 if report['status']=='pass' else 1


if __name__=='__main__':raise SystemExit(main())
