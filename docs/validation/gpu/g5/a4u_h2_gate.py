"""Explicit CPU-only H2 launcher. Import/--help never builds or transports.

Run only on Claude's explicit invocation with --out. Conditions and retry seeds
follow a4u_s1_gate1; the two arms are native CPU builds, never NumPy/CUDA/EGS5.
"""
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
from chatcarlo.gpu import bundle, core, validation
from chatcarlo.gpu.source_host import sample_source_photons
from . import a4u_gates as gates
from .a4u_s1_gate1 import inputs

ROOT=Path(__file__).resolve().parents[4]
SCENES=('s1','s2','s3')
N_RUNS=10
N_HISTORIES=1000000
BATCH_SIZE=200000
MODEL_NOTE=('G5_TEST_LEGACY_LOCAL=1 preserves the diagnostic K-only model, '
            'including its existing bound rejection, tabulated incoherent total '
            'and Rayleigh. Unbound analytic KN is the electron legacy path, '
            'not this diagnostic flag. No physics is modified by this launcher.')


def judge(old,new,*,retry=False):
    """Keep all 61 rows; apply the existing retry rule outside DIVERGENT."""
    if len(old)!=61 or len(new)!=61 or not set(gates.DIVERGENT)<=set(old):
        raise ValueError('H2 requires all 61 quantities, including DIVERGENT')
    report=gates.branches(old,new)
    if any(len(v)!=N_RUNS for v in old.values()):raise ValueError('H2 requires 10 outer runs')
    stats=validation.independent_judge(old,new)
    nondivergent={k:v for k,v in old.items() if k not in gates.DIVERGENT}
    regression=validation.independent_judge(nondivergent,{k:new[k] for k in nondivergent})
    for k,row in report['quantities'].items():
        stat=stats['quantities'][k]
        sem=float(np.hypot(stat['reference_SEM'],stat['twin_SEM']))
        z=stat['z']
        if sem==0 and row['difference']==0:z=0.
        row.update(K_only_SEM=stat['reference_SEM'],all_shell_SEM=stat['twin_SEM'],
                   SEM=sem,z=z,statistical_status=stat['status'])
    report['branch_status']=report['status']
    report['regression']=regression
    # A single 3<|z|<4 exception may rerun once, exactly as S1. Never turn a
    # judge's failure into pass; retain both attempts and their unexpected lists.
    if regression['status']!='pass':report['status']=regression['status']
    if retry and report['status']=='retry':report['status']='fail'
    report['retry']=retry
    return report


def sample(driver,source,spectrum,names,sid,seed,source_seed,fluorescence):
    """The same per-source-history quantities as gate1.collect, in native keV."""
    rng=np.random.default_rng(source_seed);quantities={}
    for start in range(0,N_HISTORIES,BATCH_SIZE):
        n=min(BATCH_SIZE,N_HISTORIES-start)
        photons=sample_source_photons(source,n,rng,spectrum=spectrum)
        result=driver.run(*photons,seed=seed,start=start,fluorescence=fluorescence)
        for name,value in validation.quantities(result,names,sid,N_HISTORIES).items():
            quantities[name]=quantities.get(name,0.)+value
    return quantities


def gate(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    sources={name:bundle.sha(core.HERE/name) for name in (*core.SOURCES,'core.py','validation.py','source_host.py')}
    # Bundles/builds are prepared only inside this explicitly invoked function.
    with tempfile.TemporaryDirectory(prefix='chatcarlo-a4u-h2-') as temp:
        prepared={};scene_hashes={}
        for sid in SCENES:
            scene,plane,sha=inputs(sid);path=Path(temp)/sid
            bundle.create(scene,plane,path)
            arrays,source,spectrum,settings,manifest=bundle.load(path)
            drivers={name:core.Core(arrays,'cpu',build_dir=path/name,_test_legacy_physics=legacy)
                     for name,legacy in (('legacy',True),('production',False))}
            prepared[sid]=(source,spectrum,settings,manifest,drivers)
            scene_hashes[sid]=sha
        attempts=[]
        for retry in (False,True):
            old={};new={};offset=100 if retry else 0
            for sid in SCENES:
                source,spectrum,settings,manifest,drivers=prepared[sid]
                for i in range(N_RUNS):
                    for name,values,seed in (('legacy',old,5000+offset+i),('production',new,6000+offset+i)):
                        driver=drivers[name];source_seed=7000+offset+i
                        identity=dict(scene_sha256=scene_hashes[sid],bundle=manifest,sources=sources,
                                      device='cpu',build_options=list(driver.set_options),
                                      seed=seed,source_seed=source_seed,n_histories=N_HISTORIES,
                                      batch_size=BATCH_SIZE,n_workers=1,
                                      divergence_sha256=gates.DIVERGENCE_SHA256)
                        filename=out/f'h2_{"retry" if retry else "initial"}_{sid}_{name}_{i:02d}.json'
                        if filename.exists():
                            record=json.loads(filename.read_text())
                            if record['identity']!=identity:raise ValueError('H2 resume identity mismatch')
                            q=record['quantities']
                        else:
                            q=sample(driver,source,spectrum,settings['material_names'],sid,seed,source_seed,settings['fluorescence'])
                            filename.write_text(json.dumps(dict(identity=identity,quantities=q),indent=2,allow_nan=False)+'\n')
                        for k,v in q.items():values.setdefault(sid+':'+k,[]).append(v)
            report=judge(old,new,retry=retry)
            report.update(scene_sha256=scene_hashes,
                          seeds=dict(legacy=list(range(5000+offset,5010+offset)),
                                     production=list(range(6000+offset,6010+offset)),
                                     source=list(range(7000+offset,7010+offset))),
                          n_histories_per_run=N_HISTORIES,n_runs=N_RUNS,batch_size=BATCH_SIZE,
                          outer_runs=dict(legacy=old,production=new))
            attempts.append(report)
            if report['status']!='retry':break
    final=dict(report);final.update(attempts=attempts,model_note=MODEL_NOTE,
                                    divergent=list(gates.DIVERGENT),sources=sources)
    (out/'h2.json').write_text(json.dumps(final,indent=2,allow_nan=False)+'\n')
    return final


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True,help='run/resume directory; final report is OUT/h2.json')
    args=parser.parse_args()
    report=gate(args.out)
    print(json.dumps({k:report[k] for k in ('status','unexpected','divergence_sha256')},indent=2))
    return 0 if report['status']=='pass' else 1


if __name__=='__main__':raise SystemExit(main())
