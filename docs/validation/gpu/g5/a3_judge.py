"""G5 A3 host judgement (Claude): H3 10 MV via the P7 -> P3b-3 crop, H4 4 MV via P7 condition C.

Both verify EGS5 input hashes, N, seed and physics set before any statistic. Existing thresholds only."""
import json
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path[:0]=[str(ROOT),str(HERE)]
import a2_judge as J
from chatcarlo.gpu import contracts,physics_sets,therapy,therapy_bundle,therapy_entry,therapy_inputs,therapy_layout,therapy_run
from prototypes.mv_electron.entry import compare
from prototypes.mv_electron.gpu import p5b_core as core,p7_core as h
from prototypes.mv_electron import run_p3b3 as p3
P3=ROOT/'docs/validation/mv/p3b'
EXPECTED=dict(mohan10=dict(N=46000000,seed=20261101,retry=20261111),mohan4=dict(N=26000000,seed=20261102,retry=20261112))
REFS=json.loads((HERE/'a3_refs.json').read_text())


def verify_files(root,hashes):
    for name,value in hashes.items():
        if therapy_bundle.sha(Path(root)/name)!=value:raise ValueError(f'reference hash mismatch: {name}')


def stacked_p6(field,N):
    """100 x 3420 P7 MeV -> the P3b-3 regions; the only path from P7 to the 10 MV comparison."""
    x=np.asarray(field)
    if x.shape!=(100,3420) or not np.isfinite(x).all() or np.any(x<0):raise ValueError('100 x 3420 outer field batches required')
    return dict(h.unpack_regions(therapy_layout.crop_p6(x),'p6'),N=np.array(N),nbatch=np.array(100))


def egs10():
    verify_files(P3,REFS['p3b'])
    lock=json.loads((P3/'pilot_lock_p3b3.json').read_text())
    if lock['N_main']!=46000000:raise ValueError('pilot lock N')
    egs=core.read_npz(P3/'egs5/main_egs/result.npz')
    summary=json.loads((P3/'egs5/main_egs/summary.json').read_text())
    return p3.validate(egs,'egs',46000000,20291763,summary),lock['Dref']


def judge10(field,N=46000000):
    egs,dref=egs10();return compare.statistics(stacked_p6(field,N),egs,dref)


def judge4(field,N=26000000):
    verify_files(ROOT/'docs/validation/mv/p7',REFS['p7C'])
    with np.load(ROOT/'docs/validation/mv/p7/egs5/main_egs_C/result.npz',allow_pickle=False) as z:ref={k:z[k] for k in z.files}
    return J.p7_judge(field,N,'C',ref)


def load(run_dir):
    """Re-run the entry's approval, batch and receipt checks; nothing unverified reaches a statistic."""
    run=Path(run_dir);cond=json.loads((run/'condition.json').read_text())
    physics_sets.check(cond['physics_set'],physics_sets.declaration(therapy.NAME))
    e=EXPECTED[cond['spectrum']]
    if cond['histories']!=e['N'] or cond['seed'] not in (e['seed'],e['retry']):raise ValueError('N or seed differs from the registered gate')
    contracts.require(run,cond,therapy_entry.verify)
    _,arrays,fg,_,_=therapy_inputs.load(run/'bundle',cond['spectrum'])
    therapy_run.collect(run/'results',cond,arrays,fg)
    with np.load(run/'results/dose.npz',allow_pickle=False) as z:return cond,z['field_MeV']


def require_retry(cond,first_path):
    """The retry seed counts only after a saved first verdict with exactly one accidental failure."""
    e=EXPECTED[cond['spectrum']]
    if cond['seed']==e['seed']:return
    first=json.loads(Path(first_path).read_text())
    if first['seed']!=e['seed'] or first['spectrum']!=cond['spectrum'] or first['retry_eligible'] is not True:raise ValueError('retry not eligible')


def main(run_dir,out,first=None):
    cond,field=load(run_dir);require_retry(cond,first)
    r=judge10(field) if cond['spectrum']=='mohan10' else judge4(field)
    stats=r['comparison'] if 'comparison' in r else r['statistics']
    r.update(spectrum=cond['spectrum'],seed=cond['seed'],
             retry_eligible=cond['seed']==EXPECTED[cond['spectrum']]['seed'] and J.retry_eligible(stats,kind='p7'))
    Path(out).write_text(json.dumps(r,indent=1,ensure_ascii=False,default=lambda v:v.item() if hasattr(v,'item') else v.tolist()))
    print(r.get('status',''),str(r.get('abstract',stats))[:800])


if __name__=='__main__':main(*sys.argv[1:])
