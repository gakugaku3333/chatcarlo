"""Exactly 50 paired outer batches; every accepted chunk passes health and ABI checks."""
import json
from pathlib import Path
import numpy as np
from . import seed_geometry as core,seed_analysis,seed_checks,contracts,seed_condition,therapy,physics_sets
from .seed_core import observation
from .seed_stats import PairStore
from .condition import sha256
from .therapy_bundle import sha


def shapes():return dict(fine=core.FINE.shape,coarse=core.COARSE.shape,density=(3,))

def expected(cond,materials,inputs,batch,field,side):
    g,a,_=core.geometry(inputs['densities'],inputs['mohan10_edges'],inputs['mohan10_cum'],angle=core.ANGLES[field],beam='prostate',with_seeds=side)
    a['strength']=np.ascontiguousarray(inputs['strength'],float)
    args=(*therapy.pack(materials),np.ascontiguousarray(inputs['constants'],float))
    key=seed_condition.identity(cond,batch)
    return observation(g,a,args,cond['seed'],key['start'],key['count'],field,physics_sets.declaration(therapy.NAME, specialization=physics_sets.ELECTRON))


def require_gates(report):
    required={'B0','B1','V0','V1a','V1b','V1c','V2a','V2b','V3','V4_rectangle','V4_prostate','V5','V5_GPU','V6','V6b','G3_grazing','G3','G5'}
    if not isinstance(report,dict) or not required<=report.keys():raise ValueError('missing B0/B1/B2 gates')
    for k in required:
        v=report[k]
        if v.get('status')=='pass':continue
        if k=='V2b' and v.get('status')=='insufficient-statistics' and v.get('fallback')=='V2a':continue
        raise ValueError('seed gate failed '+k)


def preflight(out,cond):
    out=Path(out);path=out/'prostate.npz'
    if any(p.name!='batch_table.npz' for p in out.glob('batch_*')):raise ValueError('unexpected batch name/range')
    store=PairStore(path,cond,shapes())
    if store.completed:
        gate=json.loads((out/'gates.json').read_text())
        if gate['condition_sha256']!=sha256(cond):raise ValueError('gate condition mismatch')
        require_gates(gate['gates'])
    verify_receipts(store,cond)
    return store


def _run(out,cond,materials,inputs,driver,checks,progress):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);store=preflight(out,cond);masks=seed_analysis.load_masks()
    # Fresh VM/device always reruns checks, even for resume.
    report=checks();require_gates(report)
    contracts.write(out/'gates.json',dict(condition_sha256=sha256(cond),gates=report));progress()
    for batch in range(len(store.completed),cond['batches']):
        key=seed_condition.identity(cond,batch);sums=[[np.zeros(g.shape) for g in (core.FINE,core.COARSE)] for _ in range(2)]
        doors=set();receipts=[]
        for field in range(7):
            for side in (True,False):
                r=driver(batch,field,side,key['start'],key['count'])
                want=expected(cond,materials,inputs,batch,field,side)
                contracts.check_observation(r['observed'],want,'native seed door')
                h=seed_checks.raw(r['d'],r['i'],r['ledger'],r['source_counts'],key['start'])
                for j in range(2):
                    v=np.asarray(r['grids'][j])
                    if v.shape!=(core.FINE,core.COARSE)[j].shape or not np.isfinite(v).all() or np.any(v<0):raise ValueError('grid shape/nonfinite/negative')
                    sums[not side][j]+=v
                if abs(np.sum(r['grids'][1])-h['dep'].sum())>1e-9*h['sources'].sum():raise ValueError('coarse sum / deposited energy')
                receipts.append(dict(field=field,with_seeds=side,observed=r['observed'],health='pass',kernel_s=r['seconds'],rejected=int(r['source_counts'][:,0].sum())))
                doors.add((field,side));print('A2B_DOOR',batch,field,int(side),flush=True)
        row=np.stack([seed_analysis.densities(sums[k][0],key['accepted_per_side'],masks) for k in range(2)])
        pairs={name:tuple(sums[k][j]/key['accepted_per_side'] for k in range(2)) for j,name in enumerate(('fine','coarse'))};pairs['density']=tuple(row)
        store.commit(batch,pairs,doors,row,receipts);progress()
        print('A2B_BATCH',batch,flush=True)
    return collect(out,cond)


def collect(out,cond):
    out=Path(out);store=preflight(out,cond)
    if store.completed!=list(range(50)):raise ValueError('50 complete batches required')
    rows=np.asarray(store.rows)
    for b,receipts in enumerate(store.receipts):
        key=seed_condition.identity(cond,b)
        if {(r['field'],r['with_seeds']) for r in receipts}!=set((f,s) for f in range(7) for s in (True,False)) or len(receipts)!=14:raise ValueError('receipt pair/replay')
        for r in receipts:
            o=r['observed']
            if r['health']!='pass' or o['start']!=key['start'] or o['count']!=key['count'] or o['seed']!=cond['seed']*8+r['field']:raise ValueError('receipt history range')
            physics_sets.check(o['physics_set'],cond['physics_set'])
    result=dict(condition_sha256=sha256(cond),batches=50,histories_per_side=cond['histories_per_side'],checkpoint_sha256=sha(out/'prostate.npz'),statistics=seed_analysis.analyse(rows),density_unit='MeV/cm3 per accepted photon',near_sem='between paired outer batches; old reference near SEM assumes independent cells')
    np.savez_compressed(out/'batch_table.npz',density_rows=rows,seed=cond['seed'],accepted_per_side=np.full(50,7*cond['per_batch_per_field']),source_sha256=result['checkpoint_sha256'])
    contracts.write(out/'summary.json',result);return result

def verify_receipts(store,cond):
    for b,receipts in enumerate(store.receipts):
        key=seed_condition.identity(cond,b)
        if len(receipts)!=14 or {(r['field'],r['with_seeds']) for r in receipts}!=set((f,s) for f in range(7) for s in (True,False)):raise ValueError('receipt duplicate/missing pair')
        for r in receipts:
            if r['health']!='pass':raise ValueError('unhealthy batch')
            if 'native_templates' not in cond:raise ValueError('missing sealed native templates')
            matches=[t for t in cond['native_templates'] if (t['field'],t['with_seeds'])==(r['field'],r['with_seeds'])]
            if len(matches)!=1:raise ValueError('native templates')
            want=dict(matches[0]['observed'],start=key['start'],count=key['count'])
            contracts.check_observation(r['observed'],want,'collected native receipt')

def run(out,cond,materials,inputs,driver,checks,progress=lambda:None):
    import fcntl
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'execution.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise ValueError('seed worker already running') from exc
        return _run(out,cond,materials,inputs,driver,checks,progress)
