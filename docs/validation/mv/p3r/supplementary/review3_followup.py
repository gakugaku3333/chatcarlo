"""Follow-up to the independent review of run_p3r.py's analysis part (reviews/2026-10-12-mv-p3r-review3-codex-analysis-code.md).
run_p3r.py is left frozen (its hash is part of every saved config). This script
 1/2. checks the frozen regions and the identity of every analysed run,
 3.   recomputes the relative-difference SEM with the missing a/b factor,
 4.   recomputes the G3 SEM with the n-1 variance,
 5.   reports the energy that left the slab.
Run from the repository root: PYTHONPATH=. .venv/bin/python docs/validation/mv/p3r/supplementary/review3_followup.py"""
import json
import numpy as np
import prototypes.mv_electron.run_p3r as R
from prototypes.mv_electron.p2_common import sha, ROOT
base,p3s,OUT=R.base,R.p3s,R.OUT
out={}
# 1. frozen regions: the lock must be the file hashed by the P3s analysis (2026-10-11) and by this analysis.
lock=base.OUT/'pilot_lock.json';h=sha(lock)
earlier=json.loads((p3s.OUT/'analysis.json').read_text())['input_hashes']
this=json.loads((OUT/'analysis.json').read_text())['input_hashes']
key=[k for k in earlier if k.endswith('pilot_lock.json')][0]
out['lock']=dict(sha256=h,same_as_P3s_analysis=earlier[key]==h,same_as_P3r_analysis=this['lock']==h)
# 2. identity of the analysed runs.
frozen=json.loads((OUT/'checks.json').read_text())['source_hashes'];runs={}
for name,folder in (('p3r_egs','egs5'),('p3r_proto','runs'),('p3r_proto_local','runs')):
    c=json.loads((OUT/folder/name/'config.json').read_text());r=R.load(OUT/folder/name/'result.npz');want=R.RUNS[name]
    runs[name]=dict(N=int(r['N'])==c['N']==want['N'],seed=int(r['seed'])==c['seed']==want['seed'],
        spectrum=c['spectrum_sha256']==R.SPECTRUM_SHA==sha(R.SPECTRUM),sources_as_checked=c['source_hashes']==frozen,
        mode=('brems_mode' not in want) or (int(r['brems_mode'])==c['brems_mode']==want['brems_mode']),
        result_hash_as_analysed=sha(OUT/folder/name/'result.npz')==this[{'p3r_egs':'egs','p3r_proto':'proto','p3r_proto_local':'local'}[name]])
for name,seed in (('p3a_egs',20291420),('p3a_egs_seed2',20291422)):
    c=json.loads((base.OUT/'egs5'/name/'config.json').read_text());r=R.load(base.OUT/'egs5'/name/'result.npz')
    runs[name]=dict(N=int(r['N'])==c['N']==10000000,seed=int(r['seed'])==c['seed']==seed,spectrum=c['spectrum_sha256']==R.SPECTRUM_SHA,
                    pegs_input_same_as_p3r=c['input']==json.loads((OUT/'egs5/p3r_egs/config.json').read_text())['input'])
c=json.loads((p3s.OUT/'runs/p3s_own/config.json').read_text());r=R.load(p3s.OUT/'runs/p3s_own/result.npz')
runs['p3s_own']=dict(N=int(r['N'])==c['N']==10000000,seed=int(r['seed'])==c['seed']==20291530,spectrum=c['spectrum_sha256']==R.SPECTRUM_SHA,rule=str(r['rule'])=='own')
out['runs']=runs;out['all_identity_checks_pass']=bool(all(all(v.values()) for v in runs.values()) and all(v for k,v in out['lock'].items() if k!='sha256'))
# 3. relative difference with the a/b factor.
b=json.loads((OUT/'analysis.json').read_text())['brems_per_history']
def rel(x,y):
    q=x['value']/y['value'];return dict(relative_pct=100*(q-1),SEM_pct=100*q*float(np.hypot(x['SEM']/x['value'],y['SEM']/y['value'])))
out['brems_prototype_vs_EGS5']=dict(count=rel(b['prototype']['count'],b['EGS5']['count']),energy=rel(b['prototype']['energy_MeV'],b['EGS5']['energy_MeV']))
# 4. G3 with the n-1 variance.
g=json.loads((OUT/'checks.json').read_text())['checks']['G3'];f=np.sqrt(g['N']/(g['N']-1))
out['G3']=dict(dN_SEM=g['dN_SEM']*f,z_N=g['z_N']/f,dE_SEM_MeV=g['dE_SEM_MeV']*f,z_E=g['z_E']/f)
# 5. energy leaving the slab, as a fraction of the incident energy (ratio SEM over 100 batches).
proto,local=R.load(OUT/'runs/p3r_proto/result.npz'),R.load(OUT/'runs/p3r_proto_local/result.npz');cont=R.load(p3s.OUT/'runs/p3s_own/result.npz')
esc={k:base.ratio_sem(r['batch_totals'][:,2],r['batch_totals'][:,0]) for k,r in (('prototype',proto),('prototype_local',local),('prototype_continuous',cont))}
for k,path in (('EGS5',OUT/'egs5/p3r_egs/result.npz'),('EGS5_local_1',base.OUT/'egs5/p3a_egs/result.npz'),('EGS5_local_2',base.OUT/'egs5/p3a_egs_seed2/result.npz')):
    r=R.load(path);dep=r['coarse'].sum(axis=1)*.2*(int(r['N'])/100)/R.MEV_TO_GY_G
    # EGS5 prints the escape total only for the whole run; per batch it is incident - deposited (rest-mass terms cancel to < 1e-9).
    esc[k]=base.ratio_sem(r['batchin']-dep,r['batchin']);esc[k]['whole_run_escape_over_incident_from_ledger']=float(r['totals'][2]/r['totals'][0])
out['escaped_over_incident']=esc
print(json.dumps(R.plain(out),ensure_ascii=False,indent=1))
