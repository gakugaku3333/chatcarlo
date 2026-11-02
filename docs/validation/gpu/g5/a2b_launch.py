"""Claude-side single-session B3/B5 launcher. No automatic public entry switch.

prepare creates four reviewable previews. After human 3D approval, execute owns
ONE T4 and runs cold/cached old/new CLI adapters; batches resume on the same seal.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
import uuid
from a3_t4_gates import Owner,evidence
from a3_entry_speed import SharedClient,ratio,cache_used,SITE_CUSTOMIZE,timed_main
import a2b_legacy_adapter as legacy
import a2b_judge
from chatcarlo.gpu import seed_entry,seed_colab,seed_condition,contracts,therapy

# A2b uses RawModule, whereas A3 used RawKernel. Record the actual module compile
# duration and NVRTC calls; normalize the row kind only inside cache_proof below.
INSTRUMENT=SITE_CUSTOMIZE+'''
try:
 raw_module=cp.RawModule
 class TimedModule(raw_module):
  def get_function(self,*args,**kw):
   begin=time.perf_counter()
   try:return super().get_function(*args,**kw)
   finally:
    rows.append(dict(kind='RawModule.compile',seconds=time.perf_counter()-begin));save()
 cp.RawModule=TimedModule
except Exception as exc:
 rows.append(dict(kind='instrument_error',error=str(exc)));save()
'''


class SeedSharedClient(SharedClient):
    def status_code(self,code):
        return ('snapshot.tar' in code and 'G5_ENTRY ' in code and 'Popen' not in code) or super().status_code(code)


def cache_proof(a,b):
    import copy
    a,b=copy.deepcopy(a),copy.deepcopy(b)
    for record in (a,b):
        for row in record['compiler'].get('rows',[]):
            if row['kind']=='RawModule.compile':row['kind']='RawKernel.compile'
    return cache_used(a,b)


def prepare(scene,bundle,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for version,main in (('old',legacy.main),('new',seed_entry.main)):
        for phase in ('cold','cached'):
            d=out/(version+'_'+phase)
            if not d.exists():main(['preview',str(scene),'--bundle',str(bundle),'--out',str(d)])
    return dict(previews=[str(out/(v+'_'+p)/'preview.html') for v in ('old','new') for p in ('cold','cached')],next='Review 3D, then approve with the matching old/new CLI; execute only after all four approvals.')


def measure_kernel(owner,worker,out):
    code=f"""import pathlib,subprocess,sys
r=pathlib.Path({worker!r});script=r/'kernel_measure.py'
script.write_text("import sys;sys.path.insert(0,"+repr(str(r/'source'))+");import cupy as cp;import a2b_speed;from chatcarlo.gpu import seed_bundle,contracts\\nm,i,_=seed_bundle.load("+repr(str(r/'bundle'))+")\\nr=a2b_speed.kernel(cp,m,i)\\ncontracts.write("+repr(str(r/'kernel.json'))+",r)\\n"+"from pathlib import Path;Path("+repr(str(r/'kernel.done'))+").write_text('done')\\n")
with (r/'kernel.log').open('wb') as log:
 p=subprocess.Popen([sys.executable,str(script)],cwd=r/'source',stdout=log,stderr=log,start_new_session=True)
(r/'kernel.pid').write_text(str(p.pid))
"""
    owner.raw(code)
    while True:
        state=owner.json(f"""import pathlib,json
r=pathlib.Path({worker!r});pid=int((r/'kernel.pid').read_text())
try:alive=(pathlib.Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
except OSError:alive=False
print('A3_JSON '+json.dumps(dict(done=(r/'kernel.done').exists(),alive=alive,log=(r/'kernel.log').read_text()[-2000:])))
""")
        if state['done']:
            owner.call('download',worker+'/kernel.json',str(out/'kernel_speed.json'))
            return json.loads((out/'kernel_speed.json').read_text())
        if not state['alive']:raise RuntimeError('B5 kernel failed: '+state['log'])
        time.sleep(60)


def execute(scene,bundle,reference,out):
    out=Path(out);therapy.space(out)
    # Reference must be frozen and independently reviewed before any production.
    ref=json.loads(Path(reference).read_text())
    if ref['batches']!=600 or ref['deltas']!=a2b_judge.DELTAS:raise ValueError('frozen six-run B3 reference required')
    for row in ref['runs']:
        if a2b_judge.sha(row['path'])!=row['sha256']:raise ValueError('reference changed')
    for version in ('old','new'):
        for phase in ('cold','cached'):
            d=out/(version+'_'+phase)
            with patch.object(seed_condition,'condition',legacy.condition if version=='old' else legacy.REAL_CONDITION):
                g=seed_condition.verify(d);contracts.require(d,g['condition'],seed_condition.verify)
    remote='/content/a2b-'+uuid.uuid4().hex;owner=Owner('g5-a2b-'+uuid.uuid4().hex[:10],out);watch=None
    heartbeat=out/'heartbeat.json';real_call=owner.call
    def call(*args,**kw):
        contracts.write(heartbeat,dict(timestamp=time.time(),operation=args[0]));return real_call(*args,**kw)
    owner.call=call
    contracts.write(heartbeat,dict(timestamp=time.time()))
    watch=subprocess.Popen([sys.executable,'-m','chatcarlo.gpu.therapy_colab','watchdog',str(heartbeat)],start_new_session=True)
    rows={};proof={};roots={};kernel=None
    try:
        owner.call('new','--gpu','T4')
        for phase in ('cold','cached'):
            for version,main in (('old',legacy.main),('new',seed_entry.main)):
                # Seed driver chooses its unique worker root. A3 instrumentation
                # follows all launched worker roots, using one fixed cache/version.
                instrument=remote+'/'+version+'_'+phase
                client=SeedSharedClient(owner,instrument,remote+'/cache/'+version)
                owner.raw(f"from pathlib import Path\np=Path({instrument!r});(p/'instrument').mkdir(parents=True,exist_ok=False)\n(p/'instrument/sitecustomize.py').write_text({INSTRUMENT!r})\n")
                before=evidence(owner,client)
                d=out/(version+'_'+phase);argv=['run',str(scene),'--bundle',str(bundle),'--out',str(d)]
                if version=='old':elapsed=timed_main(lambda a:legacy.main(a,client=client),argv)
                else:
                    real_run=seed_colab.run
                    with patch.object(seed_colab,'run',lambda *a,**kw:real_run(*a,**kw,client=client)):
                        elapsed=timed_main(main,argv)
                after=evidence(owner,client);roots[version+'_'+phase]=sorted(client.worker_roots)
                if version=='old' and phase=='cold':
                    candidates=[r for r in client.worker_roots if '/g5-seed-' in r]
                    if len(candidates)!=1:raise ValueError('could not identify B5 worker PID root')
                    kernel=measure_kernel(owner,candidates[0],out)
                    if kernel['status']!='pass':raise RuntimeError('B5 kernel exceeds 1.0571; stop and report')
                rows[version+'_'+phase]=dict(total_s=elapsed,cache_dir=client.cache,cache_before=before['files'],cache_after=after['files'],compiler=after['compiler'],worker_roots=roots[version+'_'+phase])
                if phase=='cached':proof[version]=cache_proof(rows[version+'_cold'],rows[version+'_cached'])
                contracts.write(out/'entry_speed.json',dict(rows=rows,cache=proof,status='running'))
        ratios={p:ratio(rows['old_'+p]['total_s'],rows['new_'+p]['total_s']) for p in ('cold','cached')}
        speed=dict(status='pass' if all(r['status']=='pass' for r in ratios.values()) and all(p['status']=='pass' for p in proof.values()) else 'undetermined' if any(p['status']!='pass' for p in proof.values()) else 'fail',rows=rows,cache=proof,ratios=ratios)
        contracts.write(out/'entry_speed.json',speed)
        verdict=a2b_judge.judge(reference,out/'new_cached/results/summary.json',out/'B3.json')
        return dict(B3=verdict,B5_entry=speed,B5_kernel=kernel)
    finally:
        try:owner.call('stop')
        finally:
            watch.terminate()
            try:watch.wait(timeout=10)
            except subprocess.TimeoutExpired:watch.kill();watch.wait(timeout=10)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','execute'));p.add_argument('--scene',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--reference',type=Path)
    a=p.parse_args()
    if a.command=='execute' and a.reference is None:p.error('--reference required')
    result=prepare(a.scene,a.bundle,a.out) if a.command=='prepare' else execute(a.scene,a.bundle,a.reference,a.out)
    print(json.dumps(result,indent=2))
