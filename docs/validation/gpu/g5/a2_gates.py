"""G5 A2-2 on one T4 session (Claude). Host: launch/collect/judge. VM: --vm stages in order.

checks (identity, cases, scorer, faults, smoke) -> speed (E3) -> production batches.
A failed check stops before statistics. Results are collected as tars at every 10-minute poll.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from types import SimpleNamespace
import numpy as np
import a2_common as common
from a0_speed import construct,measure,resources,sha,write_json

HERE=Path(__file__).resolve().parent
GPU=common.ROOT/'chatcarlo/gpu'
P7=Path.home()/'.cache/chatcarlo/p7/work/export'
ORDER=('stack','A','C','B')
SPEED=('p7A','p7B','p7C','stack')
SMOKE=dict(A=20301831,stack=830101)
T4_FAULTS=(4,5,6,7,11,12,13,14,15,16)


class T4:
    """One compiled build of the unified kernel; same launch shape as production."""
    def __init__(self,cp,materials,arrays,fg=None,half=(0.,0.),ssd=100.,*,cap=18,fault=0,check=0,blocks=40,threads=256):
        from chatcarlo.gpu import therapy,physics_sets
        self.cp,self.blocks,self.threads=cp,blocks,threads
        self.sizes=tuple(map(int,arrays['sizes']));self.half,self.ssd=half,ssd
        data,meta=therapy.pack(materials);self.declared=physics_sets.declaration(therapy.NAME);s=self.declared
        self.args=(cp.asarray(data),cp.asarray(meta),cp.asarray(np.ascontiguousarray(arrays['constants'],float)))
        self.geometry=cp.asarray(np.concatenate([arrays[k] for k in ('faces','rho','bins','strength','edges','cum')]))
        self.mapping=cp.asarray(np.r_[arrays['ids'],arrays['owner']])
        self.fg=np.uint64(0) if fg is None else cp.asarray(np.frombuffer(bytes(fg),np.uint8))
        o=('-DP5_GPU','-DP5_RNG_PHILOX','--fmad=false',f'-DP8G_STACK_CAP={cap}',f'-DT2_FAULT={fault}',f'-DP8G2_CHECK={check}',
           f'-DT2_FIELD={int(fg is not None)}',f'-DG5_SET_ID={s["id"]}',f'-DG5_SET_TAG={s["tag"]}ULL')
        src=therapy.cuda_source();self.kernels={}
        for n in ('p8g2_kernel','p8g2_init','t2_identity','t2_score_kernel'):
            self.kernels[n]=cp.RawKernel(src,n,options=o,backend='nvrtc');self.kernels[n].compile()
        self.kernel=self.kernels['p8g2_kernel']
        self.kernels['p8g2_init']((1,),(1,),self.args+(np.int32(len(materials)),));cp.cuda.Device().synchronize()
        out=cp.zeros(2,np.uint64);self.kernels['t2_identity']((1,),(1,),(out,));cp.cuda.Device().synchronize();x=cp.asnumpy(out)
        if int(x[0])!=s['id'] or int(x[1])!=s['tag']:raise ValueError('compiled identity mismatch')

    def run(self,n,seed,start=0,maxsteps=10000000,injected=None,energy=1.,history=0):
        cp=self.cp;nl,nb,nc,nf,_=self.sizes;dw=nb+nc+nf+37;iw=79+nb
        field=cp.zeros(3420);ledger=cp.zeros(15);adds=cp.zeros(3420,np.uint64);m=n if history else 1
        hd=cp.zeros((m,dw));hi=cp.zeros((m,iw),np.int64);groups=cp.zeros((100,dw));ints=cp.zeros((100,iw),np.uint64)
        absolute=cp.zeros((100,dw));nxt=cp.zeros(1,np.int32);visits=cp.zeros(n,np.int64);hits=cp.zeros(80,np.uint64)
        inj=np.uint64(0) if injected is None else cp.asarray(np.ascontiguousarray(injected,float))
        a=self.args+(self.geometry,self.mapping,*map(np.int32,self.sizes),np.uint64(seed),np.uint64(start),np.int32(n),nxt,
                     np.int32(history),np.int32(maxsteps),np.int32(0),inj,self.fg,np.float64(self.half[0]),np.float64(self.half[1]),
                     np.float64(self.ssd),np.float64(energy),field,ledger,adds,hd,hi,groups,ints,absolute,visits,hits)
        t0,t1=cp.cuda.Event(),cp.cuda.Event();t0.record();self.kernel((self.blocks,),(self.threads,),a);t1.record();t1.synchronize()
        r={k:cp.asnumpy(v) for k,v in dict(d=hd,i=hi,visits=visits,hits=hits,groups=groups,group_ints=ints,field=field,ledger=ledger,field_additions=adds).items()}
        r['sources']=np.zeros((n,6));r['kernel_s']=float(cp.cuda.get_elapsed_time(t0,t1))/1000
        return r

    def score(self,rows):
        cp=self.cp;n=len(rows);out=cp.zeros((n,3420));sums=cp.zeros((n,15))
        self.kernels['t2_score_kernel'](((n+255)//256,),(256,),(self.fg,cp.asarray(np.ascontiguousarray(rows,float)),np.int32(n),out,sums))
        cp.cuda.Device().synchronize();return cp.asnumpy(out),cp.asnumpy(sums)


def batch_health(r,arrays,fg):
    """History tensors do not fit in production: check visits and the 100 group records on the VM."""
    from prototypes.mv_electron.gpu import p8g2_core as core
    import a2_health
    nb,nc,nf=map(int,arrays['sizes'][1:4]);n=len(r['visits']);core.check_visits(r['visits'])
    for k in ('groups','group_ints','field','ledger','field_additions'):
        if not np.isfinite(r[k]).all():raise AssertionError('nonfinite '+k)
    if np.any(r['field']<0) or np.any(r['ledger']<0) or np.any(r['groups'][:,:nb+nc+nf]<0):raise AssertionError('negative score')
    a=core.reduce_records(r['groups'],r['group_ints'],nb,nc,nf,n,arrays['mass_g_cm2'])
    t=a['totals']
    if np.any(a['violations']) or np.any(a['counts'][5:7]) or t[3]:raise AssertionError('capacity/maxsteps/violations')
    if abs(t[0]-t[1:].sum()+t[5])>1e-9*t[0]:raise AssertionError('energy balance')
    if a['maxima'][0]>1e-11:raise AssertionError('history balance maximum')
    if r['hits'][60:62].any():raise AssertionError('secondary/parent lateral position')
    if fg is not None:a2_health.ledger(r,fg)
    return dict(status='pass',histories=n,max_balance=float(a['maxima'][0]),source_MeV=float(t[0]))


def reject(action):
    try:action()
    except AssertionError as exc:return str(exc)
    raise AssertionError('fault escaped detector')


def scorer(c):
    import a2_scorer
    g=common.field_geometry();rows=a2_scorer.records();actual,sums=c.score(rows);want=a2_scorer.reference(rows,g)
    if not np.allclose(actual,want,rtol=1e-12,atol=2e-14):raise AssertionError('geometry expected bin')
    for j,(o,n) in enumerate(zip(g.offsets,g.sizes)):
        if not np.allclose(actual[:,o:o+n].sum(axis=1),sums[:,j],rtol=1e-12,atol=2e-14):raise AssertionError('scorer input/ledger sum')
    return dict(status='pass',deposits=len(rows),max_abs=float(np.max(np.abs(actual-want))))


def checks(cp,materials,arrays,results,log):
    from prototypes.mv_electron.gpu import p8g2_core as core
    import a2_health
    a=common.no_spectrum(arrays);fg=common.field_geometry();g=SimpleNamespace(nb=int(a['sizes'][1]),nc=int(a['sizes'][2]),nf=int(a['sizes'][3]))
    builds={};total=np.zeros(80,np.uint64);rows=[]
    for case in core.fixed_cases():
        c=builds.setdefault(case['cap'],T4(cp,materials,a,cap=case['cap'],check=1))
        r=c.run(1,core.SEED2,start=case['start'],maxsteps=case['maxsteps'],injected=np.array([case['state']]),energy=case['state'][6],history=1)
        core.check_visits(r['visits'])
        for t in case['targets']:
            if r['hits'][t]==0:raise AssertionError(case['name']+': target '+core.PATHS[t])
        total+=r['hits'];rows.append(dict(name=case['name'],failures=int(a2_health.unpack(r,g,a)[0]['counts'][:,5:7].sum())))
    for n,i in core.TRANSITIONS.items():
        if total[i]==0:raise AssertionError('missing transition '+n)
    for i,n in enumerate(core.PATHS):
        if total[i]==0:raise AssertionError('missing path '+n)
    write_json(results/'cases_t4.json',dict(status='pass',cases=rows,paths={n:int(total[i]) for i,n in enumerate(core.PATHS)},transitions={n:int(total[i]) for n,i in core.TRANSITIONS.items()}));log('cases pass')
    low=next(x for x in core.fixed_cases() if x['name']=='low-photon');flight=next(x for x in core.fixed_cases() if x['name']=='flight')
    N=512;kw=dict(injected=np.tile(low['state'],(N,1)),energy=.007,history=1)
    def good(c):
        r=c.run(N,core.SEED2,**kw);a2_health.check(r,g,a);return r
    normal=T4(cp,materials,a,fg,check=1,blocks=1);good(normal);ok=scorer(normal);out=[]
    ann=lambda r:[(_ for _ in ()).throw(AssertionError('annihilation must occur once')) for x in a2_health.unpack(r,g,a)[0]['counts'] if int(x[12]+x[14])!=1]
    fk=dict(start=flight['start'],injected=np.array([flight['state']]),energy=flight['state'][6],history=1)
    ann(normal.run(1,core.SEED2,**fk))
    for f in T4_FAULTS:
        c=T4(cp,materials,a,fg,fault=f,check=1,blocks=1)  # 256 slots: 512 histories must reuse them
        if f in (5,6,7):msg=reject(lambda:scorer(c));hit=None
        elif f==11:
            r=c.run(1,core.SEED2,**fk);msg=reject(lambda:ann(r));hit=int(r['hits'][30])
        else:
            r=c.run(N,core.SEED2,**kw);hit=int(r['hits'][{4:64,12:31,13:32,14:33,15:34,16:35}[f]])
            msg=reject(lambda:a2_health.check(r,g,a))
        if hit==0:raise AssertionError(f'fault site not reached {f}')
        out.append(dict(fault=f,target=hit,detector=msg));log('fault',f,msg)
    write_json(results/'faults_t4.json',dict(status='pass',normal_scorer=ok,faults=out))
    # Field-off setting must pass; values are also compared with the CPU smoke tables (report only).
    stack=T4(cp,materials,arrays);res={}
    for cond,seed in SMOKE.items():
        c=stack if cond=='stack' else None;w=None
        if cond=='A':
            w=load_water('A');c=T4(cp,materials,w['arrays'],w['fg'],w['half'],w['ssd'])
        bins=[];field=[];aa=arrays if cond=='stack' else w['arrays']
        for b in range(100):
            r=c.run(1000,seed,start=b*1000);batch_health(r,aa,None if cond=='stack' else common.field_geometry())
            bins.append(r['groups'][:,:int(aa['sizes'][1])].sum(axis=0));field.append(r['field'])
        res[cond]=(np.array(field),np.array(bins))
        np.savez_compressed(results/f'smoke_{cond}_t4.npz',field=res[cond][0],bins=res[cond][1])
    write_json(results/'checks.json',dict(status='pass'));log('checks pass')


def load_water(cond):
    with np.load(WATER/f'water_{cond}.npz') as f:w={k:f[k].copy() for k in f.files}
    fg=w.pop('field_geometry').tobytes();half=float(w.pop('half'));ssd=float(w.pop('ssd'))
    return dict(arrays=w,fg=fg,half=(half,half),ssd=ssd)


def speed(cp,materials,arrays,bundles,results,log):
    cfg={c['id']:c for c in json.loads((HERE/'a0_configs.json').read_text())['configurations']};cfg['stack']=cfg['p8g2']
    rec=dict(status='running',resources={},rows=[]);save=lambda:write_json(results/'speed.json',rec);states={};N={k:cfg[k]['production_n'] for k in SPEED}
    tasks=[f'{v}-{c}' for c in SPEED for v in ('old','new')]
    for t in tasks:
        v,cid=t.split('-');c=cfg[cid]
        if v=='old':states[t]=construct(cp,c,bundles);k=states[t]['kernel']
        else:
            if cid=='stack':states[t]=T4(cp,materials,arrays)
            else:
                w=load_water(cid[-1]);states[t]=T4(cp,materials,w['arrays'],w['fg'],w['half'],w['ssd'])
            k=states[t].kernel
        rec['resources'][t]=resources(cp,k,c);save()
    def one(t,phase):
        v,cid=t.split('-');c=cfg[cid]
        if v=='old':m=measure(c,states[t],N[cid]);row=dict(kernel_s=m['kernel_s'],health=m['health'])
        else:
            r=states[t].run(N[cid],c['seed']);batch_health(r,arrays if cid=='stack' else load_water(cid[-1])['arrays'],None if cid=='stack' else common.field_geometry())
            row=dict(kernel_s=r['kernel_s'],health='pass')
        row.update(task=t,phase=phase);rec['rows'].append(row);save();log(t,phase,row['kernel_s'])
    for t in tasks:one(t,'warmup')
    for i in range(1):  # user decision 2026-10-31: one valid round
        for cid in SPEED:
            for v in (('old','new') if i%2==0 else ('new','old')):one(f'{v}-{cid}','valid')
    get=lambda t:statistics.median(r['kernel_s'] for r in rec['rows'] if r['task']==t and r['phase']=='valid')
    rec['ratio']={c:get('new-'+c)/get('old-'+c) for c in SPEED};rec['status']='complete';save();log(rec['ratio'])
    states.clear()
    return rec


def production(cp,materials,arrays,jobs,results,log):
    from a2_judge import NS,SEEDS
    from chatcarlo.gpu.therapy_bundle import sha as fsha
    root=results/'production'
    for cond,retry in jobs:
        n=NS[cond]//100;seed=SEEDS[cond][retry];d=root/f'{cond}_{seed}';d.mkdir(parents=True,exist_ok=True)
        if cond=='stack':a=arrays;c=T4(cp,materials,arrays);fg=None
        else:
            w=load_water(cond);a=w['arrays'];c=T4(cp,materials,a,w['fg'],w['half'],w['ssd']);fg=common.field_geometry()
        for b in range(100):
            row=d/f'batch_{b:03d}.json'
            if row.exists():continue
            r=c.run(n,seed,start=b*n);h=batch_health(r,a,fg);p=d/f'batch_{b:03d}.npz'
            np.savez(p,field=r['field'],ledger=r['ledger'],bins=r['groups'][:,:int(a['sizes'][1])].sum(axis=0),groups=r['groups'],group_ints=r['group_ints'],hits=r['hits'])
            tmp=row.with_suffix('.tmp');tmp.write_text(json.dumps(dict(N=n,seed=seed,start=b*n,physics_set=c.declared,health=h,sha256=fsha(p),kernel_s=r['kernel_s']),default=lambda v:v.item()));tmp.rename(row)
        log('production',cond,seed)


def vm(a):
    global WATER
    import cupy as cp
    from prototypes.mv_electron.gpu import p8g2_colab as p8
    base=a.vm;WATER=base;results=base/'results';results.mkdir(exist_ok=True)
    log=lambda *x:print(time.strftime('%H:%M:%S'),*x,flush=True)
    _,materials,arrays=p8.load(base/'bundles/p8g2')
    if 'checks' in a.stages and not (results/'checks.json').exists():checks(cp,materials,arrays,results,log)
    if 'speed' in a.stages and not (results/'speed.json').exists():speed(cp,materials,arrays,base/'bundles',results,log)
    if 'production' in a.stages:production(cp,materials,arrays,[(j.split(':')[0],int(j.split(':')[1])) for j in a.jobs.split(',')],results,log)
    (base/'G5A2_GATES_DONE').write_text('done\n')


def launch(a):
    from a1_launch import call
    from a2_t4 import water
    from chatcarlo.gpu.colab_driver import split_bundle
    out=a.output.resolve();common.space(out,int(1.4*1024**3));out.mkdir(parents=True,exist_ok=True)
    hb=out/'heartbeat.json';write_json(hb,dict(time=time.time(),operation='preflight'))
    watch=subprocess.Popen([sys.executable,str(HERE/'a2_t4.py'),'--watchdog',str(hb),'--output',str(out)],start_new_session=True)
    base='/content/g5a2g-'+a.session;attempted=False;remote=lambda code,**kw:call(['exec','-s',a.session],hb,input=code,**kw)
    d=base+'/docs/validation/gpu/g5/'
    try:
        with tempfile.TemporaryDirectory(prefix='g5a2g-',dir=out) as t:
            tmp=Path(t)
            for c in 'ABC':water(tmp/f'water_{c}.npz',c)
            files={n:split_bundle(p,tmp/('parts_'+n)) for n,p in (('p7',P7),('p8g2',common.SOURCE))}
            scripts=[HERE/n for n in ('a2_gates.py','a2_common.py','a2_health.py','a2_scorer.py','a2_judge.py','a0_speed.py','a0_bootstrap.py','a0_configs.json')]
            runtime=[GPU/n for n in ('therapy.py','therapy_bundle.py','physics_sets.py','catalog.py','g5_base.h','physics_set.h')]+sorted(GPU.glob('t2_*'))
            water_files=[tmp/f'water_{c}.npz' for c in 'ABC']
            job=dict(files=files,scripts={p.name:sha(p) for p in scripts},runtime={p.name:sha(p) for p in runtime},water={p.name:sha(p) for p in water_files})
            write_json(tmp/'job.json',job);write_json(out/'job.json',job);attempted=True
            call(['new','-s',a.session,'--gpu','T4'],hb)
            remote(f"import pathlib\nb=pathlib.Path({base!r})\nfor s in ('chatcarlo/gpu','docs/validation/gpu/g5','parts/p7','parts/p8g2'):(b/s).mkdir(parents=True,exist_ok=True)\n")
            for p in scripts:call(['upload','-s',a.session,str(p),d+p.name],hb)
            for p in runtime:call(['upload','-s',a.session,str(p),base+'/chatcarlo/gpu/'+p.name],hb)
            for p in [*water_files,tmp/'job.json']:call(['upload','-s',a.session,str(p),base+'/'+p.name],hb)
            for n in files:
                for p in sorted((tmp/('parts_'+n)).iterdir()):call(['upload','-s',a.session,str(p),f'{base}/parts/{n}/{p.name}'],hb)
            prior=out/'collected/production'
            if prior.exists():
                with tarfile.open(tmp/'prior.tar','w') as z:z.add(prior,arcname='results/production')
                call(['upload','-s',a.session,str(tmp/'prior.tar'),base+'/prior.tar'],hb)
            code=f'''import pathlib,hashlib,json,subprocess,sys,os,tarfile
base=pathlib.Path({base!r});d=base/'docs/validation/gpu/g5'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
job=json.loads((base/'job.json').read_text())
for n,x in job['scripts'].items():assert sha(d/n)==x,n
for n,x in job['runtime'].items():assert sha(base/'chatcarlo/gpu'/n)==x,n
for n,x in job['water'].items():assert sha(base/n)==x,n
(base/'chatcarlo/__init__.py').write_text('');(base/'chatcarlo/gpu/__init__.py').write_text('')
for bundle,files in job['files'].items():
 target=base/'bundles'/bundle;target.mkdir(parents=True,exist_ok=True)
 for name,item in files.items():
  with (target/name).open('wb') as out:
   for part in item['parts']:
    src=base/'parts'/bundle/part['name'];assert sha(src)==part['sha256']
    out.write(src.read_bytes())
  assert sha(target/name)==item['sha256'],name
sys.path[:0]=[str(base),str(d)]
from a0_bootstrap import extract_sources
extract_sources(base/'bundles/p7/p7_sources.zip',base);extract_sources(base/'bundles/p8g2/p8g2_source.zip',base)
if (base/'prior.tar').exists():
 with tarfile.open(base/'prior.tar') as t:t.extractall(base)
log=(base/'worker.log').open('ab')
p=subprocess.Popen(['nohup',sys.executable,str(d/'a2_gates.py'),'--vm',str(base),'--stages',{a.stages!r},'--jobs',{a.jobs!r}],cwd=base,stdout=log,stderr=log,start_new_session=True,env=dict(os.environ,PYTHONPATH=str(base)+':'+str(d)))
(base/'worker.pid').write_text(str(p.pid));print('LAUNCHED')
'''
            print(remote(code,timeout=1800),flush=True)
        while True:
            s=json.loads(remote(f'''import pathlib,json,os,tarfile
base=pathlib.Path({base!r});r=base/'results';pid=int((base/'worker.pid').read_text())
try:
 os.kill(pid,0);alive=(pathlib.Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
except OSError:alive=False
done=(base/'G5A2_GATES_DONE').exists();sent=base/'sent.txt';old=set(sent.read_text().split()) if sent.exists() else set()
new=[p for p in r.rglob('*') if p.is_file() and not p.suffix=='.tmp' and (str(p) not in old or p.parent==r)]
with tarfile.open(base/'results.tar','w') as t:
 for p in new:t.add(p,arcname=str(p.relative_to(r)))
 t.add(base/'worker.log',arcname='worker.log')
sent.write_text(' '.join(old|{{str(p) for p in new}}))
print(json.dumps(dict(done=done,alive=alive,new=len(new))))
''').strip().splitlines()[-1])
            call(['download','-s',a.session,base+'/results.tar',str(out/'results.tar')],hb)
            with tarfile.open(out/'results.tar') as t:t.extractall(out/'collected',filter='data')
            write_json(out/'progress.json',dict(s,time=time.time()))
            common.space(out)
            if s['done']:return
            if not s['alive']:raise RuntimeError('worker died; see collected/worker.log')
            end=time.monotonic()+600
            while time.monotonic()<end:
                write_json(hb,dict(time=time.time(),operation='poll wait'));time.sleep(10)
    finally:
        try:
            if attempted:call(['stop','-s',a.session],hb)
        finally:
            watch.terminate();watch.wait(timeout=30)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--vm',type=Path);p.add_argument('--session',default='g5-a2-gates')
    p.add_argument('--stages',default='checks,speed,production');p.add_argument('--jobs',default='stack:0,A:0,C:0,B:0')
    p.add_argument('--output',type=Path,default=common.CACHE/'gates');a=p.parse_args();a.stages=a.stages
    (vm if a.vm else launch)(a)


if __name__=='__main__':main()
