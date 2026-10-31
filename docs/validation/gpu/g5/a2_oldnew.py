"""G5 A2-2 report only: new vs saved old GPU compared directly (EGS5 not in between); T4 vs CPU smoke."""
import io,json,subprocess,tarfile
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE.parents[3]),str(HERE)]
import a2_judge as J
from prototypes.mv_electron import analysis_p7 as p7
from prototypes.mv_electron.gpu import p7_core
ROOT=HERE.parents[3];ARCH=Path.home()/'.cache/chatcarlo/archive';RUN=HERE/'a2_run';out={}
keep=[f'results/production_{c}/result.npz' for c in 'ABC'];tables={}
with subprocess.Popen(['zstd','-dc',str(ARCH/'p7_results.tar.zst')],stdout=subprocess.PIPE) as z,tarfile.open(fileobj=z.stdout,mode='r|') as t:
    for m in t:
        if m.name in keep:
            with np.load(io.BytesIO(t.extractfile(m).read()),allow_pickle=False) as f:tables[m.name.split('/')[1][-1]]={k:f[k] for k in f.files}
for c,old in tables.items():
    lock=json.loads((ROOT/f'docs/validation/mv/p7/pilot_lock_{c}.json').read_text());s=lock['spec']
    new=np.load(RUN/f'production/{c}.npz')['table']
    st=p7.compare(p7.dose(J.p7_expand(new,J.NS[c]),s),p7.dose(old,s),lock['Dref'],s)
    rows=[(j,x['delta'],x['SEM']) for j in ('J1','J2','J3') for x in st[j]]
    out[c]=dict(rows=rows,max_abs_z=max(abs(d/e) for _,d,e in rows),max_J12_percent=max(abs(d) for j,d,_ in rows if j!='J3'),max_J3_cm=max(abs(d) for j,d,_ in rows if j=='J3'))
col=Path.home()/'.cache/chatcarlo/g5/a2/gates/collected'
for k in ('A','stack'):
    a=np.load(RUN/f'smoke_{k}.npz');b=np.load(col/f'smoke_{k}_t4.npz');s={}
    for f in ('field','bins'):
        x,y=a[f].sum(0),b[f].sum(0);m=x>0
        s[f]=dict(total_ratio=float(y.sum()/x.sum()),max_rel=float(np.max(np.abs(y[m]/x[m]-1))) if m.any() else 0.,zero_mismatch=int(np.sum((x>0)!=(y>0))))
    out['smoke_'+k]=s
(RUN/'oldnew.json').write_text(json.dumps(out,indent=1,default=float))
for k,v in out.items():print(k,{a:b for a,b in v.items() if a!='rows'})
from prototypes.mv_electron import run_p8e as e
from prototypes.mv_electron.gpu import p8f_core
g=e.layout(e.LAYERS,[1.,.26,1.85,11.35]);bat={}
with subprocess.Popen(['zstd','-dc',str(ARCH/'p8f.tar.zst')],stdout=subprocess.PIPE) as z,tarfile.open(fileobj=z.stdout,mode='r|') as t:
    for m in t:
        if m.name.startswith('p8f/results/production/batch_') and m.name.endswith('.npz'):
            with np.load(io.BytesIO(t.extractfile(m).read()),allow_pickle=False) as f:bat[int(Path(m.name).stem[-3:])]=f['p8_batches'].sum(axis=0)
old=J.stack_dose(np.array([bat[b] for b in range(100)]),J.NS['stack'],g['mass_g_cm2'],g)
new=J.stack_dose(np.load(RUN/'production/stack.npz')['table'],J.NS['stack'],g['mass_g_cm2'],g)
r=p8f_core.judge(new,old,g,dref=p8f_core.DREF);d=np.array(r['difference']);s=np.array(r['SEM'])
out['stack']=dict(difference=d,SEM=s,max_abs_z=float(np.max(np.abs(d/s))),max_percent=float(100*np.max(np.abs(d))))
out={k:{a:(b.tolist() if hasattr(b,'tolist') else b) for a,b in v.items()} for k,v in out.items()}
(RUN/'oldnew.json').write_text(json.dumps(out,indent=1,default=float));print('stack',out['stack']['max_abs_z'],out['stack']['max_percent'])
for c in 'ABC':print(c,[round(d/e,2) for _,d,e in out[c]['rows']])
