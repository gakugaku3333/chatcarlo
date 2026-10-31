"""G5 A2-2 host judgement of collected production batches (Claude). Writes a2_run/verdict.json and 100-batch tables."""
import json
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE.parents[3]),str(HERE)]
import a2_judge as J
import a2_common as common
from prototypes.mv_electron import run_p8e as e
COL=Path.home()/'.cache/chatcarlo/g5/a2/gates/collected/production';OUT=HERE/'a2_run/production'


def load(c,retry):
    d=COL/f'{c}_{J.SEEDS[c][int(retry)]}';paths=sorted(d.glob('batch_*.npz'))
    return paths,[json.loads(p.with_suffix('.json').read_text()) for p in paths]


def main(retried=()):
    out={};g=e.layout(e.LAYERS,[1.,.26,1.85,11.35])
    ref,_=e.load_egs_result(common.ROOT/'docs/validation/mv/p8e',Path.home()/'.cache/chatcarlo/p8e/egs',g,[1.,.26,1.85,11.35])
    for c in ('A','B','C','stack'):
        for retry in ((False,True) if c in retried else (False,)):
            paths,rows=load(c,retry)
            if c=='stack':
                r=J.analyse_stack(paths,rows,g,ref,retry=retry);el=J.retry_eligible(r,kind='stack',retried=retry)
                bad=[i for i in range(19) if abs(r['difference'][i])>.01 or r['SEM'][i]>.0033]
                sm=dict(status=r['status'],max_abs_difference=float(np.max(np.abs(r['difference']))),max_SEM=float(np.max(r['SEM'])),off_blocks=bad,difference=list(map(float,r['difference'])),SEM=list(map(float,r['SEM'])))
            else:
                with np.load(common.ROOT/f'docs/validation/mv/p7/egs5/main_egs_{c}/result.npz',allow_pickle=False) as z:rf={k:z[k] for k in z.files}
                r=J.analyse_p7(paths,rows,c,rf,retry=retry);s=r['statistics'];el=J.retry_eligible(s,kind='p7',retried=retry)
                sm=dict(statistics=s,symmetry=r['symmetry'])
            sm['retry_eligible']=bool(el);out[f'{c}{"_retry" if retry else ""}']=sm
            bs=[np.load(p) for p in paths]
            np.savez_compressed(OUT/f'{c}{"_retry" if retry else ""}.npz',table=np.stack([b['bins' if c=='stack' else 'field'] for b in bs]))
    (HERE/'a2_run/verdict.json').write_text(json.dumps(out,indent=1,default=lambda v:v.item() if hasattr(v,'item') else v.tolist()))
    for k,v in out.items():
        print(k,v.get('status'),v.get('max_abs_difference'),v.get('max_SEM'),v.get('off_blocks'),'retry',v['retry_eligible'])
        if 'statistics' in v:
            for j,l in v['statistics'].items():
                if isinstance(l,list):
                    for x in l:print(' ',j,{a:b for a,b in x.items() if a in('name','delta','SEM','status','classification','verdict')} if isinstance(x,dict) else x)
                else:print(' ',j,str(l)[:200])
            print('  sym',str(v['symmetry'])[:300])


if __name__=='__main__':main(tuple(sys.argv[1:]))
