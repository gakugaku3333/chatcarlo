"""G5 A2 same-session old/new speed measurement on T4 (E4 trial; report only)."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import numpy as np
from a0_speed import construct,measure,resources,write_json

HERE=Path(__file__).resolve().parent
TRIAL=['old-p7A','new-p7A','old-stack','new-stack']
ORDER=[TRIAL,TRIAL[1::-1]+TRIAL[:1:-1]]


class NewCore:
    """CUDA launch of the unified t2 kernel with the same launch shape as the old kernels."""
    def __init__(self,cp,materials,arrays,field_geometry=None,half=(0.,0.),ssd=100.,blocks=40,threads=256,capacity=18):
        from chatcarlo.gpu import therapy,physics_sets
        self.cp,self.blocks,self.threads=cp,blocks,threads
        self.sizes=tuple(map(int,arrays['sizes']));self.half,self.ssd=half,ssd
        data,meta=therapy.pack(materials);s=physics_sets.declaration(therapy.NAME)
        self.args=(cp.asarray(data),cp.asarray(meta),cp.asarray(np.ascontiguousarray(arrays['constants'],float)))
        self.geometry=cp.asarray(np.concatenate([arrays[k] for k in ('faces','rho','bins','strength','edges','cum')]))
        self.mapping=cp.asarray(np.r_[arrays['ids'],arrays['owner']])
        self.fg=np.uint64(0) if field_geometry is None else cp.asarray(np.frombuffer(field_geometry,np.uint8))
        options=('-DP5_GPU','-DP5_RNG_PHILOX','--fmad=false',f'-DP8G_STACK_CAP={capacity}','-DT2_FAULT=0','-DP8G2_CHECK=0',
                 f'-DT2_FIELD={int(field_geometry is not None)}',f'-DG5_SET_ID={s["id"]}',f'-DG5_SET_TAG={s["tag"]}ULL')
        source=therapy.cuda_source()
        self.kernel=cp.RawKernel(source,'p8g2_kernel',options=options,backend='nvrtc');self.kernel.compile()
        init=cp.RawKernel(source,'p8g2_init',options=options,backend='nvrtc');init.compile()
        init((1,),(1,),self.args+(np.int32(len(materials)),));cp.cuda.Device().synchronize()

    def run(self,n,seed,start=0):
        cp=self.cp;nl,nb,nc,nf,_=self.sizes;dw=nb+nc+nf+37;iw=79+nb
        field=cp.zeros(3420);ledger=cp.zeros(15);adds=cp.zeros(3420,np.uint64)
        hd=cp.zeros((1,dw));hi=cp.zeros((1,iw),np.int64);groups=cp.zeros((100,dw));ints=cp.zeros((100,iw),np.uint64)
        absolute=cp.zeros((100,dw));next_=cp.zeros(1,np.int32);visits=cp.zeros(n,np.int64);hits=cp.zeros(80,np.uint64)
        args=self.args+(self.geometry,self.mapping,*map(np.int32,self.sizes),np.uint64(seed),np.uint64(start),np.int32(n),next_,
                        np.int32(0),np.int32(10000000),np.int32(0),np.uint64(0),self.fg,np.float64(self.half[0]),np.float64(self.half[1]),
                        np.float64(self.ssd),np.float64(1.),field,ledger,adds,hd,hi,groups,ints,absolute,visits,hits)
        first,last=cp.cuda.Event(),cp.cuda.Event();first.record();self.kernel((self.blocks,),(self.threads,),args);last.record();last.synchronize()
        v=cp.asnumpy(visits)
        if not np.all(v==1):raise ValueError('visits not all one')
        return dict(kernel_s=float(cp.cuda.get_elapsed_time(first,last))/1000,field_sum=float(cp.asnumpy(field).sum()),
                    groups_finite=bool(np.isfinite(cp.asnumpy(groups)).all()))


def build_new(cp,extra,cid):
    from prototypes.mv_electron.gpu import p8g2_colab as p8
    _,materials,arrays=p8.load(extra['p8g2'])
    if cid=='stack':return NewCore(cp,materials,arrays)
    with np.load(extra['water']) as f:
        w={k:f[k].copy() for k in f.files}
    fg=w.pop('field_geometry').tobytes();half=float(w.pop('half'));ssd=float(w.pop('ssd'))
    return NewCore(cp,materials,w,fg,(half,half),ssd)


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundles',type=Path);p.add_argument('--water',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--profile');a=p.parse_args()
    import cupy as cp
    cfg={c['id']:c for c in json.loads((HERE/'a0_configs.json').read_text())['configurations']}
    cfg['stack']=cfg['p8g2'];extra=dict(p8g2=a.bundles/'p8g2',water=a.water);N=200000
    if a.profile:
        build_new(cp,extra,a.profile).run(20000,cfg[a.profile]['seed']);return
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    rec=dict(status='running',environment=cp.cuda.Device().name if hasattr(cp.cuda.Device(),'name') else '',construction={},resources={},rows=[])
    save=lambda:write_json(out/'measurements.json',rec)
    states={}
    for task in TRIAL:
        version,cid=task.split('-');begin=time.perf_counter()
        try:
            if version=='old':
                c=cfg[cid if cid=='p7A' else 'p8g2'];states[task]=construct(cp,c,a.bundles);kernel=states[task]['kernel']
                rec['resources'][task]=resources(cp,kernel,c)
            else:
                states[task]=build_new(cp,extra,cid);k=states[task].kernel
                rec['resources'][task]=resources(cp,k,cfg[cid])
            rec['construction'][task]=time.perf_counter()-begin
        except Exception as exc:
            rec['construction'][task]='failed: '+str(exc)[-1500:];rec['status']='failed';save();raise
        save()
    def one(task,phase):
        version,cid=task.split('-');c=cfg[cid]
        if version=='old':
            m=measure(c,states[task],N);row=dict(kernel_s=m['kernel_s'],health=m['health'])
        else:
            row=states[task].run(N,c['seed'])
        row.update(task=task,phase=phase);rec['rows'].append(row);save()
    for t in TRIAL:one(t,'warmup')
    for t in ORDER[0]:one(t,'valid')
    get=lambda t:statistics.median(r['kernel_s'] for r in rec['rows'] if r['task']==t and r['phase']=='valid')
    rec['ratio']={c:get('new-'+c)/get('old-'+c) for c in ('p7A','stack')};rec['status']='complete';save()
    for cid in ('p7A','stack'):
        stem=out/('new_'+cid+'_full')
        cmd=['ncu','--set','full','--target-processes','all','--kernel-name','p8g2_kernel','--launch-count','1','--force-overwrite',
             '--export',str(stem),'--csv','--log-file',str(stem.with_suffix('.csv')),sys.executable,str(Path(__file__).resolve()),
             '--bundles',str(a.bundles),'--water',str(a.water),'--profile',cid]
        try:rec.setdefault('ncu',{})[cid]=subprocess.run(cmd,timeout=1200,capture_output=True,text=True).returncode
        except (OSError,subprocess.TimeoutExpired) as exc:rec.setdefault('ncu',{})[cid]=str(exc)
        if stem.with_suffix('.ncu-rep').exists():
            with stem.with_suffix('.metrics.csv').open('w') as f:subprocess.run(['ncu','--import',str(stem.with_suffix('.ncu-rep')),'--csv'],stdout=f,timeout=300)
        save()
    (out/'G5A2_DONE').write_text('done\n')


if __name__=='__main__':main()
