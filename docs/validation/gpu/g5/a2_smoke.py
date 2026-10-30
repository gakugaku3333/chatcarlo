"""Small CPU statistics, 100 outer batches; health is a gate, values report-only."""
import argparse
from pathlib import Path
import tempfile
import time
import numpy as np
import a2_common as common
import a2_health
from chatcarlo.gpu import therapy,catalog


def run(work,source=common.SOURCE):
    _,m,a=common.inputs(source);reports={}
    for condition,seed in (('A',20301831),('stack',830101)):
        arrays=a;kwargs={}
        if condition=='A':
            arrays,c=common.water_arrays(a,'A')
            kwargs=dict(field_geometry=common.field_geometry(),half=(c['field_half'],)*2,ssd=c['ssd'])
        cpu=therapy.CPU(work,m,arrays,**kwargs)
        field=[];bins=[];health=[];receipts=[];begin=time.perf_counter()
        for b in range(100):
            therapy.space(work)
            r=cpu.histories(1000,seed,start=b*1000,slots=8)
            h=a2_health.check(r,cpu.g,arrays)
            if condition=='A':a2_health.ledger(r,kwargs['field_geometry'])
            if r['hits'][60:62].any():raise AssertionError('secondary/parent lateral position')
            field.append(r['field']);bins.append(r['d'][:,:cpu.g.nb].sum(axis=0));health.append(h)
            receipts.append(dict(N=1000,seed=seed,start=b*1000,physics_set=cpu.observed,health=h,sha256=catalog.digest(np.r_[r['field'],bins[-1]])))
        report=dict(status='pass',N=100000,seed=seed,seconds=time.perf_counter()-begin,health=health,receipts=receipts,
                    mean_deposited_MeV=float(np.sum(bins)/100000),observed=cpu.observed)
        table=common.SUMMARY/f'smoke_{condition}.npz'
        np.savez_compressed(table,field=np.array(field),bins=np.array(bins),N=np.array(100000),seed=np.array(seed),starts=np.arange(100)*1000)
        report['table_sha256']=catalog.digest(np.array(field));reports[condition]=report
        print(condition,'100000 histories healthy',flush=True)
        del cpu
    common.write(common.SUMMARY/'smoke_cpu.json',dict(status='pass',conditions=reports));return reports


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path);args=p.parse_args()
    run(args.work or Path(tempfile.mkdtemp(prefix='a2-smoke-',dir='/private/tmp')))
