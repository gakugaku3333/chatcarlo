"""Bounded spawn batches, parent-only atomic saves and deterministic reduction."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import time
import numpy as np
from . import run_p3a as a, run_p3r as r
from .batch_kernels_p3u import KERNELS

MAX_WORKERS=2
TIMING_KEYS=('seconds','batch_seconds','wall_seconds','workers','fresh','max_rss')
THREAD_KEYS=('NUMBA_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
SOURCES=tuple(dict.fromkeys(tuple(x for x in r.SOURCES if x.endswith('.py'))+
    ('photon_transport_p3t.py','batch_runner_p3u.py','batch_kernels_p3u.py')))


def source_hashes():
    return {str((a.CODE/x).relative_to(a.ROOT)):a.sha(a.CODE/x) for x in SOURCES}


def batch_seeds(seed,nbatch):
    return [int(x.generate_state(1)[0]) for x in np.random.SeedSequence(seed).spawn(nbatch)]


def same(left,right,skip=TIMING_KEYS):
    if set(left)!=set(right):return False
    for k in set(left)-set(skip):
        x=np.asarray(left[k]);y=np.asarray(right[k])
        if x.dtype!=y.dtype or x.shape!=y.shape or x.tobytes()!=y.tobytes():return False
    return True


def validate_batch(batch,kernel,n,seed):
    kernel=KERNELS[kernel] if isinstance(kernel,str) else kernel
    try:
        if set(batch)!=set(kernel.keys+('N','seed','seconds')):raise ValueError('keys')
        if not all(np.issubdtype(np.asarray(x).dtype,np.number) and np.all(np.isfinite(x))
                   for x in batch.values()):raise ValueError('non-finite or nonnumeric output')
        for key,expected in (('N',n),('seed',seed)):
            if np.asarray(batch[key]).shape!=() or batch[key]!=expected:raise ValueError(key)
        if np.asarray(batch['seconds']).shape!=() or batch['seconds']<=0:raise ValueError('seconds')
        if not kernel.healthy(batch):raise ValueError('unhealthy output')
    except (ValueError,TypeError,KeyError,IndexError) as exc:
        raise RuntimeError('invalid batch: '+str(exc)) from exc


def worker(number,kernel,params,n,seed):
    k=KERNELS[kernel];t=time.perf_counter()
    values=k.run(n,seed,params)
    fingerprint=k.fingerprint(params)
    if (kernel=='fake' and 'worker_fingerprint' in params and
        os.environ.get('CHATCARLO_P3U_FAKE_FAULTS','1')=='1'):
        fingerprint['input']=params['worker_fingerprint']
    return number,values,time.perf_counter()-t,fingerprint,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def integer(x):
    return isinstance(x,(int,np.integer)) and not isinstance(x,(bool,np.bool_))


def arguments(kernel,params,N,workers,nbatch,stop_after):
    if not integer(workers) or not 1<=workers<=MAX_WORKERS:raise ValueError('workers must be 1 or 2')
    if not integer(N) or N<=0 or not integer(nbatch) or nbatch<=0 or N%nbatch:
        raise ValueError('positive N divisible by positive nbatch required')
    if stop_after is not None and (not integer(stop_after) or not 1<=stop_after<=nbatch):
        raise ValueError('stop_after outside 1..nbatch')
    if not isinstance(kernel,str) or kernel not in KERNELS:raise ValueError('unknown kernel')
    KERNELS[kernel].check_params(params,N//nbatch)


def read_batch(path,kernel,n,seed):
    try:batch=a.load(path)
    except Exception as exc:raise RuntimeError('cannot read batch: '+str(exc)) from exc
    validate_batch(batch,kernel,n,seed)
    return batch


def aggregate(work,kernel,N,seed,seeds):
    rows=[read_batch(work/f'batch_{b:03}.npz',kernel,N//len(seeds),s) for b,s in enumerate(seeds)]
    for row in rows[1:]:
        for key in rows[0]:
            if row[key].dtype!=rows[0][key].dtype or row[key].shape!=rows[0][key].shape:
                raise RuntimeError('batch dtype or shape differs: '+key)
    return dict(**{key:np.stack([row[key] for row in rows]) for key in KERNELS[kernel].keys},
                N=np.asarray(N),seed=np.asarray(seed),nbatch=np.asarray(len(seeds)),
                batch_seeds=np.asarray(seeds),batch_seconds=np.array([float(row['seconds']) for row in rows]))


def run_batches(work,kernel,params,N,seed,workers,nbatch=100,stop_after=None):
    started=time.perf_counter()
    arguments(kernel,params,N,workers,nbatch,stop_after)
    work=Path(work);k=KERNELS[kernel];n=N//nbatch
    with a.job_lock(work):
        for key in THREAD_KEYS:
            if os.environ.get(key,'1')!='1':raise ValueError(key+' must be 1')
        for key in THREAD_KEYS:os.environ.setdefault(key,'1')
        config=a.plain(dict(kernel=kernel,params=params,N=N,seed=seed,nbatch=nbatch,
                            source_hashes=source_hashes(),fingerprint=k.fingerprint(params)))
        target=work/'config.json'
        new_run=not target.exists()
        if not new_run:
            if json.loads(target.read_text())!=config:raise RuntimeError('saved run config mismatch')
        else:a.write(target,config)
        if (work/'terminal.json').exists():raise RuntimeError('run previously failed; no rerun permitted')
        seeds=batch_seeds(seed,nbatch)

        def verify():
            if source_hashes()!=config['source_hashes']:raise RuntimeError('source hashes changed; stop')
            if a.plain(k.fingerprint(params))!=config['fingerprint']:raise RuntimeError('input fingerprint changed; stop')

        result=work/'result.npz'
        if result.exists():
            verify();expected=aggregate(work,kernel,N,seed,seeds)
            try:saved=a.load(result)
            except Exception as exc:raise RuntimeError('cannot read result: '+str(exc)) from exc
            if set(saved)!=set(expected)|{'wall_seconds','workers','fresh','max_rss'}:
                raise RuntimeError('invalid result keys')
            if not same(saved,dict(expected,**{key:saved[key] for key in ('wall_seconds','workers','fresh','max_rss')})):
                raise RuntimeError('completed result differs from batches')
            return result
        limit=nbatch if stop_after is None else stop_after
        missing=[]
        for b,s in enumerate(seeds):
            path=work/f'batch_{b:03}.npz'
            if path.exists():read_batch(path,kernel,n,s)
            elif b<limit:missing.append(b)
        fresh=new_run and len(missing)==nbatch and stop_after is None
        rss=[]
        if missing:
            pool=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'))
            pending={};todo=iter(missing)
            def submit():
                b=next(todo,None)
                if b is not None:pending[pool.submit(worker,b,kernel,params,n,seeds[b])]=b
            try:
                for _ in range(workers):submit()
                while pending:
                    done,_=wait(pending,return_when=FIRST_COMPLETED)
                    for future in sorted(done,key=lambda f:pending[f]):
                        b=pending.pop(future)
                        number,values,seconds,fingerprint,max_rss=future.result()
                        if number!=b:raise RuntimeError('worker batch number mismatch')
                        if a.plain(fingerprint)!=config['fingerprint']:raise RuntimeError('worker fingerprint mismatch')
                        if source_hashes()!=config['source_hashes']:raise RuntimeError('source hashes changed; stop')
                        batch=dict(values,N=np.asarray(n),seed=np.asarray(seeds[b]),seconds=np.asarray(seconds))
                        if not a.finite(batch) or not k.healthy(batch):
                            reason='non-finite output' if not a.finite(batch) else 'unhealthy output'
                            a.write(work/'terminal.json',dict(batch=b,reason=reason))
                            raise RuntimeError(reason)
                        validate_batch(batch,kernel,n,seeds[b])
                        a.save(work/f'batch_{b:03}.npz',**batch)
                        rss.append(max_rss)
                        submit()
            except Exception as exc:
                raise RuntimeError('batch run interrupted: '+str(exc)) from exc
            finally:pool.shutdown(wait=True)
        if stop_after is not None:return None
        verify();values=aggregate(work,kernel,N,seed,seeds)
        values.update(wall_seconds=time.perf_counter()-started,workers=workers,fresh=fresh,
                      max_rss=max(rss,default=0))
        try:
            a.save(result,**values)
            # Include the completed atomic result save in the recorded wall time.
            values['wall_seconds']=time.perf_counter()-started
            a.save(result,**values)
        except Exception as exc:raise RuntimeError('result save interrupted: '+str(exc)) from exc
        return result
