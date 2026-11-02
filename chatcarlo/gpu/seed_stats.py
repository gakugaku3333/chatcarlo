"""Five paired moments; only complete seven-door pairs may be committed."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

class Moments:
    def __init__(self,shape):self.n=0;self.values=np.zeros((5,*shape))
    def add(self,a,b):
        a=np.asarray(a);b=np.asarray(b)
        if a.shape!=self.values.shape[1:] or b.shape!=a.shape or not np.isfinite([a,b]).all():raise ValueError('moment shape/nonfinite')
        self.values+=np.stack((a,b,a*a,b*b,a*b));self.n+=1
    def result(self):
        if not self.n:raise ValueError('no batches')
        sa,sb,saa,sbb,sab=self.values;a=sa/self.n;b=sb/self.n
        nan=np.full(a.shape,np.nan)
        if self.n<2:va=vb=cov=nan
        else:
            va=np.maximum(0.,(saa-sa*a)/(self.n-1)/self.n);vb=np.maximum(0.,(sbb-sb*b)/(self.n-1)/self.n);cov=(sab-sa*b)/(self.n-1)/self.n
        with np.errstate(divide='ignore',invalid='ignore'):
            ratio=np.where(b!=0,a/b,np.nan)
            paired=np.maximum(0.,va+ratio*ratio*vb-2*ratio*cov)
            independent=va+ratio*ratio*vb
            rho=np.where((va>0)&(vb>0),cov/np.sqrt(va*vb),np.nan)
            ratio_sem=np.where(b!=0,np.sqrt(paired)/abs(b),np.nan)
            unpaired=np.where(b!=0,np.sqrt(independent)/abs(b),np.nan)
            change=np.where(unpaired>0,ratio_sem/unpaired,np.nan)
        return dict(mean_with=a,mean_without=b,sem_with=np.sqrt(va),sem_without=np.sqrt(vb),difference=a-b,difference_sem=np.sqrt(np.maximum(0.,va+vb-2*cov)),ratio=ratio,ratio_sem=ratio_sem,correlation=rho,unpaired_ratio_sem=unpaired,sem_change=change,covariance=cov)

class Checkpoint:
    def __init__(self,path,identity,shapes):
        self.limit=identity.get("batches",50)
        self.path=Path(path);self.identity=json.dumps(identity,sort_keys=True,allow_nan=False);self.completed=[]
        self.moments={k:Moments(tuple(s)) for k,s in shapes.items()}
        if self.path.exists():
            with np.load(self.path,allow_pickle=False) as f:
                if f['identity'].item()!=self.identity or set(f.files)!={'identity','completed',*shapes}:raise ValueError('checkpoint identity/keys')
                self.completed=f['completed'].tolist()
                if len(self.completed)>self.limit or self.completed!=list(range(len(self.completed))):raise ValueError('checkpoint batches')
                for k,m in self.moments.items():
                    if f[k].shape!=m.values.shape or not np.isfinite(f[k]).all():raise ValueError('checkpoint moments')
                    m.values=f[k].copy();m.n=len(self.completed)
    def commit(self,number,pairs,doors):
        if type(number) is not int or not 0<=number<self.limit:raise ValueError('batch range')
        if number in self.completed:raise ValueError('completed batch replay')
        if number!=len(self.completed) or doors!=set((f,s) for f in range(7) for s in (True,False)) or set(pairs)!=set(self.moments):raise ValueError('incomplete seven-field pair')
        # Validate everything before mutation; publish all five moments atomically.
        updated={}
        for k,m in self.moments.items():
            new=Moments(m.values.shape[1:]);new.values=m.values.copy();new.n=m.n;new.add(*pairs[k]);updated[k]=new
        self.path.parent.mkdir(parents=True,exist_ok=True);tmp=self.path.with_name(self.path.name+'.pending')
        try:
            with tmp.open('wb') as f:
                np.savez(f,identity=np.array(self.identity),completed=np.array([*self.completed,number],np.int64),**{k:m.values for k,m in updated.items()});f.flush();os.fsync(f.fileno())
            tmp.replace(self.path)
        finally:
            if tmp.exists():tmp.unlink()
        self.moments=updated;self.completed.append(number)
    @property
    def digest(self):return hashlib.sha256(self.path.read_bytes()).hexdigest()

def normalize(fields,counts):
    fields=np.asarray(fields);counts=np.asarray(counts)
    if len(fields)!=7 or counts.shape!=(7,) or np.any(counts<=0):raise ValueError('seven accepted-photon counts')
    return fields.sum(axis=0)/counts.sum()

class PairStore:
    """One atomic checkpoint: paired grids, compact rows and native-call receipts.

    A killed partial door never becomes visible as an accepted batch.
    """
    def __init__(self,path,identity,shapes):
        self.path=Path(path);self.identity=identity;self.rows=[];self.receipts=[]
        self.moments={k:Moments(shape) for k,shape in shapes.items()}
        if self.path.exists():
            with np.load(self.path,allow_pickle=False) as f:
                if json.loads(str(f['identity']))!=identity:raise ValueError('checkpoint identity')
                values=f['rows']
                if values.ndim!=3 or values.shape[1:]!=(2,3) or not np.isfinite(values).all() or np.any(values<0):raise ValueError('checkpoint rows')
                self.rows=values.tolist();self.receipts=json.loads(str(f['receipts']))
                if f['completed'].tolist()!=list(range(len(self.rows))) or len(self.rows)>identity['batches'] or len(self.receipts)!=len(self.rows):raise ValueError('checkpoint batches')
                if set(f.files)!={'identity','rows','receipts','completed',*shapes}:raise ValueError('checkpoint members')
                for k,m in self.moments.items():
                    if f[k].shape!=m.values.shape or not np.isfinite(f[k]).all():raise ValueError('checkpoint moments')
                    m.values=f[k].copy();m.n=len(self.rows)
    @property
    def completed(self):return list(range(len(self.rows)))
    def commit(self,number,pairs,doors,row,receipts):
        if type(number) is not int or not 0<=number<self.identity['batches'] or number!=len(self.rows):raise ValueError('batch range/replay')
        if doors!=set((f,s) for f in range(7) for s in (True,False)) or len(receipts)!=14 or set(pairs)!=set(self.moments):raise ValueError('incomplete seven-field pair')
        row=np.asarray(row,float)
        if row.shape!=(2,3) or not np.isfinite(row).all() or np.any(row<0):raise ValueError('batch density rows')
        updated={}
        for k,m in self.moments.items():
            u=Moments(m.values.shape[1:]);u.values=m.values.copy();u.n=m.n;u.add(*pairs[k]);updated[k]=u
        from .therapy import space
        space(self.path.parent,sum(m.values.nbytes for m in updated.values())+16*1024**2)
        self.path.parent.mkdir(parents=True,exist_ok=True);tmp=self.path.with_suffix('.pending')
        try:
            with tmp.open('wb') as f:
                np.savez_compressed(f,identity=json.dumps(self.identity,sort_keys=True),rows=np.array([*self.rows,row.tolist()]),receipts=json.dumps([*self.receipts,receipts],sort_keys=True),completed=np.arange(number+1),**{k:m.values for k,m in updated.items()});f.flush();os.fsync(f.fileno())
            tmp.replace(self.path)
        finally:
            if tmp.exists():tmp.unlink()
        self.moments=updated;self.rows.append(row.tolist());self.receipts.append(receipts)
