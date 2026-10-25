import numpy as np,os,json
from prototypes.mv_electron.gpu.p9c_stats import Moments
RUNS=[os.path.expanduser(f'~/.cache/chatcarlo/p9c2/{d}/prostate.npz') for d in os.environ.get('P9C_RUNS','prod_prostate,prod_prostate2').split(',')]
def load(key,shape):
    m=Moments(shape);n=0;ids=[]
    for p in RUNS:
        with np.load(p) as f:
            k=len(f['completed']);assert k>=99 and f['completed'].tolist()==list(range(k));m.values=m.values+f[key];n+=k;ids.append(json.loads(str(f['identity']))['seed'])
    assert len(set(ids))==len(ids),ids
    m.n=n;return m
