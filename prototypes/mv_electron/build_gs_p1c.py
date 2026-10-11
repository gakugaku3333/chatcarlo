"""P1c checkpoint adapter; numerical GS definitions unchanged."""
import time
import numpy as np
from .gs_backend import coefficients,from_coefficients
def build(provider,dest):
    from pathlib import Path
    import json
    dest=Path(dest);checkpoint=dest.parent/'gs_checkpoint';checkpoint.mkdir(exist_ok=True)
    start=time.perf_counter();previous_seconds=0.;resumed=False;ts=np.geomspace(.01,20,49)
    ks=np.array([.0005,.001,.003125,.00625,.0125,.025,.05,.1,.2])
    from .single_table import build as single_build
    saved=checkpoint/'single.npz'
    if saved.exists():
        resumed=True;a=np.load(saved);previous_seconds+=float(a['seconds']) if 'seconds' in a.files else 0.;u,rates,single=(a[k] for k in ['u','rates','single']);lookup_diagnostics=json.loads(str(a['info']))
    else:
        section_start=time.perf_counter();u,rates,single,lookup_diagnostics=single_build(provider)
        np.savez(saved,u=u,rates=rates,single=single,info=json.dumps(lookup_diagnostics),seconds=time.perf_counter()-section_start)
    q=np.empty((len(ts),len(ks),len(u)))
    p0s=np.empty((len(ts),len(ks)));g1=np.empty(len(ts));tot=np.empty(len(ts));norms=[];orders=[]
    for i,t in enumerate(ts):
        saved=checkpoint/f'energy_{i:02}.npz'
        if saved.exists():
            resumed=True;a=np.load(saved);previous_seconds+=float(a['seconds']) if 'seconds' in a.files else 0.;q[i]=a['y'];p0s[i]=a['p0'];g1[i]=a['G1'];tot[i]=a['total'];norms.extend(a['norms']);orders.extend(a['orders']);continue
        section_start=time.perf_counter();L=1024
        while True:
            G,total=coefficients(provider,t,L)
            try:rows=[from_coefficients(G,total,k,provider=provider,T=t) for k in ks];break
            except ValueError:
                L*=2
                if L>32768:raise
        for j,(y,cdf,p0,G,total,d) in enumerate(rows):
            q[i,j]=np.interp(u,cdf,y);p0s[i,j]=p0;norms.append(d['norm']);orders.append(L)
        g1[i]=G[1];tot[i]=total
        # Single CDF has its own converged energy grid, from this same provider.
        np.savez(saved,y=q[i],p0=p0s[i],G1=g1[i],total=tot[i],norms=np.array(norms[-len(ks):]),orders=np.array(orders[-len(ks):]),seconds=time.perf_counter()-section_start)
        print(provider.name,'GS',i+1,len(ts),float(t),L,flush=True)
    np.savez(dest,T=ts,K=ks,u=u,y=q,p0=p0s,G1=g1,total=tot,rates=rates,single=single,lookup_diagnostics=__import__('json').dumps(lookup_diagnostics),build_seconds=previous_seconds+time.perf_counter()-start,build_resumed=resumed,normalization=norms,L=orders,backend=provider.name)
    return dest
