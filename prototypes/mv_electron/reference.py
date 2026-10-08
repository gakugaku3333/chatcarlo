"""T3(i) fixed-energy single-collision reference; full loss reference uses transport's event ledger."""
import numpy as np
from numba import njit
from .physics import *
from .sampling import gs_mu,gs_mu_cached
from .backend_sampling import table_rates,table_single
@njit
def fixed(T,path,n,seed,condensed,K,ts,ks,us,ys,backend_rates=None,backend_single=None):
    np.random.seed(seed);out=np.zeros((n,4))
    if backend_rates is None:
        eta,B,total,G=elastic(T);rate=np.sum(total)
    else:
        eta=np.zeros(2);total=np.zeros(2);rate,G=table_rates(T,ts,backend_rates)
    for i in range(n):
        x=0.;y=0.;z=0.;ux=0.;uy=0.;uz=1.;rem=path
        while rem>0:
            if condensed:
                length=min(K/G,rem);u=np.random.random();s1=u*length;s2=length-s1
                x+=s1*ux;y+=s1*uy;z+=s1*uz
                mu=gs_mu(T,G*length,ts,ks,us,ys) if backend_rates is None else gs_mu_cached(T,G*length,ts,ks,us,ys,rate,G)
                ux,uy,uz=rotate(ux,uy,uz,mu,2*np.pi*np.random.random())
                x+=s2*ux;y+=s2*uy;z+=s2*uz;rem-=length
            else:
                step=-np.log(np.random.random())/rate
                length=min(step,rem);x+=length*ux;y+=length*uy;z+=length*uz;rem-=length
                if step<length+1e-14:
                    mu=single_cached(eta,total) if backend_rates is None else table_single(T,ts,us,backend_single)
                    ux,uy,uz=rotate(ux,uy,uz,mu,2*np.pi*np.random.random())
        out[i,0]=x*x+y*y;out[i,1]=z;out[i,2]=uz;out[i,3]=(3*uz*uz-1)/2
    return out
