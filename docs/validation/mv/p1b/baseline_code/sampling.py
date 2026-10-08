"""Numba scalar inverse GS CDF interpolation, with analytic zero-collision mass."""
import numpy as np
from numba import njit
from .physics import elastic_rates
@njit
def bracket(grid,x):
    i=np.searchsorted(grid,x)-1;i=max(0,min(i,len(grid)-2))
    return i,(x-grid[i])/(grid[i+1]-grid[i])
@njit
def gs_mu_cached(T,K,ts,ks,us,ys,total,G1):
    p0=np.exp(-K/G1*total)
    if np.random.random()<p0:return 1.
    u=np.random.random();i=np.searchsorted(ts,T)-1;i=max(0,min(i,len(ts)-2));a=np.log(T/ts[i])/np.log(ts[i+1]/ts[i]);j,b=bracket(ks,K);h,c=bracket(us,u)
    y00=(1-c)*ys[i,j,h]+c*ys[i,j,h+1]
    y01=(1-c)*ys[i,j+1,h]+c*ys[i,j+1,h+1]
    y10=(1-c)*ys[i+1,j,h]+c*ys[i+1,j,h+1]
    y11=(1-c)*ys[i+1,j+1,h]+c*ys[i+1,j+1,h+1]
    return 1-((1-a)*((1-b)*y00+b*y01)+a*((1-b)*y10+b*y11))
@njit
def sample_moments(T,K,n,seed,ts,ks,us,ys):
    np.random.seed(seed);m=np.empty(n)
    for i in range(n):m[i]=gs_mu(T,K,ts,ks,us,ys)
    return m

@njit
def gs_mu(T,K,ts,ks,us,ys):
    total,G1=elastic_rates(T)
    return gs_mu_cached(T,K,ts,ks,us,ys,total,G1)
