"""Compiled rate and single-scatter access to the same selected effective DCS."""
import numpy as np
from numba import njit
from .sampling import bracket
@njit
def table_rates(T,ts,rates):
    grid=ts;shift=0
    if rates.shape[1]==3:grid=rates[:,0];shift=1
    i=np.searchsorted(grid,T)-1;i=max(0,min(i,len(grid)-2))
    a=np.log(T/grid[i])/np.log(grid[i+1]/grid[i])
    return np.exp((1-a)*np.log(rates[i,shift])+a*np.log(rates[i+1,shift])),np.exp((1-a)*np.log(rates[i,shift+1])+a*np.log(rates[i+1,shift+1]))
@njit
def table_single(T,ts,us,ys):
    grid=ts;shift=0
    if ys.shape[1]==len(us)+1:grid=ys[:,0];shift=1
    i=np.searchsorted(grid,T)-1;i=max(0,min(i,len(grid)-2));a=np.log(T/grid[i])/np.log(grid[i+1]/grid[i]);h,c=bracket(us,np.random.random())
    return 1-((1-a)*((1-c)*ys[i,h+shift]+c*ys[i,h+shift+1])+a*((1-c)*ys[i+1,h+shift]+c*ys[i+1,h+shift+1]))
