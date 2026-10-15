// All three entry points consume this scheduler, including visit diagnostics.
__device__ bool p5e_next(int n,unsigned long long *counter,const int *order,
                        int mode,int *cursor,int *k){
    unsigned long long j;
    if(mode==1) j=atomicAdd(counter,1ULL);
    else {j=*cursor; *cursor+=gridDim.x*blockDim.x;}
    if(j>=static_cast<unsigned long long>(n))return false;
    *k=mode==2?order[j]:int(j); return true;
}
#ifndef INFINITY
#define INFINITY (__longlong_as_double(0x7ff0000000000000ULL))
#endif

#ifndef P5B_MERGE_FAULT
#define P5B_MERGE_FAULT 0
#endif
extern "C" __global__ void p5e_per_history_kernel(P5B_TABLE_ARGS,
    uint64_t seed,uint64_t start,int n,int maxsteps,
    unsigned long long *counter,const int *order,int mode,P5B_HISTORY_ARGS) {
    P5B_TABLE_INIT; int cursor=blockIdx.x*blockDim.x+threadIdx.x, i;
    while(p5e_next(n,counter,order,mode,&cursor,&i)){
    P5bLocal L;p5b_history(t,seed,start+uint64_t(i),maxsteps,&L);
    p5b_store(&L,i,P5B_HISTORY_PASS);
    }
}
extern "C" __global__ void p5e_run_kernel(P5B_TABLE_ARGS,
    uint64_t seed,uint64_t start,int n,int maxsteps,
    unsigned long long *counter,const int *order,int mode,
    double *batches,double *fine_batches,double *totals,double *maxima,double *brems,
    unsigned long long *violations,unsigned long long *counts,unsigned long long *spectrum,
    unsigned long long *step_batches,int *maxdepth,unsigned long long *nan_count) {
    P5B_TABLE_INIT; int cursor=blockIdx.x*blockDim.x+threadIdx.x, i;
    while(p5e_next(n,counter,order,mode,&cursor,&i)){
    P5bLocal L;p5b_history(t,seed,start+uint64_t(i),maxsteps,&L);
    int row=p5b_row(i,n),drop=-1;
#if P5B_MERGE_FAULT == 1
    for(int j=0;j<150;++j)if(L.batches[j]!=0.)drop=j;
#endif
    for(int j=0;j<150;++j)if(j!=drop)atomicAdd(batches+row*150+j,L.batches[j]);
    for(int j=0;j<600;++j)atomicAdd(fine_batches+row*600+j,L.fine_batches[j]);
    for(int j=0;j<6;++j)atomicAdd(totals+j,L.totals[j]);
    for(int j=0;j<10;++j)atomicAdd(brems+j,L.brems[j]);
    for(int j=0;j<3;++j){
        if(!p5b_nan(L.maxima[j]))atomicMax(reinterpret_cast<unsigned long long*>(maxima+j),
            static_cast<unsigned long long>(__double_as_longlong(L.maxima[j])));
        atomicAdd(violations+j,static_cast<unsigned long long>(L.violations[j]));
    }
    for(int j=0;j<16;++j)atomicAdd(counts+j,static_cast<unsigned long long>(L.counts[j]));
    atomicAdd(spectrum+L.nspec_index,1ULL);
    atomicAdd(step_batches+row,static_cast<unsigned long long>(L.steps));
    atomicMax(maxdepth,L.maxdepth);atomicAdd(nan_count,static_cast<unsigned long long>(p5b_nans(&L)));
    }
}

extern "C" __global__ void p5e_visits_kernel(uint64_t start,int n,
    unsigned long long *counter,const int *order,int mode,
    unsigned long long *visits,uint64_t *global_h,int *rows){
    int cursor=blockIdx.x*blockDim.x+threadIdx.x,k;
    while(p5e_next(n,counter,order,mode,&cursor,&k)){
        atomicAdd(visits+k,1ULL);global_h[k]=start+uint64_t(k);rows[k]=p5b_row(k,n);
    }
}
